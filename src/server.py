from flask import Flask, request
import time
import os
import shutil
import glob
import logging
import traceback
import threading


import image_set_aux
import image_set_actions as isa
import extract_patches as ep


from io_utils import json_io, w3c_io
from models.yolov4 import yolov4_image_set_driver



from lock_queue import LockQueue


MAX_STORED_SCHEDULER_UPDATES = 10

cv = threading.Condition()
sch_ctx = {}


app = Flask(__name__)


@app.route(os.environ.get("CC_PATH") + '/add_request', methods=['POST'])
def add_request():
    logger = logging.getLogger(__name__)
    logger.info("POST to add_request")
    content_type = request.headers.get('Content-Type')

    if (content_type == 'application/json'):
        json_request = request.json
        print("request", json_request)
        req_type = json_request["request_type"]
        item = {
            "username": json_request["username"],
            "farm_name": json_request["farm_name"],
            "field_name": json_request["field_name"],
            "mission_date": json_request["mission_date"]
        }
        ok = True
        if req_type == "restart":
            sch_ctx["restart_queue"].enqueue(item)
        elif req_type == "prediction":
            sch_ctx["prediction_queue"].enqueue(item)
        elif req_type == "training":
            sch_ctx["training_queue"].enqueue(item)
        else:
            ok = False

        with cv:
            cv.notify_all()

        if ok:
            return {"message": "ok"}
        else:
            return {"message": "bad_request"}

    else:
        return {"message": 'Content-Type not supported!'}
        


def set_scheduler_status(username, farm_name, field_name, mission_date, status, extra_items={}):

    scheduler_status_path = os.path.join("usr", "shared", "scheduler_status.json")
    scheduler_status = json_io.load_json(scheduler_status_path)
    # updates = scheduler_status["updates"]
    # if len(updates) > MAX_STORED_SCHEDULER_UPDATES:
    #     updates.pop(0)
        
    update_num = scheduler_status["update_num"] + 1
    scheduler_status = {
        "update_num": update_num,
        "username": username,
        "farm_name": farm_name,
        "field_name": field_name,
        "mission_date": mission_date,
        "status": status,
        "timestamp": int(time.time())
    }

    for k, v in extra_items.items():
        scheduler_status[k] = v
    # scheduler_status["update_num"] = scheduler_status["update_num"] + 1
    # scheduler_status["username"] = username
    # scheduler_status["farm_name"] = farm_name
    # scheduler_status["field_name"] = field_name
    # scheduler_status["mission_date"] = mission_date
    # scheduler_status["status"] = status
    json_io.save_json(scheduler_status_path, scheduler_status)

    isa.emit_scheduler_status_change(scheduler_status)



    

def needs_training(image_set_dir):
    annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
    annotations = json_io.load_json(annotations_path)
    num_training_images = 0
    for image_name in annotations.keys():
        if annotations[image_name]["status"] == "completed_for_training":
            num_training_images += 1
    
    model_dir = os.path.join(image_set_dir, "model")
    status_path = os.path.join(model_dir, "status.json")
    status = json_io.load_json(status_path)
    return status["num_images_fully_trained_on"] < num_training_images


def check_train(username, farm_name, field_name, mission_date):
    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
    training_dir = os.path.join(image_set_dir, "model", "training")

    if not needs_training(image_set_dir):
        return

    usr_block_path = os.path.join(training_dir, "usr_block.json")
    sys_block_path = os.path.join(training_dir, "sys_block.json")
    if os.path.exists(usr_block_path) or os.path.exists(sys_block_path):
        return

    restart_req_path = os.path.join(training_dir, "restart_request.json")
    if os.path.exists(restart_req_path):
        return


    sch_ctx["training_queue"].enqueue({
        "username": username,
        "farm_name": farm_name,
        "field_name": field_name,
        "mission_date": mission_date
    })


def process_train(item):

    logger = logging.getLogger(__name__)

    username = item["username"]
    farm_name = item["farm_name"]
    field_name = item["field_name"]
    mission_date = item["mission_date"]

    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")
    training_dir = os.path.join(model_dir, "training")

    try:
        upload_status_path = os.path.join(image_set_dir, "upload_status.json")
        upload_status = json_io.load_json(upload_status_path)
        if upload_status["status"] == "uploaded":

            if needs_training(image_set_dir):

                usr_block_path = os.path.join(training_dir, "usr_block.json")
                sys_block_path = os.path.join(training_dir, "sys_block.json")
                if os.path.exists(usr_block_path) or os.path.exists(sys_block_path):
                    return False

                restart_req_path = os.path.join(training_dir, "restart_request.json")
                if os.path.exists(restart_req_path):
                    # restart_model(username, farm_name, field_name, mission_date)
                    return False


                annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
                annotations = w3c_io.load_annotations(annotations_path, {"plant": 0})

                training_image_names = []
                for image_name in annotations.keys():
                    if annotations[image_name]["status"] == "completed_for_training":
                        training_image_names.append(image_name) 

                changed_training_image_names = ep.update_patches(image_set_dir, annotations, training_image_names)

                if len(changed_training_image_names) > 0:
                    image_set_aux.update_training_tf_records(image_set_dir, changed_training_image_names, annotations)
                    image_set_aux.reset_loss_record(image_set_dir)

                if os.path.exists(usr_block_path) or os.path.exists(sys_block_path):
                    return False
                if os.path.exists(restart_req_path):
                    return False


                logging.info("Starting to train {}".format(item))
                set_scheduler_status(username, farm_name, field_name, mission_date, isa.TRAINING)
                
                
                training_finished = yolov4_image_set_driver.train(image_set_dir, sch_ctx)
                if training_finished:

                    status_path = os.path.join(model_dir, "status.json")
                    status = json_io.load_json(status_path)
                    status["num_images_fully_trained_on"] = len(training_image_names)
                    json_io.save_json(status_path, status)

                    set_scheduler_status(username, farm_name, field_name, mission_date, isa.FINISHED_TRAINING)

                    logger.info("Finished training {}".format(item))

                    return False

                else:
                    return True

    except Exception as e:
        trace = traceback.format_exc()
        logger.error("Exception occurred in process_train")
        logger.error(e)
        logger.error(trace)

        try:
            json_io.save_json(sys_block_path, {"error_message": str(e)})

            set_scheduler_status(username, farm_name, field_name, mission_date, isa.FINISHED_TRAINING,
                                 extra_items={"error_setting": "training", "error_message": str(e)})
        except Exception as e:
            trace = traceback.format_exc()
            logger.error("Exception occurred while handling original exception")
            logger.error(e)
            logger.error(trace)

        

        





def check_predict(username, farm_name, field_name, mission_date):

    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")
    prediction_dir = os.path.join(model_dir, "prediction")
    prediction_requests_dirs = [
        os.path.join(prediction_dir, "image_requests"),
        os.path.join(prediction_dir, "image_set_requests", "pending")
    ]

    for prediction_requests_dir in prediction_requests_dirs:
        prediction_request_paths = glob.glob(os.path.join(prediction_requests_dir, "*.json"))
        #for prediction_request_path in prediction_request_paths:
        if len(prediction_request_paths) > 0:
            sch_ctx["prediction_queue"].enqueue({
                "username": username,
                "farm_name": farm_name,
                "field_name": field_name,
                "mission_date": mission_date
            })
            return


def predict_on_images(username, farm_name, field_name, mission_date, image_names, save_result):

    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)


    annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
    annotations_json = json_io.load_json(annotations_path)
    annotations = w3c_io.convert_json_annotations(annotations_json, {"plant": 0})

    training_image_names = []
    for image_name in annotations.keys():
        if annotations[image_name]["status"] == "completed_for_training":
            training_image_names.append(image_name) 

    # first make sure that training records are up to date, so that if the inference
    # request is changing the patch data for a training image we will reset the loss record and the
    # model will train later
    changed_training_image_names = ep.update_patches(image_set_dir, annotations, training_image_names)
    if len(changed_training_image_names) > 0:
        image_set_aux.update_training_tf_records(image_set_dir, changed_training_image_names, annotations)
        image_set_aux.reset_loss_record(image_set_dir)

    ep.update_patches(image_set_dir, annotations, image_names=image_names)

    image_set_aux.update_prediction_tf_records(image_set_dir, image_names=image_names)
    
    return yolov4_image_set_driver.predict(image_set_dir, annotations_json, annotations, image_names=image_names, save_result=save_result)


def process_predict(item):

    logger = logging.getLogger(__name__)

    username = item["username"]
    farm_name = item["farm_name"]
    field_name = item["field_name"]
    mission_date = item["mission_date"]

    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)

    model_dir = os.path.join(image_set_dir, "model")

    prediction_dir = os.path.join(model_dir, "prediction")
    prediction_requests_dirs = [
        os.path.join(prediction_dir, "image_requests"),
        os.path.join(prediction_dir, "image_set_requests", "pending")
    ]

    for prediction_requests_dir in prediction_requests_dirs:
        prediction_request_paths = glob.glob(os.path.join(prediction_requests_dir, "*.json"))
        try:
            while len(prediction_request_paths) > 0:
            
                
                prediction_request_path = prediction_request_paths[0]
                request = json_io.load_json(prediction_request_path)

                logger.info("Starting to predict for: {}".format(item))
                set_scheduler_status(username, farm_name, field_name, mission_date, isa.PREDICTING)

                end_time = predict_on_images(
                        username,
                        farm_name,
                        field_name,
                        mission_date,
                        request["image_names"],
                        request["save_result"]
                )

                request["end_time"] = end_time
                os.remove(prediction_request_path)
                if request["save_result"]:
                    json_io.save_json(os.path.join(model_dir, "results", str(end_time), "request.json"), request)

                logger.info("Finished predicting for {}".format(item))
                set_scheduler_status(username, farm_name, field_name, mission_date, isa.FINISHED_PREDICTING, 
                                extra_items={"prediction_image_names": ",".join(request["image_names"])})

                if request["save_result"]:
                    isa.emit_results_change(username, farm_name, field_name, mission_date)

                prediction_request_paths = glob.glob(os.path.join(prediction_requests_dir, "*.json"))
                


        except Exception as e:

            trace = traceback.format_exc()
            logger.error("Exception occurred in process_predict")
            logger.error(e)
            logger.error(trace)

            try:
                if os.path.exists(prediction_request_path):
                    os.remove(prediction_request_path)
                if os.path.basename(prediction_requests_dir) == "pending":
                    request["aborted_time"] = int(time.time())
                    request["error_message"] = str(e)
                    request["error_info"] = str(trace)
                    json_io.save_json(
                        os.path.join(prediction_dir, "image_set_requests", "aborted", os.path.basename(prediction_request_path)),
                        request)


                set_scheduler_status(username, farm_name, field_name, mission_date, isa.FINISHED_PREDICTING, 
                            extra_items={"error_setting": "prediction", "error_message": str(e)})

                if request["save_result"]:
                    isa.emit_results_change(username, farm_name, field_name, mission_date)


            except Exception as e:
                trace = traceback.format_exc()
                logger.error("Exception occurred while handling original exception")
                logger.error(e)
                logger.error(trace)
                




            # if save_result:
            #     json_io.save_json(os.path.join(model_dir, "results", str(end_time), "request.json"), request)


def check_restart(username, farm_name, field_name, mission_date):

    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
    restart_req_path = os.path.join(image_set_dir, "model", "training", "restart_request.json")
    if os.path.exists(restart_req_path):
        sch_ctx["restart_queue"].enqueue({
            "username": username,
            "farm_name": farm_name,
            "field_name": field_name,
            "mission_date": mission_date
        })



def process_restart(item):

    logger = logging.getLogger(__name__)

    username = item["username"]
    farm_name = item["farm_name"]
    field_name = item["field_name"]
    mission_date = item["mission_date"]

    image_set_dir = os.path.join("usr", "data", username, "image_sets", farm_name, field_name, mission_date)
    model_dir = os.path.join(image_set_dir, "model")

    restart_req_path = os.path.join(model_dir, "training", "restart_request.json")
    if os.path.exists(restart_req_path):

        logger.info("Restarting {}".format(item))
        set_scheduler_status(username, farm_name, field_name, mission_date, isa.RESTARTING)

        
        loss_record_path = os.path.join(model_dir, "training", "loss_record.json")

        loss_record = {
            "training_loss": { "values": [],
                            "best": 100000000,
                            "epochs_since_improvement": 100000000}, 
            "validation_loss": {"values": [],
                                "best": 100000000,
                                "epochs_since_improvement": 100000000},
        }
        json_io.save_json(loss_record_path, loss_record)

        
        weights_dir = os.path.join(model_dir, "weights")

        default_weights_path = os.path.join("usr", "shared", "weights", "default_weights.h5")

        shutil.copy(default_weights_path, os.path.join(weights_dir, "cur_weights.h5"))
        shutil.copy(default_weights_path, os.path.join(weights_dir, "best_weights.h5"))


        results_dir = os.path.join(model_dir, "results")
        results = glob.glob(os.path.join(results_dir, "*"))
        for result in results:
            shutil.rmtree(result)

        prediction_dir = os.path.join(model_dir, "prediction")
        shutil.rmtree(prediction_dir)
        os.makedirs(prediction_dir)

        os.makedirs(os.path.join(prediction_dir, "image_requests"))
        os.makedirs(os.path.join(prediction_dir, "images"))
        image_set_requests = os.path.join(prediction_dir, "image_set_requests")
        os.makedirs(image_set_requests)
        os.makedirs(os.path.join(image_set_requests, "aborted"))
        os.makedirs(os.path.join(image_set_requests, "pending"))


        training_dir = os.path.join(model_dir, "training")
        training_records_dir = os.path.join(training_dir, "training_tf_records")
        if os.path.exists(training_records_dir):
            shutil.rmtree(training_records_dir)
            os.makedirs(training_records_dir)
        validation_records_dir = os.path.join(training_dir, "validation_tf_records")
        if os.path.exists(validation_records_dir):
            shutil.rmtree(validation_records_dir)
            os.makedirs(validation_records_dir)

        patches_dir = os.path.join(image_set_dir, "patches")
        shutil.rmtree(patches_dir)
        os.makedirs(patches_dir)



        annotations_path = os.path.join(image_set_dir, "annotations", "annotations_w3c.json")
        annotations = json_io.load_json(annotations_path)
        for image_name in annotations.keys():
            if annotations[image_name]["status"] == "completed_for_training":
                annotations[image_name]["status"] = "completed_for_testing"
        json_io.save_json(annotations_path, annotations)


        status_path = os.path.join(model_dir, "status.json")
        status = json_io.load_json(status_path)
        status["num_images_fully_trained_on"] = 0
        json_io.save_json(status_path, status)

        os.remove(restart_req_path)

        isa.emit_results_change(username, farm_name, field_name, mission_date)

        return


def one_pass():

    for username_path in glob.glob(os.path.join("usr", "data", "*")):
        username = os.path.basename(username_path)
        for user_dir in glob.glob(os.path.join(username_path, "*")):
            if os.path.basename(user_dir) == "image_sets":
                for farm_path in glob.glob(os.path.join(username_path, "image_sets", "*")):
                    farm_name = os.path.basename(farm_path)
                    for field_path in glob.glob(os.path.join(farm_path, "*")):
                        field_name = os.path.basename(field_path)
                        for mission_path in glob.glob(os.path.join(field_path, "*")):
                            mission_date = os.path.basename(mission_path)

                            check_restart(username, farm_name, field_name, mission_date)
                            check_predict(username, farm_name, field_name, mission_date)
                            check_train(username, farm_name, field_name, mission_date)



def sweep():

    logger = logging.getLogger(__name__)
    logger.info("Sweeper started")

    while True:

        if (sch_ctx["restart_queue"].size() == 0 and sch_ctx["prediction_queue"].size() == 0) and sch_ctx["training_queue"].size() == 0:

            logger.info("Performing sweep")
            try:
                one_pass()
            except Exception as e:
                trace = traceback.format_exc()
                logger.error("Exception occurred while draining queue")
                logger.error(e)
                logger.error(trace)



            if an_item_is_available():
                logger.info("Sweeper is notifying")
                with cv:
                    cv.notify_all()

            logger.info("Sweep complete")

        
        time.sleep(60 * 60)



def drain():
    logger = logging.getLogger(__name__)
    logger.info("Drain has started")

    while True:
        try:
            restart_queue_size = sch_ctx["restart_queue"].size()
            while restart_queue_size > 0:
                item = sch_ctx["restart_queue"].dequeue()
                process_restart(item)
                restart_queue_size = sch_ctx["restart_queue"].size()


            pred_queue_size = sch_ctx["prediction_queue"].size()
            while pred_queue_size  > 0:
                item = sch_ctx["prediction_queue"].dequeue()
                process_predict(item)
                pred_queue_size = sch_ctx["prediction_queue"].size()


            train_queue_size = sch_ctx["training_queue"].size()
            if train_queue_size > 0:
                item = sch_ctx["training_queue"].dequeue()
                re_enqueue = process_train(item)
                if re_enqueue:
                    sch_ctx["training_queue"].enqueue(item)
        
        except Exception as e:
            trace = traceback.format_exc()
            logger.error("Exception occurred while draining queue")
            logger.error(e)
            logger.error(trace)


        #time.sleep(5)    

        if (sch_ctx["restart_queue"].size() == 0 and sch_ctx["prediction_queue"].size() == 0) and sch_ctx["training_queue"].size() == 0:

            logger.info("Drain has finished")
            set_scheduler_status("---", "---", "---", "---", isa.IDLE)
            return

             
def an_item_is_available():
    return (sch_ctx["restart_queue"].size() > 0 or sch_ctx["prediction_queue"].size() > 0) or sch_ctx["training_queue"].size() > 0


def work():
    logger = logging.getLogger(__name__)
    logger.info("Worker thread started")


    while True:
        drain()
        logger.info("Worker waiting")
        with cv:
            #while not an_item_is_available():
            #    cv.wait()
            cv.wait_for(an_item_is_available)
        logger.info("Worker woken up")



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    sch_ctx["restart_queue"] = LockQueue()
    sch_ctx["prediction_queue"] = LockQueue()
    sch_ctx["training_queue"] = LockQueue()


    one_pass()

    worker = threading.Thread(name="worker", target=work)
    worker.start()


    sweeper = threading.Thread(name="sweeper", target=sweep)
    sweeper.start()


    app.run(host=os.environ.get("CC_IP"), port=os.environ.get("CC_PY_PORT"))