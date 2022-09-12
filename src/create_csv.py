import argparse
import os
import shutil
import glob
import time


from models.common import inference_metrics

RESULT_LIFETIME = 2 * 3600

def remove_old_results(results_dir):

    retrieval_dir = os.path.join(results_dir, "retrieval")
    if os.path.exists(retrieval_dir):
        for result_path in glob.glob(os.path.join(retrieval_dir, "*")):
            alive_time = time.time() - os.path.getmtime(result_path)
            if alive_time > RESULT_LIFETIME:
                shutil.rmtree(result_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("username", type=str)
    parser.add_argument("farm_name", type=str)
    parser.add_argument("field_name", type=str)
    parser.add_argument("mission_date", type=str)
    parser.add_argument("results_timestamp", type=str)
    parser.add_argument("download_uuid", type=str)
    parser.add_argument("annotation_version", type=str)

    args = parser.parse_args()


    remove_old_results(os.path.join("usr", "data", args.username,
                                    "image_sets", args.farm_name, args.field_name, args.mission_date,
                                    "model", "results", args.results_timestamp))

    inference_metrics.create_csv(args.username, 
                                args.farm_name, 
                                args.field_name, 
                                args.mission_date, 
                                args.results_timestamp,
                                args.download_uuid,
                                args.annotation_version)
