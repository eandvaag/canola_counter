import tensorflow as tf
import os
import glob
import logging

from io_utils import json_io


# def _get_best_loss(loss_record, key):

#     loss_vals = [(epoch, loss_record[epoch][key]) for epoch in loss_record.keys()]
#     sorted_loss_vals = sorted(loss_vals, key=lambda x: x[1])

#     return sorted_loss_vals[0][0]


def get_weights_path(config):

    logger = logging.getLogger(__name__)
    idx_path = glob.glob(os.path.join(config.weights_dir, "epoch-*.index"))[0]
    checkpoint = idx_path[:-6]
    logger.info("using checkpoint {}".format(checkpoint))
    #exit()
    return checkpoint
    
    # logger = logging.getLogger(__name__)

    # loss_record_path = os.path.join(config.model_dir, "loss_record.json")
    # loss_record = json_io.load_json(loss_record_path)

    # if config.load_method == "latest":
    #     checkpoint = tf.train.latest_checkpoint(config.weights_dir)

    # elif config.load_method == "best_train":
    #     best_epoch = _get_best_loss(loss_record, "train_loss")
    #     checkpoint = os.path.join(config.weights_dir, "epoch-" + best_epoch) 

    # elif config.load_method == "best_val":
    #     best_epoch = _get_best_loss(loss_record, "val_loss")
    #     checkpoint = os.path.join(config.weights_dir, "epoch-" + best_epoch)

    # else:
    #     raise RuntimeError("Unknown method for loading model: '{}'.".format(config.load_method))

    # logger.info("Loading weights according to criteria: '{}'. Best epoch was {}.".format(
    #             config.load_method, os.path.basename(checkpoint)))
    # return checkpoint