# import numpy as np
# import tensorflow as tf
# import random as python_random

# np.random.seed(123)
# python_random.seed(123)
# tf.random.set_seed(1234)


import argparse
import logging

import handler

from io_utils import json_io




def main():
    logging.basicConfig(level=logging.INFO)
    handler.handle_request(req)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="A tool for detecting plants in UAV images")
    parser.add_argument('input', type=str, help="")
    args = parser.parse_args()


    req = json_io.load_json(args.input)

    main()
