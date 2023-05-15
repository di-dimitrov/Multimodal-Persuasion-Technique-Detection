import os
import argparse
import logging
import json

"""
This script checks whether the results format for subtask 1 is correct. 
It also provides some warnings about possible errors.

The submission of the result file should be in json format. 
It should be a list of objects:
{
  id     -> identifier of the test sample,
  label  -> propandistic or not-propagandistic
}

"""

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

def check_format(file_path,KEYS,CLASSES):
    if not os.path.exists(file_path):
        logging.error("File doesnt exists: {}".format(file_path))
        return False

    try:
        with open(file_path) as p:
            submission = json.load(p)
    except:
        logging.error("File is not a valid json file: {}".format(file_path))
        return False
    for i, obj in enumerate(submission):
        for key in KEYS:
            if key not in obj:
                logging.error("Missing entry in {}:{}".format(file_path, i))
                return False
        label = obj['label']
        if label not in CLASSES:
            logging.error("Unknown Label in {}:{}".format(file_path, i))
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_files_path", "-p", nargs='+', required=True,
                        help="The absolute path to the files you want to check.", type=str)

    args = parser.parse_args()
    logging.info("Subtask 1: Checking files: {}".format(args.pred_files_path))
    CLASSES = ['propandistic', 'not-propagandistic']
    KEYS = ['id', 'label']

    for pred_file_path in args.pred_files_path:
        check_format(pred_file_path, KEYS, CLASSES)
