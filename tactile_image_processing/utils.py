import os
import argparse
import shutil
import json


def make_dir(dir, check=True):
    if check:
        check_dir(dir)
    os.makedirs(dir, exist_ok=True)


def check_dir(dir):
    if os.path.isdir(dir):
        str_input = input(f"\n{dir} \nSave Directory already exists, would you like to continue (y,n)? ")
        if not str2bool(str_input):
            exit()
        else:
            # clear out existing files
            empty_dir(dir)


def empty_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def save_json_obj(obj, name):
    with open(name + ".json", "w") as fp:
        json.dump(obj, fp)


def load_json_obj(name):
    with open(name + ".json", "r") as fp:
        return json.load(fp)
