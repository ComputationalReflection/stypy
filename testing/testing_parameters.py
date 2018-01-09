import os


def go_to_parent_folder(num_levels, path):
    for i in range(num_levels):
        path = os.path.dirname(path)

    return path


"""Root folder of the project, to place the parameters file in any folder easily.
It is relative, so the project can be moved to a new location with ease.
(0 folder levels below this file. DO NOT ALTER FOLDER / FILE STRUCTURE OF THE PROJECT!)"""
ROOT_PATH = go_to_parent_folder(1, os.path.realpath(__file__)).replace("\\", "/")

# Paths
LOG_PATH = ROOT_PATH + "/stypy/logfiles"
MODEL_TESTING_PROGRAMS_PATH = ROOT_PATH + "/model_testing"
CODE_GENERATION_TESTING_PROGRAMS_PATH = ROOT_PATH + "/"
STYPY_OVER_STYPY_PROGRAMS_PATH = ROOT_PATH + "/stypy_over_stypy_testing"

PYTHON_EXE = "C:/Python27/python.exe"