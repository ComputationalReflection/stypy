#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time

from stypy import stypy_main
from stypy import stypy_parameters
from stypy.errors.type_error import StypyTypeError
from stypy.errors.type_warning import TypeWarning
from stypy.sgmc.sgmc_main import SGMC

"""
Stypy command-line tool. This is the user interface to launch the stypy_main.py file.
"""

# Python implementation executable to run stypy
python_compiler_path = stypy_parameters.PYTHON_EXE

"""
Options of the tool:
    -strict: Treat warnings as errors
    -print_ts: Print type store of the generated type inference program at the end. This can be used to have a quick
    review of the inferenced types
"""
accepted_options = ["-strict", "-print_ts"]


def __show_usage():
    """
    Usage of the tool to show to users
    :return:
    """
    sys.stderr.write('\nUsage: stypy.py <full path of the input .py file> ' + str(accepted_options) + '\n')
    sys.stderr.write('Please use .\ to refer to python files in the same directory as the compiler\n')
    sys.stderr.write('Options:\n')
    sys.stderr.write('\t-strict: Treat warnings as errors\n')
    sys.stderr.write('\t-print_ts: Prints the analyzed program final type store (for debugging purposes)')


def __check_args(args):
    """
    Argument checking function
    :param args:
    :return:
    """
    options_ = []
    if len(args) < 2 or len(args) > 2 + len(accepted_options):
        __show_usage()
        sys.exit(1)

    if not os.path.exists(args[1]):
        sys.stderr.write('ERROR: Input file was not found!')
        sys.exit(1)

    if len(args) >= 3:
        for option in args[2:]:
            if option not in accepted_options:
                sys.stderr.write("ERROR: Unknown option: '" + option + "'")
                __show_usage()
                sys.exit(1)
            options_ += [option]

        return options_

    return []


def print_msgs(obj_list):
    """
    Prints the tool output (warnings and errors sorted by source line)
    :param obj_list:
    :return:
    """
    sorted(obj_list, key=lambda obj_: obj_.localization.line)
    counter = 1
    for obj in obj_list:
        print (str(counter) + ": " + str(obj) + "\n")
        counter += 1


def stypy_compilation_main(args):
    # Run type inference using a Stypy object with the main source file
    # More Stypy objects from this one will be spawned when the main source file use other modules
    route = SGMC.get_sgmc_route(args[1])
    destination_file = SGMC.sgmc_cache_absolute_path + route
    try:
        stypy = stypy_main.Stypy(args[1], python_compiler_path,
                                 type_inference_program_file=destination_file,
                                 is_main=True)
        stypy.analyze()
        return stypy.get_analyzed_program_type_store(), stypy.get_last_type_checking_running_time()
    except SystemExit:
        print ("Stypy cannot process the input file due to syntax errors in the source code.")
        return None, None


if __name__ == "__main__":
    # sys.argv = ['stypy_checker', 'code_tryouts.py']
    # ['stypy.py', './stypy.py']
    options = __check_args(sys.argv)

    if "-strict" in options:
        TypeWarning.warnings_as_errors = True

    tinit = time.time()
    ti_type_store, analysis_run_time = stypy_compilation_main(sys.argv)
    tend = time.time()
    if not (ti_type_store is None and analysis_run_time is None):
        errors = StypyTypeError.get_error_msgs()
        warnings = TypeWarning.get_warning_msgs()

        if len(errors) > 0:
            print ("- {0} error(s) detected:\n".format(len(errors)))
            print_msgs(errors)
        else:
            print ("- No errors detected.\n")

        if len(warnings) > 0:
            print ("- {0} warning(s) detected:\n".format(len(warnings)))
            print_msgs(warnings)
        else:
            print ("- No warnings detected.\n")

        # analyzed_program = sys.argv[1].split('\\')[-1].split('/')[-1]
        # print "'" + analyzed_program + "' type checked in {:.4f} seconds.".format(tend - tinit)

        # Print type store at the end
        if "-print_ts" in options:
            print (ti_type_store)
