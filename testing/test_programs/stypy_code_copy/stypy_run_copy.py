from stypy_copy import stypy_parameters_copy
from stypy_copy import stypy_main_copy
from stypy_copy.errors_copy.type_error_copy import TypeError
from stypy_copy.errors_copy.type_warning_copy import TypeWarning
import sys, os


"""
Stypy command-line tool
"""

# Python executable to run stypy
python_compiler_path = stypy_parameters_copy.PYTHON_EXE

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
    options = []
    if len(args) < 2 or len(args) > 2 + len(accepted_options):
        __show_usage()
        sys.exit(1)

    if not os.path.exists(args[1]):
        sys.stderr.write('ERROR: Input file was not found!')
        sys.exit(1)

    if len(args) >= 3:
        for option in args[2:]:
            if not option in accepted_options:
                sys.stderr.write("ERROR: Unknown option: '" + option + "'")
                __show_usage()
                sys.exit(1)
            options += [option]

        return options

    return []


def print_msgs(obj_list):
    """
    Prints the tool output (warnings and errors sorted by source line)
    :param obj_list:
    :return:
    """
    sorted(obj_list, key=lambda obj: obj.localization.line)
    counter = 1
    for obj in obj_list:
        print str(counter) + ": " + str(obj) + "\n"
        counter += 1


def stypy_compilation_main(args):
    # Run type inference using a Stypy object with the main source file
    # More Stypy objects from this one will be spawned when the main source file use other modules
    stypy = stypy_main_copy.Stypy(args[1], python_compiler_path)
    stypy.analyze()
    return stypy.get_analyzed_program_type_store(), stypy.get_last_type_checking_running_time()


import time

if __name__ == "__main__":
    sys.argv = ['stypy.py', './stypy.py']
    options = __check_args(sys.argv)

    if "-strict" in options:
        TypeWarning.warnings_as_errors = True

    tinit = time.time()
    ti_type_store, analysis_run_time = stypy_compilation_main(sys.argv)
    tend = time.time()

    errors = TypeError.get_error_msgs()
    warnings = TypeWarning.get_warning_msgs()

    if len(errors) > 0:
        print "- {0} error(s) detected:\n".format(len(errors))
        print_msgs(errors)
    else:
        print "- No errors detected.\n"

    if len(warnings) > 0:
        print "- {0} warning(s) detected:\n".format(len(warnings))
        print_msgs(warnings)
    else:
        print "- No warnings detected.\n"

    # analyzed_program = sys.argv[1].split('\\')[-1].split('/')[-1]
    # print "'" + analyzed_program + "' type checked in {:.4f} seconds.".format(tend - tinit)

    # Print type store at the end
    if "-print_ts" in options:
        print ti_type_store
