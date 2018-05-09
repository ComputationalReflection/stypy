
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from stypy_copy import stypy_main_copy, stypy_parameters_copy
2: from stypy_copy.errors_copy.type_error_copy import TypeError
3: from stypy_copy.errors_copy.type_warning_copy import TypeWarning
4: import sys, os
5: 
6: #
7: # '''
8: # Stypy command-line tool
9: # '''
10: #
11: # # Python executable to run stypy
12: # python_compiler_path = stypy_parameters.PYTHON_EXE
13: #
14: # '''
15: # Options of the tool:
16: #     -strict: Treat warnings as errors
17: #     -print_ts: Print type store of the generated type inference program at the end. This can be used to have a quick
18: #     review of the inferenced types
19: # '''
20: # accepted_options = ["-strict", "-print_ts"]
21: #
22: #
23: # def __show_usage():
24: #     '''
25: #     Usage of the tool to show to users
26: #     :return:
27: #     '''
28: #     sys.stderr.write('\nUsage: stypy.py <full path of the input .py file> ' + str(accepted_options) + '\n')
29: #     sys.stderr.write('Please use .\ to refer to python files in the same directory as the compiler\n')
30: #     sys.stderr.write('Options:\n')
31: #     sys.stderr.write('\t-strict: Treat warnings as errors\n')
32: #     sys.stderr.write('\t-print_ts: Prints the analyzed program final type store (for debugging purposes)')
33: #
34: #
35: # def __check_args(args):
36: #     '''
37: #     Argument checking function
38: #     :param args:
39: #     :return:
40: #     '''
41: #     options = []
42: #     if len(args) < 2 or len(args) > 2 + len(accepted_options):
43: #         __show_usage()
44: #         sys.exit(1)
45: #
46: #     if not os.path.exists(args[1]):
47: #         sys.stderr.write('ERROR: Input file was not found!')
48: #         sys.exit(1)
49: #
50: #     if len(args) >= 3:
51: #         for option in args[2:]:
52: #             if not option in accepted_options:
53: #                 sys.stderr.write("ERROR: Unknown option: '" + option + "'")
54: #                 __show_usage()
55: #                 sys.exit(1)
56: #             options += [option]
57: #
58: #         return options
59: #
60: #     return []
61: #
62: #
63: # def print_msgs(obj_list):
64: #     '''
65: #     Prints the tool output (warnings and errors sorted by source line)
66: #     :param obj_list:
67: #     :return:
68: #     '''
69: #     sorted(obj_list, key=lambda obj: obj.localization.line)
70: #     counter = 1
71: #     for obj in obj_list:
72: #         print str(counter) + ": " + str(obj) + "\n"
73: #         counter += 1
74: #
75: #
76: # def stypy_compilation_main(args):
77: #     # Run type inference using a Stypy object with the main source file
78: #     # More Stypy objects from this one will be spawned when the main source file use other modules
79: #     stypy = stypy_main.Stypy(args[1], python_compiler_path)
80: #     stypy.analyze()
81: #     return stypy.get_analyzed_program_type_store(), stypy.get_last_type_checking_running_time()
82: #
83: #
84: # import time
85: #
86: # if __name__ == "__main__":
87: #     sys.argv = ['stypy.py', './stypy.py']
88: #     options = __check_args(sys.argv)
89: #
90: #     if "-strict" in options:
91: #         TypeWarning.warnings_as_errors = True
92: #
93: #     tinit = time.time()
94: #     ti_type_store, analysis_run_time = stypy_compilation_main(sys.argv)
95: #     tend = time.time()
96: #
97: #     errors = TypeError.get_error_msgs()
98: #     warnings = TypeWarning.get_warning_msgs()
99: #
100: #     if len(errors) > 0:
101: #         print "- {0} error(s) detected:\n".format(len(errors))
102: #         print_msgs(errors)
103: #     else:
104: #         print "- No errors detected.\n"
105: #
106: #     if len(warnings) > 0:
107: #         print "- {0} warning(s) detected:\n".format(len(warnings))
108: #         print_msgs(warnings)
109: #     else:
110: #         print "- No warnings detected.\n"
111: #
112: #     # analyzed_program = sys.argv[1].split('\\')[-1].split('/')[-1]
113: #     # print "'" + analyzed_program + "' type checked in {:.4f} seconds.".format(tend - tinit)
114: #
115: #     # Print type store at the end
116: #     if "-print_ts" in options:
117: #         print ti_type_store
118: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from stypy_copy import stypy_main_copy, stypy_parameters_copy' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/')
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy', sys_modules_2.module_type_store, module_type_store, ['stypy_main_copy', 'stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_2, sys_modules_2.module_type_store, module_type_store)
    else:
        from stypy_copy import stypy_main_copy, stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy', None, module_type_store, ['stypy_main_copy', 'stypy_parameters_copy'], [stypy_main_copy, stypy_parameters_copy])

else:
    # Assigning a type to the variable 'stypy_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy', import_1)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/')
import_3 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.errors_copy.type_error_copy')

if (type(import_3) is not StypyTypeError):

    if (import_3 != 'pyd_module'):
        __import__(import_3)
        sys_modules_4 = sys.modules[import_3]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.errors_copy.type_error_copy', sys_modules_4.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_4, sys_modules_4.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.type_error_copy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.errors_copy.type_error_copy', import_3)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from stypy_copy.errors_copy.type_warning_copy import TypeWarning' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/')
import_5 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.errors_copy.type_warning_copy')

if (type(import_5) is not StypyTypeError):

    if (import_5 != 'pyd_module'):
        __import__(import_5)
        sys_modules_6 = sys.modules[import_5]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.errors_copy.type_warning_copy', sys_modules_6.module_type_store, module_type_store, ['TypeWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_6, sys_modules_6.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.type_warning_copy import TypeWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.errors_copy.type_warning_copy', None, module_type_store, ['TypeWarning'], [TypeWarning])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.type_warning_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.errors_copy.type_warning_copy', import_5)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# Multiple import statement. import sys (1/2) (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)
# Multiple import statement. import os (2/2) (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
