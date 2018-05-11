
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from stypy_copy import stypy_main_copy
2: from stypy_copy import stypy_parameters_copy
3: from stypy_copy.errors_copy.type_error_copy import TypeError
4: from stypy_copy.errors_copy.type_warning_copy import TypeWarning
5: import sys, os
6: 
7: 
8: '''
9: Stypy command-line tool
10: '''
11: 
12: # Python executable to run stypy
13: python_compiler_path = stypy_parameters_copy.PYTHON_EXE
14: 
15: '''
16: Options of the tool:
17:     -strict: Treat warnings as errors
18:     -print_ts: Print type store of the generated type inference program at the end. This can be used to have a quick
19:     review of the inferenced types
20: '''
21: accepted_options = ["-strict", "-print_ts"]
22: 
23: 
24: def __show_usage():
25:     '''
26:     Usage of the tool to show to users
27:     :return:
28:     '''
29:     sys.stderr.write('\nUsage: stypy.py <full path of the input .py file> ' + str(accepted_options) + '\n')
30:     sys.stderr.write('Please use .\ to refer to python files in the same directory as the compiler\n')
31:     sys.stderr.write('Options:\n')
32:     sys.stderr.write('\t-strict: Treat warnings as errors\n')
33:     sys.stderr.write('\t-print_ts: Prints the analyzed program final type store (for debugging purposes)')
34: 
35: 
36: def __check_args(args):
37:     '''
38:     Argument checking function
39:     :param args:
40:     :return:
41:     '''
42:     options = []
43:     if len(args) < 2 or len(args) > 2 + len(accepted_options):
44:         __show_usage()
45:         sys.exit(1)
46: 
47:     if not os.path.exists(args[1]):
48:         sys.stderr.write('ERROR: Input file was not found!')
49:         sys.exit(1)
50: 
51:     if len(args) >= 3:
52:         for option in args[2:]:
53:             if not option in accepted_options:
54:                 sys.stderr.write("ERROR: Unknown option: '" + option + "'")
55:                 __show_usage()
56:                 sys.exit(1)
57:             options += [option]
58: 
59:         return options
60: 
61:     return []
62: 
63: 
64: def print_msgs(obj_list):
65:     '''
66:     Prints the tool output (warnings and errors sorted by source line)
67:     :param obj_list:
68:     :return:
69:     '''
70:     sorted(obj_list, key=lambda obj: obj.localization.line)
71:     counter = 1
72:     for obj in obj_list:
73:         print str(counter) + ": " + str(obj) + "\n"
74:         counter += 1
75: 
76: 
77: def stypy_compilation_main(args):
78:     # Run type inference using a Stypy object with the main source file
79:     # More Stypy objects from this one will be spawned when the main source file use other modules
80:     stypy = stypy_main_copy.Stypy(args[1], python_compiler_path)
81:     stypy.analyze()
82:     return stypy.get_analyzed_program_type_store(), stypy.get_last_type_checking_running_time()
83: 
84: 
85: import time
86: 
87: if __name__ == "__main__":
88:     sys.argv = ['stypy.py', './stypy.py']
89:     options = __check_args(sys.argv)
90: 
91:     if "-strict" in options:
92:         TypeWarning.warnings_as_errors = True
93: 
94:     tinit = time.time()
95:     ti_type_store, analysis_run_time = stypy_compilation_main(sys.argv)
96:     tend = time.time()
97: 
98:     errors = TypeError.get_error_msgs()
99:     warnings = TypeWarning.get_warning_msgs()
100: 
101:     if len(errors) > 0:
102:         print "- {0} error(s) detected:\n".format(len(errors))
103:         print_msgs(errors)
104:     else:
105:         print "- No errors detected.\n"
106: 
107:     if len(warnings) > 0:
108:         print "- {0} warning(s) detected:\n".format(len(warnings))
109:         print_msgs(warnings)
110:     else:
111:         print "- No warnings detected.\n"
112: 
113:     # analyzed_program = sys.argv[1].split('\\')[-1].split('/')[-1]
114:     # print "'" + analyzed_program + "' type checked in {:.4f} seconds.".format(tend - tinit)
115: 
116:     # Print type store at the end
117:     if "-print_ts" in options:
118:         print ti_type_store
119: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from stypy_copy import stypy_main_copy' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/stypy_code_copy/')
import_4 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy')

if (type(import_4) is not StypyTypeError):

    if (import_4 != 'pyd_module'):
        __import__(import_4)
        sys_modules_5 = sys.modules[import_4]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy', sys_modules_5.module_type_store, module_type_store, ['stypy_main_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_5, sys_modules_5.module_type_store, module_type_store)
    else:
        from stypy_copy import stypy_main_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy', None, module_type_store, ['stypy_main_copy'], [stypy_main_copy])

else:
    # Assigning a type to the variable 'stypy_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy', import_4)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/stypy_code_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from stypy_copy import stypy_parameters_copy' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/stypy_code_copy/')
import_6 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy')

if (type(import_6) is not StypyTypeError):

    if (import_6 != 'pyd_module'):
        __import__(import_6)
        sys_modules_7 = sys.modules[import_6]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy', sys_modules_7.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_7, sys_modules_7.module_type_store, module_type_store)
    else:
        from stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'stypy_copy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy', import_6)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/stypy_code_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/stypy_code_copy/')
import_8 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.errors_copy.type_error_copy')

if (type(import_8) is not StypyTypeError):

    if (import_8 != 'pyd_module'):
        __import__(import_8)
        sys_modules_9 = sys.modules[import_8]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.errors_copy.type_error_copy', sys_modules_9.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_9, sys_modules_9.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.type_error_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.errors_copy.type_error_copy', import_8)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/stypy_code_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from stypy_copy.errors_copy.type_warning_copy import TypeWarning' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/stypy_code_copy/')
import_10 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.errors_copy.type_warning_copy')

if (type(import_10) is not StypyTypeError):

    if (import_10 != 'pyd_module'):
        __import__(import_10)
        sys_modules_11 = sys.modules[import_10]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.errors_copy.type_warning_copy', sys_modules_11.module_type_store, module_type_store, ['TypeWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_11, sys_modules_11.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.type_warning_copy import TypeWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.errors_copy.type_warning_copy', None, module_type_store, ['TypeWarning'], [TypeWarning])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.type_warning_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.errors_copy.type_warning_copy', import_10)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/stypy_code_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# Multiple import statement. import sys (1/2) (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)
# Multiple import statement. import os (2/2) (line 5)
import os

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os', os, module_type_store)

str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, (-1)), 'str', '\nStypy command-line tool\n')

# Assigning a Attribute to a Name (line 13):

# Assigning a Attribute to a Name (line 13):
# Getting the type of 'stypy_parameters_copy' (line 13)
stypy_parameters_copy_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 23), 'stypy_parameters_copy')
# Obtaining the member 'PYTHON_EXE' of a type (line 13)
PYTHON_EXE_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 23), stypy_parameters_copy_13, 'PYTHON_EXE')
# Assigning a type to the variable 'python_compiler_path' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'python_compiler_path', PYTHON_EXE_14)
str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, (-1)), 'str', '\nOptions of the tool:\n    -strict: Treat warnings as errors\n    -print_ts: Print type store of the generated type inference program at the end. This can be used to have a quick\n    review of the inferenced types\n')

# Assigning a List to a Name (line 21):

# Assigning a List to a Name (line 21):

# Obtaining an instance of the builtin type 'list' (line 21)
list_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 20), 'str', '-strict')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_16, str_17)
# Adding element type (line 21)
str_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 31), 'str', '-print_ts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_16, str_18)

# Assigning a type to the variable 'accepted_options' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'accepted_options', list_16)

@norecursion
def __show_usage(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__show_usage'
    module_type_store = module_type_store.open_function_context('__show_usage', 24, 0, False)
    
    # Passed parameters checking function
    __show_usage.stypy_localization = localization
    __show_usage.stypy_type_of_self = None
    __show_usage.stypy_type_store = module_type_store
    __show_usage.stypy_function_name = '__show_usage'
    __show_usage.stypy_param_names_list = []
    __show_usage.stypy_varargs_param_name = None
    __show_usage.stypy_kwargs_param_name = None
    __show_usage.stypy_call_defaults = defaults
    __show_usage.stypy_call_varargs = varargs
    __show_usage.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__show_usage', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__show_usage', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__show_usage(...)' code ##################

    str_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, (-1)), 'str', '\n    Usage of the tool to show to users\n    :return:\n    ')
    
    # Call to write(...): (line 29)
    # Processing the call arguments (line 29)
    str_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 21), 'str', '\nUsage: stypy.py <full path of the input .py file> ')
    
    # Call to str(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'accepted_options' (line 29)
    accepted_options_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 82), 'accepted_options', False)
    # Processing the call keyword arguments (line 29)
    kwargs_26 = {}
    # Getting the type of 'str' (line 29)
    str_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 78), 'str', False)
    # Calling str(args, kwargs) (line 29)
    str_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 29, 78), str_24, *[accepted_options_25], **kwargs_26)
    
    # Applying the binary operator '+' (line 29)
    result_add_28 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 21), '+', str_23, str_call_result_27)
    
    str_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 102), 'str', '\n')
    # Applying the binary operator '+' (line 29)
    result_add_30 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 100), '+', result_add_28, str_29)
    
    # Processing the call keyword arguments (line 29)
    kwargs_31 = {}
    # Getting the type of 'sys' (line 29)
    sys_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'sys', False)
    # Obtaining the member 'stderr' of a type (line 29)
    stderr_21 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), sys_20, 'stderr')
    # Obtaining the member 'write' of a type (line 29)
    write_22 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), stderr_21, 'write')
    # Calling write(args, kwargs) (line 29)
    write_call_result_32 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), write_22, *[result_add_30], **kwargs_31)
    
    
    # Call to write(...): (line 30)
    # Processing the call arguments (line 30)
    str_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 21), 'str', 'Please use .\\ to refer to python files in the same directory as the compiler\n')
    # Processing the call keyword arguments (line 30)
    kwargs_37 = {}
    # Getting the type of 'sys' (line 30)
    sys_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'sys', False)
    # Obtaining the member 'stderr' of a type (line 30)
    stderr_34 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 4), sys_33, 'stderr')
    # Obtaining the member 'write' of a type (line 30)
    write_35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 4), stderr_34, 'write')
    # Calling write(args, kwargs) (line 30)
    write_call_result_38 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), write_35, *[str_36], **kwargs_37)
    
    
    # Call to write(...): (line 31)
    # Processing the call arguments (line 31)
    str_42 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 21), 'str', 'Options:\n')
    # Processing the call keyword arguments (line 31)
    kwargs_43 = {}
    # Getting the type of 'sys' (line 31)
    sys_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'sys', False)
    # Obtaining the member 'stderr' of a type (line 31)
    stderr_40 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 4), sys_39, 'stderr')
    # Obtaining the member 'write' of a type (line 31)
    write_41 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 4), stderr_40, 'write')
    # Calling write(args, kwargs) (line 31)
    write_call_result_44 = invoke(stypy.reporting.localization.Localization(__file__, 31, 4), write_41, *[str_42], **kwargs_43)
    
    
    # Call to write(...): (line 32)
    # Processing the call arguments (line 32)
    str_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 21), 'str', '\t-strict: Treat warnings as errors\n')
    # Processing the call keyword arguments (line 32)
    kwargs_49 = {}
    # Getting the type of 'sys' (line 32)
    sys_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'sys', False)
    # Obtaining the member 'stderr' of a type (line 32)
    stderr_46 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 4), sys_45, 'stderr')
    # Obtaining the member 'write' of a type (line 32)
    write_47 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 4), stderr_46, 'write')
    # Calling write(args, kwargs) (line 32)
    write_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 32, 4), write_47, *[str_48], **kwargs_49)
    
    
    # Call to write(...): (line 33)
    # Processing the call arguments (line 33)
    str_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'str', '\t-print_ts: Prints the analyzed program final type store (for debugging purposes)')
    # Processing the call keyword arguments (line 33)
    kwargs_55 = {}
    # Getting the type of 'sys' (line 33)
    sys_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'sys', False)
    # Obtaining the member 'stderr' of a type (line 33)
    stderr_52 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 4), sys_51, 'stderr')
    # Obtaining the member 'write' of a type (line 33)
    write_53 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 4), stderr_52, 'write')
    # Calling write(args, kwargs) (line 33)
    write_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), write_53, *[str_54], **kwargs_55)
    
    
    # ################# End of '__show_usage(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__show_usage' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_57)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__show_usage'
    return stypy_return_type_57

# Assigning a type to the variable '__show_usage' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), '__show_usage', __show_usage)

@norecursion
def __check_args(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__check_args'
    module_type_store = module_type_store.open_function_context('__check_args', 36, 0, False)
    
    # Passed parameters checking function
    __check_args.stypy_localization = localization
    __check_args.stypy_type_of_self = None
    __check_args.stypy_type_store = module_type_store
    __check_args.stypy_function_name = '__check_args'
    __check_args.stypy_param_names_list = ['args']
    __check_args.stypy_varargs_param_name = None
    __check_args.stypy_kwargs_param_name = None
    __check_args.stypy_call_defaults = defaults
    __check_args.stypy_call_varargs = varargs
    __check_args.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__check_args', ['args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__check_args', localization, ['args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__check_args(...)' code ##################

    str_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, (-1)), 'str', '\n    Argument checking function\n    :param args:\n    :return:\n    ')
    
    # Assigning a List to a Name (line 42):
    
    # Assigning a List to a Name (line 42):
    
    # Obtaining an instance of the builtin type 'list' (line 42)
    list_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 42)
    
    # Assigning a type to the variable 'options' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'options', list_59)
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'args' (line 43)
    args_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'args', False)
    # Processing the call keyword arguments (line 43)
    kwargs_62 = {}
    # Getting the type of 'len' (line 43)
    len_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 7), 'len', False)
    # Calling len(args, kwargs) (line 43)
    len_call_result_63 = invoke(stypy.reporting.localization.Localization(__file__, 43, 7), len_60, *[args_61], **kwargs_62)
    
    int_64 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'int')
    # Applying the binary operator '<' (line 43)
    result_lt_65 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 7), '<', len_call_result_63, int_64)
    
    
    
    # Call to len(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'args' (line 43)
    args_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 28), 'args', False)
    # Processing the call keyword arguments (line 43)
    kwargs_68 = {}
    # Getting the type of 'len' (line 43)
    len_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'len', False)
    # Calling len(args, kwargs) (line 43)
    len_call_result_69 = invoke(stypy.reporting.localization.Localization(__file__, 43, 24), len_66, *[args_67], **kwargs_68)
    
    int_70 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 36), 'int')
    
    # Call to len(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'accepted_options' (line 43)
    accepted_options_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 44), 'accepted_options', False)
    # Processing the call keyword arguments (line 43)
    kwargs_73 = {}
    # Getting the type of 'len' (line 43)
    len_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 40), 'len', False)
    # Calling len(args, kwargs) (line 43)
    len_call_result_74 = invoke(stypy.reporting.localization.Localization(__file__, 43, 40), len_71, *[accepted_options_72], **kwargs_73)
    
    # Applying the binary operator '+' (line 43)
    result_add_75 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 36), '+', int_70, len_call_result_74)
    
    # Applying the binary operator '>' (line 43)
    result_gt_76 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 24), '>', len_call_result_69, result_add_75)
    
    # Applying the binary operator 'or' (line 43)
    result_or_keyword_77 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 7), 'or', result_lt_65, result_gt_76)
    
    # Testing if the type of an if condition is none (line 43)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 43, 4), result_or_keyword_77):
        pass
    else:
        
        # Testing the type of an if condition (line 43)
        if_condition_78 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 4), result_or_keyword_77)
        # Assigning a type to the variable 'if_condition_78' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'if_condition_78', if_condition_78)
        # SSA begins for if statement (line 43)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __show_usage(...): (line 44)
        # Processing the call keyword arguments (line 44)
        kwargs_80 = {}
        # Getting the type of '__show_usage' (line 44)
        show_usage_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), '__show_usage', False)
        # Calling __show_usage(args, kwargs) (line 44)
        show_usage_call_result_81 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), show_usage_79, *[], **kwargs_80)
        
        
        # Call to exit(...): (line 45)
        # Processing the call arguments (line 45)
        int_84 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 17), 'int')
        # Processing the call keyword arguments (line 45)
        kwargs_85 = {}
        # Getting the type of 'sys' (line 45)
        sys_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'sys', False)
        # Obtaining the member 'exit' of a type (line 45)
        exit_83 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), sys_82, 'exit')
        # Calling exit(args, kwargs) (line 45)
        exit_call_result_86 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), exit_83, *[int_84], **kwargs_85)
        
        # SSA join for if statement (line 43)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Call to exists(...): (line 47)
    # Processing the call arguments (line 47)
    
    # Obtaining the type of the subscript
    int_90 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 31), 'int')
    # Getting the type of 'args' (line 47)
    args_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'args', False)
    # Obtaining the member '__getitem__' of a type (line 47)
    getitem___92 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 26), args_91, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 47)
    subscript_call_result_93 = invoke(stypy.reporting.localization.Localization(__file__, 47, 26), getitem___92, int_90)
    
    # Processing the call keyword arguments (line 47)
    kwargs_94 = {}
    # Getting the type of 'os' (line 47)
    os_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 47)
    path_88 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), os_87, 'path')
    # Obtaining the member 'exists' of a type (line 47)
    exists_89 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), path_88, 'exists')
    # Calling exists(args, kwargs) (line 47)
    exists_call_result_95 = invoke(stypy.reporting.localization.Localization(__file__, 47, 11), exists_89, *[subscript_call_result_93], **kwargs_94)
    
    # Applying the 'not' unary operator (line 47)
    result_not__96 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 7), 'not', exists_call_result_95)
    
    # Testing if the type of an if condition is none (line 47)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 47, 4), result_not__96):
        pass
    else:
        
        # Testing the type of an if condition (line 47)
        if_condition_97 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 4), result_not__96)
        # Assigning a type to the variable 'if_condition_97' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'if_condition_97', if_condition_97)
        # SSA begins for if statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 48)
        # Processing the call arguments (line 48)
        str_101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 25), 'str', 'ERROR: Input file was not found!')
        # Processing the call keyword arguments (line 48)
        kwargs_102 = {}
        # Getting the type of 'sys' (line 48)
        sys_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 48)
        stderr_99 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), sys_98, 'stderr')
        # Obtaining the member 'write' of a type (line 48)
        write_100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), stderr_99, 'write')
        # Calling write(args, kwargs) (line 48)
        write_call_result_103 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), write_100, *[str_101], **kwargs_102)
        
        
        # Call to exit(...): (line 49)
        # Processing the call arguments (line 49)
        int_106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 17), 'int')
        # Processing the call keyword arguments (line 49)
        kwargs_107 = {}
        # Getting the type of 'sys' (line 49)
        sys_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'sys', False)
        # Obtaining the member 'exit' of a type (line 49)
        exit_105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), sys_104, 'exit')
        # Calling exit(args, kwargs) (line 49)
        exit_call_result_108 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), exit_105, *[int_106], **kwargs_107)
        
        # SSA join for if statement (line 47)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Call to len(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'args' (line 51)
    args_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'args', False)
    # Processing the call keyword arguments (line 51)
    kwargs_111 = {}
    # Getting the type of 'len' (line 51)
    len_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 7), 'len', False)
    # Calling len(args, kwargs) (line 51)
    len_call_result_112 = invoke(stypy.reporting.localization.Localization(__file__, 51, 7), len_109, *[args_110], **kwargs_111)
    
    int_113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 20), 'int')
    # Applying the binary operator '>=' (line 51)
    result_ge_114 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 7), '>=', len_call_result_112, int_113)
    
    # Testing if the type of an if condition is none (line 51)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 51, 4), result_ge_114):
        pass
    else:
        
        # Testing the type of an if condition (line 51)
        if_condition_115 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 4), result_ge_114)
        # Assigning a type to the variable 'if_condition_115' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'if_condition_115', if_condition_115)
        # SSA begins for if statement (line 51)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Obtaining the type of the subscript
        int_116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 27), 'int')
        slice_117 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 52, 22), int_116, None, None)
        # Getting the type of 'args' (line 52)
        args_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 22), 'args')
        # Obtaining the member '__getitem__' of a type (line 52)
        getitem___119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 22), args_118, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 52)
        subscript_call_result_120 = invoke(stypy.reporting.localization.Localization(__file__, 52, 22), getitem___119, slice_117)
        
        # Assigning a type to the variable 'subscript_call_result_120' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'subscript_call_result_120', subscript_call_result_120)
        # Testing if the for loop is going to be iterated (line 52)
        # Testing the type of a for loop iterable (line 52)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 52, 8), subscript_call_result_120)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 52, 8), subscript_call_result_120):
            # Getting the type of the for loop variable (line 52)
            for_loop_var_121 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 52, 8), subscript_call_result_120)
            # Assigning a type to the variable 'option' (line 52)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'option', for_loop_var_121)
            # SSA begins for a for statement (line 52)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'option' (line 53)
            option_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 19), 'option')
            # Getting the type of 'accepted_options' (line 53)
            accepted_options_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 29), 'accepted_options')
            # Applying the binary operator 'in' (line 53)
            result_contains_124 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 19), 'in', option_122, accepted_options_123)
            
            # Applying the 'not' unary operator (line 53)
            result_not__125 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 15), 'not', result_contains_124)
            
            # Testing if the type of an if condition is none (line 53)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 53, 12), result_not__125):
                pass
            else:
                
                # Testing the type of an if condition (line 53)
                if_condition_126 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 12), result_not__125)
                # Assigning a type to the variable 'if_condition_126' (line 53)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'if_condition_126', if_condition_126)
                # SSA begins for if statement (line 53)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 54)
                # Processing the call arguments (line 54)
                str_130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 33), 'str', "ERROR: Unknown option: '")
                # Getting the type of 'option' (line 54)
                option_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 62), 'option', False)
                # Applying the binary operator '+' (line 54)
                result_add_132 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 33), '+', str_130, option_131)
                
                str_133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 71), 'str', "'")
                # Applying the binary operator '+' (line 54)
                result_add_134 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 69), '+', result_add_132, str_133)
                
                # Processing the call keyword arguments (line 54)
                kwargs_135 = {}
                # Getting the type of 'sys' (line 54)
                sys_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'sys', False)
                # Obtaining the member 'stderr' of a type (line 54)
                stderr_128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), sys_127, 'stderr')
                # Obtaining the member 'write' of a type (line 54)
                write_129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), stderr_128, 'write')
                # Calling write(args, kwargs) (line 54)
                write_call_result_136 = invoke(stypy.reporting.localization.Localization(__file__, 54, 16), write_129, *[result_add_134], **kwargs_135)
                
                
                # Call to __show_usage(...): (line 55)
                # Processing the call keyword arguments (line 55)
                kwargs_138 = {}
                # Getting the type of '__show_usage' (line 55)
                show_usage_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), '__show_usage', False)
                # Calling __show_usage(args, kwargs) (line 55)
                show_usage_call_result_139 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), show_usage_137, *[], **kwargs_138)
                
                
                # Call to exit(...): (line 56)
                # Processing the call arguments (line 56)
                int_142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'int')
                # Processing the call keyword arguments (line 56)
                kwargs_143 = {}
                # Getting the type of 'sys' (line 56)
                sys_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'sys', False)
                # Obtaining the member 'exit' of a type (line 56)
                exit_141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), sys_140, 'exit')
                # Calling exit(args, kwargs) (line 56)
                exit_call_result_144 = invoke(stypy.reporting.localization.Localization(__file__, 56, 16), exit_141, *[int_142], **kwargs_143)
                
                # SSA join for if statement (line 53)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'options' (line 57)
            options_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'options')
            
            # Obtaining an instance of the builtin type 'list' (line 57)
            list_146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'list')
            # Adding type elements to the builtin type 'list' instance (line 57)
            # Adding element type (line 57)
            # Getting the type of 'option' (line 57)
            option_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 24), 'option')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 23), list_146, option_147)
            
            # Applying the binary operator '+=' (line 57)
            result_iadd_148 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 12), '+=', options_145, list_146)
            # Assigning a type to the variable 'options' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'options', result_iadd_148)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'options' (line 59)
        options_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'options')
        # Assigning a type to the variable 'stypy_return_type' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type', options_149)
        # SSA join for if statement (line 51)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Obtaining an instance of the builtin type 'list' (line 61)
    list_150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 61)
    
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type', list_150)
    
    # ################# End of '__check_args(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__check_args' in the type store
    # Getting the type of 'stypy_return_type' (line 36)
    stypy_return_type_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_151)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__check_args'
    return stypy_return_type_151

# Assigning a type to the variable '__check_args' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), '__check_args', __check_args)

@norecursion
def print_msgs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'print_msgs'
    module_type_store = module_type_store.open_function_context('print_msgs', 64, 0, False)
    
    # Passed parameters checking function
    print_msgs.stypy_localization = localization
    print_msgs.stypy_type_of_self = None
    print_msgs.stypy_type_store = module_type_store
    print_msgs.stypy_function_name = 'print_msgs'
    print_msgs.stypy_param_names_list = ['obj_list']
    print_msgs.stypy_varargs_param_name = None
    print_msgs.stypy_kwargs_param_name = None
    print_msgs.stypy_call_defaults = defaults
    print_msgs.stypy_call_varargs = varargs
    print_msgs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'print_msgs', ['obj_list'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'print_msgs', localization, ['obj_list'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'print_msgs(...)' code ##################

    str_152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, (-1)), 'str', '\n    Prints the tool output (warnings and errors sorted by source line)\n    :param obj_list:\n    :return:\n    ')
    
    # Call to sorted(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'obj_list' (line 70)
    obj_list_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'obj_list', False)
    # Processing the call keyword arguments (line 70)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 70, 25, True)
        # Passed parameters checking function
        _stypy_temp_lambda_1.stypy_localization = localization
        _stypy_temp_lambda_1.stypy_type_of_self = None
        _stypy_temp_lambda_1.stypy_type_store = module_type_store
        _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
        _stypy_temp_lambda_1.stypy_param_names_list = ['obj']
        _stypy_temp_lambda_1.stypy_varargs_param_name = None
        _stypy_temp_lambda_1.stypy_kwargs_param_name = None
        _stypy_temp_lambda_1.stypy_call_defaults = defaults
        _stypy_temp_lambda_1.stypy_call_varargs = varargs
        _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_1', ['obj'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'obj' (line 70)
        obj_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 37), 'obj', False)
        # Obtaining the member 'localization' of a type (line 70)
        localization_156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 37), obj_155, 'localization')
        # Obtaining the member 'line' of a type (line 70)
        line_157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 37), localization_156, 'line')
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'stypy_return_type', line_157)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_158)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_158

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 70)
    _stypy_temp_lambda_1_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), '_stypy_temp_lambda_1')
    keyword_160 = _stypy_temp_lambda_1_159
    kwargs_161 = {'key': keyword_160}
    # Getting the type of 'sorted' (line 70)
    sorted_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'sorted', False)
    # Calling sorted(args, kwargs) (line 70)
    sorted_call_result_162 = invoke(stypy.reporting.localization.Localization(__file__, 70, 4), sorted_153, *[obj_list_154], **kwargs_161)
    
    
    # Assigning a Num to a Name (line 71):
    
    # Assigning a Num to a Name (line 71):
    int_163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 14), 'int')
    # Assigning a type to the variable 'counter' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'counter', int_163)
    
    # Getting the type of 'obj_list' (line 72)
    obj_list_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'obj_list')
    # Assigning a type to the variable 'obj_list_164' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'obj_list_164', obj_list_164)
    # Testing if the for loop is going to be iterated (line 72)
    # Testing the type of a for loop iterable (line 72)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 72, 4), obj_list_164)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 72, 4), obj_list_164):
        # Getting the type of the for loop variable (line 72)
        for_loop_var_165 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 72, 4), obj_list_164)
        # Assigning a type to the variable 'obj' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'obj', for_loop_var_165)
        # SSA begins for a for statement (line 72)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to str(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'counter' (line 73)
        counter_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'counter', False)
        # Processing the call keyword arguments (line 73)
        kwargs_168 = {}
        # Getting the type of 'str' (line 73)
        str_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 14), 'str', False)
        # Calling str(args, kwargs) (line 73)
        str_call_result_169 = invoke(stypy.reporting.localization.Localization(__file__, 73, 14), str_166, *[counter_167], **kwargs_168)
        
        str_170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 29), 'str', ': ')
        # Applying the binary operator '+' (line 73)
        result_add_171 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 14), '+', str_call_result_169, str_170)
        
        
        # Call to str(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'obj' (line 73)
        obj_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 40), 'obj', False)
        # Processing the call keyword arguments (line 73)
        kwargs_174 = {}
        # Getting the type of 'str' (line 73)
        str_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 36), 'str', False)
        # Calling str(args, kwargs) (line 73)
        str_call_result_175 = invoke(stypy.reporting.localization.Localization(__file__, 73, 36), str_172, *[obj_173], **kwargs_174)
        
        # Applying the binary operator '+' (line 73)
        result_add_176 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 34), '+', result_add_171, str_call_result_175)
        
        str_177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 47), 'str', '\n')
        # Applying the binary operator '+' (line 73)
        result_add_178 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 45), '+', result_add_176, str_177)
        
        
        # Getting the type of 'counter' (line 74)
        counter_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'counter')
        int_180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 19), 'int')
        # Applying the binary operator '+=' (line 74)
        result_iadd_181 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 8), '+=', counter_179, int_180)
        # Assigning a type to the variable 'counter' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'counter', result_iadd_181)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'print_msgs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_msgs' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_182)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_msgs'
    return stypy_return_type_182

# Assigning a type to the variable 'print_msgs' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'print_msgs', print_msgs)

@norecursion
def stypy_compilation_main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'stypy_compilation_main'
    module_type_store = module_type_store.open_function_context('stypy_compilation_main', 77, 0, False)
    
    # Passed parameters checking function
    stypy_compilation_main.stypy_localization = localization
    stypy_compilation_main.stypy_type_of_self = None
    stypy_compilation_main.stypy_type_store = module_type_store
    stypy_compilation_main.stypy_function_name = 'stypy_compilation_main'
    stypy_compilation_main.stypy_param_names_list = ['args']
    stypy_compilation_main.stypy_varargs_param_name = None
    stypy_compilation_main.stypy_kwargs_param_name = None
    stypy_compilation_main.stypy_call_defaults = defaults
    stypy_compilation_main.stypy_call_varargs = varargs
    stypy_compilation_main.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'stypy_compilation_main', ['args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'stypy_compilation_main', localization, ['args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'stypy_compilation_main(...)' code ##################

    
    # Assigning a Call to a Name (line 80):
    
    # Assigning a Call to a Name (line 80):
    
    # Call to Stypy(...): (line 80)
    # Processing the call arguments (line 80)
    
    # Obtaining the type of the subscript
    int_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 39), 'int')
    # Getting the type of 'args' (line 80)
    args_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 34), 'args', False)
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 34), args_186, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_188 = invoke(stypy.reporting.localization.Localization(__file__, 80, 34), getitem___187, int_185)
    
    # Getting the type of 'python_compiler_path' (line 80)
    python_compiler_path_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 43), 'python_compiler_path', False)
    # Processing the call keyword arguments (line 80)
    kwargs_190 = {}
    # Getting the type of 'stypy_main_copy' (line 80)
    stypy_main_copy_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'stypy_main_copy', False)
    # Obtaining the member 'Stypy' of a type (line 80)
    Stypy_184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), stypy_main_copy_183, 'Stypy')
    # Calling Stypy(args, kwargs) (line 80)
    Stypy_call_result_191 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), Stypy_184, *[subscript_call_result_188, python_compiler_path_189], **kwargs_190)
    
    # Assigning a type to the variable 'stypy' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy', Stypy_call_result_191)
    
    # Call to analyze(...): (line 81)
    # Processing the call keyword arguments (line 81)
    kwargs_194 = {}
    # Getting the type of 'stypy' (line 81)
    stypy_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy', False)
    # Obtaining the member 'analyze' of a type (line 81)
    analyze_193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 4), stypy_192, 'analyze')
    # Calling analyze(args, kwargs) (line 81)
    analyze_call_result_195 = invoke(stypy.reporting.localization.Localization(__file__, 81, 4), analyze_193, *[], **kwargs_194)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 82)
    tuple_196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 82)
    # Adding element type (line 82)
    
    # Call to get_analyzed_program_type_store(...): (line 82)
    # Processing the call keyword arguments (line 82)
    kwargs_199 = {}
    # Getting the type of 'stypy' (line 82)
    stypy_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'stypy', False)
    # Obtaining the member 'get_analyzed_program_type_store' of a type (line 82)
    get_analyzed_program_type_store_198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 11), stypy_197, 'get_analyzed_program_type_store')
    # Calling get_analyzed_program_type_store(args, kwargs) (line 82)
    get_analyzed_program_type_store_call_result_200 = invoke(stypy.reporting.localization.Localization(__file__, 82, 11), get_analyzed_program_type_store_198, *[], **kwargs_199)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 11), tuple_196, get_analyzed_program_type_store_call_result_200)
    # Adding element type (line 82)
    
    # Call to get_last_type_checking_running_time(...): (line 82)
    # Processing the call keyword arguments (line 82)
    kwargs_203 = {}
    # Getting the type of 'stypy' (line 82)
    stypy_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 52), 'stypy', False)
    # Obtaining the member 'get_last_type_checking_running_time' of a type (line 82)
    get_last_type_checking_running_time_202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 52), stypy_201, 'get_last_type_checking_running_time')
    # Calling get_last_type_checking_running_time(args, kwargs) (line 82)
    get_last_type_checking_running_time_call_result_204 = invoke(stypy.reporting.localization.Localization(__file__, 82, 52), get_last_type_checking_running_time_202, *[], **kwargs_203)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 11), tuple_196, get_last_type_checking_running_time_call_result_204)
    
    # Assigning a type to the variable 'stypy_return_type' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type', tuple_196)
    
    # ################# End of 'stypy_compilation_main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'stypy_compilation_main' in the type store
    # Getting the type of 'stypy_return_type' (line 77)
    stypy_return_type_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_205)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'stypy_compilation_main'
    return stypy_return_type_205

# Assigning a type to the variable 'stypy_compilation_main' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'stypy_compilation_main', stypy_compilation_main)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 85, 0))

# 'import time' statement (line 85)
import time

import_module(stypy.reporting.localization.Localization(__file__, 85, 0), 'time', time, module_type_store)


if (__name__ == '__main__'):
    
    # Assigning a List to a Attribute (line 88):
    
    # Assigning a List to a Attribute (line 88):
    
    # Obtaining an instance of the builtin type 'list' (line 88)
    list_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 88)
    # Adding element type (line 88)
    str_207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 16), 'str', 'stypy.py')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), list_206, str_207)
    # Adding element type (line 88)
    str_208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 28), 'str', './stypy.py')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), list_206, str_208)
    
    # Getting the type of 'sys' (line 88)
    sys_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'sys')
    # Setting the type of the member 'argv' of a type (line 88)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 4), sys_209, 'argv', list_206)
    
    # Assigning a Call to a Name (line 89):
    
    # Assigning a Call to a Name (line 89):
    
    # Call to __check_args(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'sys' (line 89)
    sys_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 27), 'sys', False)
    # Obtaining the member 'argv' of a type (line 89)
    argv_212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 27), sys_211, 'argv')
    # Processing the call keyword arguments (line 89)
    kwargs_213 = {}
    # Getting the type of '__check_args' (line 89)
    check_args_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 14), '__check_args', False)
    # Calling __check_args(args, kwargs) (line 89)
    check_args_call_result_214 = invoke(stypy.reporting.localization.Localization(__file__, 89, 14), check_args_210, *[argv_212], **kwargs_213)
    
    # Assigning a type to the variable 'options' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'options', check_args_call_result_214)
    
    str_215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 7), 'str', '-strict')
    # Getting the type of 'options' (line 91)
    options_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'options')
    # Applying the binary operator 'in' (line 91)
    result_contains_217 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 7), 'in', str_215, options_216)
    
    # Testing if the type of an if condition is none (line 91)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 91, 4), result_contains_217):
        pass
    else:
        
        # Testing the type of an if condition (line 91)
        if_condition_218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 4), result_contains_217)
        # Assigning a type to the variable 'if_condition_218' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'if_condition_218', if_condition_218)
        # SSA begins for if statement (line 91)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 92):
        
        # Assigning a Name to a Attribute (line 92):
        # Getting the type of 'True' (line 92)
        True_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 41), 'True')
        # Getting the type of 'TypeWarning' (line 92)
        TypeWarning_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'TypeWarning')
        # Setting the type of the member 'warnings_as_errors' of a type (line 92)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), TypeWarning_220, 'warnings_as_errors', True_219)
        # SSA join for if statement (line 91)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 94):
    
    # Assigning a Call to a Name (line 94):
    
    # Call to time(...): (line 94)
    # Processing the call keyword arguments (line 94)
    kwargs_223 = {}
    # Getting the type of 'time' (line 94)
    time_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'time', False)
    # Obtaining the member 'time' of a type (line 94)
    time_222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), time_221, 'time')
    # Calling time(args, kwargs) (line 94)
    time_call_result_224 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), time_222, *[], **kwargs_223)
    
    # Assigning a type to the variable 'tinit' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'tinit', time_call_result_224)
    
    # Assigning a Call to a Tuple (line 95):
    
    # Assigning a Call to a Name:
    
    # Call to stypy_compilation_main(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'sys' (line 95)
    sys_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 62), 'sys', False)
    # Obtaining the member 'argv' of a type (line 95)
    argv_227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 62), sys_226, 'argv')
    # Processing the call keyword arguments (line 95)
    kwargs_228 = {}
    # Getting the type of 'stypy_compilation_main' (line 95)
    stypy_compilation_main_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 39), 'stypy_compilation_main', False)
    # Calling stypy_compilation_main(args, kwargs) (line 95)
    stypy_compilation_main_call_result_229 = invoke(stypy.reporting.localization.Localization(__file__, 95, 39), stypy_compilation_main_225, *[argv_227], **kwargs_228)
    
    # Assigning a type to the variable 'call_assignment_1' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'call_assignment_1', stypy_compilation_main_call_result_229)
    
    # Assigning a Call to a Name (line 95):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_1' (line 95)
    call_assignment_1_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'call_assignment_1', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_231 = stypy_get_value_from_tuple(call_assignment_1_230, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_2' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'call_assignment_2', stypy_get_value_from_tuple_call_result_231)
    
    # Assigning a Name to a Name (line 95):
    # Getting the type of 'call_assignment_2' (line 95)
    call_assignment_2_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'call_assignment_2')
    # Assigning a type to the variable 'ti_type_store' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'ti_type_store', call_assignment_2_232)
    
    # Assigning a Call to a Name (line 95):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_1' (line 95)
    call_assignment_1_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'call_assignment_1', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_234 = stypy_get_value_from_tuple(call_assignment_1_233, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_3' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'call_assignment_3', stypy_get_value_from_tuple_call_result_234)
    
    # Assigning a Name to a Name (line 95):
    # Getting the type of 'call_assignment_3' (line 95)
    call_assignment_3_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'call_assignment_3')
    # Assigning a type to the variable 'analysis_run_time' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 19), 'analysis_run_time', call_assignment_3_235)
    
    # Assigning a Call to a Name (line 96):
    
    # Assigning a Call to a Name (line 96):
    
    # Call to time(...): (line 96)
    # Processing the call keyword arguments (line 96)
    kwargs_238 = {}
    # Getting the type of 'time' (line 96)
    time_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'time', False)
    # Obtaining the member 'time' of a type (line 96)
    time_237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), time_236, 'time')
    # Calling time(args, kwargs) (line 96)
    time_call_result_239 = invoke(stypy.reporting.localization.Localization(__file__, 96, 11), time_237, *[], **kwargs_238)
    
    # Assigning a type to the variable 'tend' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'tend', time_call_result_239)
    
    # Assigning a Call to a Name (line 98):
    
    # Assigning a Call to a Name (line 98):
    
    # Call to get_error_msgs(...): (line 98)
    # Processing the call keyword arguments (line 98)
    kwargs_242 = {}
    # Getting the type of 'TypeError' (line 98)
    TypeError_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 13), 'TypeError', False)
    # Obtaining the member 'get_error_msgs' of a type (line 98)
    get_error_msgs_241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 13), TypeError_240, 'get_error_msgs')
    # Calling get_error_msgs(args, kwargs) (line 98)
    get_error_msgs_call_result_243 = invoke(stypy.reporting.localization.Localization(__file__, 98, 13), get_error_msgs_241, *[], **kwargs_242)
    
    # Assigning a type to the variable 'errors' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'errors', get_error_msgs_call_result_243)
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to get_warning_msgs(...): (line 99)
    # Processing the call keyword arguments (line 99)
    kwargs_246 = {}
    # Getting the type of 'TypeWarning' (line 99)
    TypeWarning_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 15), 'TypeWarning', False)
    # Obtaining the member 'get_warning_msgs' of a type (line 99)
    get_warning_msgs_245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 15), TypeWarning_244, 'get_warning_msgs')
    # Calling get_warning_msgs(args, kwargs) (line 99)
    get_warning_msgs_call_result_247 = invoke(stypy.reporting.localization.Localization(__file__, 99, 15), get_warning_msgs_245, *[], **kwargs_246)
    
    # Assigning a type to the variable 'warnings' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'warnings', get_warning_msgs_call_result_247)
    
    
    # Call to len(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'errors' (line 101)
    errors_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'errors', False)
    # Processing the call keyword arguments (line 101)
    kwargs_250 = {}
    # Getting the type of 'len' (line 101)
    len_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 7), 'len', False)
    # Calling len(args, kwargs) (line 101)
    len_call_result_251 = invoke(stypy.reporting.localization.Localization(__file__, 101, 7), len_248, *[errors_249], **kwargs_250)
    
    int_252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 21), 'int')
    # Applying the binary operator '>' (line 101)
    result_gt_253 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 7), '>', len_call_result_251, int_252)
    
    # Testing if the type of an if condition is none (line 101)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 101, 4), result_gt_253):
        str_267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 14), 'str', '- No errors detected.\n')
    else:
        
        # Testing the type of an if condition (line 101)
        if_condition_254 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 4), result_gt_253)
        # Assigning a type to the variable 'if_condition_254' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'if_condition_254', if_condition_254)
        # SSA begins for if statement (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to format(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Call to len(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'errors' (line 102)
        errors_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 54), 'errors', False)
        # Processing the call keyword arguments (line 102)
        kwargs_259 = {}
        # Getting the type of 'len' (line 102)
        len_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 50), 'len', False)
        # Calling len(args, kwargs) (line 102)
        len_call_result_260 = invoke(stypy.reporting.localization.Localization(__file__, 102, 50), len_257, *[errors_258], **kwargs_259)
        
        # Processing the call keyword arguments (line 102)
        kwargs_261 = {}
        str_255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 14), 'str', '- {0} error(s) detected:\n')
        # Obtaining the member 'format' of a type (line 102)
        format_256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 14), str_255, 'format')
        # Calling format(args, kwargs) (line 102)
        format_call_result_262 = invoke(stypy.reporting.localization.Localization(__file__, 102, 14), format_256, *[len_call_result_260], **kwargs_261)
        
        
        # Call to print_msgs(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'errors' (line 103)
        errors_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'errors', False)
        # Processing the call keyword arguments (line 103)
        kwargs_265 = {}
        # Getting the type of 'print_msgs' (line 103)
        print_msgs_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'print_msgs', False)
        # Calling print_msgs(args, kwargs) (line 103)
        print_msgs_call_result_266 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), print_msgs_263, *[errors_264], **kwargs_265)
        
        # SSA branch for the else part of an if statement (line 101)
        module_type_store.open_ssa_branch('else')
        str_267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 14), 'str', '- No errors detected.\n')
        # SSA join for if statement (line 101)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Call to len(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'warnings' (line 107)
    warnings_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'warnings', False)
    # Processing the call keyword arguments (line 107)
    kwargs_270 = {}
    # Getting the type of 'len' (line 107)
    len_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 7), 'len', False)
    # Calling len(args, kwargs) (line 107)
    len_call_result_271 = invoke(stypy.reporting.localization.Localization(__file__, 107, 7), len_268, *[warnings_269], **kwargs_270)
    
    int_272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 23), 'int')
    # Applying the binary operator '>' (line 107)
    result_gt_273 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 7), '>', len_call_result_271, int_272)
    
    # Testing if the type of an if condition is none (line 107)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 107, 4), result_gt_273):
        str_287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 14), 'str', '- No warnings detected.\n')
    else:
        
        # Testing the type of an if condition (line 107)
        if_condition_274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 4), result_gt_273)
        # Assigning a type to the variable 'if_condition_274' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'if_condition_274', if_condition_274)
        # SSA begins for if statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to format(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Call to len(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'warnings' (line 108)
        warnings_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 56), 'warnings', False)
        # Processing the call keyword arguments (line 108)
        kwargs_279 = {}
        # Getting the type of 'len' (line 108)
        len_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 52), 'len', False)
        # Calling len(args, kwargs) (line 108)
        len_call_result_280 = invoke(stypy.reporting.localization.Localization(__file__, 108, 52), len_277, *[warnings_278], **kwargs_279)
        
        # Processing the call keyword arguments (line 108)
        kwargs_281 = {}
        str_275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 14), 'str', '- {0} warning(s) detected:\n')
        # Obtaining the member 'format' of a type (line 108)
        format_276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 14), str_275, 'format')
        # Calling format(args, kwargs) (line 108)
        format_call_result_282 = invoke(stypy.reporting.localization.Localization(__file__, 108, 14), format_276, *[len_call_result_280], **kwargs_281)
        
        
        # Call to print_msgs(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'warnings' (line 109)
        warnings_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'warnings', False)
        # Processing the call keyword arguments (line 109)
        kwargs_285 = {}
        # Getting the type of 'print_msgs' (line 109)
        print_msgs_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'print_msgs', False)
        # Calling print_msgs(args, kwargs) (line 109)
        print_msgs_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), print_msgs_283, *[warnings_284], **kwargs_285)
        
        # SSA branch for the else part of an if statement (line 107)
        module_type_store.open_ssa_branch('else')
        str_287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 14), 'str', '- No warnings detected.\n')
        # SSA join for if statement (line 107)
        module_type_store = module_type_store.join_ssa_context()
        

    
    str_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 7), 'str', '-print_ts')
    # Getting the type of 'options' (line 117)
    options_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 22), 'options')
    # Applying the binary operator 'in' (line 117)
    result_contains_290 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 7), 'in', str_288, options_289)
    
    # Testing if the type of an if condition is none (line 117)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 117, 4), result_contains_290):
        pass
    else:
        
        # Testing the type of an if condition (line 117)
        if_condition_291 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 4), result_contains_290)
        # Assigning a type to the variable 'if_condition_291' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'if_condition_291', if_condition_291)
        # SSA begins for if statement (line 117)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'ti_type_store' (line 118)
        ti_type_store_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 14), 'ti_type_store')
        # SSA join for if statement (line 117)
        module_type_store = module_type_store.join_ssa_context()
        



# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
