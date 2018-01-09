
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Main entry point'''
2: 
3: import sys
4: if sys.argv[0].endswith("__main__.py"):
5:     sys.argv[0] = "python -m unittest"
6: 
7: __unittest = True
8: 
9: from .main import main, TestProgram, USAGE_AS_MAIN
10: TestProgram.USAGE = USAGE_AS_MAIN
11: 
12: main(module=None)
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_193090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Main entry point')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)



# Call to endswith(...): (line 4)
# Processing the call arguments (line 4)
str_193097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 24), 'str', '__main__.py')
# Processing the call keyword arguments (line 4)
kwargs_193098 = {}

# Obtaining the type of the subscript
int_193091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 12), 'int')
# Getting the type of 'sys' (line 4)
sys_193092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 3), 'sys', False)
# Obtaining the member 'argv' of a type (line 4)
argv_193093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 3), sys_193092, 'argv')
# Obtaining the member '__getitem__' of a type (line 4)
getitem___193094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 3), argv_193093, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 4)
subscript_call_result_193095 = invoke(stypy.reporting.localization.Localization(__file__, 4, 3), getitem___193094, int_193091)

# Obtaining the member 'endswith' of a type (line 4)
endswith_193096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 3), subscript_call_result_193095, 'endswith')
# Calling endswith(args, kwargs) (line 4)
endswith_call_result_193099 = invoke(stypy.reporting.localization.Localization(__file__, 4, 3), endswith_193096, *[str_193097], **kwargs_193098)

# Testing the type of an if condition (line 4)
if_condition_193100 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 4, 0), endswith_call_result_193099)
# Assigning a type to the variable 'if_condition_193100' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'if_condition_193100', if_condition_193100)
# SSA begins for if statement (line 4)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Str to a Subscript (line 5):
str_193101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 18), 'str', 'python -m unittest')
# Getting the type of 'sys' (line 5)
sys_193102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'sys')
# Obtaining the member 'argv' of a type (line 5)
argv_193103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), sys_193102, 'argv')
int_193104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'int')
# Storing an element on a container (line 5)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 4), argv_193103, (int_193104, str_193101))
# SSA join for if statement (line 4)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Name (line 7):
# Getting the type of 'True' (line 7)
True_193105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 13), 'True')
# Assigning a type to the variable '__unittest' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__unittest', True_193105)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from unittest.main import main, TestProgram, USAGE_AS_MAIN' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/unittest/')
import_193106 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'unittest.main')

if (type(import_193106) is not StypyTypeError):

    if (import_193106 != 'pyd_module'):
        __import__(import_193106)
        sys_modules_193107 = sys.modules[import_193106]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'unittest.main', sys_modules_193107.module_type_store, module_type_store, ['main', 'TestProgram', 'USAGE_AS_MAIN'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_193107, sys_modules_193107.module_type_store, module_type_store)
    else:
        from unittest.main import main, TestProgram, USAGE_AS_MAIN

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'unittest.main', None, module_type_store, ['main', 'TestProgram', 'USAGE_AS_MAIN'], [main, TestProgram, USAGE_AS_MAIN])

else:
    # Assigning a type to the variable 'unittest.main' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'unittest.main', import_193106)

remove_current_file_folder_from_path('C:/Python27/lib/unittest/')


# Assigning a Name to a Attribute (line 10):
# Getting the type of 'USAGE_AS_MAIN' (line 10)
USAGE_AS_MAIN_193108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 20), 'USAGE_AS_MAIN')
# Getting the type of 'TestProgram' (line 10)
TestProgram_193109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'TestProgram')
# Setting the type of the member 'USAGE' of a type (line 10)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 0), TestProgram_193109, 'USAGE', USAGE_AS_MAIN_193108)

# Call to main(...): (line 12)
# Processing the call keyword arguments (line 12)
# Getting the type of 'None' (line 12)
None_193111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'None', False)
keyword_193112 = None_193111
kwargs_193113 = {'module': keyword_193112}
# Getting the type of 'main' (line 12)
main_193110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'main', False)
# Calling main(args, kwargs) (line 12)
main_call_result_193114 = invoke(stypy.reporting.localization.Localization(__file__, 12, 0), main_193110, *[], **kwargs_193113)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
