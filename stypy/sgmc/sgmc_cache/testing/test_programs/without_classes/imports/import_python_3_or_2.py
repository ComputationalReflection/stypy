
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: import sys
3: 
4: if sys.version_info[0] >= 3:
5:     import pickle
6:     basestring = str
7:     import builtins
8: else:
9:     import cPickle as pickle
10:     import __builtin__ as builtins
11: 
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import sys' statement (line 2)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'sys', sys, module_type_store)




# Obtaining the type of the subscript
int_5089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 20), 'int')
# Getting the type of 'sys' (line 4)
sys_5090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 4)
version_info_5091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 3), sys_5090, 'version_info')
# Obtaining the member '__getitem__' of a type (line 4)
getitem___5092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 3), version_info_5091, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 4)
subscript_call_result_5093 = invoke(stypy.reporting.localization.Localization(__file__, 4, 3), getitem___5092, int_5089)

int_5094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 26), 'int')
# Applying the binary operator '>=' (line 4)
result_ge_5095 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 3), '>=', subscript_call_result_5093, int_5094)

# Testing the type of an if condition (line 4)
if_condition_5096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 4, 0), result_ge_5095)
# Assigning a type to the variable 'if_condition_5096' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'if_condition_5096', if_condition_5096)
# SSA begins for if statement (line 4)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 4))

# 'import pickle' statement (line 5)
import pickle

import_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'pickle', pickle, module_type_store)


# Assigning a Name to a Name (line 6):
# Getting the type of 'str' (line 6)
str_5097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 17), 'str')
# Assigning a type to the variable 'basestring' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'basestring', str_5097)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 4))

# 'import builtins' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')
import_5098 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'builtins')

if (type(import_5098) is not StypyTypeError):

    if (import_5098 != 'pyd_module'):
        __import__(import_5098)
        sys_modules_5099 = sys.modules[import_5098]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'builtins', sys_modules_5099.module_type_store, module_type_store)
    else:
        import builtins

        import_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'builtins', builtins, module_type_store)

else:
    # Assigning a type to the variable 'builtins' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'builtins', import_5098)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')

# SSA branch for the else part of an if statement (line 4)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 4))

# 'import cPickle' statement (line 9)
import cPickle as pickle

import_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'pickle', pickle, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 4))

# 'import __builtin__' statement (line 10)
import __builtin__ as builtins

import_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'builtins', builtins, module_type_store)

# SSA join for if statement (line 4)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
