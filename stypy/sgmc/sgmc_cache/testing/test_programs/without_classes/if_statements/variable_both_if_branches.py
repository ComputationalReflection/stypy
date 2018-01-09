
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: a = 0
2: if a < 0:
3:     x = "test"
4: else:
5:     x = 4
6: 
7: c = x[2]  # Not detected
8: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 1):
int_4923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 4), 'int')
# Assigning a type to the variable 'a' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'a', int_4923)


# Getting the type of 'a' (line 2)
a_4924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 3), 'a')
int_4925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 7), 'int')
# Applying the binary operator '<' (line 2)
result_lt_4926 = python_operator(stypy.reporting.localization.Localization(__file__, 2, 3), '<', a_4924, int_4925)

# Testing the type of an if condition (line 2)
if_condition_4927 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2, 0), result_lt_4926)
# Assigning a type to the variable 'if_condition_4927' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'if_condition_4927', if_condition_4927)
# SSA begins for if statement (line 2)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Str to a Name (line 3):
str_4928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 8), 'str', 'test')
# Assigning a type to the variable 'x' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'x', str_4928)
# SSA branch for the else part of an if statement (line 2)
module_type_store.open_ssa_branch('else')

# Assigning a Num to a Name (line 5):
int_4929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 8), 'int')
# Assigning a type to the variable 'x' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'x', int_4929)
# SSA join for if statement (line 2)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Subscript to a Name (line 7):

# Obtaining the type of the subscript
int_4930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 6), 'int')
# Getting the type of 'x' (line 7)
x_4931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'x')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___4932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), x_4931, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_4933 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), getitem___4932, int_4930)

# Assigning a type to the variable 'c' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'c', subscript_call_result_4933)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
