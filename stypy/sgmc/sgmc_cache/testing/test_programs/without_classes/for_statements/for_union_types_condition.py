
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: if True:
4:     x = "hi"
5: 
6: else:
7:     x = 3
8: 
9: 
10: for c in x:
11:     r = c
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Getting the type of 'True' (line 3)
True_5449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 3), 'True')
# Testing the type of an if condition (line 3)
if_condition_5450 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3, 0), True_5449)
# Assigning a type to the variable 'if_condition_5450' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'if_condition_5450', if_condition_5450)
# SSA begins for if statement (line 3)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Str to a Name (line 4):
str_5451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 8), 'str', 'hi')
# Assigning a type to the variable 'x' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'x', str_5451)
# SSA branch for the else part of an if statement (line 3)
module_type_store.open_ssa_branch('else')

# Assigning a Num to a Name (line 7):
int_5452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 8), 'int')
# Assigning a type to the variable 'x' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'x', int_5452)
# SSA join for if statement (line 3)
module_type_store = module_type_store.join_ssa_context()


# Getting the type of 'x' (line 10)
x_5453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'x')
# Testing the type of a for loop iterable (line 10)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 10, 0), x_5453)
# Getting the type of the for loop variable (line 10)
for_loop_var_5454 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 10, 0), x_5453)
# Assigning a type to the variable 'c' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'c', for_loop_var_5454)
# SSA begins for a for statement (line 10)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Name to a Name (line 11):
# Getting the type of 'c' (line 11)
c_5455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'c')
# Assigning a type to the variable 'r' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'r', c_5455)
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
