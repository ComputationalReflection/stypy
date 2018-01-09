
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: a = 0
2: 
3: if a > 0:
4:     x = 3.4
5: else:
6:     x = True
7: 
8: c = x[2]  # Error detected
9: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 1):
int_7727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 4), 'int')
# Assigning a type to the variable 'a' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'a', int_7727)


# Getting the type of 'a' (line 3)
a_7728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 3), 'a')
int_7729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 7), 'int')
# Applying the binary operator '>' (line 3)
result_gt_7730 = python_operator(stypy.reporting.localization.Localization(__file__, 3, 3), '>', a_7728, int_7729)

# Testing the type of an if condition (line 3)
if_condition_7731 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3, 0), result_gt_7730)
# Assigning a type to the variable 'if_condition_7731' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'if_condition_7731', if_condition_7731)
# SSA begins for if statement (line 3)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Num to a Name (line 4):
float_7732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 8), 'float')
# Assigning a type to the variable 'x' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'x', float_7732)
# SSA branch for the else part of an if statement (line 3)
module_type_store.open_ssa_branch('else')

# Assigning a Name to a Name (line 6):
# Getting the type of 'True' (line 6)
True_7733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'True')
# Assigning a type to the variable 'x' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'x', True_7733)
# SSA join for if statement (line 3)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Subscript to a Name (line 8):

# Obtaining the type of the subscript
int_7734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 6), 'int')
# Getting the type of 'x' (line 8)
x_7735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'x')
# Obtaining the member '__getitem__' of a type (line 8)
getitem___7736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), x_7735, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 8)
subscript_call_result_7737 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), getitem___7736, int_7734)

# Assigning a type to the variable 'c' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'c', subscript_call_result_7737)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
