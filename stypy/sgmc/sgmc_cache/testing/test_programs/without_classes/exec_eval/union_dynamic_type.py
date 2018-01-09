
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: x = eval("3+2")
4: y = eval("5.6 + 4.3")
5: 
6: r = "hi"
7: 
8: if True:
9:     r = x
10: 
11: if True:
12:     r = y
13: 
14: y2 = r.endswith("i")
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Call to a Name (line 3):

# Call to eval(...): (line 3)
# Processing the call arguments (line 3)
str_2646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 9), 'str', '3+2')
# Processing the call keyword arguments (line 3)
kwargs_2647 = {}
# Getting the type of 'eval' (line 3)
eval_2645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'eval', False)
# Calling eval(args, kwargs) (line 3)
eval_call_result_2648 = invoke(stypy.reporting.localization.Localization(__file__, 3, 4), eval_2645, *[str_2646], **kwargs_2647)

# Assigning a type to the variable 'x' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'x', eval_call_result_2648)

# Assigning a Call to a Name (line 4):

# Call to eval(...): (line 4)
# Processing the call arguments (line 4)
str_2650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 9), 'str', '5.6 + 4.3')
# Processing the call keyword arguments (line 4)
kwargs_2651 = {}
# Getting the type of 'eval' (line 4)
eval_2649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'eval', False)
# Calling eval(args, kwargs) (line 4)
eval_call_result_2652 = invoke(stypy.reporting.localization.Localization(__file__, 4, 4), eval_2649, *[str_2650], **kwargs_2651)

# Assigning a type to the variable 'y' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'y', eval_call_result_2652)

# Assigning a Str to a Name (line 6):
str_2653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'str', 'hi')
# Assigning a type to the variable 'r' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r', str_2653)

# Getting the type of 'True' (line 8)
True_2654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 3), 'True')
# Testing the type of an if condition (line 8)
if_condition_2655 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 8, 0), True_2654)
# Assigning a type to the variable 'if_condition_2655' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'if_condition_2655', if_condition_2655)
# SSA begins for if statement (line 8)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 9):
# Getting the type of 'x' (line 9)
x_2656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'x')
# Assigning a type to the variable 'r' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'r', x_2656)
# SSA join for if statement (line 8)
module_type_store = module_type_store.join_ssa_context()


# Getting the type of 'True' (line 11)
True_2657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 3), 'True')
# Testing the type of an if condition (line 11)
if_condition_2658 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 11, 0), True_2657)
# Assigning a type to the variable 'if_condition_2658' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'if_condition_2658', if_condition_2658)
# SSA begins for if statement (line 11)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 12):
# Getting the type of 'y' (line 12)
y_2659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'y')
# Assigning a type to the variable 'r' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'r', y_2659)
# SSA join for if statement (line 11)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Call to a Name (line 14):

# Call to endswith(...): (line 14)
# Processing the call arguments (line 14)
str_2662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'str', 'i')
# Processing the call keyword arguments (line 14)
kwargs_2663 = {}
# Getting the type of 'r' (line 14)
r_2660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'r', False)
# Obtaining the member 'endswith' of a type (line 14)
endswith_2661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), r_2660, 'endswith')
# Calling endswith(args, kwargs) (line 14)
endswith_call_result_2664 = invoke(stypy.reporting.localization.Localization(__file__, 14, 5), endswith_2661, *[str_2662], **kwargs_2663)

# Assigning a type to the variable 'y2' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'y2', endswith_call_result_2664)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
