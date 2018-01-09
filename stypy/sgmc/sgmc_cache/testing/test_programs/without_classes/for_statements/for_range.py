
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: a = 3
4: b = False
5: for i in range(1000):
6:     b = "hi"
7:     a = a - 1
8: 
9: 
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 3):
int_5402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 4), 'int')
# Assigning a type to the variable 'a' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'a', int_5402)

# Assigning a Name to a Name (line 4):
# Getting the type of 'False' (line 4)
False_5403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'False')
# Assigning a type to the variable 'b' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'b', False_5403)


# Call to range(...): (line 5)
# Processing the call arguments (line 5)
int_5405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'int')
# Processing the call keyword arguments (line 5)
kwargs_5406 = {}
# Getting the type of 'range' (line 5)
range_5404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 9), 'range', False)
# Calling range(args, kwargs) (line 5)
range_call_result_5407 = invoke(stypy.reporting.localization.Localization(__file__, 5, 9), range_5404, *[int_5405], **kwargs_5406)

# Testing the type of a for loop iterable (line 5)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 5, 0), range_call_result_5407)
# Getting the type of the for loop variable (line 5)
for_loop_var_5408 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 5, 0), range_call_result_5407)
# Assigning a type to the variable 'i' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'i', for_loop_var_5408)
# SSA begins for a for statement (line 5)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Str to a Name (line 6):
str_5409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 8), 'str', 'hi')
# Assigning a type to the variable 'b' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'b', str_5409)

# Assigning a BinOp to a Name (line 7):
# Getting the type of 'a' (line 7)
a_5410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'a')
int_5411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 12), 'int')
# Applying the binary operator '-' (line 7)
result_sub_5412 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 8), '-', a_5410, int_5411)

# Assigning a type to the variable 'a' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'a', result_sub_5412)
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
