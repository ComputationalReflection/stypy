
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: c = False
3: out_and_if = 1
4: if not (c):
5:     out_and_if = "1"
6: 
7: result = out_and_if * 3
8: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Name to a Name (line 2):
# Getting the type of 'False' (line 2)
False_4914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'False')
# Assigning a type to the variable 'c' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'c', False_4914)

# Assigning a Num to a Name (line 3):
int_4915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 13), 'int')
# Assigning a type to the variable 'out_and_if' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'out_and_if', int_4915)


# Getting the type of 'c' (line 4)
c_4916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 8), 'c')
# Applying the 'not' unary operator (line 4)
result_not__4917 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 3), 'not', c_4916)

# Testing the type of an if condition (line 4)
if_condition_4918 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 4, 0), result_not__4917)
# Assigning a type to the variable 'if_condition_4918' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'if_condition_4918', if_condition_4918)
# SSA begins for if statement (line 4)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Str to a Name (line 5):
str_4919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'str', '1')
# Assigning a type to the variable 'out_and_if' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'out_and_if', str_4919)
# SSA join for if statement (line 4)
module_type_store = module_type_store.join_ssa_context()


# Assigning a BinOp to a Name (line 7):
# Getting the type of 'out_and_if' (line 7)
out_and_if_4920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'out_and_if')
int_4921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 22), 'int')
# Applying the binary operator '*' (line 7)
result_mul_4922 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 9), '*', out_and_if_4920, int_4921)

# Assigning a type to the variable 'result' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'result', result_mul_4922)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
