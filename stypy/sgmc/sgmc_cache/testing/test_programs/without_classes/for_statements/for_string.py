
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: string = "this is an string"
4: s = ""
5: a = 0
6: for c in string:
7:     s = s + c
8:     a = a + 1
9: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_5413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 9), 'str', 'this is an string')
# Assigning a type to the variable 'string' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'string', str_5413)

# Assigning a Str to a Name (line 4):
str_5414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 4), 'str', '')
# Assigning a type to the variable 's' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 's', str_5414)

# Assigning a Num to a Name (line 5):
int_5415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 4), 'int')
# Assigning a type to the variable 'a' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'a', int_5415)

# Getting the type of 'string' (line 6)
string_5416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 9), 'string')
# Testing the type of a for loop iterable (line 6)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 6, 0), string_5416)
# Getting the type of the for loop variable (line 6)
for_loop_var_5417 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 6, 0), string_5416)
# Assigning a type to the variable 'c' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'c', for_loop_var_5417)
# SSA begins for a for statement (line 6)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a BinOp to a Name (line 7):
# Getting the type of 's' (line 7)
s_5418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 's')
# Getting the type of 'c' (line 7)
c_5419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'c')
# Applying the binary operator '+' (line 7)
result_add_5420 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 8), '+', s_5418, c_5419)

# Assigning a type to the variable 's' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 's', result_add_5420)

# Assigning a BinOp to a Name (line 8):
# Getting the type of 'a' (line 8)
a_5421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'a')
int_5422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 12), 'int')
# Applying the binary operator '+' (line 8)
result_add_5423 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 8), '+', a_5421, int_5422)

# Assigning a type to the variable 'a' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'a', result_add_5423)
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
