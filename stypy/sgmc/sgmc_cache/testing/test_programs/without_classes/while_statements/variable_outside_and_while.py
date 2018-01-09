
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: a = 3
3: b = False
4: while a > 0:
5:     b = "hi"
6:     a = a - 1
7: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 2):
int_5456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 4), 'int')
# Assigning a type to the variable 'a' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'a', int_5456)

# Assigning a Name to a Name (line 3):
# Getting the type of 'False' (line 3)
False_5457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'False')
# Assigning a type to the variable 'b' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'b', False_5457)


# Getting the type of 'a' (line 4)
a_5458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 6), 'a')
int_5459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'int')
# Applying the binary operator '>' (line 4)
result_gt_5460 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 6), '>', a_5458, int_5459)

# Testing the type of an if condition (line 4)
is_suitable_condition(stypy.reporting.localization.Localization(__file__, 4, 0), result_gt_5460)
# SSA begins for while statement (line 4)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')

# Assigning a Str to a Name (line 5):
str_5461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 8), 'str', 'hi')
# Assigning a type to the variable 'b' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'b', str_5461)

# Assigning a BinOp to a Name (line 6):
# Getting the type of 'a' (line 6)
a_5462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'a')
int_5463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'int')
# Applying the binary operator '-' (line 6)
result_sub_5464 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 8), '-', a_5462, int_5463)

# Assigning a type to the variable 'a' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'a', result_sub_5464)
# SSA join for while statement (line 4)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
