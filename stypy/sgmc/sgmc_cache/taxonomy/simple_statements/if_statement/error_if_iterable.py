
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: 
4: l = 4
5: 
6: # Type error
7: if 3 in l:
8:     print "a"
9: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 4):
int_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 4), 'int')
# Assigning a type to the variable 'l' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'l', int_1)


int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 3), 'int')
# Getting the type of 'l' (line 7)
l_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'l')
# Applying the binary operator 'in' (line 7)
result_contains_4 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 3), 'in', int_2, l_3)

# Testing the type of an if condition (line 7)
if_condition_5 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 7, 0), result_contains_4)
# Assigning a type to the variable 'if_condition_5' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'if_condition_5', if_condition_5)
# SSA begins for if statement (line 7)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'str', 'a')
# SSA join for if statement (line 7)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
