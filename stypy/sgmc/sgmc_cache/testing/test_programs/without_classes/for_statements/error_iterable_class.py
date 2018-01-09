
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: for elem in list:  # Error reported
2:     r = elem
3: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Getting the type of 'list' (line 1)
list_7738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 12), 'list')
# Testing the type of a for loop iterable (line 1)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1, 0), list_7738)
# Getting the type of the for loop variable (line 1)
for_loop_var_7739 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1, 0), list_7738)
# Assigning a type to the variable 'elem' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'elem', for_loop_var_7739)
# SSA begins for a for statement (line 1)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Name to a Name (line 2):
# Getting the type of 'elem' (line 2)
elem_7740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 8), 'elem')
# Assigning a type to the variable 'r' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'r', elem_7740)
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
