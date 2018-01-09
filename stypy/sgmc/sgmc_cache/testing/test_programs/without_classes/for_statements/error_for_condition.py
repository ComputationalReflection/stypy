
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: nargs = "hi"
4: 
5: for i in range(nargs):
6:     x = i
7: 
8: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_5356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 8), 'str', 'hi')
# Assigning a type to the variable 'nargs' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'nargs', str_5356)


# Call to range(...): (line 5)
# Processing the call arguments (line 5)
# Getting the type of 'nargs' (line 5)
nargs_5358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 15), 'nargs', False)
# Processing the call keyword arguments (line 5)
kwargs_5359 = {}
# Getting the type of 'range' (line 5)
range_5357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 9), 'range', False)
# Calling range(args, kwargs) (line 5)
range_call_result_5360 = invoke(stypy.reporting.localization.Localization(__file__, 5, 9), range_5357, *[nargs_5358], **kwargs_5359)

# Testing the type of a for loop iterable (line 5)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 5, 0), range_call_result_5360)
# Getting the type of the for loop variable (line 5)
for_loop_var_5361 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 5, 0), range_call_result_5360)
# Assigning a type to the variable 'i' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'i', for_loop_var_5361)
# SSA begins for a for statement (line 5)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Name to a Name (line 6):
# Getting the type of 'i' (line 6)
i_5362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'i')
# Assigning a type to the variable 'x' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'x', i_5362)
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
