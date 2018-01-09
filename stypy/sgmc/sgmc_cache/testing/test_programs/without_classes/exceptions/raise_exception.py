
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: if True:
4:     raise TypeError('arg is not a code object')

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Getting the type of 'True' (line 3)
True_2671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 3), 'True')
# Testing the type of an if condition (line 3)
if_condition_2672 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3, 0), True_2671)
# Assigning a type to the variable 'if_condition_2672' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'if_condition_2672', if_condition_2672)
# SSA begins for if statement (line 3)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to TypeError(...): (line 4)
# Processing the call arguments (line 4)
str_2674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 20), 'str', 'arg is not a code object')
# Processing the call keyword arguments (line 4)
kwargs_2675 = {}
# Getting the type of 'TypeError' (line 4)
TypeError_2673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 10), 'TypeError', False)
# Calling TypeError(args, kwargs) (line 4)
TypeError_call_result_2676 = invoke(stypy.reporting.localization.Localization(__file__, 4, 10), TypeError_2673, *[str_2674], **kwargs_2675)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 4, 4), TypeError_call_result_2676, 'raise parameter', BaseException)
# SSA join for if statement (line 3)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
