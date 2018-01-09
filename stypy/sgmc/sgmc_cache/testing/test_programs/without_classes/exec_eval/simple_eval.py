
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: x ="str"
4: 
5: x = eval('4+7/2')
6: 
7: y = eval(1)
8: 
9: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_2633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 3), 'str', 'str')
# Assigning a type to the variable 'x' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'x', str_2633)

# Assigning a Call to a Name (line 5):

# Call to eval(...): (line 5)
# Processing the call arguments (line 5)
str_2635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 9), 'str', '4+7/2')
# Processing the call keyword arguments (line 5)
kwargs_2636 = {}
# Getting the type of 'eval' (line 5)
eval_2634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'eval', False)
# Calling eval(args, kwargs) (line 5)
eval_call_result_2637 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), eval_2634, *[str_2635], **kwargs_2636)

# Assigning a type to the variable 'x' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'x', eval_call_result_2637)

# Assigning a Call to a Name (line 7):

# Call to eval(...): (line 7)
# Processing the call arguments (line 7)
int_2639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 9), 'int')
# Processing the call keyword arguments (line 7)
kwargs_2640 = {}
# Getting the type of 'eval' (line 7)
eval_2638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'eval', False)
# Calling eval(args, kwargs) (line 7)
eval_call_result_2641 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), eval_2638, *[int_2639], **kwargs_2640)

# Assigning a type to the variable 'y' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'y', eval_call_result_2641)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
