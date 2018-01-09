
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: a = range(1)
3: b = range(1, 10)
4: c = range(1, 10, 2)
5: 
6: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Call to a Name (line 2):

# Call to range(...): (line 2)
# Processing the call arguments (line 2)
int_690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'int')
# Processing the call keyword arguments (line 2)
kwargs_691 = {}
# Getting the type of 'range' (line 2)
range_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'range', False)
# Calling range(args, kwargs) (line 2)
range_call_result_692 = invoke(stypy.reporting.localization.Localization(__file__, 2, 4), range_689, *[int_690], **kwargs_691)

# Assigning a type to the variable 'a' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'a', range_call_result_692)

# Assigning a Call to a Name (line 3):

# Call to range(...): (line 3)
# Processing the call arguments (line 3)
int_694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'int')
int_695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 13), 'int')
# Processing the call keyword arguments (line 3)
kwargs_696 = {}
# Getting the type of 'range' (line 3)
range_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'range', False)
# Calling range(args, kwargs) (line 3)
range_call_result_697 = invoke(stypy.reporting.localization.Localization(__file__, 3, 4), range_693, *[int_694, int_695], **kwargs_696)

# Assigning a type to the variable 'b' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'b', range_call_result_697)

# Assigning a Call to a Name (line 4):

# Call to range(...): (line 4)
# Processing the call arguments (line 4)
int_699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'int')
int_700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 13), 'int')
int_701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 17), 'int')
# Processing the call keyword arguments (line 4)
kwargs_702 = {}
# Getting the type of 'range' (line 4)
range_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'range', False)
# Calling range(args, kwargs) (line 4)
range_call_result_703 = invoke(stypy.reporting.localization.Localization(__file__, 4, 4), range_698, *[int_699, int_700, int_701], **kwargs_702)

# Assigning a type to the variable 'c' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'c', range_call_result_703)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
