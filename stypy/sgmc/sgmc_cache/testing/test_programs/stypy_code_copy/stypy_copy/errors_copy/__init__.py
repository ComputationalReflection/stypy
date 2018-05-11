
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: __author__ = 'Usuario'
2: __all__ = ['type_warning_copy', 'singleton_copy', 'stack_trace_copy', 'type_error_copy', 'undefined_type_error_copy', 'unsupported_features_copy']
3: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 1):
str_4195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 13), 'str', 'Usuario')
# Assigning a type to the variable '__author__' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), '__author__', str_4195)

# Assigning a List to a Name (line 2):
__all__ = ['type_warning_copy', 'singleton_copy', 'stack_trace_copy', 'type_error_copy', 'undefined_type_error_copy', 'unsupported_features_copy']
module_type_store.set_exportable_members(['type_warning_copy', 'singleton_copy', 'stack_trace_copy', 'type_error_copy', 'undefined_type_error_copy', 'unsupported_features_copy'])

# Obtaining an instance of the builtin type 'list' (line 2)
list_4196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 2)
# Adding element type (line 2)
str_4197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'str', 'type_warning_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_4196, str_4197)
# Adding element type (line 2)
str_4198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 32), 'str', 'singleton_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_4196, str_4198)
# Adding element type (line 2)
str_4199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 50), 'str', 'stack_trace_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_4196, str_4199)
# Adding element type (line 2)
str_4200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 70), 'str', 'type_error_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_4196, str_4200)
# Adding element type (line 2)
str_4201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 89), 'str', 'undefined_type_error_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_4196, str_4201)
# Adding element type (line 2)
str_4202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 118), 'str', 'unsupported_features_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_4196, str_4202)

# Assigning a type to the variable '__all__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__all__', list_4196)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
