
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: __author__ = 'Usuario'
2: __all__ = ['localization_copy', 'no_recursion_copy', 'type_inference_proxy_copy', 'type_inference_proxy_management_copy',
3:            'undefined_type_copy', 'union_type_copy']
4: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 1):
str_14232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 13), 'str', 'Usuario')
# Assigning a type to the variable '__author__' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), '__author__', str_14232)

# Assigning a List to a Name (line 2):
__all__ = ['localization_copy', 'no_recursion_copy', 'type_inference_proxy_copy', 'type_inference_proxy_management_copy', 'undefined_type_copy', 'union_type_copy']
module_type_store.set_exportable_members(['localization_copy', 'no_recursion_copy', 'type_inference_proxy_copy', 'type_inference_proxy_management_copy', 'undefined_type_copy', 'union_type_copy'])

# Obtaining an instance of the builtin type 'list' (line 2)
list_14233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 2)
# Adding element type (line 2)
str_14234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'str', 'localization_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_14233, str_14234)
# Adding element type (line 2)
str_14235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 32), 'str', 'no_recursion_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_14233, str_14235)
# Adding element type (line 2)
str_14236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 53), 'str', 'type_inference_proxy_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_14233, str_14236)
# Adding element type (line 2)
str_14237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 82), 'str', 'type_inference_proxy_management_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_14233, str_14237)
# Adding element type (line 2)
str_14238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'str', 'undefined_type_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_14233, str_14238)
# Adding element type (line 2)
str_14239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 34), 'str', 'union_type_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_14233, str_14239)

# Assigning a type to the variable '__all__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__all__', list_14233)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
