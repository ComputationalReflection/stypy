
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: __author__ = 'Usuario'
2: __all__ = ['instantiation_copy', 'type_inference_copy', 'type_introspection_copy', 'non_python_type_copy', 'type_copy']
3: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 1):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 13), 'str', 'Usuario')
# Assigning a type to the variable '__author__' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), '__author__', str_1)

# Assigning a List to a Name (line 2):
__all__ = ['instantiation_copy', 'type_inference_copy', 'type_introspection_copy', 'non_python_type_copy', 'type_copy']
module_type_store.set_exportable_members(['instantiation_copy', 'type_inference_copy', 'type_introspection_copy', 'non_python_type_copy', 'type_copy'])

# Obtaining an instance of the builtin type 'list' (line 2)
list_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 2)
# Adding element type (line 2)
str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'str', 'instantiation_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_2, str_3)
# Adding element type (line 2)
str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 33), 'str', 'type_inference_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_2, str_4)
# Adding element type (line 2)
str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 56), 'str', 'type_introspection_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_2, str_5)
# Adding element type (line 2)
str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 83), 'str', 'non_python_type_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_2, str_6)
# Adding element type (line 2)
str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 107), 'str', 'type_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_2, str_7)

# Assigning a type to the variable '__all__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__all__', list_2)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
