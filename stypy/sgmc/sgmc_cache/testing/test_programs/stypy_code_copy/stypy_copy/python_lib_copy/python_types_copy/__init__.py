
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: __author__ = 'Usuario'
2: __all__ = ['instantiation', 'type_inference', 'type_introspection', 'non_python_type', 'type']
3: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 1):
str_8867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 13), 'str', 'Usuario')
# Assigning a type to the variable '__author__' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), '__author__', str_8867)

# Assigning a List to a Name (line 2):
__all__ = ['instantiation', 'type_inference', 'type_introspection', 'non_python_type', 'type']
module_type_store.set_exportable_members(['instantiation', 'type_inference', 'type_introspection', 'non_python_type', 'type'])

# Obtaining an instance of the builtin type 'list' (line 2)
list_8868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 2)
# Adding element type (line 2)
str_8869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'str', 'instantiation')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_8868, str_8869)
# Adding element type (line 2)
str_8870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 28), 'str', 'type_inference')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_8868, str_8870)
# Adding element type (line 2)
str_8871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 46), 'str', 'type_introspection')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_8868, str_8871)
# Adding element type (line 2)
str_8872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 68), 'str', 'non_python_type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_8868, str_8872)
# Adding element type (line 2)
str_8873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 87), 'str', 'type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_8868, str_8873)

# Assigning a type to the variable '__all__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__all__', list_8868)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
