
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: __all__ = ["python_src_generation_copy", "python_src_print_copy", "python_type_data_autogeneration_copy",
2:            "type_annotation_copy", "type_inference_copy"]
3: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 1):
__all__ = ['python_src_generation_copy', 'python_src_print_copy', 'python_type_data_autogeneration_copy', 'type_annotation_copy', 'type_inference_copy']
module_type_store.set_exportable_members(['python_src_generation_copy', 'python_src_print_copy', 'python_type_data_autogeneration_copy', 'type_annotation_copy', 'type_inference_copy'])

# Obtaining an instance of the builtin type 'list' (line 1)
list_15271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 1)
# Adding element type (line 1)
str_15272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 11), 'str', 'python_src_generation_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_15271, str_15272)
# Adding element type (line 1)
str_15273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 41), 'str', 'python_src_print_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_15271, str_15273)
# Adding element type (line 1)
str_15274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 66), 'str', 'python_type_data_autogeneration_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_15271, str_15274)
# Adding element type (line 1)
str_15275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'str', 'type_annotation_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_15271, str_15275)
# Adding element type (line 1)
str_15276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 35), 'str', 'type_inference_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_15271, str_15276)

# Assigning a type to the variable '__all__' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), '__all__', list_15271)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
