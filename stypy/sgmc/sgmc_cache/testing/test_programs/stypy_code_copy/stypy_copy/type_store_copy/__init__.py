
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: __all__ = ['function_context_copy', 'type_annotation_record_copy', 'typestore_copy']

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 1):
__all__ = ['function_context_copy', 'type_annotation_record_copy', 'typestore_copy']
module_type_store.set_exportable_members(['function_context_copy', 'type_annotation_record_copy', 'typestore_copy'])

# Obtaining an instance of the builtin type 'list' (line 1)
list_18867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 1)
# Adding element type (line 1)
str_18868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 11), 'str', 'function_context_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_18867, str_18868)
# Adding element type (line 1)
str_18869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 36), 'str', 'type_annotation_record_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_18867, str_18869)
# Adding element type (line 1)
str_18870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 67), 'str', 'typestore_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_18867, str_18870)

# Assigning a type to the variable '__all__' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), '__all__', list_18867)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
