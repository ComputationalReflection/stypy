
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: __all__ = ['import']

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 1):
__all__ = ['import']
module_type_store.set_exportable_members(['import'])

# Obtaining an instance of the builtin type 'list' (line 1)
list_4965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 1)
# Adding element type (line 1)
str_4966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 11), 'str', 'import')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_4965, str_4966)

# Assigning a type to the variable '__all__' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), '__all__', list_4965)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
