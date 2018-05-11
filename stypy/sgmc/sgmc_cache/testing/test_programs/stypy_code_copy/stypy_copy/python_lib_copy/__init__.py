
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: __author__ = 'Redondo'
2: __all__ = ['member_call_copy', 'module_imports_copy', 'python_types_copy', 'type_rules_copy']
3: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 1):
str_4920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 13), 'str', 'Redondo')
# Assigning a type to the variable '__author__' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), '__author__', str_4920)

# Assigning a List to a Name (line 2):
__all__ = ['member_call_copy', 'module_imports_copy', 'python_types_copy', 'type_rules_copy']
module_type_store.set_exportable_members(['member_call_copy', 'module_imports_copy', 'python_types_copy', 'type_rules_copy'])

# Obtaining an instance of the builtin type 'list' (line 2)
list_4921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 2)
# Adding element type (line 2)
str_4922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'str', 'member_call_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_4921, str_4922)
# Adding element type (line 2)
str_4923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 31), 'str', 'module_imports_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_4921, str_4923)
# Adding element type (line 2)
str_4924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 54), 'str', 'python_types_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_4921, str_4924)
# Adding element type (line 2)
str_4925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 75), 'str', 'type_rules_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_4921, str_4925)

# Assigning a type to the variable '__all__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__all__', list_4921)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
