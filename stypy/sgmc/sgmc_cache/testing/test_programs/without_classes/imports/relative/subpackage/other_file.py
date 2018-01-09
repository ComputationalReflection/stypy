
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: __all__ = ["x", "y"]
3: 
4: x = 2
5: y = 5
6: 
7: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 2):
__all__ = ['x', 'y']
module_type_store.set_exportable_members(['x', 'y'])

# Obtaining an instance of the builtin type 'list' (line 2)
list_5216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 2)
# Adding element type (line 2)
str_5217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'str', 'x')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_5216, str_5217)
# Adding element type (line 2)
str_5218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 16), 'str', 'y')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 10), list_5216, str_5218)

# Assigning a type to the variable '__all__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__all__', list_5216)

# Assigning a Num to a Name (line 4):
int_5219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 4), 'int')
# Assigning a type to the variable 'x' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'x', int_5219)

# Assigning a Num to a Name (line 5):
int_5220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 4), 'int')
# Assigning a type to the variable 'y' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'y', int_5220)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
