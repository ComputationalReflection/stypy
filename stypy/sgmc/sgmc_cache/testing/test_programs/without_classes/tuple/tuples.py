
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: 
4: a = (4, 5)
5: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Tuple to a Name (line 4):

# Obtaining an instance of the builtin type 'tuple' (line 4)
tuple_5353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4)
# Adding element type (line 4)
int_5354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 5), tuple_5353, int_5354)
# Adding element type (line 4)
int_5355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 5), tuple_5353, int_5355)

# Assigning a type to the variable 'a' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'a', tuple_5353)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
