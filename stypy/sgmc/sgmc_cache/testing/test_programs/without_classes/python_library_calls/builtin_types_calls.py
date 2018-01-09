
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: l = [1, 3, False, "aa"]
3: 
4: x = l[0]
5: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 2):

# Obtaining an instance of the builtin type 'list' (line 2)
list_704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 2)
# Adding element type (line 2)
int_705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 4), list_704, int_705)
# Adding element type (line 2)
int_706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 4), list_704, int_706)
# Adding element type (line 2)
# Getting the type of 'False' (line 2)
False_707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 11), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 4), list_704, False_707)
# Adding element type (line 2)
str_708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 18), 'str', 'aa')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 4), list_704, str_708)

# Assigning a type to the variable 'l' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'l', list_704)

# Assigning a Subscript to a Name (line 4):

# Obtaining the type of the subscript
int_709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 6), 'int')
# Getting the type of 'l' (line 4)
l_710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'l')
# Obtaining the member '__getitem__' of a type (line 4)
getitem___711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), l_710, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 4)
subscript_call_result_712 = invoke(stypy.reporting.localization.Localization(__file__, 4, 4), getitem___711, int_709)

# Assigning a type to the variable 'x' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'x', subscript_call_result_712)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
