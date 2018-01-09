
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: l = [1,2,4,5]
4: 
5: a = 2
6: 
7: x = l[1]
8: 
9: y = l[a]
10: 
11: z = l[a+1]
12: 
13: s = "abcd"
14: 
15: c = s[2]
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 3):

# Obtaining an instance of the builtin type 'list' (line 3)
list_6417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 3)
# Adding element type (line 3)
int_6418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 4), list_6417, int_6418)
# Adding element type (line 3)
int_6419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 7), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 4), list_6417, int_6419)
# Adding element type (line 3)
int_6420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 4), list_6417, int_6420)
# Adding element type (line 3)
int_6421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 4), list_6417, int_6421)

# Assigning a type to the variable 'l' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'l', list_6417)

# Assigning a Num to a Name (line 5):
int_6422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 4), 'int')
# Assigning a type to the variable 'a' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'a', int_6422)

# Assigning a Subscript to a Name (line 7):

# Obtaining the type of the subscript
int_6423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 6), 'int')
# Getting the type of 'l' (line 7)
l_6424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'l')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___6425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), l_6424, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_6426 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), getitem___6425, int_6423)

# Assigning a type to the variable 'x' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'x', subscript_call_result_6426)

# Assigning a Subscript to a Name (line 9):

# Obtaining the type of the subscript
# Getting the type of 'a' (line 9)
a_6427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 6), 'a')
# Getting the type of 'l' (line 9)
l_6428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'l')
# Obtaining the member '__getitem__' of a type (line 9)
getitem___6429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), l_6428, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 9)
subscript_call_result_6430 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), getitem___6429, a_6427)

# Assigning a type to the variable 'y' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'y', subscript_call_result_6430)

# Assigning a Subscript to a Name (line 11):

# Obtaining the type of the subscript
# Getting the type of 'a' (line 11)
a_6431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 6), 'a')
int_6432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 8), 'int')
# Applying the binary operator '+' (line 11)
result_add_6433 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 6), '+', a_6431, int_6432)

# Getting the type of 'l' (line 11)
l_6434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'l')
# Obtaining the member '__getitem__' of a type (line 11)
getitem___6435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), l_6434, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 11)
subscript_call_result_6436 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), getitem___6435, result_add_6433)

# Assigning a type to the variable 'z' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'z', subscript_call_result_6436)

# Assigning a Str to a Name (line 13):
str_6437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 4), 'str', 'abcd')
# Assigning a type to the variable 's' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 's', str_6437)

# Assigning a Subscript to a Name (line 15):

# Obtaining the type of the subscript
int_6438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 6), 'int')
# Getting the type of 's' (line 15)
s_6439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 's')
# Obtaining the member '__getitem__' of a type (line 15)
getitem___6440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), s_6439, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 15)
subscript_call_result_6441 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), getitem___6440, int_6438)

# Assigning a type to the variable 'c' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'c', subscript_call_result_6441)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
