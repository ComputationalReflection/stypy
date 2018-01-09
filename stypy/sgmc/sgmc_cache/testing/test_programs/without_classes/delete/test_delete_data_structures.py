
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: a = 3
4: 
5: t = ("z","a")
6: 
7: l = range(5)
8: 
9: d = {
10:     'a': 1,
11:     'b': 2,
12: }
13: 
14: # Right
15: del l[1]
16: del d['b']
17: 
18: if True:
19:     del d['a']
20: else:
21:     del l[2]
22: 
23: # Wrong
24: del t[1]
25: del a['b']
26: 
27: if True:
28:     del t['a']
29: else:
30:     del a[2]

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 3):
int_6291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 4), 'int')
# Assigning a type to the variable 'a' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'a', int_6291)

# Assigning a Tuple to a Name (line 5):

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_6292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
str_6293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 5), 'str', 'z')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 5), tuple_6292, str_6293)
# Adding element type (line 5)
str_6294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 9), 'str', 'a')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 5), tuple_6292, str_6294)

# Assigning a type to the variable 't' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 't', tuple_6292)

# Assigning a Call to a Name (line 7):

# Call to range(...): (line 7)
# Processing the call arguments (line 7)
int_6296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'int')
# Processing the call keyword arguments (line 7)
kwargs_6297 = {}
# Getting the type of 'range' (line 7)
range_6295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'range', False)
# Calling range(args, kwargs) (line 7)
range_call_result_6298 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), range_6295, *[int_6296], **kwargs_6297)

# Assigning a type to the variable 'l' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'l', range_call_result_6298)

# Assigning a Dict to a Name (line 9):

# Obtaining an instance of the builtin type 'dict' (line 9)
dict_6299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 9)
# Adding element type (key, value) (line 9)
str_6300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'str', 'a')
int_6301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 9), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 4), dict_6299, (str_6300, int_6301))
# Adding element type (key, value) (line 9)
str_6302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 4), 'str', 'b')
int_6303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 9), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 4), dict_6299, (str_6302, int_6303))

# Assigning a type to the variable 'd' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'd', dict_6299)
# Deleting a member
# Getting the type of 'l' (line 15)
l_6304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'l')

# Obtaining the type of the subscript
int_6305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 6), 'int')
# Getting the type of 'l' (line 15)
l_6306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'l')
# Obtaining the member '__getitem__' of a type (line 15)
getitem___6307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), l_6306, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 15)
subscript_call_result_6308 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), getitem___6307, int_6305)

del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 0), l_6304, subscript_call_result_6308)
# Deleting a member
# Getting the type of 'd' (line 16)
d_6309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'd')

# Obtaining the type of the subscript
str_6310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 6), 'str', 'b')
# Getting the type of 'd' (line 16)
d_6311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'd')
# Obtaining the member '__getitem__' of a type (line 16)
getitem___6312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), d_6311, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 16)
subscript_call_result_6313 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), getitem___6312, str_6310)

del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 0), d_6309, subscript_call_result_6313)

# Getting the type of 'True' (line 18)
True_6314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 3), 'True')
# Testing the type of an if condition (line 18)
if_condition_6315 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 0), True_6314)
# Assigning a type to the variable 'if_condition_6315' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'if_condition_6315', if_condition_6315)
# SSA begins for if statement (line 18)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
# Deleting a member
# Getting the type of 'd' (line 19)
d_6316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'd')

# Obtaining the type of the subscript
str_6317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 10), 'str', 'a')
# Getting the type of 'd' (line 19)
d_6318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'd')
# Obtaining the member '__getitem__' of a type (line 19)
getitem___6319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), d_6318, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 19)
subscript_call_result_6320 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), getitem___6319, str_6317)

del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 4), d_6316, subscript_call_result_6320)
# SSA branch for the else part of an if statement (line 18)
module_type_store.open_ssa_branch('else')
# Deleting a member
# Getting the type of 'l' (line 21)
l_6321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'l')

# Obtaining the type of the subscript
int_6322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 10), 'int')
# Getting the type of 'l' (line 21)
l_6323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'l')
# Obtaining the member '__getitem__' of a type (line 21)
getitem___6324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), l_6323, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 21)
subscript_call_result_6325 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), getitem___6324, int_6322)

del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 4), l_6321, subscript_call_result_6325)
# SSA join for if statement (line 18)
module_type_store = module_type_store.join_ssa_context()

# Deleting a member
# Getting the type of 't' (line 24)
t_6326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 't')

# Obtaining the type of the subscript
int_6327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 6), 'int')
# Getting the type of 't' (line 24)
t_6328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 't')
# Obtaining the member '__getitem__' of a type (line 24)
getitem___6329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), t_6328, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 24)
subscript_call_result_6330 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), getitem___6329, int_6327)

del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 0), t_6326, subscript_call_result_6330)
# Deleting a member
# Getting the type of 'a' (line 25)
a_6331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'a')

# Obtaining the type of the subscript
str_6332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 6), 'str', 'b')
# Getting the type of 'a' (line 25)
a_6333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'a')
# Obtaining the member '__getitem__' of a type (line 25)
getitem___6334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), a_6333, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 25)
subscript_call_result_6335 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), getitem___6334, str_6332)

del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 0), a_6331, subscript_call_result_6335)

# Getting the type of 'True' (line 27)
True_6336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 3), 'True')
# Testing the type of an if condition (line 27)
if_condition_6337 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 0), True_6336)
# Assigning a type to the variable 'if_condition_6337' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'if_condition_6337', if_condition_6337)
# SSA begins for if statement (line 27)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
# Deleting a member
# Getting the type of 't' (line 28)
t_6338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 't')

# Obtaining the type of the subscript
str_6339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 10), 'str', 'a')
# Getting the type of 't' (line 28)
t_6340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 't')
# Obtaining the member '__getitem__' of a type (line 28)
getitem___6341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), t_6340, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 28)
subscript_call_result_6342 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), getitem___6341, str_6339)

del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 4), t_6338, subscript_call_result_6342)
# SSA branch for the else part of an if statement (line 27)
module_type_store.open_ssa_branch('else')
# Deleting a member
# Getting the type of 'a' (line 30)
a_6343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'a')

# Obtaining the type of the subscript
int_6344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 10), 'int')
# Getting the type of 'a' (line 30)
a_6345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'a')
# Obtaining the member '__getitem__' of a type (line 30)
getitem___6346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), a_6345, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 30)
subscript_call_result_6347 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), getitem___6346, int_6344)

del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 4), a_6343, subscript_call_result_6347)
# SSA join for if statement (line 27)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
