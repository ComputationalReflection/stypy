
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: kwargs = {
3:     'constant_values': [(1, 1.0),  (2, 2.0), (3, 3.0)]
4: }
5: pad_width = [('a', 'a'), ('b', 'b'), ('c', 'c')]
6: 
7: for axis, ((pad_before, pad_after), (before_val, after_val)) in enumerate(
8:         zip(pad_width, kwargs['constant_values'])):
9:     print axis
10:     print pad_before
11:     print pad_after
12:     print before_val
13:     print after_val

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Dict to a Name (line 2):

# Obtaining an instance of the builtin type 'dict' (line 2)
dict_5363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 2)
# Adding element type (key, value) (line 2)
str_5364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 4), 'str', 'constant_values')

# Obtaining an instance of the builtin type 'list' (line 3)
list_5365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 3)
# Adding element type (line 3)

# Obtaining an instance of the builtin type 'tuple' (line 3)
tuple_5366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3)
# Adding element type (line 3)
int_5367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 25), tuple_5366, int_5367)
# Adding element type (line 3)
float_5368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 28), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 25), tuple_5366, float_5368)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 23), list_5365, tuple_5366)
# Adding element type (line 3)

# Obtaining an instance of the builtin type 'tuple' (line 3)
tuple_5369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3)
# Adding element type (line 3)
int_5370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 36), tuple_5369, int_5370)
# Adding element type (line 3)
float_5371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 39), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 36), tuple_5369, float_5371)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 23), list_5365, tuple_5369)
# Adding element type (line 3)

# Obtaining an instance of the builtin type 'tuple' (line 3)
tuple_5372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 46), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 3)
# Adding element type (line 3)
int_5373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 46), tuple_5372, int_5373)
# Adding element type (line 3)
float_5374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 46), tuple_5372, float_5374)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 23), list_5365, tuple_5372)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 9), dict_5363, (str_5364, list_5365))

# Assigning a type to the variable 'kwargs' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'kwargs', dict_5363)

# Assigning a List to a Name (line 5):

# Obtaining an instance of the builtin type 'list' (line 5)
list_5375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_5376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
str_5377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'str', 'a')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), tuple_5376, str_5377)
# Adding element type (line 5)
str_5378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 19), 'str', 'a')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), tuple_5376, str_5378)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 12), list_5375, tuple_5376)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_5379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
str_5380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 26), tuple_5379, str_5380)
# Adding element type (line 5)
str_5381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 31), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 26), tuple_5379, str_5381)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 12), list_5375, tuple_5379)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_5382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 38), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
str_5383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 38), 'str', 'c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 38), tuple_5382, str_5383)
# Adding element type (line 5)
str_5384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 43), 'str', 'c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 38), tuple_5382, str_5384)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 12), list_5375, tuple_5382)

# Assigning a type to the variable 'pad_width' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'pad_width', list_5375)


# Call to enumerate(...): (line 7)
# Processing the call arguments (line 7)

# Call to zip(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'pad_width' (line 8)
pad_width_5387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'pad_width', False)

# Obtaining the type of the subscript
str_5388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 30), 'str', 'constant_values')
# Getting the type of 'kwargs' (line 8)
kwargs_5389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 23), 'kwargs', False)
# Obtaining the member '__getitem__' of a type (line 8)
getitem___5390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 23), kwargs_5389, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 8)
subscript_call_result_5391 = invoke(stypy.reporting.localization.Localization(__file__, 8, 23), getitem___5390, str_5388)

# Processing the call keyword arguments (line 8)
kwargs_5392 = {}
# Getting the type of 'zip' (line 8)
zip_5386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'zip', False)
# Calling zip(args, kwargs) (line 8)
zip_call_result_5393 = invoke(stypy.reporting.localization.Localization(__file__, 8, 8), zip_5386, *[pad_width_5387, subscript_call_result_5391], **kwargs_5392)

# Processing the call keyword arguments (line 7)
kwargs_5394 = {}
# Getting the type of 'enumerate' (line 7)
enumerate_5385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 64), 'enumerate', False)
# Calling enumerate(args, kwargs) (line 7)
enumerate_call_result_5395 = invoke(stypy.reporting.localization.Localization(__file__, 7, 64), enumerate_5385, *[zip_call_result_5393], **kwargs_5394)

# Testing the type of a for loop iterable (line 7)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 7, 0), enumerate_call_result_5395)
# Getting the type of the for loop variable (line 7)
for_loop_var_5396 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 7, 0), enumerate_call_result_5395)
# Assigning a type to the variable 'axis' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'axis', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 0), for_loop_var_5396))
# Assigning a type to the variable 'pad_before' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'pad_before', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 0), for_loop_var_5396))
# Assigning a type to the variable 'pad_after' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'pad_after', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 0), for_loop_var_5396))
# Assigning a type to the variable 'before_val' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'before_val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 0), for_loop_var_5396))
# Assigning a type to the variable 'after_val' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'after_val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 0), for_loop_var_5396))
# SSA begins for a for statement (line 7)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
# Getting the type of 'axis' (line 9)
axis_5397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 10), 'axis')
# Getting the type of 'pad_before' (line 10)
pad_before_5398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 10), 'pad_before')
# Getting the type of 'pad_after' (line 11)
pad_after_5399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'pad_after')
# Getting the type of 'before_val' (line 12)
before_val_5400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'before_val')
# Getting the type of 'after_val' (line 13)
after_val_5401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'after_val')
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
