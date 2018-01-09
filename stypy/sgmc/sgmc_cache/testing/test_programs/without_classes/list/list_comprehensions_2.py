
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: modules = [("module1", "desc1"),
2:           ("module2", "desc2")]
3: 
4: normalized_path = "foo"
5: 
6: user_defined_modules = dict((module_name, module_desc) for (module_name, module_desc) in modules
7:                             if (normalized_path not in str(module_desc) and "built-in" not in
8:                                 str(module_desc)
9:                                 and module_desc is not None))
10: 
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 1):

# Obtaining an instance of the builtin type 'list' (line 1)
list_5306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 1)
# Adding element type (line 1)

# Obtaining an instance of the builtin type 'tuple' (line 1)
tuple_5307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 1)
# Adding element type (line 1)
str_5308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 12), 'str', 'module1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 12), tuple_5307, str_5308)
# Adding element type (line 1)
str_5309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 23), 'str', 'desc1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 12), tuple_5307, str_5309)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_5306, tuple_5307)
# Adding element type (line 1)

# Obtaining an instance of the builtin type 'tuple' (line 2)
tuple_5310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2)
# Adding element type (line 2)
str_5311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'str', 'module2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 11), tuple_5310, str_5311)
# Adding element type (line 2)
str_5312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 22), 'str', 'desc2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 11), tuple_5310, str_5312)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_5306, tuple_5310)

# Assigning a type to the variable 'modules' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'modules', list_5306)

# Assigning a Str to a Name (line 4):
str_5313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 18), 'str', 'foo')
# Assigning a type to the variable 'normalized_path' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'normalized_path', str_5313)

# Assigning a Call to a Name (line 6):

# Call to dict(...): (line 6)
# Processing the call arguments (line 6)
# Calculating generator expression
module_type_store = module_type_store.open_function_context('list comprehension expression', 6, 28, True)
# Calculating comprehension expression
# Getting the type of 'modules' (line 6)
modules_5335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 89), 'modules', False)
comprehension_5336 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 28), modules_5335)
# Assigning a type to the variable 'module_name' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 28), 'module_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 28), comprehension_5336))
# Assigning a type to the variable 'module_desc' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 28), 'module_desc', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 28), comprehension_5336))

# Evaluating a boolean operation

# Getting the type of 'normalized_path' (line 7)
normalized_path_5318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 32), 'normalized_path', False)

# Call to str(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'module_desc' (line 7)
module_desc_5320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 59), 'module_desc', False)
# Processing the call keyword arguments (line 7)
kwargs_5321 = {}
# Getting the type of 'str' (line 7)
str_5319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 55), 'str', False)
# Calling str(args, kwargs) (line 7)
str_call_result_5322 = invoke(stypy.reporting.localization.Localization(__file__, 7, 55), str_5319, *[module_desc_5320], **kwargs_5321)

# Applying the binary operator 'notin' (line 7)
result_contains_5323 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 32), 'notin', normalized_path_5318, str_call_result_5322)


str_5324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 76), 'str', 'built-in')

# Call to str(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'module_desc' (line 8)
module_desc_5326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 36), 'module_desc', False)
# Processing the call keyword arguments (line 8)
kwargs_5327 = {}
# Getting the type of 'str' (line 8)
str_5325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 32), 'str', False)
# Calling str(args, kwargs) (line 8)
str_call_result_5328 = invoke(stypy.reporting.localization.Localization(__file__, 8, 32), str_5325, *[module_desc_5326], **kwargs_5327)

# Applying the binary operator 'notin' (line 7)
result_contains_5329 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 76), 'notin', str_5324, str_call_result_5328)

# Applying the binary operator 'and' (line 7)
result_and_keyword_5330 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 32), 'and', result_contains_5323, result_contains_5329)

# Getting the type of 'module_desc' (line 9)
module_desc_5331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 36), 'module_desc', False)
# Getting the type of 'None' (line 9)
None_5332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 55), 'None', False)
# Applying the binary operator 'isnot' (line 9)
result_is_not_5333 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 36), 'isnot', module_desc_5331, None_5332)

# Applying the binary operator 'and' (line 7)
result_and_keyword_5334 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 32), 'and', result_and_keyword_5330, result_is_not_5333)


# Obtaining an instance of the builtin type 'tuple' (line 6)
tuple_5315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 29), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6)
# Adding element type (line 6)
# Getting the type of 'module_name' (line 6)
module_name_5316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 29), 'module_name', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 29), tuple_5315, module_name_5316)
# Adding element type (line 6)
# Getting the type of 'module_desc' (line 6)
module_desc_5317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 42), 'module_desc', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 29), tuple_5315, module_desc_5317)

list_5337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 28), 'list')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 28), list_5337, tuple_5315)
# Processing the call keyword arguments (line 6)
kwargs_5338 = {}
# Getting the type of 'dict' (line 6)
dict_5314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 23), 'dict', False)
# Calling dict(args, kwargs) (line 6)
dict_call_result_5339 = invoke(stypy.reporting.localization.Localization(__file__, 6, 23), dict_5314, *[list_5337], **kwargs_5338)

# Assigning a type to the variable 'user_defined_modules' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'user_defined_modules', dict_call_result_5339)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
