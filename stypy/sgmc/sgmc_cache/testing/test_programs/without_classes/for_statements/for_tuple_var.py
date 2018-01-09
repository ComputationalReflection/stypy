
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: arguments = {'a': 1, 'b': 2}
3: ret_str = ""
4: 
5: for key, arg in arguments.items():
6:     ret_str += str(key) + ": " + str(arg)
7: 
8: print ret_str
9: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Dict to a Name (line 2):

# Obtaining an instance of the builtin type 'dict' (line 2)
dict_5424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 2)
# Adding element type (key, value) (line 2)
str_5425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 13), 'str', 'a')
int_5426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 18), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 12), dict_5424, (str_5425, int_5426))
# Adding element type (key, value) (line 2)
str_5427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 21), 'str', 'b')
int_5428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 26), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 12), dict_5424, (str_5427, int_5428))

# Assigning a type to the variable 'arguments' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'arguments', dict_5424)

# Assigning a Str to a Name (line 3):
str_5429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', '')
# Assigning a type to the variable 'ret_str' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'ret_str', str_5429)


# Call to items(...): (line 5)
# Processing the call keyword arguments (line 5)
kwargs_5432 = {}
# Getting the type of 'arguments' (line 5)
arguments_5430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 16), 'arguments', False)
# Obtaining the member 'items' of a type (line 5)
items_5431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 16), arguments_5430, 'items')
# Calling items(args, kwargs) (line 5)
items_call_result_5433 = invoke(stypy.reporting.localization.Localization(__file__, 5, 16), items_5431, *[], **kwargs_5432)

# Testing the type of a for loop iterable (line 5)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 5, 0), items_call_result_5433)
# Getting the type of the for loop variable (line 5)
for_loop_var_5434 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 5, 0), items_call_result_5433)
# Assigning a type to the variable 'key' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 0), for_loop_var_5434))
# Assigning a type to the variable 'arg' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'arg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 0), for_loop_var_5434))
# SSA begins for a for statement (line 5)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Getting the type of 'ret_str' (line 6)
ret_str_5435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'ret_str')

# Call to str(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'key' (line 6)
key_5437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 19), 'key', False)
# Processing the call keyword arguments (line 6)
kwargs_5438 = {}
# Getting the type of 'str' (line 6)
str_5436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), 'str', False)
# Calling str(args, kwargs) (line 6)
str_call_result_5439 = invoke(stypy.reporting.localization.Localization(__file__, 6, 15), str_5436, *[key_5437], **kwargs_5438)

str_5440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 26), 'str', ': ')
# Applying the binary operator '+' (line 6)
result_add_5441 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 15), '+', str_call_result_5439, str_5440)


# Call to str(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'arg' (line 6)
arg_5443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 37), 'arg', False)
# Processing the call keyword arguments (line 6)
kwargs_5444 = {}
# Getting the type of 'str' (line 6)
str_5442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 33), 'str', False)
# Calling str(args, kwargs) (line 6)
str_call_result_5445 = invoke(stypy.reporting.localization.Localization(__file__, 6, 33), str_5442, *[arg_5443], **kwargs_5444)

# Applying the binary operator '+' (line 6)
result_add_5446 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 31), '+', result_add_5441, str_call_result_5445)

# Applying the binary operator '+=' (line 6)
result_iadd_5447 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 4), '+=', ret_str_5435, result_add_5446)
# Assigning a type to the variable 'ret_str' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'ret_str', result_iadd_5447)

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# Getting the type of 'ret_str' (line 8)
ret_str_5448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 6), 'ret_str')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
