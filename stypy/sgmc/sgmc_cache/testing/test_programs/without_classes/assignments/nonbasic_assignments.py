
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: # Non-basic type assignment
4: l = list()
5: 
6: # Function assigment (attribute assignments)
7: a = l.count
8: 
9: # Dynamic type
10: b = eval("3+4")
11: 
12: # Undefined type assignment (subscript assignment)
13: c = l[3]

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Call to a Name (line 4):

# Call to list(...): (line 4)
# Processing the call keyword arguments (line 4)
kwargs_555 = {}
# Getting the type of 'list' (line 4)
list_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'list', False)
# Calling list(args, kwargs) (line 4)
list_call_result_556 = invoke(stypy.reporting.localization.Localization(__file__, 4, 4), list_554, *[], **kwargs_555)

# Assigning a type to the variable 'l' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'l', list_call_result_556)

# Assigning a Attribute to a Name (line 7):
# Getting the type of 'l' (line 7)
l_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'l')
# Obtaining the member 'count' of a type (line 7)
count_558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), l_557, 'count')
# Assigning a type to the variable 'a' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'a', count_558)

# Assigning a Call to a Name (line 10):

# Call to eval(...): (line 10)
# Processing the call arguments (line 10)
str_560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 9), 'str', '3+4')
# Processing the call keyword arguments (line 10)
kwargs_561 = {}
# Getting the type of 'eval' (line 10)
eval_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'eval', False)
# Calling eval(args, kwargs) (line 10)
eval_call_result_562 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), eval_559, *[str_560], **kwargs_561)

# Assigning a type to the variable 'b' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'b', eval_call_result_562)

# Assigning a Subscript to a Name (line 13):

# Obtaining the type of the subscript
int_563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 6), 'int')
# Getting the type of 'l' (line 13)
l_564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'l')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), l_564, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_566 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), getitem___565, int_563)

# Assigning a type to the variable 'c' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'c', subscript_call_result_566)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
