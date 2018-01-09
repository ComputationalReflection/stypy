
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: 
4: import os
5: 
6: x = os.environ
7: 
8: r = x['COMPUTERNAME']
9: 
10: x['FOO'] = 'BAR'
11: 
12: r2 = x['FOO']
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)


# Assigning a Attribute to a Name (line 6):
# Getting the type of 'os' (line 6)
os_5340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'os')
# Obtaining the member 'environ' of a type (line 6)
environ_5341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), os_5340, 'environ')
# Assigning a type to the variable 'x' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'x', environ_5341)

# Assigning a Subscript to a Name (line 8):

# Obtaining the type of the subscript
str_5342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 6), 'str', 'COMPUTERNAME')
# Getting the type of 'x' (line 8)
x_5343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'x')
# Obtaining the member '__getitem__' of a type (line 8)
getitem___5344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), x_5343, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 8)
subscript_call_result_5345 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), getitem___5344, str_5342)

# Assigning a type to the variable 'r' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r', subscript_call_result_5345)

# Assigning a Str to a Subscript (line 10):
str_5346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'BAR')
# Getting the type of 'x' (line 10)
x_5347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'x')
str_5348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 2), 'str', 'FOO')
# Storing an element on a container (line 10)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 0), x_5347, (str_5348, str_5346))

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
str_5349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 7), 'str', 'FOO')
# Getting the type of 'x' (line 12)
x_5350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'x')
# Obtaining the member '__getitem__' of a type (line 12)
getitem___5351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), x_5350, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_5352 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), getitem___5351, str_5349)

# Assigning a type to the variable 'r2' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r2', subscript_call_result_5352)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
