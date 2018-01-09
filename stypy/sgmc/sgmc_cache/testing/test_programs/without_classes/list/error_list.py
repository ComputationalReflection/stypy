
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import math
2: 
3: l1 = ["3", "4", 5, 6]
4: 
5: r1 = math.fsum(l1)  # Not detected
6: r2 = l1[2] / 3  # Not detected
7: 
8: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import math' statement (line 1)
import math

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'math', math, module_type_store)


# Assigning a List to a Name (line 3):

# Obtaining an instance of the builtin type 'list' (line 3)
list_7773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 3)
# Adding element type (line 3)
str_7774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 6), 'str', '3')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 5), list_7773, str_7774)
# Adding element type (line 3)
str_7775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'str', '4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 5), list_7773, str_7775)
# Adding element type (line 3)
int_7776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 5), list_7773, int_7776)
# Adding element type (line 3)
int_7777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 5), list_7773, int_7777)

# Assigning a type to the variable 'l1' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'l1', list_7773)

# Assigning a Call to a Name (line 5):

# Call to fsum(...): (line 5)
# Processing the call arguments (line 5)
# Getting the type of 'l1' (line 5)
l1_7780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 15), 'l1', False)
# Processing the call keyword arguments (line 5)
kwargs_7781 = {}
# Getting the type of 'math' (line 5)
math_7778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 5), 'math', False)
# Obtaining the member 'fsum' of a type (line 5)
fsum_7779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 5), math_7778, 'fsum')
# Calling fsum(args, kwargs) (line 5)
fsum_call_result_7782 = invoke(stypy.reporting.localization.Localization(__file__, 5, 5), fsum_7779, *[l1_7780], **kwargs_7781)

# Assigning a type to the variable 'r1' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'r1', fsum_call_result_7782)

# Assigning a BinOp to a Name (line 6):

# Obtaining the type of the subscript
int_7783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 8), 'int')
# Getting the type of 'l1' (line 6)
l1_7784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'l1')
# Obtaining the member '__getitem__' of a type (line 6)
getitem___7785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 5), l1_7784, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 6)
subscript_call_result_7786 = invoke(stypy.reporting.localization.Localization(__file__, 6, 5), getitem___7785, int_7783)

int_7787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'int')
# Applying the binary operator 'div' (line 6)
result_div_7788 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 5), 'div', subscript_call_result_7786, int_7787)

# Assigning a type to the variable 'r2' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r2', result_div_7788)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
