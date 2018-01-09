
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: a = 3
4: b = 4
5: c = 8
6: d = 10
7: 
8: c1 = a < b < c
9: c2 = c < b > a
10: c3 = a < b < c < d
11: 
12: 
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 3):
int_5485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 4), 'int')
# Assigning a type to the variable 'a' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'a', int_5485)

# Assigning a Num to a Name (line 4):
int_5486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 4), 'int')
# Assigning a type to the variable 'b' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'b', int_5486)

# Assigning a Num to a Name (line 5):
int_5487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 4), 'int')
# Assigning a type to the variable 'c' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'c', int_5487)

# Assigning a Num to a Name (line 6):
int_5488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'int')
# Assigning a type to the variable 'd' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'd', int_5488)

# Assigning a Compare to a Name (line 8):

# Getting the type of 'a' (line 8)
a_5489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'a')
# Getting the type of 'b' (line 8)
b_5490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 9), 'b')
# Applying the binary operator '<' (line 8)
result_lt_5491 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 5), '<', a_5489, b_5490)
# Getting the type of 'c' (line 8)
c_5492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 13), 'c')
# Applying the binary operator '<' (line 8)
result_lt_5493 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 5), '<', b_5490, c_5492)
# Applying the binary operator '&' (line 8)
result_and__5494 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 5), '&', result_lt_5491, result_lt_5493)

# Assigning a type to the variable 'c1' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'c1', result_and__5494)

# Assigning a Compare to a Name (line 9):

# Getting the type of 'c' (line 9)
c_5495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'c')
# Getting the type of 'b' (line 9)
b_5496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 9), 'b')
# Applying the binary operator '<' (line 9)
result_lt_5497 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 5), '<', c_5495, b_5496)
# Getting the type of 'a' (line 9)
a_5498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 13), 'a')
# Applying the binary operator '>' (line 9)
result_gt_5499 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 5), '>', b_5496, a_5498)
# Applying the binary operator '&' (line 9)
result_and__5500 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 5), '&', result_lt_5497, result_gt_5499)

# Assigning a type to the variable 'c2' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'c2', result_and__5500)

# Assigning a Compare to a Name (line 10):

# Getting the type of 'a' (line 10)
a_5501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'a')
# Getting the type of 'b' (line 10)
b_5502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'b')
# Applying the binary operator '<' (line 10)
result_lt_5503 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 5), '<', a_5501, b_5502)
# Getting the type of 'c' (line 10)
c_5504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 13), 'c')
# Applying the binary operator '<' (line 10)
result_lt_5505 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 5), '<', b_5502, c_5504)
# Applying the binary operator '&' (line 10)
result_and__5506 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 5), '&', result_lt_5503, result_lt_5505)
# Getting the type of 'd' (line 10)
d_5507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 17), 'd')
# Applying the binary operator '<' (line 10)
result_lt_5508 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 5), '<', c_5504, d_5507)
# Applying the binary operator '&' (line 10)
result_and__5509 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 5), '&', result_and__5506, result_lt_5508)

# Assigning a type to the variable 'c3' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'c3', result_and__5509)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
