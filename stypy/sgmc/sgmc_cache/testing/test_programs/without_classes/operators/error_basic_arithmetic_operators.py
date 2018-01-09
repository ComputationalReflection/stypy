
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: a = "4" + 5
3: 
4: b = 4
5: 
6: c = a * b
7: 
8: d = c / a
9: 
10: e = a + b + c
11: 
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a BinOp to a Name (line 2):
str_6984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 4), 'str', '4')
int_6985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'int')
# Applying the binary operator '+' (line 2)
result_add_6986 = python_operator(stypy.reporting.localization.Localization(__file__, 2, 4), '+', str_6984, int_6985)

# Assigning a type to the variable 'a' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'a', result_add_6986)

# Assigning a Num to a Name (line 4):
int_6987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 4), 'int')
# Assigning a type to the variable 'b' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'b', int_6987)

# Assigning a BinOp to a Name (line 6):
# Getting the type of 'a' (line 6)
a_6988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'a')
# Getting the type of 'b' (line 6)
b_6989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'b')
# Applying the binary operator '*' (line 6)
result_mul_6990 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 4), '*', a_6988, b_6989)

# Assigning a type to the variable 'c' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'c', result_mul_6990)

# Assigning a BinOp to a Name (line 8):
# Getting the type of 'c' (line 8)
c_6991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'c')
# Getting the type of 'a' (line 8)
a_6992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'a')
# Applying the binary operator 'div' (line 8)
result_div_6993 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 4), 'div', c_6991, a_6992)

# Assigning a type to the variable 'd' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'd', result_div_6993)

# Assigning a BinOp to a Name (line 10):
# Getting the type of 'a' (line 10)
a_6994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'a')
# Getting the type of 'b' (line 10)
b_6995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'b')
# Applying the binary operator '+' (line 10)
result_add_6996 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 4), '+', a_6994, b_6995)

# Getting the type of 'c' (line 10)
c_6997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'c')
# Applying the binary operator '+' (line 10)
result_add_6998 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 10), '+', result_add_6996, c_6997)

# Assigning a type to the variable 'e' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'e', result_add_6998)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
