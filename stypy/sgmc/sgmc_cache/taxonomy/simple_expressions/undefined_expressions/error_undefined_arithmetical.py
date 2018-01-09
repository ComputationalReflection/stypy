
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: import math
3: 
4: a = 4
5: 
6: # Type error
7: x = a + b
8: 
9: c = x / 2
10: 
11: # Type error
12: print math.pow(base, exponent)
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import math' statement (line 2)
import math

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'math', math, module_type_store)


# Assigning a Num to a Name (line 4):
int_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 4), 'int')
# Assigning a type to the variable 'a' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'a', int_1)

# Assigning a BinOp to a Name (line 7):
# Getting the type of 'a' (line 7)
a_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'a')
# Getting the type of 'b' (line 7)
b_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'b')
# Applying the binary operator '+' (line 7)
result_add_4 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 4), '+', a_2, b_3)

# Assigning a type to the variable 'x' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'x', result_add_4)

# Assigning a BinOp to a Name (line 9):
# Getting the type of 'x' (line 9)
x_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'x')
int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 8), 'int')
# Applying the binary operator 'div' (line 9)
result_div_7 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 4), 'div', x_5, int_6)

# Assigning a type to the variable 'c' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'c', result_div_7)

# Call to pow(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'base' (line 12)
base_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'base', False)
# Getting the type of 'exponent' (line 12)
exponent_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 21), 'exponent', False)
# Processing the call keyword arguments (line 12)
kwargs_12 = {}
# Getting the type of 'math' (line 12)
math_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 6), 'math', False)
# Obtaining the member 'pow' of a type (line 12)
pow_9 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 6), math_8, 'pow')
# Calling pow(args, kwargs) (line 12)
pow_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 12, 6), pow_9, *[base_10, exponent_11], **kwargs_12)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
