
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: a = 3
3: b = 4
4: c = 8
5: 
6: c1 = a < b
7: c2 = c < b
8: 
9: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 2):
int_5510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 4), 'int')
# Assigning a type to the variable 'a' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'a', int_5510)

# Assigning a Num to a Name (line 3):
int_5511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 4), 'int')
# Assigning a type to the variable 'b' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'b', int_5511)

# Assigning a Num to a Name (line 4):
int_5512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 4), 'int')
# Assigning a type to the variable 'c' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'c', int_5512)

# Assigning a Compare to a Name (line 6):

# Getting the type of 'a' (line 6)
a_5513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'a')
# Getting the type of 'b' (line 6)
b_5514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 9), 'b')
# Applying the binary operator '<' (line 6)
result_lt_5515 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 5), '<', a_5513, b_5514)

# Assigning a type to the variable 'c1' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'c1', result_lt_5515)

# Assigning a Compare to a Name (line 7):

# Getting the type of 'c' (line 7)
c_5516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'c')
# Getting the type of 'b' (line 7)
b_5517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'b')
# Applying the binary operator '<' (line 7)
result_lt_5518 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 5), '<', c_5516, b_5517)

# Assigning a type to the variable 'c2' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'c2', result_lt_5518)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
