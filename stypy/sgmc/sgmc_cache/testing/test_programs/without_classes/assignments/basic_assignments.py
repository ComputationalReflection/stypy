
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: a = 3
3: n = 4.5
4: b = "str"
5: 
6: c = a + a + n
7: 
8: z = a
9: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 2):
int_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 4), 'int')
# Assigning a type to the variable 'a' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'a', int_110)

# Assigning a Num to a Name (line 3):
float_111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 4), 'float')
# Assigning a type to the variable 'n' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'n', float_111)

# Assigning a Str to a Name (line 4):
str_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 4), 'str', 'str')
# Assigning a type to the variable 'b' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'b', str_112)

# Assigning a BinOp to a Name (line 6):
# Getting the type of 'a' (line 6)
a_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'a')
# Getting the type of 'a' (line 6)
a_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'a')
# Applying the binary operator '+' (line 6)
result_add_115 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 4), '+', a_113, a_114)

# Getting the type of 'n' (line 6)
n_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 12), 'n')
# Applying the binary operator '+' (line 6)
result_add_117 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 10), '+', result_add_115, n_116)

# Assigning a type to the variable 'c' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'c', result_add_117)

# Assigning a Name to a Name (line 8):
# Getting the type of 'a' (line 8)
a_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'a')
# Assigning a type to the variable 'z' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'z', a_118)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
