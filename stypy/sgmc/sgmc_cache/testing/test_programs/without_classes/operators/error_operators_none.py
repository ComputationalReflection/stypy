
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: none1 = None
2: a = 4 + none1
3: b = 4 + None
4: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Name to a Name (line 1):
# Getting the type of 'None' (line 1)
None_7927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 8), 'None')
# Assigning a type to the variable 'none1' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'none1', None_7927)

# Assigning a BinOp to a Name (line 2):
int_7928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 4), 'int')
# Getting the type of 'none1' (line 2)
none1_7929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 8), 'none1')
# Applying the binary operator '+' (line 2)
result_add_7930 = python_operator(stypy.reporting.localization.Localization(__file__, 2, 4), '+', int_7928, none1_7929)

# Assigning a type to the variable 'a' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'a', result_add_7930)

# Assigning a BinOp to a Name (line 3):
int_7931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 4), 'int')
# Getting the type of 'None' (line 3)
None_7932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'None')
# Applying the binary operator '+' (line 3)
result_add_7933 = python_operator(stypy.reporting.localization.Localization(__file__, 3, 4), '+', int_7931, None_7932)

# Assigning a type to the variable 'b' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'b', result_add_7933)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
