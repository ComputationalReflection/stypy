
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: a = True
4: b = False
5: 
6: c = b or a
7: 
8: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Name to a Name (line 3):
# Getting the type of 'True' (line 3)
True_5480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'True')
# Assigning a type to the variable 'a' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'a', True_5480)

# Assigning a Name to a Name (line 4):
# Getting the type of 'False' (line 4)
False_5481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'False')
# Assigning a type to the variable 'b' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'b', False_5481)

# Assigning a BoolOp to a Name (line 6):

# Evaluating a boolean operation
# Getting the type of 'b' (line 6)
b_5482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'b')
# Getting the type of 'a' (line 6)
a_5483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 9), 'a')
# Applying the binary operator 'or' (line 6)
result_or_keyword_5484 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 4), 'or', b_5482, a_5483)

# Assigning a type to the variable 'c' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'c', result_or_keyword_5484)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
