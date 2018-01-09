
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: x = z
4: 
5: if x:
6:     y = True
7: else:
8:     y = "hi"
9: 
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Name to a Name (line 3):
# Getting the type of 'z' (line 3)
z_4888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'z')
# Assigning a type to the variable 'x' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'x', z_4888)

# Getting the type of 'x' (line 5)
x_4889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 3), 'x')
# Testing the type of an if condition (line 5)
if_condition_4890 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 5, 0), x_4889)
# Assigning a type to the variable 'if_condition_4890' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'if_condition_4890', if_condition_4890)
# SSA begins for if statement (line 5)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 6):
# Getting the type of 'True' (line 6)
True_4891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'True')
# Assigning a type to the variable 'y' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'y', True_4891)
# SSA branch for the else part of an if statement (line 5)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 8):
str_4892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 8), 'str', 'hi')
# Assigning a type to the variable 'y' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'y', str_4892)
# SSA join for if statement (line 5)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
