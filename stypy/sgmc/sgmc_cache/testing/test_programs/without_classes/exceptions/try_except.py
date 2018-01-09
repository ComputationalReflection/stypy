
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: try:
3:     a = 3
4: except KeyError as k:
5:     a = "3"
6: 
7: z = None

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################



# SSA begins for try-except statement (line 2)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Assigning a Num to a Name (line 3):
int_2689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 8), 'int')
# Assigning a type to the variable 'a' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'a', int_2689)
# SSA branch for the except part of a try statement (line 2)
# SSA branch for the except 'KeyError' branch of a try statement (line 2)
# Storing handler type
module_type_store.open_ssa_branch('except')
# Getting the type of 'KeyError' (line 4)
KeyError_2690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 7), 'KeyError')
# Assigning a type to the variable 'k' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'k', KeyError_2690)

# Assigning a Str to a Name (line 5):
str_2691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 8), 'str', '3')
# Assigning a type to the variable 'a' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'a', str_2691)
# SSA join for try-except statement (line 2)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Name (line 7):
# Getting the type of 'None' (line 7)
None_2692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'None')
# Assigning a type to the variable 'z' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'z', None_2692)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
