
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: try:
2:     a = 3
3: except KeyError as k:
4:     a = "3"
5: except Exception as e:
6:     a = list()
7: else:
8:     a = dict()
9: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################



# SSA begins for try-except statement (line 1)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Assigning a Num to a Name (line 2):
int_2693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 8), 'int')
# Assigning a type to the variable 'a' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'a', int_2693)
# SSA branch for the except part of a try statement (line 1)
# SSA branch for the except 'KeyError' branch of a try statement (line 1)
# Storing handler type
module_type_store.open_ssa_branch('except')
# Getting the type of 'KeyError' (line 3)
KeyError_2694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 7), 'KeyError')
# Assigning a type to the variable 'k' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'k', KeyError_2694)

# Assigning a Str to a Name (line 4):
str_2695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 8), 'str', '3')
# Assigning a type to the variable 'a' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'a', str_2695)
# SSA branch for the except 'Exception' branch of a try statement (line 1)
# Storing handler type
module_type_store.open_ssa_branch('except')
# Getting the type of 'Exception' (line 5)
Exception_2696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 7), 'Exception')
# Assigning a type to the variable 'e' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'e', Exception_2696)

# Assigning a Call to a Name (line 6):

# Call to list(...): (line 6)
# Processing the call keyword arguments (line 6)
kwargs_2698 = {}
# Getting the type of 'list' (line 6)
list_2697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'list', False)
# Calling list(args, kwargs) (line 6)
list_call_result_2699 = invoke(stypy.reporting.localization.Localization(__file__, 6, 8), list_2697, *[], **kwargs_2698)

# Assigning a type to the variable 'a' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'a', list_call_result_2699)
# SSA branch for the else branch of a try statement (line 1)
module_type_store.open_ssa_branch('except else')

# Assigning a Call to a Name (line 8):

# Call to dict(...): (line 8)
# Processing the call keyword arguments (line 8)
kwargs_2701 = {}
# Getting the type of 'dict' (line 8)
dict_2700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'dict', False)
# Calling dict(args, kwargs) (line 8)
dict_call_result_2702 = invoke(stypy.reporting.localization.Localization(__file__, 8, 8), dict_2700, *[], **kwargs_2701)

# Assigning a type to the variable 'a' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'a', dict_call_result_2702)
# SSA join for try-except statement (line 1)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
