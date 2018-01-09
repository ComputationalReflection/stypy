
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: try:
2:     a = 3
3: except KeyError as k:
4:     a = "3"
5: except Exception as e:
6:     a = list()
7: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################



# SSA begins for try-except statement (line 1)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Assigning a Num to a Name (line 2):
int_2727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 8), 'int')
# Assigning a type to the variable 'a' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'a', int_2727)
# SSA branch for the except part of a try statement (line 1)
# SSA branch for the except 'KeyError' branch of a try statement (line 1)
# Storing handler type
module_type_store.open_ssa_branch('except')
# Getting the type of 'KeyError' (line 3)
KeyError_2728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 7), 'KeyError')
# Assigning a type to the variable 'k' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'k', KeyError_2728)

# Assigning a Str to a Name (line 4):
str_2729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 8), 'str', '3')
# Assigning a type to the variable 'a' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'a', str_2729)
# SSA branch for the except 'Exception' branch of a try statement (line 1)
# Storing handler type
module_type_store.open_ssa_branch('except')
# Getting the type of 'Exception' (line 5)
Exception_2730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 7), 'Exception')
# Assigning a type to the variable 'e' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'e', Exception_2730)

# Assigning a Call to a Name (line 6):

# Call to list(...): (line 6)
# Processing the call keyword arguments (line 6)
kwargs_2732 = {}
# Getting the type of 'list' (line 6)
list_2731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'list', False)
# Calling list(args, kwargs) (line 6)
list_call_result_2733 = invoke(stypy.reporting.localization.Localization(__file__, 6, 8), list_2731, *[], **kwargs_2732)

# Assigning a type to the variable 'a' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'a', list_call_result_2733)
# SSA join for try-except statement (line 1)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
