
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import math
2: 
3: try:
4:     a = 3
5: except:
6:     a = "3"
7: 
8: r1 = a[6]  # No control about the possible values of a
9: r2 = math.fsum(a) # Not detected
10: 
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import math' statement (line 1)
import math

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'math', math, module_type_store)



# SSA begins for try-except statement (line 3)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Assigning a Num to a Name (line 4):
int_8011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 8), 'int')
# Assigning a type to the variable 'a' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'a', int_8011)
# SSA branch for the except part of a try statement (line 3)
# SSA branch for the except '<any exception>' branch of a try statement (line 3)
module_type_store.open_ssa_branch('except')

# Assigning a Str to a Name (line 6):
str_8012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 8), 'str', '3')
# Assigning a type to the variable 'a' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'a', str_8012)
# SSA join for try-except statement (line 3)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Subscript to a Name (line 8):

# Obtaining the type of the subscript
int_8013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 7), 'int')
# Getting the type of 'a' (line 8)
a_8014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'a')
# Obtaining the member '__getitem__' of a type (line 8)
getitem___8015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), a_8014, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 8)
subscript_call_result_8016 = invoke(stypy.reporting.localization.Localization(__file__, 8, 5), getitem___8015, int_8013)

# Assigning a type to the variable 'r1' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r1', subscript_call_result_8016)

# Assigning a Call to a Name (line 9):

# Call to fsum(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'a' (line 9)
a_8019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 15), 'a', False)
# Processing the call keyword arguments (line 9)
kwargs_8020 = {}
# Getting the type of 'math' (line 9)
math_8017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'math', False)
# Obtaining the member 'fsum' of a type (line 9)
fsum_8018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), math_8017, 'fsum')
# Calling fsum(args, kwargs) (line 9)
fsum_call_result_8021 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), fsum_8018, *[a_8019], **kwargs_8020)

# Assigning a type to the variable 'r2' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r2', fsum_call_result_8021)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
