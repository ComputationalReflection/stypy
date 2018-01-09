
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: # THIS FILE IS GENERATED FROM SCIPY SETUP.PY
3: short_version = '1.0.0'
4: version = '1.0.0'
5: full_version = '1.0.0'
6: git_revision = '11509c4a98edded6c59423ac44ca1b7f28fba1fd'
7: release = True
8: 
9: if not release:
10:     version = full_version
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 16), 'str', '1.0.0')
# Assigning a type to the variable 'short_version' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'short_version', str_187)

# Assigning a Str to a Name (line 4):
str_188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'str', '1.0.0')
# Assigning a type to the variable 'version' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'version', str_188)

# Assigning a Str to a Name (line 5):
str_189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'str', '1.0.0')
# Assigning a type to the variable 'full_version' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'full_version', str_189)

# Assigning a Str to a Name (line 6):
str_190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'str', '11509c4a98edded6c59423ac44ca1b7f28fba1fd')
# Assigning a type to the variable 'git_revision' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'git_revision', str_190)

# Assigning a Name to a Name (line 7):
# Getting the type of 'True' (line 7)
True_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 10), 'True')
# Assigning a type to the variable 'release' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'release', True_191)


# Getting the type of 'release' (line 9)
release_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 7), 'release')
# Applying the 'not' unary operator (line 9)
result_not__193 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 3), 'not', release_192)

# Testing the type of an if condition (line 9)
if_condition_194 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 9, 0), result_not__193)
# Assigning a type to the variable 'if_condition_194' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'if_condition_194', if_condition_194)
# SSA begins for if statement (line 9)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 10):
# Getting the type of 'full_version' (line 10)
full_version_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 14), 'full_version')
# Assigning a type to the variable 'version' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'version', full_version_195)
# SSA join for if statement (line 9)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
