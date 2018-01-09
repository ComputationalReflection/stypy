
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: # THIS FILE IS GENERATED FROM NUMPY SETUP.PY
3: #
4: # To compare versions robustly, use `numpy.lib.NumpyVersion`
5: short_version = '1.11.0'
6: version = '1.11.0'
7: full_version = '1.11.0'
8: git_revision = '4092a9e160cc247a4a45724579a0c829733688ca'
9: release = True
10: 
11: if not release:
12:     version = full_version
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 5):
str_24388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'str', '1.11.0')
# Assigning a type to the variable 'short_version' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'short_version', str_24388)

# Assigning a Str to a Name (line 6):
str_24389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 10), 'str', '1.11.0')
# Assigning a type to the variable 'version' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'version', str_24389)

# Assigning a Str to a Name (line 7):
str_24390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'str', '1.11.0')
# Assigning a type to the variable 'full_version' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'full_version', str_24390)

# Assigning a Str to a Name (line 8):
str_24391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 15), 'str', '4092a9e160cc247a4a45724579a0c829733688ca')
# Assigning a type to the variable 'git_revision' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'git_revision', str_24391)

# Assigning a Name to a Name (line 9):
# Getting the type of 'True' (line 9)
True_24392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 10), 'True')
# Assigning a type to the variable 'release' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'release', True_24392)


# Getting the type of 'release' (line 11)
release_24393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 7), 'release')
# Applying the 'not' unary operator (line 11)
result_not__24394 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 3), 'not', release_24393)

# Testing the type of an if condition (line 11)
if_condition_24395 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 11, 0), result_not__24394)
# Assigning a type to the variable 'if_condition_24395' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'if_condition_24395', if_condition_24395)
# SSA begins for if statement (line 11)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 12):
# Getting the type of 'full_version' (line 12)
full_version_24396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'full_version')
# Assigning a type to the variable 'version' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'version', full_version_24396)
# SSA join for if statement (line 11)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
