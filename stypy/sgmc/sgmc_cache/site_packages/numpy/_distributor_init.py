
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Inialization specific to Windows ATLAS SSE2 build
2: '''
3: 
4: # Add check for SSE2 on Windows
5: try:
6:     from ctypes import windll, wintypes
7: except (ImportError, ValueError):
8:     pass
9: else:
10:     has_feature = windll.kernel32.IsProcessorFeaturePresent
11:     has_feature.argtypes = [wintypes.DWORD]
12:     if not has_feature(10):
13:         msg = ("This version of numpy needs a CPU capable of SSE2, "
14:                 "but Windows says that is not so.\n",
15:                 "Please reinstall numpy using a different distribution")
16:         raise RuntimeError(msg)
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_24397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', ' Inialization specific to Windows ATLAS SSE2 build\n')


# SSA begins for try-except statement (line 5)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 4))

# 'from ctypes import windll, wintypes' statement (line 6)
from ctypes import windll, wintypes

import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'ctypes', None, module_type_store, ['windll', 'wintypes'], [windll, wintypes])

# SSA branch for the except part of a try statement (line 5)
# SSA branch for the except 'Tuple' branch of a try statement (line 5)
module_type_store.open_ssa_branch('except')
pass
# SSA branch for the else branch of a try statement (line 5)
module_type_store.open_ssa_branch('except else')

# Assigning a Attribute to a Name (line 10):
# Getting the type of 'windll' (line 10)
windll_24398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 18), 'windll')
# Obtaining the member 'kernel32' of a type (line 10)
kernel32_24399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 18), windll_24398, 'kernel32')
# Obtaining the member 'IsProcessorFeaturePresent' of a type (line 10)
IsProcessorFeaturePresent_24400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 18), kernel32_24399, 'IsProcessorFeaturePresent')
# Assigning a type to the variable 'has_feature' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'has_feature', IsProcessorFeaturePresent_24400)

# Assigning a List to a Attribute (line 11):

# Obtaining an instance of the builtin type 'list' (line 11)
list_24401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)
# Getting the type of 'wintypes' (line 11)
wintypes_24402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 28), 'wintypes')
# Obtaining the member 'DWORD' of a type (line 11)
DWORD_24403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 28), wintypes_24402, 'DWORD')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 27), list_24401, DWORD_24403)

# Getting the type of 'has_feature' (line 11)
has_feature_24404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'has_feature')
# Setting the type of the member 'argtypes' of a type (line 11)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), has_feature_24404, 'argtypes', list_24401)



# Call to has_feature(...): (line 12)
# Processing the call arguments (line 12)
int_24406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 23), 'int')
# Processing the call keyword arguments (line 12)
kwargs_24407 = {}
# Getting the type of 'has_feature' (line 12)
has_feature_24405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 11), 'has_feature', False)
# Calling has_feature(args, kwargs) (line 12)
has_feature_call_result_24408 = invoke(stypy.reporting.localization.Localization(__file__, 12, 11), has_feature_24405, *[int_24406], **kwargs_24407)

# Applying the 'not' unary operator (line 12)
result_not__24409 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 7), 'not', has_feature_call_result_24408)

# Testing the type of an if condition (line 12)
if_condition_24410 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 4), result_not__24409)
# Assigning a type to the variable 'if_condition_24410' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'if_condition_24410', if_condition_24410)
# SSA begins for if statement (line 12)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Tuple to a Name (line 13):

# Obtaining an instance of the builtin type 'tuple' (line 13)
tuple_24411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 13)
# Adding element type (line 13)
str_24412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'This version of numpy needs a CPU capable of SSE2, but Windows says that is not so.\n')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 15), tuple_24411, str_24412)
# Adding element type (line 13)
str_24413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 16), 'str', 'Please reinstall numpy using a different distribution')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 15), tuple_24411, str_24413)

# Assigning a type to the variable 'msg' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'msg', tuple_24411)

# Call to RuntimeError(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of 'msg' (line 16)
msg_24415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 27), 'msg', False)
# Processing the call keyword arguments (line 16)
kwargs_24416 = {}
# Getting the type of 'RuntimeError' (line 16)
RuntimeError_24414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'RuntimeError', False)
# Calling RuntimeError(args, kwargs) (line 16)
RuntimeError_call_result_24417 = invoke(stypy.reporting.localization.Localization(__file__, 16, 14), RuntimeError_24414, *[msg_24415], **kwargs_24416)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 16, 8), RuntimeError_call_result_24417, 'raise parameter', BaseException)
# SSA join for if statement (line 12)
module_type_store = module_type_store.join_ssa_context()

# SSA join for try-except statement (line 5)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
