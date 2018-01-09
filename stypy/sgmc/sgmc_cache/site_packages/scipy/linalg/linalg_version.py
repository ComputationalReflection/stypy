
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: major = 0
4: minor = 4
5: micro = 9
6: 
7: linalg_version = '%(major)d.%(minor)d.%(micro)d' % (locals())
8: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 3):
int_22532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 8), 'int')
# Assigning a type to the variable 'major' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'major', int_22532)

# Assigning a Num to a Name (line 4):
int_22533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 8), 'int')
# Assigning a type to the variable 'minor' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'minor', int_22533)

# Assigning a Num to a Name (line 5):
int_22534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 8), 'int')
# Assigning a type to the variable 'micro' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'micro', int_22534)

# Assigning a BinOp to a Name (line 7):
str_22535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 17), 'str', '%(major)d.%(minor)d.%(micro)d')

# Call to locals(...): (line 7)
# Processing the call keyword arguments (line 7)
kwargs_22537 = {}
# Getting the type of 'locals' (line 7)
locals_22536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 52), 'locals', False)
# Calling locals(args, kwargs) (line 7)
locals_call_result_22538 = invoke(stypy.reporting.localization.Localization(__file__, 7, 52), locals_22536, *[], **kwargs_22537)

# Applying the binary operator '%' (line 7)
result_mod_22539 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 17), '%', str_22535, locals_call_result_22538)

# Assigning a type to the variable 'linalg_version' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'linalg_version', result_mod_22539)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
