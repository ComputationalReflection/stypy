
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: major = 0
4: minor = 4
5: micro = 0
6: version = '%(major)d.%(minor)d.%(micro)d' % (locals())
7: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 3):
int_52270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 8), 'int')
# Assigning a type to the variable 'major' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'major', int_52270)

# Assigning a Num to a Name (line 4):
int_52271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 8), 'int')
# Assigning a type to the variable 'minor' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'minor', int_52271)

# Assigning a Num to a Name (line 5):
int_52272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 8), 'int')
# Assigning a type to the variable 'micro' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'micro', int_52272)

# Assigning a BinOp to a Name (line 6):
str_52273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 10), 'str', '%(major)d.%(minor)d.%(micro)d')

# Call to locals(...): (line 6)
# Processing the call keyword arguments (line 6)
kwargs_52275 = {}
# Getting the type of 'locals' (line 6)
locals_52274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 45), 'locals', False)
# Calling locals(args, kwargs) (line 6)
locals_call_result_52276 = invoke(stypy.reporting.localization.Localization(__file__, 6, 45), locals_52274, *[], **kwargs_52275)

# Applying the binary operator '%' (line 6)
result_mod_52277 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 10), '%', str_52273, locals_call_result_52276)

# Assigning a type to the variable 'version' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'version', result_mod_52277)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
