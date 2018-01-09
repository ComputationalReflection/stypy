
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import os
2: 
3: __revision__ = "$Id$"
4: 
5: # If DISTUTILS_DEBUG is anything other than the empty string, we run in
6: # debug mode.
7: DEBUG = os.environ.get('DISTUTILS_DEBUG')
8: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import os' statement (line 1)
import os

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'os', os, module_type_store)


# Assigning a Str to a Name (line 3):
str_307493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__revision__', str_307493)

# Assigning a Call to a Name (line 7):

# Call to get(...): (line 7)
# Processing the call arguments (line 7)
str_307497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 23), 'str', 'DISTUTILS_DEBUG')
# Processing the call keyword arguments (line 7)
kwargs_307498 = {}
# Getting the type of 'os' (line 7)
os_307494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'os', False)
# Obtaining the member 'environ' of a type (line 7)
environ_307495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 8), os_307494, 'environ')
# Obtaining the member 'get' of a type (line 7)
get_307496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 8), environ_307495, 'get')
# Calling get(args, kwargs) (line 7)
get_call_result_307499 = invoke(stypy.reporting.localization.Localization(__file__, 7, 8), get_307496, *[str_307497], **kwargs_307498)

# Assigning a type to the variable 'DEBUG' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'DEBUG', get_call_result_307499)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
