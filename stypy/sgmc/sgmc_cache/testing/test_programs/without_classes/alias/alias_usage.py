
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from math import sin
2: 
3: from math import acos as my_acos
4: 
5: y = my_acos(0.5)
6: z = my_acos
7: 
8: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from math import sin' statement (line 1)
from math import sin

import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'math', None, module_type_store, ['sin'], [sin])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from math import my_acos' statement (line 3)
from math import acos as my_acos

import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'math', None, module_type_store, ['acos'], [my_acos])
# Adding an alias
module_type_store.add_alias('my_acos', 'acos')


# Assigning a Call to a Name (line 5):

# Call to my_acos(...): (line 5)
# Processing the call arguments (line 5)
float_6209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 12), 'float')
# Processing the call keyword arguments (line 5)
kwargs_6210 = {}
# Getting the type of 'my_acos' (line 5)
my_acos_6208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'my_acos', False)
# Calling my_acos(args, kwargs) (line 5)
my_acos_call_result_6211 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), my_acos_6208, *[float_6209], **kwargs_6210)

# Assigning a type to the variable 'y' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'y', my_acos_call_result_6211)

# Assigning a Name to a Name (line 6):
# Getting the type of 'my_acos' (line 6)
my_acos_6212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'my_acos')
# Assigning a type to the variable 'z' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'z', my_acos_6212)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
