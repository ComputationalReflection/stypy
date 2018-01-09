
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from math import *
2: 
3: x = pi
4: f = pow(3, 5)
5: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from math import ' statement (line 1)
from math import *

import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'math', None, module_type_store, ['*'], None)


# Assigning a Name to a Name (line 3):
# Getting the type of 'pi' (line 3)
pi_4994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'pi')
# Assigning a type to the variable 'x' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'x', pi_4994)

# Assigning a Call to a Name (line 4):

# Call to pow(...): (line 4)
# Processing the call arguments (line 4)
int_4996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 8), 'int')
int_4997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 11), 'int')
# Processing the call keyword arguments (line 4)
kwargs_4998 = {}
# Getting the type of 'pow' (line 4)
pow_4995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'pow', False)
# Calling pow(args, kwargs) (line 4)
pow_call_result_4999 = invoke(stypy.reporting.localization.Localization(__file__, 4, 4), pow_4995, *[int_4996, int_4997], **kwargs_4998)

# Assigning a type to the variable 'f' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'f', pow_call_result_4999)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
