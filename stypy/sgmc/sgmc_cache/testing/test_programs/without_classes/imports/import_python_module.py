
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import math
2: 
3: x = math.pi
4: f = math.pow(3, 5)
5: 

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


# Assigning a Attribute to a Name (line 3):
# Getting the type of 'math' (line 3)
math_5100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'math')
# Obtaining the member 'pi' of a type (line 3)
pi_5101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 3, 4), math_5100, 'pi')
# Assigning a type to the variable 'x' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'x', pi_5101)

# Assigning a Call to a Name (line 4):

# Call to pow(...): (line 4)
# Processing the call arguments (line 4)
int_5104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 13), 'int')
int_5105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 16), 'int')
# Processing the call keyword arguments (line 4)
kwargs_5106 = {}
# Getting the type of 'math' (line 4)
math_5102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'math', False)
# Obtaining the member 'pow' of a type (line 4)
pow_5103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), math_5102, 'pow')
# Calling pow(args, kwargs) (line 4)
pow_call_result_5107 = invoke(stypy.reporting.localization.Localization(__file__, 4, 4), pow_5103, *[int_5104, int_5105], **kwargs_5106)

# Assigning a type to the variable 'f' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'f', pow_call_result_5107)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
