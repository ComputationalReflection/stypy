
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import math
2: 
3: x_pre = math.sin(4)
4: 
5: del math.sin
6: 
7: x_post = math.sin(4)

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


# Assigning a Call to a Name (line 3):

# Call to sin(...): (line 3)
# Processing the call arguments (line 3)
int_6282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 17), 'int')
# Processing the call keyword arguments (line 3)
kwargs_6283 = {}
# Getting the type of 'math' (line 3)
math_6280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'math', False)
# Obtaining the member 'sin' of a type (line 3)
sin_6281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 3, 8), math_6280, 'sin')
# Calling sin(args, kwargs) (line 3)
sin_call_result_6284 = invoke(stypy.reporting.localization.Localization(__file__, 3, 8), sin_6281, *[int_6282], **kwargs_6283)

# Assigning a type to the variable 'x_pre' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'x_pre', sin_call_result_6284)
# Deleting a member
# Getting the type of 'math' (line 5)
math_6285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'math')
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 5, 0), math_6285, 'sin')

# Assigning a Call to a Name (line 7):

# Call to sin(...): (line 7)
# Processing the call arguments (line 7)
int_6288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 18), 'int')
# Processing the call keyword arguments (line 7)
kwargs_6289 = {}
# Getting the type of 'math' (line 7)
math_6286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'math', False)
# Obtaining the member 'sin' of a type (line 7)
sin_6287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 9), math_6286, 'sin')
# Calling sin(args, kwargs) (line 7)
sin_call_result_6290 = invoke(stypy.reporting.localization.Localization(__file__, 7, 9), sin_6287, *[int_6288], **kwargs_6289)

# Assigning a type to the variable 'x_post' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'x_post', sin_call_result_6290)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
