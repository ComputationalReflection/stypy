
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: x ="str"
2: 
3: exec('x=4+7/2')
4: 
5: exec(1)
6: 
7: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 1):
str_2642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 3), 'str', 'str')
# Assigning a type to the variable 'x' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'x', str_2642)
# Dynamic code evaluation using an exec statement
str_2643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 5), 'str', 'x=4+7/2')
ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 3, 0), str_2643, 'exec parameter', 'StringType', 'FileType', 'CodeType')
enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 3, 0))
# Dynamic code evaluation using an exec statement
int_2644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 5), 'int')
ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 5, 0), int_2644, 'exec parameter', 'StringType', 'FileType', 'CodeType')
enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 5, 0))

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
