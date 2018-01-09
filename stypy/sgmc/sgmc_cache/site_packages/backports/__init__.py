
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: __path__ = __import__('pkgutil').extend_path(__path__, __name__)
2: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Call to a Name (line 1):

# Call to extend_path(...): (line 1)
# Processing the call arguments (line 1)
# Getting the type of '__path__' (line 1)
path___308676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 45), '__path__', False)
# Getting the type of '__name__' (line 1)
name___308677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 55), '__name__', False)
# Processing the call keyword arguments (line 1)
kwargs_308678 = {}

# Call to __import__(...): (line 1)
# Processing the call arguments (line 1)
str_308672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 22), 'str', 'pkgutil')
# Processing the call keyword arguments (line 1)
kwargs_308673 = {}
# Getting the type of '__import__' (line 1)
import___308671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 11), '__import__', False)
# Calling __import__(args, kwargs) (line 1)
import___call_result_308674 = invoke(stypy.reporting.localization.Localization(__file__, 1, 11), import___308671, *[str_308672], **kwargs_308673)

# Obtaining the member 'extend_path' of a type (line 1)
extend_path_308675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1, 11), import___call_result_308674, 'extend_path')
# Calling extend_path(args, kwargs) (line 1)
extend_path_call_result_308679 = invoke(stypy.reporting.localization.Localization(__file__, 1, 11), extend_path_308675, *[path___308676, name___308677], **kwargs_308678)

# Assigning a type to the variable '__path__' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), '__path__', extend_path_call_result_308679)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
