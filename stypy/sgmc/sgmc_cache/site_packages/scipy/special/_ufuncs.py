
# -*- coding: utf-8 -*-

"""


"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

@norecursion
def jv(localization, *varargs, **kwargs):
    return None

# Assigning a type to the variable 'ellip_normal' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'jv', jv)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
