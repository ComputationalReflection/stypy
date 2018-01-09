
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Distributor init file
2: 
3: Distributors: you can add custom code here to support particular distributions
4: of scipy.
5: 
6: For example, this is a good place to put any checks for hardware requirements.
7: 
8: The scipy standard source distribution will not put code in this file, so you
9: can safely replace this file with your own version.
10: '''
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, (-1)), 'str', ' Distributor init file\n\nDistributors: you can add custom code here to support particular distributions\nof scipy.\n\nFor example, this is a good place to put any checks for hardware requirements.\n\nThe scipy standard source distribution will not put code in this file, so you\ncan safely replace this file with your own version.\n')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
