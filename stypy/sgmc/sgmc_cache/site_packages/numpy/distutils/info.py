
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Enhanced distutils with Fortran compilers support and more.
3: '''
4: from __future__ import division, absolute_import, print_function
5: 
6: postpone_import = True
7: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_35804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nEnhanced distutils with Fortran compilers support and more.\n')

# Assigning a Name to a Name (line 6):
# Getting the type of 'True' (line 6)
True_35805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 18), 'True')
# Assigning a type to the variable 'postpone_import' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'postpone_import', True_35805)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
