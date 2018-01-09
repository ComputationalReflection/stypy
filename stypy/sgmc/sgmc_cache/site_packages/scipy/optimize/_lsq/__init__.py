
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''This module contains least-squares algorithms.'''
2: from __future__ import division, print_function, absolute_import
3: 
4: from .least_squares import least_squares
5: from .lsq_linear import lsq_linear
6: 
7: __all__ = ['least_squares', 'lsq_linear']
8: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_255710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'This module contains least-squares algorithms.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.optimize._lsq.least_squares import least_squares' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_255711 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.optimize._lsq.least_squares')

if (type(import_255711) is not StypyTypeError):

    if (import_255711 != 'pyd_module'):
        __import__(import_255711)
        sys_modules_255712 = sys.modules[import_255711]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.optimize._lsq.least_squares', sys_modules_255712.module_type_store, module_type_store, ['least_squares'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_255712, sys_modules_255712.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.least_squares import least_squares

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.optimize._lsq.least_squares', None, module_type_store, ['least_squares'], [least_squares])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.least_squares' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.optimize._lsq.least_squares', import_255711)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.optimize._lsq.lsq_linear import lsq_linear' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
import_255713 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.optimize._lsq.lsq_linear')

if (type(import_255713) is not StypyTypeError):

    if (import_255713 != 'pyd_module'):
        __import__(import_255713)
        sys_modules_255714 = sys.modules[import_255713]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.optimize._lsq.lsq_linear', sys_modules_255714.module_type_store, module_type_store, ['lsq_linear'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_255714, sys_modules_255714.module_type_store, module_type_store)
    else:
        from scipy.optimize._lsq.lsq_linear import lsq_linear

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.optimize._lsq.lsq_linear', None, module_type_store, ['lsq_linear'], [lsq_linear])

else:
    # Assigning a type to the variable 'scipy.optimize._lsq.lsq_linear' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.optimize._lsq.lsq_linear', import_255713)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')


# Assigning a List to a Name (line 7):
__all__ = ['least_squares', 'lsq_linear']
module_type_store.set_exportable_members(['least_squares', 'lsq_linear'])

# Obtaining an instance of the builtin type 'list' (line 7)
list_255715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_255716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'str', 'least_squares')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_255715, str_255716)
# Adding element type (line 7)
str_255717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 28), 'str', 'lsq_linear')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 10), list_255715, str_255717)

# Assigning a type to the variable '__all__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__all__', list_255715)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
