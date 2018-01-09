
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: from ._subplots import *
5: from ._axes import *
6: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from matplotlib.axes._subplots import ' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/axes/')
import_217762 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'matplotlib.axes._subplots')

if (type(import_217762) is not StypyTypeError):

    if (import_217762 != 'pyd_module'):
        __import__(import_217762)
        sys_modules_217763 = sys.modules[import_217762]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'matplotlib.axes._subplots', sys_modules_217763.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_217763, sys_modules_217763.module_type_store, module_type_store)
    else:
        from matplotlib.axes._subplots import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'matplotlib.axes._subplots', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'matplotlib.axes._subplots' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'matplotlib.axes._subplots', import_217762)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/axes/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from matplotlib.axes._axes import ' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/axes/')
import_217764 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'matplotlib.axes._axes')

if (type(import_217764) is not StypyTypeError):

    if (import_217764 != 'pyd_module'):
        __import__(import_217764)
        sys_modules_217765 = sys.modules[import_217764]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'matplotlib.axes._axes', sys_modules_217765.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_217765, sys_modules_217765.module_type_store, module_type_store)
    else:
        from matplotlib.axes._axes import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'matplotlib.axes._axes', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'matplotlib.axes._axes' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'matplotlib.axes._axes', import_217764)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/axes/')


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
