
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import absolute_import
2: 
3: from .core import use, context, available, library, reload_library
4: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from matplotlib.style.core import use, context, available, library, reload_library' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/style/')
import_288323 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'matplotlib.style.core')

if (type(import_288323) is not StypyTypeError):

    if (import_288323 != 'pyd_module'):
        __import__(import_288323)
        sys_modules_288324 = sys.modules[import_288323]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'matplotlib.style.core', sys_modules_288324.module_type_store, module_type_store, ['use', 'context', 'available', 'library', 'reload_library'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_288324, sys_modules_288324.module_type_store, module_type_store)
    else:
        from matplotlib.style.core import use, context, available, library, reload_library

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'matplotlib.style.core', None, module_type_store, ['use', 'context', 'available', 'library', 'reload_library'], [use, context, available, library, reload_library])

else:
    # Assigning a type to the variable 'matplotlib.style.core' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'matplotlib.style.core', import_288323)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/style/')


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
