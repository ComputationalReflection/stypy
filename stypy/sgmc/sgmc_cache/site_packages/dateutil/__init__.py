
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -*- coding: utf-8 -*-
2: from ._version import VERSION as __version__
3: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from dateutil._version import __version__' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/dateutil/')
import_320193 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'dateutil._version')

if (type(import_320193) is not StypyTypeError):

    if (import_320193 != 'pyd_module'):
        __import__(import_320193)
        sys_modules_320194 = sys.modules[import_320193]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'dateutil._version', sys_modules_320194.module_type_store, module_type_store, ['VERSION'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_320194, sys_modules_320194.module_type_store, module_type_store)
    else:
        from dateutil._version import VERSION as __version__

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'dateutil._version', None, module_type_store, ['VERSION'], [__version__])

else:
    # Assigning a type to the variable 'dateutil._version' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'dateutil._version', import_320193)

# Adding an alias
module_type_store.add_alias('__version__', 'VERSION')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/dateutil/')


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
