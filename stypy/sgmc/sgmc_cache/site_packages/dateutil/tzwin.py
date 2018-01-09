
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # tzwin has moved to dateutil.tz.win
2: from .tz.win import *
3: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from dateutil.tz.win import ' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/dateutil/')
import_320108 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'dateutil.tz.win')

if (type(import_320108) is not StypyTypeError):

    if (import_320108 != 'pyd_module'):
        __import__(import_320108)
        sys_modules_320109 = sys.modules[import_320108]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'dateutil.tz.win', sys_modules_320109.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_320109, sys_modules_320109.module_type_store, module_type_store)
    else:
        from dateutil.tz.win import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'dateutil.tz.win', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'dateutil.tz.win' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'dateutil.tz.win', import_320108)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/dateutil/')


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
