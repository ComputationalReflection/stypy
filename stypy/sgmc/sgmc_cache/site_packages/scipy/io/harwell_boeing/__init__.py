
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from scipy.io.harwell_boeing.hb import MalformedHeader, HBInfo, HBFile, \
4:     HBMatrixType, hb_read, hb_write
5: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy.io.harwell_boeing.hb import MalformedHeader, HBInfo, HBFile, HBMatrixType, hb_read, hb_write' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/')
import_132906 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.io.harwell_boeing.hb')

if (type(import_132906) is not StypyTypeError):

    if (import_132906 != 'pyd_module'):
        __import__(import_132906)
        sys_modules_132907 = sys.modules[import_132906]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.io.harwell_boeing.hb', sys_modules_132907.module_type_store, module_type_store, ['MalformedHeader', 'HBInfo', 'HBFile', 'HBMatrixType', 'hb_read', 'hb_write'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_132907, sys_modules_132907.module_type_store, module_type_store)
    else:
        from scipy.io.harwell_boeing.hb import MalformedHeader, HBInfo, HBFile, HBMatrixType, hb_read, hb_write

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.io.harwell_boeing.hb', None, module_type_store, ['MalformedHeader', 'HBInfo', 'HBFile', 'HBMatrixType', 'hb_read', 'hb_write'], [MalformedHeader, HBInfo, HBFile, HBMatrixType, hb_read, hb_write])

else:
    # Assigning a type to the variable 'scipy.io.harwell_boeing.hb' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.io.harwell_boeing.hb', import_132906)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/harwell_boeing/')


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
