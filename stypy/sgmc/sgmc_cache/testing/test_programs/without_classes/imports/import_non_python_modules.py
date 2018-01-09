
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: from stypy import stypy_main, stypy_parameters
3: 
4: x = stypy_main.reset_logs
5: y = stypy_parameters.TYPE_INFERENCE_PATH
6: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from stypy import stypy_main, stypy_parameters' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')
import_5021 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy')

if (type(import_5021) is not StypyTypeError):

    if (import_5021 != 'pyd_module'):
        __import__(import_5021)
        sys_modules_5022 = sys.modules[import_5021]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy', sys_modules_5022.module_type_store, module_type_store, ['stypy_main', 'stypy_parameters'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_5022, sys_modules_5022.module_type_store, module_type_store)
    else:
        from stypy import stypy_main, stypy_parameters

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy', None, module_type_store, ['stypy_main', 'stypy_parameters'], [stypy_main, stypy_parameters])

else:
    # Assigning a type to the variable 'stypy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy', import_5021)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')


# Assigning a Attribute to a Name (line 4):
# Getting the type of 'stypy_main' (line 4)
stypy_main_5023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'stypy_main')
# Obtaining the member 'reset_logs' of a type (line 4)
reset_logs_5024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), stypy_main_5023, 'reset_logs')
# Assigning a type to the variable 'x' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'x', reset_logs_5024)

# Assigning a Attribute to a Name (line 5):
# Getting the type of 'stypy_parameters' (line 5)
stypy_parameters_5025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_parameters')
# Obtaining the member 'TYPE_INFERENCE_PATH' of a type (line 5)
TYPE_INFERENCE_PATH_5026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), stypy_parameters_5025, 'TYPE_INFERENCE_PATH')
# Assigning a type to the variable 'y' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'y', TYPE_INFERENCE_PATH_5026)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
