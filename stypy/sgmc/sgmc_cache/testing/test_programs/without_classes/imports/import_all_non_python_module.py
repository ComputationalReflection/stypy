
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from module_to_import import *
2: 
3: x = global_a
4: f_parent()
5: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from module_to_import import ' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')
import_4967 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'module_to_import')

if (type(import_4967) is not StypyTypeError):

    if (import_4967 != 'pyd_module'):
        __import__(import_4967)
        sys_modules_4968 = sys.modules[import_4967]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'module_to_import', sys_modules_4968.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_4968, sys_modules_4968.module_type_store, module_type_store)
    else:
        from module_to_import import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'module_to_import', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'module_to_import' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'module_to_import', import_4967)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')


# Assigning a Name to a Name (line 3):
# Getting the type of 'global_a' (line 3)
global_a_4969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'global_a')
# Assigning a type to the variable 'x' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'x', global_a_4969)

# Call to f_parent(...): (line 4)
# Processing the call keyword arguments (line 4)
kwargs_4971 = {}
# Getting the type of 'f_parent' (line 4)
f_parent_4970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'f_parent', False)
# Calling f_parent(args, kwargs) (line 4)
f_parent_call_result_4972 = invoke(stypy.reporting.localization.Localization(__file__, 4, 0), f_parent_4970, *[], **kwargs_4971)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
