
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from module_to_import import global_a, f_parent
2: 
3: x = global_a
4: r = f_parent()
5: 
6: print x
7: print r
8: 
9: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from module_to_import import global_a, f_parent' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')
import_5114 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'module_to_import')

if (type(import_5114) is not StypyTypeError):

    if (import_5114 != 'pyd_module'):
        __import__(import_5114)
        sys_modules_5115 = sys.modules[import_5114]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'module_to_import', sys_modules_5115.module_type_store, module_type_store, ['global_a', 'f_parent'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_5115, sys_modules_5115.module_type_store, module_type_store)
    else:
        from module_to_import import global_a, f_parent

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'module_to_import', None, module_type_store, ['global_a', 'f_parent'], [global_a, f_parent])

else:
    # Assigning a type to the variable 'module_to_import' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'module_to_import', import_5114)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')


# Assigning a Name to a Name (line 3):
# Getting the type of 'global_a' (line 3)
global_a_5116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'global_a')
# Assigning a type to the variable 'x' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'x', global_a_5116)

# Assigning a Call to a Name (line 4):

# Call to f_parent(...): (line 4)
# Processing the call keyword arguments (line 4)
kwargs_5118 = {}
# Getting the type of 'f_parent' (line 4)
f_parent_5117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'f_parent', False)
# Calling f_parent(args, kwargs) (line 4)
f_parent_call_result_5119 = invoke(stypy.reporting.localization.Localization(__file__, 4, 4), f_parent_5117, *[], **kwargs_5118)

# Assigning a type to the variable 'r' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'r', f_parent_call_result_5119)
# Getting the type of 'x' (line 6)
x_5120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 6), 'x')
# Getting the type of 'r' (line 7)
r_5121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 6), 'r')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
