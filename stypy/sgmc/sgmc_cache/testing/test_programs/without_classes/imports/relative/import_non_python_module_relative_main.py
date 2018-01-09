
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from subpackage import module_to_import
2: 
3: r = module_to_import.x
4: r2 = module_to_import.y
5: 
6: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from subpackage import module_to_import' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/relative/')
import_5197 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'subpackage')

if (type(import_5197) is not StypyTypeError):

    if (import_5197 != 'pyd_module'):
        __import__(import_5197)
        sys_modules_5198 = sys.modules[import_5197]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'subpackage', sys_modules_5198.module_type_store, module_type_store, ['module_to_import'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_5198, sys_modules_5198.module_type_store, module_type_store)
    else:
        from subpackage import module_to_import

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'subpackage', None, module_type_store, ['module_to_import'], [module_to_import])

else:
    # Assigning a type to the variable 'subpackage' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'subpackage', import_5197)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/relative/')


# Assigning a Attribute to a Name (line 3):
# Getting the type of 'module_to_import' (line 3)
module_to_import_5199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'module_to_import')
# Obtaining the member 'x' of a type (line 3)
x_5200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 3, 4), module_to_import_5199, 'x')
# Assigning a type to the variable 'r' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'r', x_5200)

# Assigning a Attribute to a Name (line 4):
# Getting the type of 'module_to_import' (line 4)
module_to_import_5201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 5), 'module_to_import')
# Obtaining the member 'y' of a type (line 4)
y_5202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 5), module_to_import_5201, 'y')
# Assigning a type to the variable 'r2' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'r2', y_5202)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
