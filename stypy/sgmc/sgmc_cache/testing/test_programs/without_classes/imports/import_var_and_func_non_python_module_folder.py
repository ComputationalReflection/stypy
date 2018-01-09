
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from modules import module_to_import
2: 
3: x = module_to_import.global_a2
4: y = module_to_import.f_parent2()
5: z = module_to_import.submodule
6: w = module_to_import.submodule.submodule_var
7: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from modules import module_to_import' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')
import_5122 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'modules')

if (type(import_5122) is not StypyTypeError):

    if (import_5122 != 'pyd_module'):
        __import__(import_5122)
        sys_modules_5123 = sys.modules[import_5122]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'modules', sys_modules_5123.module_type_store, module_type_store, ['module_to_import'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_5123, sys_modules_5123.module_type_store, module_type_store)
    else:
        from modules import module_to_import

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'modules', None, module_type_store, ['module_to_import'], [module_to_import])

else:
    # Assigning a type to the variable 'modules' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'modules', import_5122)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')


# Assigning a Attribute to a Name (line 3):
# Getting the type of 'module_to_import' (line 3)
module_to_import_5124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'module_to_import')
# Obtaining the member 'global_a2' of a type (line 3)
global_a2_5125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 3, 4), module_to_import_5124, 'global_a2')
# Assigning a type to the variable 'x' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'x', global_a2_5125)

# Assigning a Call to a Name (line 4):

# Call to f_parent2(...): (line 4)
# Processing the call keyword arguments (line 4)
kwargs_5128 = {}
# Getting the type of 'module_to_import' (line 4)
module_to_import_5126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'module_to_import', False)
# Obtaining the member 'f_parent2' of a type (line 4)
f_parent2_5127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), module_to_import_5126, 'f_parent2')
# Calling f_parent2(args, kwargs) (line 4)
f_parent2_call_result_5129 = invoke(stypy.reporting.localization.Localization(__file__, 4, 4), f_parent2_5127, *[], **kwargs_5128)

# Assigning a type to the variable 'y' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'y', f_parent2_call_result_5129)

# Assigning a Attribute to a Name (line 5):
# Getting the type of 'module_to_import' (line 5)
module_to_import_5130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'module_to_import')
# Obtaining the member 'submodule' of a type (line 5)
submodule_5131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), module_to_import_5130, 'submodule')
# Assigning a type to the variable 'z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'z', submodule_5131)

# Assigning a Attribute to a Name (line 6):
# Getting the type of 'module_to_import' (line 6)
module_to_import_5132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'module_to_import')
# Obtaining the member 'submodule' of a type (line 6)
submodule_5133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), module_to_import_5132, 'submodule')
# Obtaining the member 'submodule_var' of a type (line 6)
submodule_var_5134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), submodule_5133, 'submodule_var')
# Assigning a type to the variable 'w' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'w', submodule_var_5134)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
