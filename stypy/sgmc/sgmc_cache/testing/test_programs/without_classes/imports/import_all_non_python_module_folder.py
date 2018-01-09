
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from modules.module_to_import import *
2: 
3: 
4: x = global_a2
5: y = f_parent2()
6: z = submodule
7: w = submodule.submodule_var
8: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from modules.module_to_import import ' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')
import_4973 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'modules.module_to_import')

if (type(import_4973) is not StypyTypeError):

    if (import_4973 != 'pyd_module'):
        __import__(import_4973)
        sys_modules_4974 = sys.modules[import_4973]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'modules.module_to_import', sys_modules_4974.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_4974, sys_modules_4974.module_type_store, module_type_store)
    else:
        from modules.module_to_import import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'modules.module_to_import', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'modules.module_to_import' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'modules.module_to_import', import_4973)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')


# Assigning a Name to a Name (line 4):
# Getting the type of 'global_a2' (line 4)
global_a2_4975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'global_a2')
# Assigning a type to the variable 'x' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'x', global_a2_4975)

# Assigning a Call to a Name (line 5):

# Call to f_parent2(...): (line 5)
# Processing the call keyword arguments (line 5)
kwargs_4977 = {}
# Getting the type of 'f_parent2' (line 5)
f_parent2_4976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'f_parent2', False)
# Calling f_parent2(args, kwargs) (line 5)
f_parent2_call_result_4978 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), f_parent2_4976, *[], **kwargs_4977)

# Assigning a type to the variable 'y' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'y', f_parent2_call_result_4978)

# Assigning a Name to a Name (line 6):
# Getting the type of 'submodule' (line 6)
submodule_4979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'submodule')
# Assigning a type to the variable 'z' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'z', submodule_4979)

# Assigning a Attribute to a Name (line 7):
# Getting the type of 'submodule' (line 7)
submodule_4980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'submodule')
# Obtaining the member 'submodule_var' of a type (line 7)
submodule_var_4981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), submodule_4980, 'submodule_var')
# Assigning a type to the variable 'w' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'w', submodule_var_4981)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
