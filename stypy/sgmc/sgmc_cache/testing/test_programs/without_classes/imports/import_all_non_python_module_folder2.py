
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from modules.other_module_to_import import *
2: 
3: 
4: x = global_a2
5: y = f_parent2()
6: z = submodule_func()
7: w = submodule_var
8: 
9: a = var1
10: b = var2

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from modules.other_module_to_import import ' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')
import_4982 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'modules.other_module_to_import')

if (type(import_4982) is not StypyTypeError):

    if (import_4982 != 'pyd_module'):
        __import__(import_4982)
        sys_modules_4983 = sys.modules[import_4982]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'modules.other_module_to_import', sys_modules_4983.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_4983, sys_modules_4983.module_type_store, module_type_store)
    else:
        from modules.other_module_to_import import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'modules.other_module_to_import', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'modules.other_module_to_import' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'modules.other_module_to_import', import_4982)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')


# Assigning a Name to a Name (line 4):
# Getting the type of 'global_a2' (line 4)
global_a2_4984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'global_a2')
# Assigning a type to the variable 'x' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'x', global_a2_4984)

# Assigning a Call to a Name (line 5):

# Call to f_parent2(...): (line 5)
# Processing the call keyword arguments (line 5)
kwargs_4986 = {}
# Getting the type of 'f_parent2' (line 5)
f_parent2_4985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'f_parent2', False)
# Calling f_parent2(args, kwargs) (line 5)
f_parent2_call_result_4987 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), f_parent2_4985, *[], **kwargs_4986)

# Assigning a type to the variable 'y' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'y', f_parent2_call_result_4987)

# Assigning a Call to a Name (line 6):

# Call to submodule_func(...): (line 6)
# Processing the call keyword arguments (line 6)
kwargs_4989 = {}
# Getting the type of 'submodule_func' (line 6)
submodule_func_4988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'submodule_func', False)
# Calling submodule_func(args, kwargs) (line 6)
submodule_func_call_result_4990 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), submodule_func_4988, *[], **kwargs_4989)

# Assigning a type to the variable 'z' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'z', submodule_func_call_result_4990)

# Assigning a Name to a Name (line 7):
# Getting the type of 'submodule_var' (line 7)
submodule_var_4991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'submodule_var')
# Assigning a type to the variable 'w' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'w', submodule_var_4991)

# Assigning a Name to a Name (line 9):
# Getting the type of 'var1' (line 9)
var1_4992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'var1')
# Assigning a type to the variable 'a' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'a', var1_4992)

# Assigning a Name to a Name (line 10):
# Getting the type of 'var2' (line 10)
var2_4993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'var2')
# Assigning a type to the variable 'b' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'b', var2_4993)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
