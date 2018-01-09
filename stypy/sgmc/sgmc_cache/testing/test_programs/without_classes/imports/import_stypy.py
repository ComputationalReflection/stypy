
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import stypy
2: 
3: r1 = stypy.TypeError
4: r2 = stypy.ENABLE_CODING_ADVICES
5: 
6: 
7: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import stypy' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')
import_5108 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy')

if (type(import_5108) is not StypyTypeError):

    if (import_5108 != 'pyd_module'):
        __import__(import_5108)
        sys_modules_5109 = sys.modules[import_5108]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy', sys_modules_5109.module_type_store, module_type_store)
    else:
        import stypy

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy', stypy, module_type_store)

else:
    # Assigning a type to the variable 'stypy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy', import_5108)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')


# Assigning a Attribute to a Name (line 3):
# Getting the type of 'stypy' (line 3)
stypy_5110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 5), 'stypy')
# Obtaining the member 'TypeError' of a type (line 3)
TypeError_5111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 3, 5), stypy_5110, 'TypeError')
# Assigning a type to the variable 'r1' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'r1', TypeError_5111)

# Assigning a Attribute to a Name (line 4):
# Getting the type of 'stypy' (line 4)
stypy_5112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 5), 'stypy')
# Obtaining the member 'ENABLE_CODING_ADVICES' of a type (line 4)
ENABLE_CODING_ADVICES_5113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 5), stypy_5112, 'ENABLE_CODING_ADVICES')
# Assigning a type to the variable 'r2' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'r2', ENABLE_CODING_ADVICES_5113)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
