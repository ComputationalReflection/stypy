
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: import math, os, module_to_import
4: 
5: r = math.cos
6: r2 = os.path
7: r3 = module_to_import.global_a
8: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# Multiple import statement. import math (1/3) (line 3)
import math

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'math', math, module_type_store)
# Multiple import statement. import os (2/3) (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)
# Multiple import statement. import module_to_import (3/3) (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')
import_5000 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'module_to_import')

if (type(import_5000) is not StypyTypeError):

    if (import_5000 != 'pyd_module'):
        __import__(import_5000)
        sys_modules_5001 = sys.modules[import_5000]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'module_to_import', sys_modules_5001.module_type_store, module_type_store)
    else:
        import module_to_import

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'module_to_import', module_to_import, module_type_store)

else:
    # Assigning a type to the variable 'module_to_import' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'module_to_import', import_5000)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')


# Assigning a Attribute to a Name (line 5):
# Getting the type of 'math' (line 5)
math_5002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'math')
# Obtaining the member 'cos' of a type (line 5)
cos_5003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), math_5002, 'cos')
# Assigning a type to the variable 'r' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'r', cos_5003)

# Assigning a Attribute to a Name (line 6):
# Getting the type of 'os' (line 6)
os_5004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'os')
# Obtaining the member 'path' of a type (line 6)
path_5005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 5), os_5004, 'path')
# Assigning a type to the variable 'r2' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r2', path_5005)

# Assigning a Attribute to a Name (line 7):
# Getting the type of 'module_to_import' (line 7)
module_to_import_5006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'module_to_import')
# Obtaining the member 'global_a' of a type (line 7)
global_a_5007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 5), module_to_import_5006, 'global_a')
# Assigning a type to the variable 'r3' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r3', global_a_5007)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
