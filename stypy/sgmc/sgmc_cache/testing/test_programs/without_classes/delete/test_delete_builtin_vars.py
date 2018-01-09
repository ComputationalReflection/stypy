
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: 
4: 
5: import os
6: 
7: r1 = os
8: 
9: del os
10: 
11: r2 = os

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import os' statement (line 5)
import os

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os', os, module_type_store)


# Assigning a Name to a Name (line 7):
# Getting the type of 'os' (line 7)
os_6278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'os')
# Assigning a type to the variable 'r1' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r1', os_6278)
# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 9, 0), module_type_store, 'os')

# Assigning a Name to a Name (line 11):
# Getting the type of 'os' (line 11)
os_6279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'os')
# Assigning a type to the variable 'r2' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r2', os_6279)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
