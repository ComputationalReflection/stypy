
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: package_path = "foo"
3: 
4: if package_path is None:
5:     raise AssertionError
6: 
7: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_2665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 15), 'str', 'foo')
# Assigning a type to the variable 'package_path' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'package_path', str_2665)

# Type idiom detected: calculating its left and rigth part (line 4)
# Getting the type of 'package_path' (line 4)
package_path_2666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 3), 'package_path')
# Getting the type of 'None' (line 4)
None_2667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 19), 'None')

(may_be_2668, more_types_in_union_2669) = may_be_none(package_path_2666, None_2667)

if may_be_2668:

    if more_types_in_union_2669:
        # Runtime conditional SSA (line 4)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    # Getting the type of 'AssertionError' (line 5)
    AssertionError_2670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 10), 'AssertionError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 5, 4), AssertionError_2670, 'raise parameter', BaseException)

    if more_types_in_union_2669:
        # SSA join for if statement (line 4)
        module_type_store = module_type_store.join_ssa_context()




# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
