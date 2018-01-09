
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "If with non-boolean condition"
4: 
5: if __name__ == '__main__':
6:     a = "str"
7: 
8:     if a:
9:         print a
10:     else:
11:         print "else"
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'If with non-boolean condition')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Str to a Name (line 6):
    str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 8), 'str', 'str')
    # Assigning a type to the variable 'a' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'a', str_2)
    
    # Getting the type of 'a' (line 8)
    a_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 7), 'a')
    # Testing the type of an if condition (line 8)
    if_condition_4 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 8, 4), a_3)
    # Assigning a type to the variable 'if_condition_4' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'if_condition_4', if_condition_4)
    # SSA begins for if statement (line 8)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'a' (line 9)
    a_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 14), 'a')
    # SSA branch for the else part of an if statement (line 8)
    module_type_store.open_ssa_branch('else')
    str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 14), 'str', 'else')
    # SSA join for if statement (line 8)
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
