
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Use an dict type instead of a dict instance"
4: 
5: if __name__ == '__main__':
6:     # Type error
7:     for i in dict:
8:         print i
9: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Use an dict type instead of a dict instance')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Getting the type of 'dict' (line 7)
    dict_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 13), 'dict')
    # Testing the type of a for loop iterable (line 7)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 7, 4), dict_2)
    # Getting the type of the for loop variable (line 7)
    for_loop_var_3 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 7, 4), dict_2)
    # Assigning a type to the variable 'i' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'i', for_loop_var_3)
    # SSA begins for a for statement (line 7)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    # Getting the type of 'i' (line 8)
    i_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 14), 'i')
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
