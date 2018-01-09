
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "The statement uses a non-iterable data as if it were one"
4: 
5: if __name__ == '__main__':
6:     l = 4
7: 
8:     # Type error
9:     for elem in l:
10:         print "a"
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'The statement uses a non-iterable data as if it were one')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Num to a Name (line 6):
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 8), 'int')
    # Assigning a type to the variable 'l' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'l', int_2)
    
    # Getting the type of 'l' (line 9)
    l_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 16), 'l')
    # Testing the type of a for loop iterable (line 9)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 9, 4), l_3)
    # Getting the type of the for loop variable (line 9)
    for_loop_var_4 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 9, 4), l_3)
    # Assigning a type to the variable 'elem' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'elem', for_loop_var_4)
    # SSA begins for a for statement (line 9)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'str', 'a')
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
