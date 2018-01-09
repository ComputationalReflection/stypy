
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
9:     while 3 in l:
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
    
    
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'int')
    # Getting the type of 'l' (line 9)
    l_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 15), 'l')
    # Applying the binary operator 'in' (line 9)
    result_contains_5 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 10), 'in', int_3, l_4)
    
    # Testing the type of an if condition (line 9)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 9, 4), result_contains_5)
    # SSA begins for while statement (line 9)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 14), 'str', 'a')
    # SSA join for while statement (line 9)
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
