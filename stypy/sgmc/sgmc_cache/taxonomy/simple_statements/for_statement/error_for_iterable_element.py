
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "The statement uses an iterable, but performs invalid operations over the iterable elements type"
3: 
4: if __name__ == '__main__':
5:     l = range(5)
6: 
7:     for n in l:
8:         # Type error
9:         print n + "string"
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'The statement uses an iterable, but performs invalid operations over the iterable elements type')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 5):
    
    # Call to range(...): (line 5)
    # Processing the call arguments (line 5)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
    # Processing the call keyword arguments (line 5)
    kwargs_4 = {}
    # Getting the type of 'range' (line 5)
    range_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'range', False)
    # Calling range(args, kwargs) (line 5)
    range_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 5, 8), range_2, *[int_3], **kwargs_4)
    
    # Assigning a type to the variable 'l' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'l', range_call_result_5)
    
    # Getting the type of 'l' (line 7)
    l_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 13), 'l')
    # Testing the type of a for loop iterable (line 7)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 7, 4), l_6)
    # Getting the type of the for loop variable (line 7)
    for_loop_var_7 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 7, 4), l_6)
    # Assigning a type to the variable 'n' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'n', for_loop_var_7)
    # SSA begins for a for statement (line 7)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    # Getting the type of 'n' (line 9)
    n_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 14), 'n')
    str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 18), 'str', 'string')
    # Applying the binary operator '+' (line 9)
    result_add_10 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 14), '+', n_8, str_9)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
