
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "No execution path has an execution flow free of type errors"
3: 
4: if __name__ == '__main__':
5:     var = list()
6:     for i in range(4):
7:         var = str(var) + str(i)
8: 
9:     # Type error
10:     print var / 3
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'No execution path has an execution flow free of type errors')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 5):
    
    # Call to list(...): (line 5)
    # Processing the call keyword arguments (line 5)
    kwargs_3 = {}
    # Getting the type of 'list' (line 5)
    list_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 10), 'list', False)
    # Calling list(args, kwargs) (line 5)
    list_call_result_4 = invoke(stypy.reporting.localization.Localization(__file__, 5, 10), list_2, *[], **kwargs_3)
    
    # Assigning a type to the variable 'var' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'var', list_call_result_4)
    
    
    # Call to range(...): (line 6)
    # Processing the call arguments (line 6)
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 19), 'int')
    # Processing the call keyword arguments (line 6)
    kwargs_7 = {}
    # Getting the type of 'range' (line 6)
    range_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'range', False)
    # Calling range(args, kwargs) (line 6)
    range_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 6, 13), range_5, *[int_6], **kwargs_7)
    
    # Testing the type of a for loop iterable (line 6)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 6, 4), range_call_result_8)
    # Getting the type of the for loop variable (line 6)
    for_loop_var_9 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 6, 4), range_call_result_8)
    # Assigning a type to the variable 'i' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'i', for_loop_var_9)
    # SSA begins for a for statement (line 6)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 7):
    
    # Call to str(...): (line 7)
    # Processing the call arguments (line 7)
    # Getting the type of 'var' (line 7)
    var_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 18), 'var', False)
    # Processing the call keyword arguments (line 7)
    kwargs_12 = {}
    # Getting the type of 'str' (line 7)
    str_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 14), 'str', False)
    # Calling str(args, kwargs) (line 7)
    str_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 7, 14), str_10, *[var_11], **kwargs_12)
    
    
    # Call to str(...): (line 7)
    # Processing the call arguments (line 7)
    # Getting the type of 'i' (line 7)
    i_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 29), 'i', False)
    # Processing the call keyword arguments (line 7)
    kwargs_16 = {}
    # Getting the type of 'str' (line 7)
    str_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 25), 'str', False)
    # Calling str(args, kwargs) (line 7)
    str_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 7, 25), str_14, *[i_15], **kwargs_16)
    
    # Applying the binary operator '+' (line 7)
    result_add_18 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 14), '+', str_call_result_13, str_call_result_17)
    
    # Assigning a type to the variable 'var' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'var', result_add_18)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'var' (line 10)
    var_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 10), 'var')
    int_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 16), 'int')
    # Applying the binary operator 'div' (line 10)
    result_div_21 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 10), 'div', var_19, int_20)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
