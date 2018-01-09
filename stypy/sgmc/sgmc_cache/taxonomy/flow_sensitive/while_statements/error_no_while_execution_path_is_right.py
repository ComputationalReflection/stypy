
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "No execution path has an execution flow free of type errors"
3: 
4: if __name__ == '__main__':
5:     var = list()
6:     i = 0
7:     while i < 4:
8:         var = str(var) + str(i)
9:         i += 1
10: 
11:     # Type error
12:     print var / 3
13: 

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
    
    # Assigning a Num to a Name (line 6):
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 8), 'int')
    # Assigning a type to the variable 'i' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'i', int_5)
    
    
    # Getting the type of 'i' (line 7)
    i_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 10), 'i')
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'int')
    # Applying the binary operator '<' (line 7)
    result_lt_8 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 10), '<', i_6, int_7)
    
    # Testing the type of an if condition (line 7)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 7, 4), result_lt_8)
    # SSA begins for while statement (line 7)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a BinOp to a Name (line 8):
    
    # Call to str(...): (line 8)
    # Processing the call arguments (line 8)
    # Getting the type of 'var' (line 8)
    var_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 18), 'var', False)
    # Processing the call keyword arguments (line 8)
    kwargs_11 = {}
    # Getting the type of 'str' (line 8)
    str_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 14), 'str', False)
    # Calling str(args, kwargs) (line 8)
    str_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 8, 14), str_9, *[var_10], **kwargs_11)
    
    
    # Call to str(...): (line 8)
    # Processing the call arguments (line 8)
    # Getting the type of 'i' (line 8)
    i_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 29), 'i', False)
    # Processing the call keyword arguments (line 8)
    kwargs_15 = {}
    # Getting the type of 'str' (line 8)
    str_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 25), 'str', False)
    # Calling str(args, kwargs) (line 8)
    str_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 8, 25), str_13, *[i_14], **kwargs_15)
    
    # Applying the binary operator '+' (line 8)
    result_add_17 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 14), '+', str_call_result_12, str_call_result_16)
    
    # Assigning a type to the variable 'var' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'var', result_add_17)
    
    # Getting the type of 'i' (line 9)
    i_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'i')
    int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'int')
    # Applying the binary operator '+=' (line 9)
    result_iadd_20 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 8), '+=', i_18, int_19)
    # Assigning a type to the variable 'i' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'i', result_iadd_20)
    
    # SSA join for while statement (line 7)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'var' (line 12)
    var_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'var')
    int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'int')
    # Applying the binary operator 'div' (line 12)
    result_div_23 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 10), 'div', var_21, int_22)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
