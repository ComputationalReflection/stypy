
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "No execution path has an execution flow free of type errors"
3: 
4: if __name__ == '__main__':
5:     var = 3
6:     if var > 0:
7:         var = "str value"
8:     else:
9:         var = list()
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
    
    # Assigning a Num to a Name (line 5):
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 10), 'int')
    # Assigning a type to the variable 'var' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'var', int_2)
    
    
    # Getting the type of 'var' (line 6)
    var_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 7), 'var')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'int')
    # Applying the binary operator '>' (line 6)
    result_gt_5 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 7), '>', var_3, int_4)
    
    # Testing the type of an if condition (line 6)
    if_condition_6 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 6, 4), result_gt_5)
    # Assigning a type to the variable 'if_condition_6' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'if_condition_6', if_condition_6)
    # SSA begins for if statement (line 6)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 7):
    str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'str', 'str value')
    # Assigning a type to the variable 'var' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'var', str_7)
    # SSA branch for the else part of an if statement (line 6)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 9):
    
    # Call to list(...): (line 9)
    # Processing the call keyword arguments (line 9)
    kwargs_9 = {}
    # Getting the type of 'list' (line 9)
    list_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 14), 'list', False)
    # Calling list(args, kwargs) (line 9)
    list_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 9, 14), list_8, *[], **kwargs_9)
    
    # Assigning a type to the variable 'var' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'var', list_call_result_10)
    # SSA join for if statement (line 6)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'var' (line 12)
    var_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'var')
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'int')
    # Applying the binary operator 'div' (line 12)
    result_div_13 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 10), 'div', var_11, int_12)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
