
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "At least one (but not all) execution paths has an execution flow free of type errors"
4: 
5: if __name__ == '__main__':
6:     try:
7:         a = 3
8:         b = "5"
9:     except KeyError as k:
10:         a = 3.4
11:     except Exception as e:
12:         a = list()
13:     else:
14:         # Type warning
15:         b = b / 5
16:         a = dict()
17: 
18:     # Type warning
19:     r1 = len(a)
20: 
21: 
22: 
23: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'At least one (but not all) execution paths has an execution flow free of type errors')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    
    # SSA begins for try-except statement (line 6)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Num to a Name (line 7):
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 12), 'int')
    # Assigning a type to the variable 'a' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'a', int_2)
    
    # Assigning a Str to a Name (line 8):
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 12), 'str', '5')
    # Assigning a type to the variable 'b' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'b', str_3)
    # SSA branch for the except part of a try statement (line 6)
    # SSA branch for the except 'KeyError' branch of a try statement (line 6)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'KeyError' (line 9)
    KeyError_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'KeyError')
    # Assigning a type to the variable 'k' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'k', KeyError_4)
    
    # Assigning a Num to a Name (line 10):
    float_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 12), 'float')
    # Assigning a type to the variable 'a' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'a', float_5)
    # SSA branch for the except 'Exception' branch of a try statement (line 6)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 11)
    Exception_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'Exception')
    # Assigning a type to the variable 'e' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'e', Exception_6)
    
    # Assigning a Call to a Name (line 12):
    
    # Call to list(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_8 = {}
    # Getting the type of 'list' (line 12)
    list_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'list', False)
    # Calling list(args, kwargs) (line 12)
    list_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 12, 12), list_7, *[], **kwargs_8)
    
    # Assigning a type to the variable 'a' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'a', list_call_result_9)
    # SSA branch for the else branch of a try statement (line 6)
    module_type_store.open_ssa_branch('except else')
    
    # Assigning a BinOp to a Name (line 15):
    # Getting the type of 'b' (line 15)
    b_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'b')
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 16), 'int')
    # Applying the binary operator 'div' (line 15)
    result_div_12 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 12), 'div', b_10, int_11)
    
    # Assigning a type to the variable 'b' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'b', result_div_12)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to dict(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_14 = {}
    # Getting the type of 'dict' (line 16)
    dict_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'dict', False)
    # Calling dict(args, kwargs) (line 16)
    dict_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 16, 12), dict_13, *[], **kwargs_14)
    
    # Assigning a type to the variable 'a' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'a', dict_call_result_15)
    # SSA join for try-except statement (line 6)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 19):
    
    # Call to len(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'a' (line 19)
    a_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), 'a', False)
    # Processing the call keyword arguments (line 19)
    kwargs_18 = {}
    # Getting the type of 'len' (line 19)
    len_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 9), 'len', False)
    # Calling len(args, kwargs) (line 19)
    len_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 19, 9), len_16, *[a_17], **kwargs_18)
    
    # Assigning a type to the variable 'r1' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'r1', len_call_result_19)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
