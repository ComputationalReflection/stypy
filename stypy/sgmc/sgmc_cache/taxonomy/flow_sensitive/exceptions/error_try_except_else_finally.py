
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "The finally execution path has an execution flow free of type errors"
4: 
5: if __name__ == '__main__':
6:     try:
7:         a = 3
8:     except KeyError as k:
9:         a = "3"
10:     except Exception as e:
11:         a = list()
12:     else:
13:         a = dict()
14:     finally:
15:         a = 3.2
16: 
17:     # Type error
18:     r1 = len(a)
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'The finally execution path has an execution flow free of type errors')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Try-finally block (line 6)
    
    
    # SSA begins for try-except statement (line 6)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Num to a Name (line 7):
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 12), 'int')
    # Assigning a type to the variable 'a' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'a', int_2)
    # SSA branch for the except part of a try statement (line 6)
    # SSA branch for the except 'KeyError' branch of a try statement (line 6)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'KeyError' (line 8)
    KeyError_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 11), 'KeyError')
    # Assigning a type to the variable 'k' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'k', KeyError_3)
    
    # Assigning a Str to a Name (line 9):
    str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 12), 'str', '3')
    # Assigning a type to the variable 'a' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'a', str_4)
    # SSA branch for the except 'Exception' branch of a try statement (line 6)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 10)
    Exception_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'Exception')
    # Assigning a type to the variable 'e' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'e', Exception_5)
    
    # Assigning a Call to a Name (line 11):
    
    # Call to list(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_7 = {}
    # Getting the type of 'list' (line 11)
    list_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'list', False)
    # Calling list(args, kwargs) (line 11)
    list_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 11, 12), list_6, *[], **kwargs_7)
    
    # Assigning a type to the variable 'a' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'a', list_call_result_8)
    # SSA branch for the else branch of a try statement (line 6)
    module_type_store.open_ssa_branch('except else')
    
    # Assigning a Call to a Name (line 13):
    
    # Call to dict(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_10 = {}
    # Getting the type of 'dict' (line 13)
    dict_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'dict', False)
    # Calling dict(args, kwargs) (line 13)
    dict_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 13, 12), dict_9, *[], **kwargs_10)
    
    # Assigning a type to the variable 'a' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'a', dict_call_result_11)
    # SSA join for try-except statement (line 6)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 6)
    
    # Assigning a Num to a Name (line 15):
    float_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'float')
    # Assigning a type to the variable 'a' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'a', float_12)
    
    
    # Assigning a Call to a Name (line 18):
    
    # Call to len(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'a' (line 18)
    a_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 13), 'a', False)
    # Processing the call keyword arguments (line 18)
    kwargs_15 = {}
    # Getting the type of 'len' (line 18)
    len_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 9), 'len', False)
    # Calling len(args, kwargs) (line 18)
    len_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 18, 9), len_13, *[a_14], **kwargs_15)
    
    # Assigning a type to the variable 'r1' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'r1', len_call_result_16)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
