
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__imod__ usage with python types"
3: 
4: if __name__ == '__main__':
5:     l = list()
6: 
7:     # Type error
8:     l %= 3
9: 
10:     t = tuple()
11: 
12:     # Type error
13:     t %= "str"
14: 
15:     x = dict()
16: 
17:     # Type error
18:     x %= 4
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__imod__ usage with python types')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 5):
    
    # Call to list(...): (line 5)
    # Processing the call keyword arguments (line 5)
    kwargs_3 = {}
    # Getting the type of 'list' (line 5)
    list_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'list', False)
    # Calling list(args, kwargs) (line 5)
    list_call_result_4 = invoke(stypy.reporting.localization.Localization(__file__, 5, 8), list_2, *[], **kwargs_3)
    
    # Assigning a type to the variable 'l' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'l', list_call_result_4)
    
    # Getting the type of 'l' (line 8)
    l_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'l')
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 9), 'int')
    # Applying the binary operator '%=' (line 8)
    result_imod_7 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 4), '%=', l_5, int_6)
    # Assigning a type to the variable 'l' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'l', result_imod_7)
    
    
    # Assigning a Call to a Name (line 10):
    
    # Call to tuple(...): (line 10)
    # Processing the call keyword arguments (line 10)
    kwargs_9 = {}
    # Getting the type of 'tuple' (line 10)
    tuple_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'tuple', False)
    # Calling tuple(args, kwargs) (line 10)
    tuple_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 10, 8), tuple_8, *[], **kwargs_9)
    
    # Assigning a type to the variable 't' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 't', tuple_call_result_10)
    
    # Getting the type of 't' (line 13)
    t_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 't')
    str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 9), 'str', 'str')
    # Applying the binary operator '%=' (line 13)
    result_imod_13 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 4), '%=', t_11, str_12)
    # Assigning a type to the variable 't' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 't', result_imod_13)
    
    
    # Assigning a Call to a Name (line 15):
    
    # Call to dict(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_15 = {}
    # Getting the type of 'dict' (line 15)
    dict_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'dict', False)
    # Calling dict(args, kwargs) (line 15)
    dict_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 15, 8), dict_14, *[], **kwargs_15)
    
    # Assigning a type to the variable 'x' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'x', dict_call_result_16)
    
    # Getting the type of 'x' (line 18)
    x_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'x')
    int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 9), 'int')
    # Applying the binary operator '%=' (line 18)
    result_imod_19 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 4), '%=', x_17, int_18)
    # Assigning a type to the variable 'x' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'x', result_imod_19)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
