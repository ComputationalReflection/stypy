
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "__imul__ usage with python types"
3: 
4: if __name__ == '__main__':
5:     l = list()
6: 
7:     l *= 3
8: 
9:     t = tuple()
10: 
11:     # Type error
12:     t *= "str"
13: 
14:     x = "str"
15: 
16:     x *= 4
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', '__imul__ usage with python types')
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
    
    # Getting the type of 'l' (line 7)
    l_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'l')
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 9), 'int')
    # Applying the binary operator '*=' (line 7)
    result_imul_7 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 4), '*=', l_5, int_6)
    # Assigning a type to the variable 'l' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'l', result_imul_7)
    
    
    # Assigning a Call to a Name (line 9):
    
    # Call to tuple(...): (line 9)
    # Processing the call keyword arguments (line 9)
    kwargs_9 = {}
    # Getting the type of 'tuple' (line 9)
    tuple_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'tuple', False)
    # Calling tuple(args, kwargs) (line 9)
    tuple_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 9, 8), tuple_8, *[], **kwargs_9)
    
    # Assigning a type to the variable 't' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 't', tuple_call_result_10)
    
    # Getting the type of 't' (line 12)
    t_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 't')
    str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 9), 'str', 'str')
    # Applying the binary operator '*=' (line 12)
    result_imul_13 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 4), '*=', t_11, str_12)
    # Assigning a type to the variable 't' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 't', result_imul_13)
    
    
    # Assigning a Str to a Name (line 14):
    str_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 8), 'str', 'str')
    # Assigning a type to the variable 'x' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'x', str_14)
    
    # Getting the type of 'x' (line 16)
    x_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'x')
    int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 9), 'int')
    # Applying the binary operator '*=' (line 16)
    result_imul_17 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 4), '*=', x_15, int_16)
    # Assigning a type to the variable 'x' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'x', result_imul_17)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
