
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "range builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Integer) -> <built-in function range>
7:     # (Overloads__trunc__) -> <built-in function range>
8:     # (Integer, Integer) -> <built-in function range>
9:     # (Overloads__trunc__, Integer) -> <built-in function range>
10:     # (Integer, Overloads__trunc__) -> <built-in function range>
11:     # (Overloads__trunc__, Overloads__trunc__) -> <built-in function range>
12:     # (Integer, Integer, Integer) -> <built-in function range>
13:     # (Overloads__trunc__, Integer, Integer) -> <built-in function range>
14:     # (Integer, Overloads__trunc__, Integer) -> <built-in function range>
15:     # (Integer, Integer, Overloads__trunc__) -> <built-in function range>
16:     # (Integer, Overloads__trunc__, Overloads__trunc__) -> <built-in function range>
17:     # (Overloads__trunc__, Overloads__trunc__, Integer) -> <built-in function range>
18:     # (Overloads__trunc__, Integer, Overloads__trunc__) -> <built-in function range>
19:     # (Overloads__trunc__, Overloads__trunc__, Overloads__trunc__) -> <built-in function range>
20: 
21: 
22: 
23: 
24: 
25: 
26:     # Call the builtin with correct parameters
27:     ret = range(3)
28:     ret = range(3, 6)
29: 
30:     # Call the builtin with incorrect types of parameters
31: 
32:     # Type error
33:     ret = range("str")
34:     # Type error
35:     ret = range()
36: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'range builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 27):
    
    # Call to range(...): (line 27)
    # Processing the call arguments (line 27)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'int')
    # Processing the call keyword arguments (line 27)
    kwargs_4 = {}
    # Getting the type of 'range' (line 27)
    range_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 10), 'range', False)
    # Calling range(args, kwargs) (line 27)
    range_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 27, 10), range_2, *[int_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'ret', range_call_result_5)
    
    # Assigning a Call to a Name (line 28):
    
    # Call to range(...): (line 28)
    # Processing the call arguments (line 28)
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 16), 'int')
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 19), 'int')
    # Processing the call keyword arguments (line 28)
    kwargs_9 = {}
    # Getting the type of 'range' (line 28)
    range_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 10), 'range', False)
    # Calling range(args, kwargs) (line 28)
    range_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 28, 10), range_6, *[int_7, int_8], **kwargs_9)
    
    # Assigning a type to the variable 'ret' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'ret', range_call_result_10)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to range(...): (line 33)
    # Processing the call arguments (line 33)
    str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 16), 'str', 'str')
    # Processing the call keyword arguments (line 33)
    kwargs_13 = {}
    # Getting the type of 'range' (line 33)
    range_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 10), 'range', False)
    # Calling range(args, kwargs) (line 33)
    range_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 33, 10), range_11, *[str_12], **kwargs_13)
    
    # Assigning a type to the variable 'ret' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'ret', range_call_result_14)
    
    # Assigning a Call to a Name (line 35):
    
    # Call to range(...): (line 35)
    # Processing the call keyword arguments (line 35)
    kwargs_16 = {}
    # Getting the type of 'range' (line 35)
    range_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 10), 'range', False)
    # Calling range(args, kwargs) (line 35)
    range_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 35, 10), range_15, *[], **kwargs_16)
    
    # Assigning a type to the variable 'ret' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'ret', range_call_result_17)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
