
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "xrange builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Integer) -> <type 'xrange'>
7:     # (Overloads__trunc__) -> <type 'xrange'>
8:     # (Integer, Integer) -> <type 'xrange'>
9:     # (Overloads__trunc__, Integer) -> <type 'xrange'>
10:     # (Integer, Overloads__trunc__) -> <type 'xrange'>
11:     # (Overloads__trunc__, Overloads__trunc__) -> <type 'xrange'>
12:     # (Integer, Integer, Integer) -> <type 'xrange'>
13:     # (Overloads__trunc__, Integer, Integer) -> <type 'xrange'>
14:     # (Integer, Overloads__trunc__, Integer) -> <type 'xrange'>
15:     # (Integer, Integer, Overloads__trunc__) -> <type 'xrange'>
16:     # (Integer, Overloads__trunc__, Overloads__trunc__) -> <type 'xrange'>
17:     # (Overloads__trunc__, Overloads__trunc__, Integer) -> <type 'xrange'>
18:     # (Overloads__trunc__, Integer, Overloads__trunc__) -> <type 'xrange'>
19:     # (Overloads__trunc__, Overloads__trunc__, Overloads__trunc__) -> <type 'xrange'>
20: 
21: 
22:     # Call the builtin
23:     ret = xrange(3)
24: 
25:     # Type error
26:     ret.unexisting_method()
27: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'xrange builtin is invoked and its return type is used to call an non existing method')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 23):
    
    # Call to xrange(...): (line 23)
    # Processing the call arguments (line 23)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 17), 'int')
    # Processing the call keyword arguments (line 23)
    kwargs_4 = {}
    # Getting the type of 'xrange' (line 23)
    xrange_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'xrange', False)
    # Calling xrange(args, kwargs) (line 23)
    xrange_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), xrange_2, *[int_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'ret', xrange_call_result_5)
    
    # Call to unexisting_method(...): (line 26)
    # Processing the call keyword arguments (line 26)
    kwargs_8 = {}
    # Getting the type of 'ret' (line 26)
    ret_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'ret', False)
    # Obtaining the member 'unexisting_method' of a type (line 26)
    unexisting_method_7 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 4), ret_6, 'unexisting_method')
    # Calling unexisting_method(args, kwargs) (line 26)
    unexisting_method_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), unexisting_method_7, *[], **kwargs_8)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
