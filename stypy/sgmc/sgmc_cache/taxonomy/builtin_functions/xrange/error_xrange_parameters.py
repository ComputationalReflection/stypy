
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "xrange method is present, but is invoked with a wrong number of parameters"
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
22:     # Call the builtin with incorrect number of parameters
23:     # Type error
24:     ret = xrange(3, 4, 5, 6)
25: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'xrange method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 24):
    
    # Call to xrange(...): (line 24)
    # Processing the call arguments (line 24)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 17), 'int')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 20), 'int')
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'int')
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 26), 'int')
    # Processing the call keyword arguments (line 24)
    kwargs_7 = {}
    # Getting the type of 'xrange' (line 24)
    xrange_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 10), 'xrange', False)
    # Calling xrange(args, kwargs) (line 24)
    xrange_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 24, 10), xrange_2, *[int_3, int_4, int_5, int_6], **kwargs_7)
    
    # Assigning a type to the variable 'ret' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'ret', xrange_call_result_8)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
