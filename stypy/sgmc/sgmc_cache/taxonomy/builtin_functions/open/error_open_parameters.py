
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "open method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Str) -> <type 'file'>
7:     # (Str, Str) -> <type 'file'>
8:     # (Str, Str, Integer) -> <type 'file'>
9:     # (Str, Str, Overloads__trunc__) -> <type 'file'>
10: 
11: 
12:     # Call the builtin with incorrect number of parameters
13:     # Type error
14:     ret = open("f.py", "r", 0, 0)
15:     # Type error
16:     ret = open()
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'open method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 14):
    
    # Call to open(...): (line 14)
    # Processing the call arguments (line 14)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'str', 'f.py')
    str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 23), 'str', 'r')
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 28), 'int')
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 31), 'int')
    # Processing the call keyword arguments (line 14)
    kwargs_7 = {}
    # Getting the type of 'open' (line 14)
    open_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'open', False)
    # Calling open(args, kwargs) (line 14)
    open_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), open_2, *[str_3, str_4, int_5, int_6], **kwargs_7)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', open_call_result_8)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to open(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_10 = {}
    # Getting the type of 'open' (line 16)
    open_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'open', False)
    # Calling open(args, kwargs) (line 16)
    open_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), open_9, *[], **kwargs_10)
    
    # Assigning a type to the variable 'ret' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', open_call_result_11)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
