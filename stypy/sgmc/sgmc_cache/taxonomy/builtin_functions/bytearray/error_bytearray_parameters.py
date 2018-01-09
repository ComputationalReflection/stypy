
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "bytearray method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'bytearray'>
7:     # (IterableDataStructureWithTypedElements(Integer, Overloads__trunc__)) -> <type 'bytearray'>
8:     # (Integer) -> <type 'bytearray'>
9:     # (Str) -> <type 'bytearray'>
10: 
11: 
12:     # Call the builtin with incorrect number of parameters
13:     # Type error
14:     ret = bytearray(3, 4)
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'bytearray method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 14):
    
    # Call to bytearray(...): (line 14)
    # Processing the call arguments (line 14)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 20), 'int')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 23), 'int')
    # Processing the call keyword arguments (line 14)
    kwargs_5 = {}
    # Getting the type of 'bytearray' (line 14)
    bytearray_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'bytearray', False)
    # Calling bytearray(args, kwargs) (line 14)
    bytearray_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), bytearray_2, *[int_3, int_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', bytearray_call_result_6)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
