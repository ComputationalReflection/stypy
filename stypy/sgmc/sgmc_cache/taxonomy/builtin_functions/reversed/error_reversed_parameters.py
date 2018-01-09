
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "reversed method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (<type buffer>) -> <type 'reversed'>
7:     # (<type bytearray>) -> <type 'reversed'>
8:     # (Str) -> <type 'reversed'>
9:     # (<type list>) -> ExtraTypeDefinitions.listreverseiterator
10:     # (<type tuple>) -> <type 'reversed'>
11:     # (<type xrange>) -> ExtraTypeDefinitions.rangeiterator
12: 
13: 
14:     # Call the builtin with incorrect number of parameters
15:     # Type error
16:     ret = reversed("str", "str")
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'reversed method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 16):
    
    # Call to reversed(...): (line 16)
    # Processing the call arguments (line 16)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 19), 'str', 'str')
    str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'str', 'str')
    # Processing the call keyword arguments (line 16)
    kwargs_5 = {}
    # Getting the type of 'reversed' (line 16)
    reversed_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'reversed', False)
    # Calling reversed(args, kwargs) (line 16)
    reversed_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), reversed_2, *[str_3, str_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', reversed_call_result_6)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
