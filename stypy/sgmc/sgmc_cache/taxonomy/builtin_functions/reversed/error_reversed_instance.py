
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "reversed builtin is invoked, but a class is used instead of an instance"
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
13:     # Type error
14:     ret = reversed(str)
15:     # Type error
16:     ret = reversed(list)
17:     # Type error
18:     ret = reversed(tuple)
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'reversed builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 14):
    
    # Call to reversed(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'str' (line 14)
    str_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'str', False)
    # Processing the call keyword arguments (line 14)
    kwargs_4 = {}
    # Getting the type of 'reversed' (line 14)
    reversed_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'reversed', False)
    # Calling reversed(args, kwargs) (line 14)
    reversed_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), reversed_2, *[str_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', reversed_call_result_5)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to reversed(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'list' (line 16)
    list_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 19), 'list', False)
    # Processing the call keyword arguments (line 16)
    kwargs_8 = {}
    # Getting the type of 'reversed' (line 16)
    reversed_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'reversed', False)
    # Calling reversed(args, kwargs) (line 16)
    reversed_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), reversed_6, *[list_7], **kwargs_8)
    
    # Assigning a type to the variable 'ret' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', reversed_call_result_9)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to reversed(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'tuple' (line 18)
    tuple_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 19), 'tuple', False)
    # Processing the call keyword arguments (line 18)
    kwargs_12 = {}
    # Getting the type of 'reversed' (line 18)
    reversed_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'reversed', False)
    # Calling reversed(args, kwargs) (line 18)
    reversed_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), reversed_10, *[tuple_11], **kwargs_12)
    
    # Assigning a type to the variable 'ret' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', reversed_call_result_13)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
