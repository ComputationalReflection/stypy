
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "reversed builtin is invoked, but incorrect parameter types are passed"
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
14:     # Call the builtin with correct parameters
15:     ret = reversed("str")
16:     ret = reversed([1, 2])
17:     ret = reversed((1, 2))
18: 
19:     # Call the builtin with incorrect types of parameters
20:     # Type error
21:     ret = reversed(3)
22:     # Type error
23:     ret = reversed()
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'reversed builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 15):
    
    # Call to reversed(...): (line 15)
    # Processing the call arguments (line 15)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 19), 'str', 'str')
    # Processing the call keyword arguments (line 15)
    kwargs_4 = {}
    # Getting the type of 'reversed' (line 15)
    reversed_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'reversed', False)
    # Calling reversed(args, kwargs) (line 15)
    reversed_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), reversed_2, *[str_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'ret', reversed_call_result_5)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to reversed(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 19), list_7, int_8)
    # Adding element type (line 16)
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 19), list_7, int_9)
    
    # Processing the call keyword arguments (line 16)
    kwargs_10 = {}
    # Getting the type of 'reversed' (line 16)
    reversed_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'reversed', False)
    # Calling reversed(args, kwargs) (line 16)
    reversed_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), reversed_6, *[list_7], **kwargs_10)
    
    # Assigning a type to the variable 'ret' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', reversed_call_result_11)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to reversed(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Obtaining an instance of the builtin type 'tuple' (line 17)
    tuple_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 17)
    # Adding element type (line 17)
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 20), tuple_13, int_14)
    # Adding element type (line 17)
    int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 20), tuple_13, int_15)
    
    # Processing the call keyword arguments (line 17)
    kwargs_16 = {}
    # Getting the type of 'reversed' (line 17)
    reversed_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'reversed', False)
    # Calling reversed(args, kwargs) (line 17)
    reversed_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), reversed_12, *[tuple_13], **kwargs_16)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', reversed_call_result_17)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to reversed(...): (line 21)
    # Processing the call arguments (line 21)
    int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 19), 'int')
    # Processing the call keyword arguments (line 21)
    kwargs_20 = {}
    # Getting the type of 'reversed' (line 21)
    reversed_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'reversed', False)
    # Calling reversed(args, kwargs) (line 21)
    reversed_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), reversed_18, *[int_19], **kwargs_20)
    
    # Assigning a type to the variable 'ret' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'ret', reversed_call_result_21)
    
    # Assigning a Call to a Name (line 23):
    
    # Call to reversed(...): (line 23)
    # Processing the call keyword arguments (line 23)
    kwargs_23 = {}
    # Getting the type of 'reversed' (line 23)
    reversed_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'reversed', False)
    # Calling reversed(args, kwargs) (line 23)
    reversed_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), reversed_22, *[], **kwargs_23)
    
    # Assigning a type to the variable 'ret' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'ret', reversed_call_result_24)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
