
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "pow builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (<type int>, Number) -> TypeOfParam(2)
7:     # (<type bool>, <type bool>) -> <type 'int'>
8:     # (<type bool>, Number) -> TypeOfParam(2)
9:     # (<type complex>, Number) -> <type 'complex'>
10:     # (<type long>, Integer) -> <type 'long'>
11:     # (<type long>, <type complex>) -> <type 'complex'>
12:     # (<type long>, <type float>) -> <type 'float'>
13:     # (<type int>, <type bool>) -> <type 'int'>
14:     # (<type float>, RealNumber) -> <type 'float'>
15:     # (<type float>, <type complex>) -> <type 'complex'>
16:     # (<type bool>, <type bool>, <type bool>) -> <type 'int'>
17:     # (<type bool>, <type bool>, types.NoneType) -> <type 'int'>
18:     # (<type bool>, <type bool>, Integer) -> TypeOfParam(2)
19:     # (<type bool>, <type complex>, types.NoneType) -> <type 'complex'>
20:     # (<type bool>, <type long>, <type bool>) -> <type 'long'>
21:     # (<type bool>, <type long>, types.NoneType) -> <type 'long'>
22:     # (<type bool>, <type long>, Integer) -> TypeOfParam(1)
23:     # (<type bool>, <type int>, <type bool>) -> <type 'int'>
24:     # (<type bool>, <type int>, types.NoneType) -> <type 'int'>
25:     # (<type bool>, <type int>, Integer) -> TypeOfParam(2)
26:     # (<type bool>, <type float>, types.NoneType) -> <type 'float'>
27:     # (<type complex>, Number, types.NoneType) -> <type 'complex'>
28:     # (<type long>, <type bool>, Integer) -> <type 'long'>
29:     # (<type long>, <type bool>, types.NoneType) -> <type 'long'>
30:     # (<type long>, <type complex>, types.NoneType) -> <type 'complex'>
31:     # (<type long>, <type long>, Integer) -> <type 'long'>
32:     # (<type long>, <type long>, types.NoneType) -> <type 'long'>
33:     # (<type long>, <type int>, Integer) -> <type 'long'>
34:     # (<type long>, <type int>, types.NoneType) -> <type 'long'>
35:     # (<type long>, <type float>, types.NoneType) -> <type 'float'>
36:     # (<type int>, <type bool>, <type bool>) -> <type 'int'>
37:     # (<type int>, <type bool>, types.NoneType) -> <type 'int'>
38:     # (<type int>, <type bool>, Integer) -> TypeOfParam(3)
39:     # (<type int>, <type complex>, types.NoneType) -> <type 'complex'>
40:     # (<type int>, <type long>, Integer) -> <type 'long'>
41:     # (<type int>, <type long>, types.NoneType) -> <type 'long'>
42:     # (<type int>, <type int>, <type bool>) -> <type 'int'>
43:     # (<type int>, <type int>, types.NoneType) -> <type 'int'>
44:     # (<type int>, <type int>, Integer) -> TypeOfParam(3)
45:     # (<type int>, RealNumber, types.NoneType) -> <type 'float'>
46:     # (<type float>, <type complex>, types.NoneType) -> <type 'complex'>
47:     # (Overloads__pow__, AnyType, AnyType) -> DynamicType
48:     # (Overloads__pow__, AnyType, AnyType, AnyType) -> DynamicType
49: 
50: 
51:     # Call the builtin
52:     ret = pow(3, 4)
53: 
54:     # Type error
55:     ret.unexisting_method()
56: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'pow builtin is invoked and its return type is used to call an non existing method')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 52):
    
    # Call to pow(...): (line 52)
    # Processing the call arguments (line 52)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 14), 'int')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 17), 'int')
    # Processing the call keyword arguments (line 52)
    kwargs_5 = {}
    # Getting the type of 'pow' (line 52)
    pow_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 10), 'pow', False)
    # Calling pow(args, kwargs) (line 52)
    pow_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 52, 10), pow_2, *[int_3, int_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'ret', pow_call_result_6)
    
    # Call to unexisting_method(...): (line 55)
    # Processing the call keyword arguments (line 55)
    kwargs_9 = {}
    # Getting the type of 'ret' (line 55)
    ret_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'ret', False)
    # Obtaining the member 'unexisting_method' of a type (line 55)
    unexisting_method_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 4), ret_7, 'unexisting_method')
    # Calling unexisting_method(args, kwargs) (line 55)
    unexisting_method_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 55, 4), unexisting_method_8, *[], **kwargs_9)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
