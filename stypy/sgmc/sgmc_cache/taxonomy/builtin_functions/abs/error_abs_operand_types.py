
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "abs builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (<type bool>) -> <type 'int'>
7:     # (<type complex>) -> <type 'float'>
8:     # (Number) -> TypeOfParam(1)
9:     # (Overloads__abs__) -> <type 'int'>
10: 
11: 
12: 
13:     # Call the builtin with correct parameters
14:     # No error
15:     ret = abs(3)
16:     # No error
17:     ret = abs(3 + 2j)
18: 
19:     # Call the builtin with incorrect parameters
20: 
21:     # Type error
22:     ret = abs(list())
23: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'abs builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 15):
    
    # Call to abs(...): (line 15)
    # Processing the call arguments (line 15)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 14), 'int')
    # Processing the call keyword arguments (line 15)
    kwargs_4 = {}
    # Getting the type of 'abs' (line 15)
    abs_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'abs', False)
    # Calling abs(args, kwargs) (line 15)
    abs_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), abs_2, *[int_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'ret', abs_call_result_5)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to abs(...): (line 17)
    # Processing the call arguments (line 17)
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'int')
    complex_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 18), 'complex')
    # Applying the binary operator '+' (line 17)
    result_add_9 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 14), '+', int_7, complex_8)
    
    # Processing the call keyword arguments (line 17)
    kwargs_10 = {}
    # Getting the type of 'abs' (line 17)
    abs_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'abs', False)
    # Calling abs(args, kwargs) (line 17)
    abs_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), abs_6, *[result_add_9], **kwargs_10)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', abs_call_result_11)
    
    # Assigning a Call to a Name (line 22):
    
    # Call to abs(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Call to list(...): (line 22)
    # Processing the call keyword arguments (line 22)
    kwargs_14 = {}
    # Getting the type of 'list' (line 22)
    list_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 14), 'list', False)
    # Calling list(args, kwargs) (line 22)
    list_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 22, 14), list_13, *[], **kwargs_14)
    
    # Processing the call keyword arguments (line 22)
    kwargs_16 = {}
    # Getting the type of 'abs' (line 22)
    abs_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 10), 'abs', False)
    # Calling abs(args, kwargs) (line 22)
    abs_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 22, 10), abs_12, *[list_call_result_15], **kwargs_16)
    
    # Assigning a type to the variable 'ret' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'ret', abs_call_result_17)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
