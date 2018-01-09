
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "super builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Type, AnyType) -> <type 'super'>
7:     # (Type, types.NoneType) -> <type 'super'>
8:     # (Type) -> <type 'super'>
9: 
10: 
11:     # Call the builtin with correct parameters
12:     ret = super(list)
13:     ret = super(list, None)
14:     ret = super(list, list())
15: 
16:     # Call the builtin with incorrect types of parameters
17:     # Type error
18:     ret = super(3)
19:     # Type error
20:     ret = super(list, object)
21: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'super builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to super(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'list' (line 12)
    list_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 16), 'list', False)
    # Processing the call keyword arguments (line 12)
    kwargs_4 = {}
    # Getting the type of 'super' (line 12)
    super_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'super', False)
    # Calling super(args, kwargs) (line 12)
    super_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), super_2, *[list_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'ret', super_call_result_5)
    
    # Assigning a Call to a Name (line 13):
    
    # Call to super(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'list' (line 13)
    list_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'list', False)
    # Getting the type of 'None' (line 13)
    None_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 22), 'None', False)
    # Processing the call keyword arguments (line 13)
    kwargs_9 = {}
    # Getting the type of 'super' (line 13)
    super_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'super', False)
    # Calling super(args, kwargs) (line 13)
    super_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), super_6, *[list_7, None_8], **kwargs_9)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', super_call_result_10)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to super(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'list' (line 14)
    list_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'list', False)
    
    # Call to list(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_14 = {}
    # Getting the type of 'list' (line 14)
    list_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 22), 'list', False)
    # Calling list(args, kwargs) (line 14)
    list_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 14, 22), list_13, *[], **kwargs_14)
    
    # Processing the call keyword arguments (line 14)
    kwargs_16 = {}
    # Getting the type of 'super' (line 14)
    super_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'super', False)
    # Calling super(args, kwargs) (line 14)
    super_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), super_11, *[list_12, list_call_result_15], **kwargs_16)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', super_call_result_17)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to super(...): (line 18)
    # Processing the call arguments (line 18)
    int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 16), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_20 = {}
    # Getting the type of 'super' (line 18)
    super_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'super', False)
    # Calling super(args, kwargs) (line 18)
    super_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), super_18, *[int_19], **kwargs_20)
    
    # Assigning a type to the variable 'ret' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', super_call_result_21)
    
    # Assigning a Call to a Name (line 20):
    
    # Call to super(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'list' (line 20)
    list_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'list', False)
    # Getting the type of 'object' (line 20)
    object_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'object', False)
    # Processing the call keyword arguments (line 20)
    kwargs_25 = {}
    # Getting the type of 'super' (line 20)
    super_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'super', False)
    # Calling super(args, kwargs) (line 20)
    super_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 20, 10), super_22, *[list_23, object_24], **kwargs_25)
    
    # Assigning a type to the variable 'ret' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'ret', super_call_result_26)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
