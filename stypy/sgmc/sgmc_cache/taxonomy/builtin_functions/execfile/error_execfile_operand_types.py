
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "execfile builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Str) -> DynamicType
7:     # (Str, <type list>) -> DynamicType
8:     # (Str, <type list>, <type dict>) -> DynamicType
9: 
10: 
11:     # Call the builtin with correct parameters
12:     # Type warning
13:     ret = execfile("f.py", dict(), dict())
14: 
15:     # Call the builtin with incorrect types of parameters
16:     # Type error
17:     ret = execfile(4)
18: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'execfile builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to execfile(...): (line 13)
    # Processing the call arguments (line 13)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 19), 'str', 'f.py')
    
    # Call to dict(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_5 = {}
    # Getting the type of 'dict' (line 13)
    dict_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 27), 'dict', False)
    # Calling dict(args, kwargs) (line 13)
    dict_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 13, 27), dict_4, *[], **kwargs_5)
    
    
    # Call to dict(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_8 = {}
    # Getting the type of 'dict' (line 13)
    dict_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 35), 'dict', False)
    # Calling dict(args, kwargs) (line 13)
    dict_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 13, 35), dict_7, *[], **kwargs_8)
    
    # Processing the call keyword arguments (line 13)
    kwargs_10 = {}
    # Getting the type of 'execfile' (line 13)
    execfile_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'execfile', False)
    # Calling execfile(args, kwargs) (line 13)
    execfile_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), execfile_2, *[str_3, dict_call_result_6, dict_call_result_9], **kwargs_10)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', execfile_call_result_11)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to execfile(...): (line 17)
    # Processing the call arguments (line 17)
    int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 19), 'int')
    # Processing the call keyword arguments (line 17)
    kwargs_14 = {}
    # Getting the type of 'execfile' (line 17)
    execfile_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'execfile', False)
    # Calling execfile(args, kwargs) (line 17)
    execfile_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), execfile_12, *[int_13], **kwargs_14)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', execfile_call_result_15)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
