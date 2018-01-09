
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "max method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (IterableObject) -> DynamicType
7:     # (Str) -> DynamicType
8:     # (IterableObject, Has__call__) -> DynamicType
9:     # (Str, Has__call__) -> DynamicType
10:     # (AnyType, AnyType) -> DynamicType
11:     # (AnyType, AnyType, Has__call__) -> DynamicType
12:     # (AnyType, AnyType, AnyType) -> DynamicType
13:     # (AnyType, AnyType, AnyType, Has__call__) -> DynamicType
14:     # (AnyType, VarArgs) -> DynamicType
15: 
16: 
17:     # Call the builtin with incorrect number of parameters
18:     # Type error
19:     ret = max()
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'max method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 19):
    
    # Call to max(...): (line 19)
    # Processing the call keyword arguments (line 19)
    kwargs_3 = {}
    # Getting the type of 'max' (line 19)
    max_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'max', False)
    # Calling max(args, kwargs) (line 19)
    max_call_result_4 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), max_2, *[], **kwargs_3)
    
    # Assigning a type to the variable 'ret' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ret', max_call_result_4)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
