
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "super method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Type, AnyType) -> <type 'super'>
7:     # (Type, types.NoneType) -> <type 'super'>
8:     # (Type) -> <type 'super'>
9: 
10: 
11:     # Call the builtin with incorrect number of parameters
12:     # Type error
13:     ret = super(list, list, None)
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'super method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to super(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'list' (line 13)
    list_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'list', False)
    # Getting the type of 'list' (line 13)
    list_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 22), 'list', False)
    # Getting the type of 'None' (line 13)
    None_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 28), 'None', False)
    # Processing the call keyword arguments (line 13)
    kwargs_6 = {}
    # Getting the type of 'super' (line 13)
    super_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'super', False)
    # Calling super(args, kwargs) (line 13)
    super_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), super_2, *[list_3, list_4, None_5], **kwargs_6)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', super_call_result_7)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
