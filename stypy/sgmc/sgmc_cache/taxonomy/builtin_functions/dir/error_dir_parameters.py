
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "dir method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'list'>
7:     # (AnyType) -> <type 'list'>
8: 
9: 
10:     # Call the builtin with incorrect number of parameters
11:     # Type error
12:     ret = dir(int, float)
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'dir method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to dir(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'int' (line 12)
    int_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'int', False)
    # Getting the type of 'float' (line 12)
    float_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'float', False)
    # Processing the call keyword arguments (line 12)
    kwargs_5 = {}
    # Getting the type of 'dir' (line 12)
    dir_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'dir', False)
    # Calling dir(args, kwargs) (line 12)
    dir_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), dir_2, *[int_3, float_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'ret', dir_call_result_6)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
