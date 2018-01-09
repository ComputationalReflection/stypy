
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "isinstance builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (AnyType, AnyType) -> <type 'bool'>
7: 
8: 
9:     # Call the builtin with correct parameters
10:     # No error
11:     ret = isinstance(3, int)
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'isinstance builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 11):
    
    # Call to isinstance(...): (line 11)
    # Processing the call arguments (line 11)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 21), 'int')
    # Getting the type of 'int' (line 11)
    int_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 24), 'int', False)
    # Processing the call keyword arguments (line 11)
    kwargs_5 = {}
    # Getting the type of 'isinstance' (line 11)
    isinstance_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 11)
    isinstance_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), isinstance_2, *[int_3, int_4], **kwargs_5)
    
    # Assigning a type to the variable 'ret' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'ret', isinstance_call_result_6)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
