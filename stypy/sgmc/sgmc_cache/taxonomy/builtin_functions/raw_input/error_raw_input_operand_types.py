
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "raw_input builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> DynamicType
7:     # (AnyType) -> DynamicType
8: 
9: 
10:     # Call the builtin with correct parameters
11:     # Type warning
12:     ret = raw_input()
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'raw_input builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to raw_input(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_3 = {}
    # Getting the type of 'raw_input' (line 12)
    raw_input_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'raw_input', False)
    # Calling raw_input(args, kwargs) (line 12)
    raw_input_call_result_4 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), raw_input_2, *[], **kwargs_3)
    
    # Assigning a type to the variable 'ret' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'ret', raw_input_call_result_4)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
