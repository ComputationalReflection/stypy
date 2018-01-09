
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "locals builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'dict'>
7: 
8: 
9:     # Call the builtin with correct parameters
10:     ret = locals()
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'locals builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 10):
    
    # Call to locals(...): (line 10)
    # Processing the call keyword arguments (line 10)
    kwargs_3 = {}
    # Getting the type of 'locals' (line 10)
    locals_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 10), 'locals', False)
    # Calling locals(args, kwargs) (line 10)
    locals_call_result_4 = invoke(stypy.reporting.localization.Localization(__file__, 10, 10), locals_2, *[], **kwargs_3)
    
    # Assigning a type to the variable 'ret' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'ret', locals_call_result_4)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
