
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "copyright builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'str'>
7: 
8: 
9:     # Call the builtin with correct parameters
10:     # No error
11:     ret = copyright()
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'copyright builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 11):
    
    # Call to copyright(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_3 = {}
    # Getting the type of 'copyright' (line 11)
    copyright_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'copyright', False)
    # Calling copyright(args, kwargs) (line 11)
    copyright_call_result_4 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), copyright_2, *[], **kwargs_3)
    
    # Assigning a type to the variable 'ret' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'ret', copyright_call_result_4)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
