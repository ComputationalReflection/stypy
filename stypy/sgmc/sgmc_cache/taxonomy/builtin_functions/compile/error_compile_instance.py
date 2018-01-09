
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "compile builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Str, Str, Str) -> types.CodeType
7:     # (Str, Str, Str, Integer) -> types.CodeType
8:     # (Str, Str, Str, Integer, Integer) -> types.CodeType
9: 
10: 
11:     # Type error
12:     ret = compile(str, str, str)
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'compile builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to compile(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'str' (line 12)
    str_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 18), 'str', False)
    # Getting the type of 'str' (line 12)
    str_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 23), 'str', False)
    # Getting the type of 'str' (line 12)
    str_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 28), 'str', False)
    # Processing the call keyword arguments (line 12)
    kwargs_6 = {}
    # Getting the type of 'compile' (line 12)
    compile_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'compile', False)
    # Calling compile(args, kwargs) (line 12)
    compile_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), compile_2, *[str_3, str_4, str_5], **kwargs_6)
    
    # Assigning a type to the variable 'ret' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'ret', compile_call_result_7)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
