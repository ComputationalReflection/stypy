
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: import math
3: 
4: __doc__ = "No execution path has an execution flow free of type errors"
5: 
6: if __name__ == '__main__':
7:     try:
8:         a = 3
9:     except:
10:         a = "3"
11: 
12:     # Type error
13:     r2 = math.fsum(a)  # Not detected
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import math' statement (line 2)
import math

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'math', math, module_type_store)


# Assigning a Str to a Name (line 4):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'str', 'No execution path has an execution flow free of type errors')
# Assigning a type to the variable '__doc__' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    
    # SSA begins for try-except statement (line 7)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Num to a Name (line 8):
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 12), 'int')
    # Assigning a type to the variable 'a' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'a', int_2)
    # SSA branch for the except part of a try statement (line 7)
    # SSA branch for the except '<any exception>' branch of a try statement (line 7)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Str to a Name (line 10):
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 12), 'str', '3')
    # Assigning a type to the variable 'a' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'a', str_3)
    # SSA join for try-except statement (line 7)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 13):
    
    # Call to fsum(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'a' (line 13)
    a_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 19), 'a', False)
    # Processing the call keyword arguments (line 13)
    kwargs_7 = {}
    # Getting the type of 'math' (line 13)
    math_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 9), 'math', False)
    # Obtaining the member 'fsum' of a type (line 13)
    fsum_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 9), math_4, 'fsum')
    # Calling fsum(args, kwargs) (line 13)
    fsum_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 13, 9), fsum_5, *[a_6], **kwargs_7)
    
    # Assigning a type to the variable 'r2' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'r2', fsum_call_result_8)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
