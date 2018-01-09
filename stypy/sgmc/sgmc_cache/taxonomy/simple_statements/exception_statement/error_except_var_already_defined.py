
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "The exception variable name was already defined into an outer scope"
4: 
5: if __name__ == '__main__':
6:     var = "e"
7: 
8:     try:
9:         pass
10:     except Exception as var:
11:         # Type error
12:         print len(var)
13: 
14: 
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'The exception variable name was already defined into an outer scope')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Str to a Name (line 6):
    str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 10), 'str', 'e')
    # Assigning a type to the variable 'var' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'var', str_2)
    
    
    # SSA begins for try-except statement (line 8)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    pass
    # SSA branch for the except part of a try statement (line 8)
    # SSA branch for the except 'Exception' branch of a try statement (line 8)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 10)
    Exception_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'Exception')
    # Assigning a type to the variable 'var' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'var', Exception_3)
    
    # Call to len(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'var' (line 12)
    var_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 18), 'var', False)
    # Processing the call keyword arguments (line 12)
    kwargs_6 = {}
    # Getting the type of 'len' (line 12)
    len_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'len', False)
    # Calling len(args, kwargs) (line 12)
    len_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 12, 14), len_4, *[var_5], **kwargs_6)
    
    # SSA join for try-except statement (line 8)
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
