
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Wrong usages of the exception variable type"
4: 
5: if __name__ == '__main__':
6:     try:
7:         pass
8:     except Exception as e:
9:         # Type error
10:         print e / 3
11:         # Type error
12:         print e.undefined
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Wrong usages of the exception variable type')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    
    # SSA begins for try-except statement (line 6)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    pass
    # SSA branch for the except part of a try statement (line 6)
    # SSA branch for the except 'Exception' branch of a try statement (line 6)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 8)
    Exception_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 11), 'Exception')
    # Assigning a type to the variable 'e' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'e', Exception_2)
    # Getting the type of 'e' (line 10)
    e_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 14), 'e')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 18), 'int')
    # Applying the binary operator 'div' (line 10)
    result_div_5 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 14), 'div', e_3, int_4)
    
    # Getting the type of 'e' (line 12)
    e_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'e')
    # Obtaining the member 'undefined' of a type (line 12)
    undefined_7 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 14), e_6, 'undefined')
    # SSA join for try-except statement (line 6)
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
