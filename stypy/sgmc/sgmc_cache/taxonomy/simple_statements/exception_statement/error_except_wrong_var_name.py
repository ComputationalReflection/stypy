
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Use a variable name from a previous exception branch"
4: 
5: if __name__ == '__main__':
6:     try:
7:         pass
8:     except AttributeError as e:
9:         pass
10:     except StandardError as var:
11:         # Type error
12:         print e / 3
13:         # Type error
14:         print e.undefined
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Use a variable name from a previous exception branch')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    
    # SSA begins for try-except statement (line 6)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    pass
    # SSA branch for the except part of a try statement (line 6)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 6)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'AttributeError' (line 8)
    AttributeError_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 11), 'AttributeError')
    # Assigning a type to the variable 'e' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'e', AttributeError_2)
    pass
    # SSA branch for the except 'StandardError' branch of a try statement (line 6)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'StandardError' (line 10)
    StandardError_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'StandardError')
    # Assigning a type to the variable 'var' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'var', StandardError_3)
    # Getting the type of 'e' (line 12)
    e_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'e')
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'int')
    # Applying the binary operator 'div' (line 12)
    result_div_6 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 14), 'div', e_4, int_5)
    
    # Getting the type of 'e' (line 14)
    e_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'e')
    # Obtaining the member 'undefined' of a type (line 14)
    undefined_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 14), e_7, 'undefined')
    # SSA join for try-except statement (line 6)
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
