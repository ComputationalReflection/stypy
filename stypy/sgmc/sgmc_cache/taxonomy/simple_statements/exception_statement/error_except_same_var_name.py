
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Use the same variable name in multiple exception branches"
4: 
5: if __name__ == '__main__':
6: 
7:     try:
8:         pass
9:     except AttributeError as var:
10:         # Type error
11:         print var / 3
12:         # Type error
13:         print var.undefined
14:     except StandardError as var:
15:         # Type error
16:         print var / 3
17:         # Type error
18:         print var.undefined
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Use the same variable name in multiple exception branches')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    
    # SSA begins for try-except statement (line 7)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    pass
    # SSA branch for the except part of a try statement (line 7)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 7)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'AttributeError' (line 9)
    AttributeError_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'AttributeError')
    # Assigning a type to the variable 'var' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'var', AttributeError_2)
    # Getting the type of 'var' (line 11)
    var_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 14), 'var')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 20), 'int')
    # Applying the binary operator 'div' (line 11)
    result_div_5 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 14), 'div', var_3, int_4)
    
    # Getting the type of 'var' (line 13)
    var_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'var')
    # Obtaining the member 'undefined' of a type (line 13)
    undefined_7 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 14), var_6, 'undefined')
    # SSA branch for the except 'StandardError' branch of a try statement (line 7)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'StandardError' (line 14)
    StandardError_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'StandardError')
    # Assigning a type to the variable 'var' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'var', StandardError_8)
    # Getting the type of 'var' (line 16)
    var_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'var')
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 20), 'int')
    # Applying the binary operator 'div' (line 16)
    result_div_11 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 14), 'div', var_9, int_10)
    
    # Getting the type of 'var' (line 18)
    var_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'var')
    # Obtaining the member 'undefined' of a type (line 18)
    undefined_13 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 14), var_12, 'undefined')
    # SSA join for try-except statement (line 7)
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
