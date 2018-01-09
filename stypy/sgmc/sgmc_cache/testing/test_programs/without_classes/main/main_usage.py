
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: if __name__ == '__main__':
3:     a = 3
4:     n = 4.5
5:     b = "str"
6: 
7:     c = a + a + n
8: 
9:     z = a
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


if (__name__ == '__main__'):
    
    # Assigning a Num to a Name (line 3):
    int_5631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 8), 'int')
    # Assigning a type to the variable 'a' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'a', int_5631)
    
    # Assigning a Num to a Name (line 4):
    float_5632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 8), 'float')
    # Assigning a type to the variable 'n' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'n', float_5632)
    
    # Assigning a Str to a Name (line 5):
    str_5633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 8), 'str', 'str')
    # Assigning a type to the variable 'b' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'b', str_5633)
    
    # Assigning a BinOp to a Name (line 7):
    # Getting the type of 'a' (line 7)
    a_5634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'a')
    # Getting the type of 'a' (line 7)
    a_5635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'a')
    # Applying the binary operator '+' (line 7)
    result_add_5636 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 8), '+', a_5634, a_5635)
    
    # Getting the type of 'n' (line 7)
    n_5637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 16), 'n')
    # Applying the binary operator '+' (line 7)
    result_add_5638 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 14), '+', result_add_5636, n_5637)
    
    # Assigning a type to the variable 'c' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'c', result_add_5638)
    
    # Assigning a Name to a Name (line 9):
    # Getting the type of 'a' (line 9)
    a_5639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'a')
    # Assigning a type to the variable 'z' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'z', a_5639)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
