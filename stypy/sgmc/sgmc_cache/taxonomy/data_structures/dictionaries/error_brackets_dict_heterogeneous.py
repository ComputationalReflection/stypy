
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Checking the type of the index with heterogeneous keys"
4: 
5: if __name__ == '__main__':
6:     d = {
7:         "one": 1,
8:         2: "two",
9:         "three": 3,
10:     }
11: 
12:     # Type error
13:     print d["3"]
14: 
15:     print d[4]
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Checking the type of the index with heterogeneous keys')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Dict to a Name (line 6):
    
    # Obtaining an instance of the builtin type 'dict' (line 6)
    dict_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 6)
    # Adding element type (key, value) (line 6)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 8), 'str', 'one')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 8), dict_2, (str_3, int_4))
    # Adding element type (key, value) (line 6)
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 8), 'int')
    str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'two')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 8), dict_2, (int_5, str_6))
    # Adding element type (key, value) (line 6)
    str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 8), 'str', 'three')
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 17), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 8), dict_2, (str_7, int_8))
    
    # Assigning a type to the variable 'd' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'd', dict_2)
    
    # Obtaining the type of the subscript
    str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 12), 'str', '3')
    # Getting the type of 'd' (line 13)
    d_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'd')
    # Obtaining the member '__getitem__' of a type (line 13)
    getitem___11 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 10), d_10, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 13)
    subscript_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), getitem___11, str_9)
    
    
    # Obtaining the type of the subscript
    int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'int')
    # Getting the type of 'd' (line 15)
    d_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'd')
    # Obtaining the member '__getitem__' of a type (line 15)
    getitem___15 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 10), d_14, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 15)
    subscript_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), getitem___15, int_13)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
