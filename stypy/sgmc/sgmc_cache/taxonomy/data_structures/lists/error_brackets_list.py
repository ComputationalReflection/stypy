
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Checking the type of the index"
4: 
5: if __name__ == '__main__':
6:     it_list = range(5)
7: 
8:     print it_list[3]
9: 
10:     # Type error
11:     print it_list["3"]
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Checking the type of the index')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 6):
    
    # Call to range(...): (line 6)
    # Processing the call arguments (line 6)
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 20), 'int')
    # Processing the call keyword arguments (line 6)
    kwargs_4 = {}
    # Getting the type of 'range' (line 6)
    range_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 14), 'range', False)
    # Calling range(args, kwargs) (line 6)
    range_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 6, 14), range_2, *[int_3], **kwargs_4)
    
    # Assigning a type to the variable 'it_list' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'it_list', range_call_result_5)
    
    # Obtaining the type of the subscript
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 18), 'int')
    # Getting the type of 'it_list' (line 8)
    it_list_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 10), 'it_list')
    # Obtaining the member '__getitem__' of a type (line 8)
    getitem___8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 10), it_list_7, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 8)
    subscript_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 8, 10), getitem___8, int_6)
    
    
    # Obtaining the type of the subscript
    str_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'str', '3')
    # Getting the type of 'it_list' (line 11)
    it_list_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'it_list')
    # Obtaining the member '__getitem__' of a type (line 11)
    getitem___12 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 10), it_list_11, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 11)
    subscript_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), getitem___12, str_10)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
