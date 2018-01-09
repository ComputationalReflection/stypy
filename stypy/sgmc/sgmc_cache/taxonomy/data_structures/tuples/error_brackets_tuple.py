
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Checking the type of the index"
4: 
5: if __name__ == '__main__':
6:     it_tuple = tuple(range(5))
7: 
8:     print it_tuple[3]
9:     # Type error
10:     print it_tuple["3"]
11: 

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
    
    # Call to tuple(...): (line 6)
    # Processing the call arguments (line 6)
    
    # Call to range(...): (line 6)
    # Processing the call arguments (line 6)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 27), 'int')
    # Processing the call keyword arguments (line 6)
    kwargs_5 = {}
    # Getting the type of 'range' (line 6)
    range_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 21), 'range', False)
    # Calling range(args, kwargs) (line 6)
    range_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 6, 21), range_3, *[int_4], **kwargs_5)
    
    # Processing the call keyword arguments (line 6)
    kwargs_7 = {}
    # Getting the type of 'tuple' (line 6)
    tuple_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 6)
    tuple_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 6, 15), tuple_2, *[range_call_result_6], **kwargs_7)
    
    # Assigning a type to the variable 'it_tuple' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'it_tuple', tuple_call_result_8)
    
    # Obtaining the type of the subscript
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 19), 'int')
    # Getting the type of 'it_tuple' (line 8)
    it_tuple_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 10), 'it_tuple')
    # Obtaining the member '__getitem__' of a type (line 8)
    getitem___11 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 10), it_tuple_10, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 8)
    subscript_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 8, 10), getitem___11, int_9)
    
    
    # Obtaining the type of the subscript
    str_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 19), 'str', '3')
    # Getting the type of 'it_tuple' (line 10)
    it_tuple_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 10), 'it_tuple')
    # Obtaining the member '__getitem__' of a type (line 10)
    getitem___15 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 10), it_tuple_14, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 10)
    subscript_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 10, 10), getitem___15, str_13)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
