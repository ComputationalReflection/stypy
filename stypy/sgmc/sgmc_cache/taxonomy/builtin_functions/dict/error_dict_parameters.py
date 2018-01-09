
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "dict method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'dict'>
7:     # (<type dict>) -> <type 'dict'>
8:     # (IterableObject) -> <type 'dict'>
9: 
10: 
11:     # Call the builtin with incorrect number of parameters
12:     # Type error
13:     ret = dict([(1, 2), (3, 4)], 5)
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'dict method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to dict(...): (line 13)
    # Processing the call arguments (line 13)
    
    # Obtaining an instance of the builtin type 'list' (line 13)
    list_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 13)
    # Adding element type (line 13)
    
    # Obtaining an instance of the builtin type 'tuple' (line 13)
    tuple_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 13)
    # Adding element type (line 13)
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 17), tuple_4, int_5)
    # Adding element type (line 13)
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 17), tuple_4, int_6)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 15), list_3, tuple_4)
    # Adding element type (line 13)
    
    # Obtaining an instance of the builtin type 'tuple' (line 13)
    tuple_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 13)
    # Adding element type (line 13)
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 25), tuple_7, int_8)
    # Adding element type (line 13)
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 25), tuple_7, int_9)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 15), list_3, tuple_7)
    
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 33), 'int')
    # Processing the call keyword arguments (line 13)
    kwargs_11 = {}
    # Getting the type of 'dict' (line 13)
    dict_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'dict', False)
    # Calling dict(args, kwargs) (line 13)
    dict_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), dict_2, *[list_3, int_10], **kwargs_11)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', dict_call_result_12)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
