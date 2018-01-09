
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "sum method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (IterableDataStructureWithTypedElements(Integer)) -> DynamicType
7:     # (IterableDataStructureWithTypedElements(Integer), Integer) -> DynamicType
8: 
9: 
10:     # Call the builtin with incorrect number of parameters
11:     # Type error
12:     ret = sum([1, 2], [1, 2], [1, 2])
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'sum method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to sum(...): (line 12)
    # Processing the call arguments (line 12)
    
    # Obtaining an instance of the builtin type 'list' (line 12)
    list_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 12)
    # Adding element type (line 12)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), list_3, int_4)
    # Adding element type (line 12)
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), list_3, int_5)
    
    
    # Obtaining an instance of the builtin type 'list' (line 12)
    list_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 12)
    # Adding element type (line 12)
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 22), list_6, int_7)
    # Adding element type (line 12)
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 22), list_6, int_8)
    
    
    # Obtaining an instance of the builtin type 'list' (line 12)
    list_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 12)
    # Adding element type (line 12)
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 30), list_9, int_10)
    # Adding element type (line 12)
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 30), list_9, int_11)
    
    # Processing the call keyword arguments (line 12)
    kwargs_12 = {}
    # Getting the type of 'sum' (line 12)
    sum_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'sum', False)
    # Calling sum(args, kwargs) (line 12)
    sum_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), sum_2, *[list_3, list_6, list_9], **kwargs_12)
    
    # Assigning a type to the variable 'ret' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'ret', sum_call_result_13)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
