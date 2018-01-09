
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "sum builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (IterableDataStructureWithTypedElements(Integer)) -> DynamicType
7:     # (IterableDataStructureWithTypedElements(Integer), Integer) -> DynamicType
8: 
9: 
10:     # Type error
11:     ret = sum(list)
12:     # Type error
13:     ret = sum(list, int)
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'sum builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 11):
    
    # Call to sum(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'list' (line 11)
    list_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 14), 'list', False)
    # Processing the call keyword arguments (line 11)
    kwargs_4 = {}
    # Getting the type of 'sum' (line 11)
    sum_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'sum', False)
    # Calling sum(args, kwargs) (line 11)
    sum_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), sum_2, *[list_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'ret', sum_call_result_5)
    
    # Assigning a Call to a Name (line 13):
    
    # Call to sum(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'list' (line 13)
    list_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'list', False)
    # Getting the type of 'int' (line 13)
    int_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 20), 'int', False)
    # Processing the call keyword arguments (line 13)
    kwargs_9 = {}
    # Getting the type of 'sum' (line 13)
    sum_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'sum', False)
    # Calling sum(args, kwargs) (line 13)
    sum_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), sum_6, *[list_7, int_8], **kwargs_9)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', sum_call_result_10)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
