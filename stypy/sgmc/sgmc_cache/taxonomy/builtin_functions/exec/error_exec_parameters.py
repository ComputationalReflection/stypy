
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "exec method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (types.CodeType) -> DynamicType
7:     # (types.CodeType, types.NoneType) -> DynamicType
8:     # (types.CodeType, <type dict>) -> DynamicType
9:     # (types.CodeType, <type dict>, types.NoneType) -> DynamicType
10:     # (Str) -> DynamicType
11:     # (Str, <type dict>) -> DynamicType
12:     # (Str, <type dict>, types.NoneType) -> DynamicType
13:     # (Str, <type dict>, <type dict>) -> DynamicType
14:     # (Str, types.NoneType, <type dict>) -> DynamicType
15:     # (<type file>) -> DynamicType
16:     # (<type file>, <type dict>) -> DynamicType
17:     # (<type file>, <type dict>, types.NoneType) -> DynamicType
18:     # (<type file>, <type dict>, <type dict>) -> DynamicType
19:     # (<type file>, types.NoneType, <type dict>) -> DynamicType
20: 
21: 
22:     # Call the builtin with incorrect number of parameters
23:     # Type error
24:     exec ("a=5", dict(), None, None)
25: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'exec method is present, but is invoked with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Dynamic code evaluation using an exec statement
    
    # Obtaining an instance of the builtin type 'tuple' (line 24)
    tuple_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 10), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 24)
    # Adding element type (line 24)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 10), 'str', 'a=5')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), tuple_2, str_3)
    # Adding element type (line 24)
    
    # Call to dict(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_5 = {}
    # Getting the type of 'dict' (line 24)
    dict_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 17), 'dict', False)
    # Calling dict(args, kwargs) (line 24)
    dict_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 24, 17), dict_4, *[], **kwargs_5)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), tuple_2, dict_call_result_6)
    # Adding element type (line 24)
    # Getting the type of 'None' (line 24)
    None_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 25), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), tuple_2, None_7)
    # Adding element type (line 24)
    # Getting the type of 'None' (line 24)
    None_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 31), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), tuple_2, None_8)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 24, 4), tuple_2, 'exec parameter', 'StringType', 'FileType', 'CodeType')
    enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 24, 4))


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
