
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "exec builtin is invoked, but incorrect parameter types are passed"
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
22:     # Call the builtin with correct parameters
23:     # Type warning
24:     exec ("a=5")
25: 
26:     # Call the builtin with incorrect types of parameters
27: 
28:     # Type error
29:     exec (4)
30: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'exec builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Dynamic code evaluation using an exec statement
    str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 10), 'str', 'a=5')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 24, 4), str_2, 'exec parameter', 'StringType', 'FileType', 'CodeType')
    enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 24, 4))
    # Dynamic code evaluation using an exec statement
    int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 10), 'int')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 29, 4), int_3, 'exec parameter', 'StringType', 'FileType', 'CodeType')
    enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 29, 4))


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
