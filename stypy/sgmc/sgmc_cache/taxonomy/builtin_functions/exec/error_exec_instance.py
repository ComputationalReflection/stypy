
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "exec builtin is invoked, but a class is used instead of an instance"
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
20:     import types
21: 
22:     # Type error
23:     exec (str)
24: 
25:     # Type error
26:     exec (types.CodeType)
27: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'exec builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 4))
    
    # 'import types' statement (line 20)
    import types

    import_module(stypy.reporting.localization.Localization(__file__, 20, 4), 'types', types, module_type_store)
    
    # Dynamic code evaluation using an exec statement
    # Getting the type of 'str' (line 23)
    str_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'str')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 23, 4), str_2, 'exec parameter', 'StringType', 'FileType', 'CodeType')
    enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 23, 4))
    # Dynamic code evaluation using an exec statement
    # Getting the type of 'types' (line 26)
    types_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 10), 'types')
    # Obtaining the member 'CodeType' of a type (line 26)
    CodeType_4 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 10), types_3, 'CodeType')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 26, 4), CodeType_4, 'exec parameter', 'StringType', 'FileType', 'CodeType')
    enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 26, 4))


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
