
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: __all__ = ['python_lib_copy', 'type_store_copy', 'errors_copy', 'code_generation_copy', 'log_copy', 'reporting_copy',
2:            'ssa_copy', 'type_store_copy', 'visitor_copy', 'python_interface_copy', 'stypy_main_copy',
3:            'stypy_parameters_copy']
4: 
5: from python_lib_copy.python_types_copy.type_inference_copy.localization_copy import Localization
6: from type_store_copy.typestore_copy import TypeStore
7: from errors_copy.type_error_copy import TypeError
8: from errors_copy.type_warning_copy import TypeWarning
9: from python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy import norecursion
10: from code_generation_copy.type_inference_programs_copy.aux_functions_copy import *
11: from python_lib_copy.python_types_copy.type_inference_copy import union_type_copy
12: from ssa_copy.ssa_copy import *
13: 
14: import stypy_parameters_copy
15: from python_interface_copy import *
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a List to a Name (line 1):
__all__ = ['python_lib_copy', 'type_store_copy', 'errors_copy', 'code_generation_copy', 'log_copy', 'reporting_copy', 'ssa_copy', 'type_store_copy', 'visitor_copy', 'python_interface_copy', 'stypy_main_copy', 'stypy_parameters_copy']
module_type_store.set_exportable_members(['python_lib_copy', 'type_store_copy', 'errors_copy', 'code_generation_copy', 'log_copy', 'reporting_copy', 'ssa_copy', 'type_store_copy', 'visitor_copy', 'python_interface_copy', 'stypy_main_copy', 'stypy_parameters_copy'])

# Obtaining an instance of the builtin type 'list' (line 1)
list_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 1)
# Adding element type (line 1)
str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 11), 'str', 'python_lib_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1, str_2)
# Adding element type (line 1)
str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 30), 'str', 'type_store_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1, str_3)
# Adding element type (line 1)
str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 49), 'str', 'errors_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1, str_4)
# Adding element type (line 1)
str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 64), 'str', 'code_generation_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1, str_5)
# Adding element type (line 1)
str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 88), 'str', 'log_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1, str_6)
# Adding element type (line 1)
str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 100), 'str', 'reporting_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1, str_7)
# Adding element type (line 1)
str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'str', 'ssa_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1, str_8)
# Adding element type (line 1)
str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 23), 'str', 'type_store_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1, str_9)
# Adding element type (line 1)
str_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 42), 'str', 'visitor_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1, str_10)
# Adding element type (line 1)
str_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 58), 'str', 'python_interface_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1, str_11)
# Adding element type (line 1)
str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 83), 'str', 'stypy_main_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1, str_12)
# Adding element type (line 1)
str_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'str', 'stypy_parameters_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1, str_13)

# Assigning a type to the variable '__all__' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), '__all__', list_1)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from python_lib_copy.python_types_copy.type_inference_copy.localization_copy import Localization' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_14 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'python_lib_copy.python_types_copy.type_inference_copy.localization_copy')

if (type(import_14) is not StypyTypeError):

    if (import_14 != 'pyd_module'):
        __import__(import_14)
        sys_modules_15 = sys.modules[import_14]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'python_lib_copy.python_types_copy.type_inference_copy.localization_copy', sys_modules_15.module_type_store, module_type_store, ['Localization'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_15, sys_modules_15.module_type_store, module_type_store)
    else:
        from python_lib_copy.python_types_copy.type_inference_copy.localization_copy import Localization

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'python_lib_copy.python_types_copy.type_inference_copy.localization_copy', None, module_type_store, ['Localization'], [Localization])

else:
    # Assigning a type to the variable 'python_lib_copy.python_types_copy.type_inference_copy.localization_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'python_lib_copy.python_types_copy.type_inference_copy.localization_copy', import_14)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from type_store_copy.typestore_copy import TypeStore' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_16 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'type_store_copy.typestore_copy')

if (type(import_16) is not StypyTypeError):

    if (import_16 != 'pyd_module'):
        __import__(import_16)
        sys_modules_17 = sys.modules[import_16]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'type_store_copy.typestore_copy', sys_modules_17.module_type_store, module_type_store, ['TypeStore'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_17, sys_modules_17.module_type_store, module_type_store)
    else:
        from type_store_copy.typestore_copy import TypeStore

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'type_store_copy.typestore_copy', None, module_type_store, ['TypeStore'], [TypeStore])

else:
    # Assigning a type to the variable 'type_store_copy.typestore_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'type_store_copy.typestore_copy', import_16)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from errors_copy.type_error_copy import TypeError' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_18 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'errors_copy.type_error_copy')

if (type(import_18) is not StypyTypeError):

    if (import_18 != 'pyd_module'):
        __import__(import_18)
        sys_modules_19 = sys.modules[import_18]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'errors_copy.type_error_copy', sys_modules_19.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_19, sys_modules_19.module_type_store, module_type_store)
    else:
        from errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'errors_copy.type_error_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'errors_copy.type_error_copy', import_18)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from errors_copy.type_warning_copy import TypeWarning' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_20 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'errors_copy.type_warning_copy')

if (type(import_20) is not StypyTypeError):

    if (import_20 != 'pyd_module'):
        __import__(import_20)
        sys_modules_21 = sys.modules[import_20]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'errors_copy.type_warning_copy', sys_modules_21.module_type_store, module_type_store, ['TypeWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_21, sys_modules_21.module_type_store, module_type_store)
    else:
        from errors_copy.type_warning_copy import TypeWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'errors_copy.type_warning_copy', None, module_type_store, ['TypeWarning'], [TypeWarning])

else:
    # Assigning a type to the variable 'errors_copy.type_warning_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'errors_copy.type_warning_copy', import_20)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy import norecursion' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_22 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy')

if (type(import_22) is not StypyTypeError):

    if (import_22 != 'pyd_module'):
        __import__(import_22)
        sys_modules_23 = sys.modules[import_22]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy', sys_modules_23.module_type_store, module_type_store, ['norecursion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_23, sys_modules_23.module_type_store, module_type_store)
    else:
        from python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy import norecursion

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy', None, module_type_store, ['norecursion'], [norecursion])

else:
    # Assigning a type to the variable 'python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy', import_22)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from code_generation_copy.type_inference_programs_copy.aux_functions_copy import ' statement (line 10)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_24 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'code_generation_copy.type_inference_programs_copy.aux_functions_copy')

if (type(import_24) is not StypyTypeError):

    if (import_24 != 'pyd_module'):
        __import__(import_24)
        sys_modules_25 = sys.modules[import_24]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'code_generation_copy.type_inference_programs_copy.aux_functions_copy', sys_modules_25.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_25, sys_modules_25.module_type_store, module_type_store)
    else:
        from code_generation_copy.type_inference_programs_copy.aux_functions_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'code_generation_copy.type_inference_programs_copy.aux_functions_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'code_generation_copy.type_inference_programs_copy.aux_functions_copy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'code_generation_copy.type_inference_programs_copy.aux_functions_copy', import_24)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from python_lib_copy.python_types_copy.type_inference_copy import union_type_copy' statement (line 11)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_26 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_26) is not StypyTypeError):

    if (import_26 != 'pyd_module'):
        __import__(import_26)
        sys_modules_27 = sys.modules[import_26]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'python_lib_copy.python_types_copy.type_inference_copy', sys_modules_27.module_type_store, module_type_store, ['union_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_27, sys_modules_27.module_type_store, module_type_store)
    else:
        from python_lib_copy.python_types_copy.type_inference_copy import union_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['union_type_copy'], [union_type_copy])

else:
    # Assigning a type to the variable 'python_lib_copy.python_types_copy.type_inference_copy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'python_lib_copy.python_types_copy.type_inference_copy', import_26)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from ssa_copy.ssa_copy import ' statement (line 12)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_28 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'ssa_copy.ssa_copy')

if (type(import_28) is not StypyTypeError):

    if (import_28 != 'pyd_module'):
        __import__(import_28)
        sys_modules_29 = sys.modules[import_28]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'ssa_copy.ssa_copy', sys_modules_29.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_29, sys_modules_29.module_type_store, module_type_store)
    else:
        from ssa_copy.ssa_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'ssa_copy.ssa_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'ssa_copy.ssa_copy' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'ssa_copy.ssa_copy', import_28)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'import stypy_parameters_copy' statement (line 14)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_30 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_parameters_copy')

if (type(import_30) is not StypyTypeError):

    if (import_30 != 'pyd_module'):
        __import__(import_30)
        sys_modules_31 = sys.modules[import_30]
        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_parameters_copy', sys_modules_31.module_type_store, module_type_store)
    else:
        import stypy_parameters_copy

        import_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_parameters_copy', stypy_parameters_copy, module_type_store)

else:
    # Assigning a type to the variable 'stypy_parameters_copy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_parameters_copy', import_30)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from python_interface_copy import ' statement (line 15)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_32 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'python_interface_copy')

if (type(import_32) is not StypyTypeError):

    if (import_32 != 'pyd_module'):
        __import__(import_32)
        sys_modules_33 = sys.modules[import_32]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'python_interface_copy', sys_modules_33.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_33, sys_modules_33.module_type_store, module_type_store)
    else:
        from python_interface_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'python_interface_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'python_interface_copy' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'python_interface_copy', import_32)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
