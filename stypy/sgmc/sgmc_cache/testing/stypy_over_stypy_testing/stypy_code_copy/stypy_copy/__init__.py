
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: __all__ = ['python_lib_copy', 'type_store_copy', 'errors_copy', 'code_generation_copy', 'log_copy', 'reporting_copy',
2:            'ssa_copy', 'type_store_copy', 'visitor_copy', 'python_interface_copy', 'stypy_main_copy',
3:            'stypy_parameters_copy']
4: 
5: from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy import Localization
6: from stypy_copy.type_store_copy.typestore_copy import TypeStore
7: from stypy_copy.errors_copy.type_error_copy import TypeError
8: from stypy_copy.errors_copy.type_warning_copy import TypeWarning
9: from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy import norecursion
10: from stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import *
11: from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy
12: from ssa_copy.ssa_copy import *
13: 
14: from python_interface_copy import *
15: 

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
list_1518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 1)
# Adding element type (line 1)
str_1519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 11), 'str', 'python_lib_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1518, str_1519)
# Adding element type (line 1)
str_1520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 30), 'str', 'type_store_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1518, str_1520)
# Adding element type (line 1)
str_1521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 49), 'str', 'errors_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1518, str_1521)
# Adding element type (line 1)
str_1522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 64), 'str', 'code_generation_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1518, str_1522)
# Adding element type (line 1)
str_1523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 88), 'str', 'log_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1518, str_1523)
# Adding element type (line 1)
str_1524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 100), 'str', 'reporting_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1518, str_1524)
# Adding element type (line 1)
str_1525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'str', 'ssa_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1518, str_1525)
# Adding element type (line 1)
str_1526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 23), 'str', 'type_store_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1518, str_1526)
# Adding element type (line 1)
str_1527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 42), 'str', 'visitor_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1518, str_1527)
# Adding element type (line 1)
str_1528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 58), 'str', 'python_interface_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1518, str_1528)
# Adding element type (line 1)
str_1529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 83), 'str', 'stypy_main_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1518, str_1529)
# Adding element type (line 1)
str_1530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'str', 'stypy_parameters_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1518, str_1530)

# Assigning a type to the variable '__all__' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), '__all__', list_1518)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy import Localization' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_1531 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy')

if (type(import_1531) is not StypyTypeError):

    if (import_1531 != 'pyd_module'):
        __import__(import_1531)
        sys_modules_1532 = sys.modules[import_1531]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy', sys_modules_1532.module_type_store, module_type_store, ['Localization'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_1532, sys_modules_1532.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy import Localization

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy', None, module_type_store, ['Localization'], [Localization])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy', import_1531)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from stypy_copy.type_store_copy.typestore_copy import TypeStore' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_1533 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.type_store_copy.typestore_copy')

if (type(import_1533) is not StypyTypeError):

    if (import_1533 != 'pyd_module'):
        __import__(import_1533)
        sys_modules_1534 = sys.modules[import_1533]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.type_store_copy.typestore_copy', sys_modules_1534.module_type_store, module_type_store, ['TypeStore'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_1534, sys_modules_1534.module_type_store, module_type_store)
    else:
        from stypy_copy.type_store_copy.typestore_copy import TypeStore

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.type_store_copy.typestore_copy', None, module_type_store, ['TypeStore'], [TypeStore])

else:
    # Assigning a type to the variable 'stypy_copy.type_store_copy.typestore_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.type_store_copy.typestore_copy', import_1533)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_1535 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.errors_copy.type_error_copy')

if (type(import_1535) is not StypyTypeError):

    if (import_1535 != 'pyd_module'):
        __import__(import_1535)
        sys_modules_1536 = sys.modules[import_1535]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.errors_copy.type_error_copy', sys_modules_1536.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_1536, sys_modules_1536.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.type_error_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.errors_copy.type_error_copy', import_1535)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from stypy_copy.errors_copy.type_warning_copy import TypeWarning' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_1537 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.errors_copy.type_warning_copy')

if (type(import_1537) is not StypyTypeError):

    if (import_1537 != 'pyd_module'):
        __import__(import_1537)
        sys_modules_1538 = sys.modules[import_1537]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.errors_copy.type_warning_copy', sys_modules_1538.module_type_store, module_type_store, ['TypeWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_1538, sys_modules_1538.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.type_warning_copy import TypeWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.errors_copy.type_warning_copy', None, module_type_store, ['TypeWarning'], [TypeWarning])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.type_warning_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.errors_copy.type_warning_copy', import_1537)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy import norecursion' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_1539 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy')

if (type(import_1539) is not StypyTypeError):

    if (import_1539 != 'pyd_module'):
        __import__(import_1539)
        sys_modules_1540 = sys.modules[import_1539]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy', sys_modules_1540.module_type_store, module_type_store, ['norecursion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_1540, sys_modules_1540.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy import norecursion

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy', None, module_type_store, ['norecursion'], [norecursion])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy', import_1539)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import ' statement (line 10)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_1541 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy')

if (type(import_1541) is not StypyTypeError):

    if (import_1541 != 'pyd_module'):
        __import__(import_1541)
        sys_modules_1542 = sys.modules[import_1541]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', sys_modules_1542.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_1542, sys_modules_1542.module_type_store, module_type_store)
    else:
        from stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', import_1541)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy' statement (line 11)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_1543 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_1543) is not StypyTypeError):

    if (import_1543 != 'pyd_module'):
        __import__(import_1543)
        sys_modules_1544 = sys.modules[import_1543]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_1544.module_type_store, module_type_store, ['union_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_1544, sys_modules_1544.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['union_type_copy'], [union_type_copy])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_1543)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from ssa_copy.ssa_copy import ' statement (line 12)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_1545 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'ssa_copy.ssa_copy')

if (type(import_1545) is not StypyTypeError):

    if (import_1545 != 'pyd_module'):
        __import__(import_1545)
        sys_modules_1546 = sys.modules[import_1545]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'ssa_copy.ssa_copy', sys_modules_1546.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_1546, sys_modules_1546.module_type_store, module_type_store)
    else:
        from ssa_copy.ssa_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'ssa_copy.ssa_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'ssa_copy.ssa_copy' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'ssa_copy.ssa_copy', import_1545)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from python_interface_copy import ' statement (line 14)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_1547 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'python_interface_copy')

if (type(import_1547) is not StypyTypeError):

    if (import_1547 != 'pyd_module'):
        __import__(import_1547)
        sys_modules_1548 = sys.modules[import_1547]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'python_interface_copy', sys_modules_1548.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_1548, sys_modules_1548.module_type_store, module_type_store)
    else:
        from python_interface_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'python_interface_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'python_interface_copy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'python_interface_copy', import_1547)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
