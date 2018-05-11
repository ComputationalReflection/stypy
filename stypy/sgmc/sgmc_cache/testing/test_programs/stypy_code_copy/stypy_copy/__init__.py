
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
list_1804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 1)
# Adding element type (line 1)
str_1805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 11), 'str', 'python_lib_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1804, str_1805)
# Adding element type (line 1)
str_1806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 30), 'str', 'type_store_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1804, str_1806)
# Adding element type (line 1)
str_1807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 49), 'str', 'errors_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1804, str_1807)
# Adding element type (line 1)
str_1808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 64), 'str', 'code_generation_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1804, str_1808)
# Adding element type (line 1)
str_1809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 88), 'str', 'log_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1804, str_1809)
# Adding element type (line 1)
str_1810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 100), 'str', 'reporting_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1804, str_1810)
# Adding element type (line 1)
str_1811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'str', 'ssa_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1804, str_1811)
# Adding element type (line 1)
str_1812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 23), 'str', 'type_store_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1804, str_1812)
# Adding element type (line 1)
str_1813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 42), 'str', 'visitor_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1804, str_1813)
# Adding element type (line 1)
str_1814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 58), 'str', 'python_interface_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1804, str_1814)
# Adding element type (line 1)
str_1815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 83), 'str', 'stypy_main_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1804, str_1815)
# Adding element type (line 1)
str_1816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'str', 'stypy_parameters_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 10), list_1804, str_1816)

# Assigning a type to the variable '__all__' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), '__all__', list_1804)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from python_lib_copy.python_types_copy.type_inference_copy.localization_copy import Localization' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_1817 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'python_lib_copy.python_types_copy.type_inference_copy.localization_copy')

if (type(import_1817) is not StypyTypeError):

    if (import_1817 != 'pyd_module'):
        __import__(import_1817)
        sys_modules_1818 = sys.modules[import_1817]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'python_lib_copy.python_types_copy.type_inference_copy.localization_copy', sys_modules_1818.module_type_store, module_type_store, ['Localization'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_1818, sys_modules_1818.module_type_store, module_type_store)
    else:
        from python_lib_copy.python_types_copy.type_inference_copy.localization_copy import Localization

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'python_lib_copy.python_types_copy.type_inference_copy.localization_copy', None, module_type_store, ['Localization'], [Localization])

else:
    # Assigning a type to the variable 'python_lib_copy.python_types_copy.type_inference_copy.localization_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'python_lib_copy.python_types_copy.type_inference_copy.localization_copy', import_1817)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from type_store_copy.typestore_copy import TypeStore' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_1819 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'type_store_copy.typestore_copy')

if (type(import_1819) is not StypyTypeError):

    if (import_1819 != 'pyd_module'):
        __import__(import_1819)
        sys_modules_1820 = sys.modules[import_1819]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'type_store_copy.typestore_copy', sys_modules_1820.module_type_store, module_type_store, ['TypeStore'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_1820, sys_modules_1820.module_type_store, module_type_store)
    else:
        from type_store_copy.typestore_copy import TypeStore

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'type_store_copy.typestore_copy', None, module_type_store, ['TypeStore'], [TypeStore])

else:
    # Assigning a type to the variable 'type_store_copy.typestore_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'type_store_copy.typestore_copy', import_1819)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from errors_copy.type_error_copy import TypeError' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_1821 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'errors_copy.type_error_copy')

if (type(import_1821) is not StypyTypeError):

    if (import_1821 != 'pyd_module'):
        __import__(import_1821)
        sys_modules_1822 = sys.modules[import_1821]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'errors_copy.type_error_copy', sys_modules_1822.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_1822, sys_modules_1822.module_type_store, module_type_store)
    else:
        from errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'errors_copy.type_error_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'errors_copy.type_error_copy', import_1821)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from errors_copy.type_warning_copy import TypeWarning' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_1823 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'errors_copy.type_warning_copy')

if (type(import_1823) is not StypyTypeError):

    if (import_1823 != 'pyd_module'):
        __import__(import_1823)
        sys_modules_1824 = sys.modules[import_1823]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'errors_copy.type_warning_copy', sys_modules_1824.module_type_store, module_type_store, ['TypeWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_1824, sys_modules_1824.module_type_store, module_type_store)
    else:
        from errors_copy.type_warning_copy import TypeWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'errors_copy.type_warning_copy', None, module_type_store, ['TypeWarning'], [TypeWarning])

else:
    # Assigning a type to the variable 'errors_copy.type_warning_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'errors_copy.type_warning_copy', import_1823)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy import norecursion' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_1825 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy')

if (type(import_1825) is not StypyTypeError):

    if (import_1825 != 'pyd_module'):
        __import__(import_1825)
        sys_modules_1826 = sys.modules[import_1825]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy', sys_modules_1826.module_type_store, module_type_store, ['norecursion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_1826, sys_modules_1826.module_type_store, module_type_store)
    else:
        from python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy import norecursion

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy', None, module_type_store, ['norecursion'], [norecursion])

else:
    # Assigning a type to the variable 'python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy', import_1825)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from code_generation_copy.type_inference_programs_copy.aux_functions_copy import ' statement (line 10)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_1827 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'code_generation_copy.type_inference_programs_copy.aux_functions_copy')

if (type(import_1827) is not StypyTypeError):

    if (import_1827 != 'pyd_module'):
        __import__(import_1827)
        sys_modules_1828 = sys.modules[import_1827]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'code_generation_copy.type_inference_programs_copy.aux_functions_copy', sys_modules_1828.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_1828, sys_modules_1828.module_type_store, module_type_store)
    else:
        from code_generation_copy.type_inference_programs_copy.aux_functions_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'code_generation_copy.type_inference_programs_copy.aux_functions_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'code_generation_copy.type_inference_programs_copy.aux_functions_copy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'code_generation_copy.type_inference_programs_copy.aux_functions_copy', import_1827)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from python_lib_copy.python_types_copy.type_inference_copy import union_type_copy' statement (line 11)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_1829 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_1829) is not StypyTypeError):

    if (import_1829 != 'pyd_module'):
        __import__(import_1829)
        sys_modules_1830 = sys.modules[import_1829]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'python_lib_copy.python_types_copy.type_inference_copy', sys_modules_1830.module_type_store, module_type_store, ['union_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_1830, sys_modules_1830.module_type_store, module_type_store)
    else:
        from python_lib_copy.python_types_copy.type_inference_copy import union_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['union_type_copy'], [union_type_copy])

else:
    # Assigning a type to the variable 'python_lib_copy.python_types_copy.type_inference_copy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'python_lib_copy.python_types_copy.type_inference_copy', import_1829)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from ssa_copy.ssa_copy import ' statement (line 12)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_1831 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'ssa_copy.ssa_copy')

if (type(import_1831) is not StypyTypeError):

    if (import_1831 != 'pyd_module'):
        __import__(import_1831)
        sys_modules_1832 = sys.modules[import_1831]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'ssa_copy.ssa_copy', sys_modules_1832.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_1832, sys_modules_1832.module_type_store, module_type_store)
    else:
        from ssa_copy.ssa_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'ssa_copy.ssa_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'ssa_copy.ssa_copy' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'ssa_copy.ssa_copy', import_1831)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from python_interface_copy import ' statement (line 14)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_1833 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'python_interface_copy')

if (type(import_1833) is not StypyTypeError):

    if (import_1833 != 'pyd_module'):
        __import__(import_1833)
        sys_modules_1834 = sys.modules[import_1833]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'python_interface_copy', sys_modules_1834.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_1834, sys_modules_1834.module_type_store, module_type_store)
    else:
        from python_interface_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'python_interface_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'python_interface_copy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'python_interface_copy', import_1833)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
