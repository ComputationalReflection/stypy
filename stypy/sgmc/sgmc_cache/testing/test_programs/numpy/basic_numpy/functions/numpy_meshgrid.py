
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.zeros((10, 10), [('x', float), ('y', float)])
6: Z['x'], Z['y'] = np.meshgrid(np.linspace(0, 1, 10),
7:                              np.linspace(0, 1, 10))
8: #
9: # l = globals().copy()
10: # for v in l:
11: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_1803 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1803) is not StypyTypeError):

    if (import_1803 != 'pyd_module'):
        __import__(import_1803)
        sys_modules_1804 = sys.modules[import_1803]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1804.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1803)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Assigning a Call to a Name (line 5):

# Call to zeros(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_1807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
int_1808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), tuple_1807, int_1808)
# Adding element type (line 5)
int_1809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), tuple_1807, int_1809)


# Obtaining an instance of the builtin type 'list' (line 5)
list_1810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_1811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
str_1812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 25), 'str', 'x')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 25), tuple_1811, str_1812)
# Adding element type (line 5)
# Getting the type of 'float' (line 5)
float_1813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 30), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 25), tuple_1811, float_1813)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 23), list_1810, tuple_1811)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_1814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 39), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
str_1815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 39), 'str', 'y')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 39), tuple_1814, str_1815)
# Adding element type (line 5)
# Getting the type of 'float' (line 5)
float_1816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 44), 'float', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 39), tuple_1814, float_1816)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 23), list_1810, tuple_1814)

# Processing the call keyword arguments (line 5)
kwargs_1817 = {}
# Getting the type of 'np' (line 5)
np_1805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'zeros' of a type (line 5)
zeros_1806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_1805, 'zeros')
# Calling zeros(args, kwargs) (line 5)
zeros_call_result_1818 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), zeros_1806, *[tuple_1807, list_1810], **kwargs_1817)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', zeros_call_result_1818)

# Assigning a Call to a Tuple (line 6):

# Assigning a Call to a Name:

# Call to meshgrid(...): (line 6)
# Processing the call arguments (line 6)

# Call to linspace(...): (line 6)
# Processing the call arguments (line 6)
int_1823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 41), 'int')
int_1824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 44), 'int')
int_1825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 47), 'int')
# Processing the call keyword arguments (line 6)
kwargs_1826 = {}
# Getting the type of 'np' (line 6)
np_1821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 29), 'np', False)
# Obtaining the member 'linspace' of a type (line 6)
linspace_1822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 29), np_1821, 'linspace')
# Calling linspace(args, kwargs) (line 6)
linspace_call_result_1827 = invoke(stypy.reporting.localization.Localization(__file__, 6, 29), linspace_1822, *[int_1823, int_1824, int_1825], **kwargs_1826)


# Call to linspace(...): (line 7)
# Processing the call arguments (line 7)
int_1830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 41), 'int')
int_1831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 44), 'int')
int_1832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 47), 'int')
# Processing the call keyword arguments (line 7)
kwargs_1833 = {}
# Getting the type of 'np' (line 7)
np_1828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 29), 'np', False)
# Obtaining the member 'linspace' of a type (line 7)
linspace_1829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 29), np_1828, 'linspace')
# Calling linspace(args, kwargs) (line 7)
linspace_call_result_1834 = invoke(stypy.reporting.localization.Localization(__file__, 7, 29), linspace_1829, *[int_1830, int_1831, int_1832], **kwargs_1833)

# Processing the call keyword arguments (line 6)
kwargs_1835 = {}
# Getting the type of 'np' (line 6)
np_1819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 17), 'np', False)
# Obtaining the member 'meshgrid' of a type (line 6)
meshgrid_1820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 17), np_1819, 'meshgrid')
# Calling meshgrid(args, kwargs) (line 6)
meshgrid_call_result_1836 = invoke(stypy.reporting.localization.Localization(__file__, 6, 17), meshgrid_1820, *[linspace_call_result_1827, linspace_call_result_1834], **kwargs_1835)

# Assigning a type to the variable 'call_assignment_1800' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'call_assignment_1800', meshgrid_call_result_1836)

# Assigning a Call to a Name (line 6):

# Call to __getitem__(...):
# Processing the call arguments
int_1839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 0), 'int')
# Processing the call keyword arguments
kwargs_1840 = {}
# Getting the type of 'call_assignment_1800' (line 6)
call_assignment_1800_1837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'call_assignment_1800', False)
# Obtaining the member '__getitem__' of a type (line 6)
getitem___1838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 0), call_assignment_1800_1837, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_1841 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1838, *[int_1839], **kwargs_1840)

# Assigning a type to the variable 'call_assignment_1801' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'call_assignment_1801', getitem___call_result_1841)

# Assigning a Name to a Subscript (line 6):
# Getting the type of 'call_assignment_1801' (line 6)
call_assignment_1801_1842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'call_assignment_1801')
# Getting the type of 'Z' (line 6)
Z_1843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'Z')
str_1844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 2), 'str', 'x')
# Storing an element on a container (line 6)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 0), Z_1843, (str_1844, call_assignment_1801_1842))

# Assigning a Call to a Name (line 6):

# Call to __getitem__(...):
# Processing the call arguments
int_1847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 0), 'int')
# Processing the call keyword arguments
kwargs_1848 = {}
# Getting the type of 'call_assignment_1800' (line 6)
call_assignment_1800_1845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'call_assignment_1800', False)
# Obtaining the member '__getitem__' of a type (line 6)
getitem___1846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 0), call_assignment_1800_1845, '__getitem__')
# Calling __getitem__(args, kwargs)
getitem___call_result_1849 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1846, *[int_1847], **kwargs_1848)

# Assigning a type to the variable 'call_assignment_1802' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'call_assignment_1802', getitem___call_result_1849)

# Assigning a Name to a Subscript (line 6):
# Getting the type of 'call_assignment_1802' (line 6)
call_assignment_1802_1850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'call_assignment_1802')
# Getting the type of 'Z' (line 6)
Z_1851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'Z')
str_1852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 10), 'str', 'y')
# Storing an element on a container (line 6)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 8), Z_1851, (str_1852, call_assignment_1802_1850))

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
