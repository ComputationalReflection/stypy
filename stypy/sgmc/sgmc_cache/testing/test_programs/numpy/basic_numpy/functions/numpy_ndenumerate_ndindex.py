
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.arange(9).reshape(3, 3)
6: for index, value in np.ndenumerate(Z):
7:     r = (index, value)
8: for index in np.ndindex(Z.shape):
9:     r2 = (index, Z[index])
10: 
11: # l = globals().copy()
12: # for v in l:
13: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_1853 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1853) is not StypyTypeError):

    if (import_1853 != 'pyd_module'):
        __import__(import_1853)
        sys_modules_1854 = sys.modules[import_1853]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1854.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1853)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to reshape(...): (line 5)
# Processing the call arguments (line 5)
int_1861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 25), 'int')
int_1862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 28), 'int')
# Processing the call keyword arguments (line 5)
kwargs_1863 = {}

# Call to arange(...): (line 5)
# Processing the call arguments (line 5)
int_1857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
# Processing the call keyword arguments (line 5)
kwargs_1858 = {}
# Getting the type of 'np' (line 5)
np_1855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'arange' of a type (line 5)
arange_1856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_1855, 'arange')
# Calling arange(args, kwargs) (line 5)
arange_call_result_1859 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), arange_1856, *[int_1857], **kwargs_1858)

# Obtaining the member 'reshape' of a type (line 5)
reshape_1860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), arange_call_result_1859, 'reshape')
# Calling reshape(args, kwargs) (line 5)
reshape_call_result_1864 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), reshape_1860, *[int_1861, int_1862], **kwargs_1863)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', reshape_call_result_1864)


# Call to ndenumerate(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'Z' (line 6)
Z_1867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 35), 'Z', False)
# Processing the call keyword arguments (line 6)
kwargs_1868 = {}
# Getting the type of 'np' (line 6)
np_1865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 20), 'np', False)
# Obtaining the member 'ndenumerate' of a type (line 6)
ndenumerate_1866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 20), np_1865, 'ndenumerate')
# Calling ndenumerate(args, kwargs) (line 6)
ndenumerate_call_result_1869 = invoke(stypy.reporting.localization.Localization(__file__, 6, 20), ndenumerate_1866, *[Z_1867], **kwargs_1868)

# Testing the type of a for loop iterable (line 6)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 6, 0), ndenumerate_call_result_1869)
# Getting the type of the for loop variable (line 6)
for_loop_var_1870 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 6, 0), ndenumerate_call_result_1869)
# Assigning a type to the variable 'index' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'index', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 0), for_loop_var_1870))
# Assigning a type to the variable 'value' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 0), for_loop_var_1870))
# SSA begins for a for statement (line 6)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Tuple to a Name (line 7):

# Obtaining an instance of the builtin type 'tuple' (line 7)
tuple_1871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7)
# Adding element type (line 7)
# Getting the type of 'index' (line 7)
index_1872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'index')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 9), tuple_1871, index_1872)
# Adding element type (line 7)
# Getting the type of 'value' (line 7)
value_1873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 16), 'value')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 9), tuple_1871, value_1873)

# Assigning a type to the variable 'r' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'r', tuple_1871)
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()



# Call to ndindex(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'Z' (line 8)
Z_1876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 24), 'Z', False)
# Obtaining the member 'shape' of a type (line 8)
shape_1877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 24), Z_1876, 'shape')
# Processing the call keyword arguments (line 8)
kwargs_1878 = {}
# Getting the type of 'np' (line 8)
np_1874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 13), 'np', False)
# Obtaining the member 'ndindex' of a type (line 8)
ndindex_1875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 13), np_1874, 'ndindex')
# Calling ndindex(args, kwargs) (line 8)
ndindex_call_result_1879 = invoke(stypy.reporting.localization.Localization(__file__, 8, 13), ndindex_1875, *[shape_1877], **kwargs_1878)

# Testing the type of a for loop iterable (line 8)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 8, 0), ndindex_call_result_1879)
# Getting the type of the for loop variable (line 8)
for_loop_var_1880 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 8, 0), ndindex_call_result_1879)
# Assigning a type to the variable 'index' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'index', for_loop_var_1880)
# SSA begins for a for statement (line 8)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a Tuple to a Name (line 9):

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_1881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
# Getting the type of 'index' (line 9)
index_1882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 10), 'index')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), tuple_1881, index_1882)
# Adding element type (line 9)

# Obtaining the type of the subscript
# Getting the type of 'index' (line 9)
index_1883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 19), 'index')
# Getting the type of 'Z' (line 9)
Z_1884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 17), 'Z')
# Obtaining the member '__getitem__' of a type (line 9)
getitem___1885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 17), Z_1884, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 9)
subscript_call_result_1886 = invoke(stypy.reporting.localization.Localization(__file__, 9, 17), getitem___1885, index_1883)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 10), tuple_1881, subscript_call_result_1886)

# Assigning a type to the variable 'r2' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'r2', tuple_1881)
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
