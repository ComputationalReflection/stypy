
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.scipy-lectures.org/intro/numpy/numpy.html
2: 
3: import numpy as np
4: 
5: x, y = np.ogrid[0:5, 0:5]
6: 
7: r1, r2 = x.shape, y.shape
8: 
9: distance = np.sqrt(x ** 2 + y ** 2)
10: 
11: x, y = np.mgrid[0:4, 0:4]
12: #
13: # l = globals().copy()
14: # for v in l:
15: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_1906 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1906) is not StypyTypeError):

    if (import_1906 != 'pyd_module'):
        __import__(import_1906)
        sys_modules_1907 = sys.modules[import_1906]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1907.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1906)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Subscript to a Tuple (line 5):

# Assigning a Subscript to a Name (line 5):

# Obtaining the type of the subscript
int_1908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'int')

# Obtaining the type of the subscript
int_1909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'int')
int_1910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 18), 'int')
slice_1911 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 5, 7), int_1909, int_1910, None)
int_1912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 21), 'int')
int_1913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
slice_1914 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 5, 7), int_1912, int_1913, None)
# Getting the type of 'np' (line 5)
np_1915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 7), 'np')
# Obtaining the member 'ogrid' of a type (line 5)
ogrid_1916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 7), np_1915, 'ogrid')
# Obtaining the member '__getitem__' of a type (line 5)
getitem___1917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 7), ogrid_1916, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 5)
subscript_call_result_1918 = invoke(stypy.reporting.localization.Localization(__file__, 5, 7), getitem___1917, (slice_1911, slice_1914))

# Obtaining the member '__getitem__' of a type (line 5)
getitem___1919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 0), subscript_call_result_1918, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 5)
subscript_call_result_1920 = invoke(stypy.reporting.localization.Localization(__file__, 5, 0), getitem___1919, int_1908)

# Assigning a type to the variable 'tuple_var_assignment_1900' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_var_assignment_1900', subscript_call_result_1920)

# Assigning a Subscript to a Name (line 5):

# Obtaining the type of the subscript
int_1921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 0), 'int')

# Obtaining the type of the subscript
int_1922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'int')
int_1923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 18), 'int')
slice_1924 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 5, 7), int_1922, int_1923, None)
int_1925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 21), 'int')
int_1926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
slice_1927 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 5, 7), int_1925, int_1926, None)
# Getting the type of 'np' (line 5)
np_1928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 7), 'np')
# Obtaining the member 'ogrid' of a type (line 5)
ogrid_1929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 7), np_1928, 'ogrid')
# Obtaining the member '__getitem__' of a type (line 5)
getitem___1930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 7), ogrid_1929, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 5)
subscript_call_result_1931 = invoke(stypy.reporting.localization.Localization(__file__, 5, 7), getitem___1930, (slice_1924, slice_1927))

# Obtaining the member '__getitem__' of a type (line 5)
getitem___1932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 0), subscript_call_result_1931, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 5)
subscript_call_result_1933 = invoke(stypy.reporting.localization.Localization(__file__, 5, 0), getitem___1932, int_1921)

# Assigning a type to the variable 'tuple_var_assignment_1901' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_var_assignment_1901', subscript_call_result_1933)

# Assigning a Name to a Name (line 5):
# Getting the type of 'tuple_var_assignment_1900' (line 5)
tuple_var_assignment_1900_1934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_var_assignment_1900')
# Assigning a type to the variable 'x' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'x', tuple_var_assignment_1900_1934)

# Assigning a Name to a Name (line 5):
# Getting the type of 'tuple_var_assignment_1901' (line 5)
tuple_var_assignment_1901_1935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'tuple_var_assignment_1901')
# Assigning a type to the variable 'y' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 3), 'y', tuple_var_assignment_1901_1935)

# Assigning a Tuple to a Tuple (line 7):

# Assigning a Attribute to a Name (line 7):
# Getting the type of 'x' (line 7)
x_1936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'x')
# Obtaining the member 'shape' of a type (line 7)
shape_1937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 9), x_1936, 'shape')
# Assigning a type to the variable 'tuple_assignment_1902' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_1902', shape_1937)

# Assigning a Attribute to a Name (line 7):
# Getting the type of 'y' (line 7)
y_1938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 18), 'y')
# Obtaining the member 'shape' of a type (line 7)
shape_1939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 18), y_1938, 'shape')
# Assigning a type to the variable 'tuple_assignment_1903' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_1903', shape_1939)

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_assignment_1902' (line 7)
tuple_assignment_1902_1940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_1902')
# Assigning a type to the variable 'r1' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r1', tuple_assignment_1902_1940)

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_assignment_1903' (line 7)
tuple_assignment_1903_1941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_1903')
# Assigning a type to the variable 'r2' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'r2', tuple_assignment_1903_1941)

# Assigning a Call to a Name (line 9):

# Assigning a Call to a Name (line 9):

# Call to sqrt(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'x' (line 9)
x_1944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 19), 'x', False)
int_1945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 24), 'int')
# Applying the binary operator '**' (line 9)
result_pow_1946 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 19), '**', x_1944, int_1945)

# Getting the type of 'y' (line 9)
y_1947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 28), 'y', False)
int_1948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 33), 'int')
# Applying the binary operator '**' (line 9)
result_pow_1949 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 28), '**', y_1947, int_1948)

# Applying the binary operator '+' (line 9)
result_add_1950 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 19), '+', result_pow_1946, result_pow_1949)

# Processing the call keyword arguments (line 9)
kwargs_1951 = {}
# Getting the type of 'np' (line 9)
np_1942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'np', False)
# Obtaining the member 'sqrt' of a type (line 9)
sqrt_1943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 11), np_1942, 'sqrt')
# Calling sqrt(args, kwargs) (line 9)
sqrt_call_result_1952 = invoke(stypy.reporting.localization.Localization(__file__, 9, 11), sqrt_1943, *[result_add_1950], **kwargs_1951)

# Assigning a type to the variable 'distance' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distance', sqrt_call_result_1952)

# Assigning a Subscript to a Tuple (line 11):

# Assigning a Subscript to a Name (line 11):

# Obtaining the type of the subscript
int_1953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 0), 'int')

# Obtaining the type of the subscript
int_1954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 16), 'int')
int_1955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'int')
slice_1956 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 11, 7), int_1954, int_1955, None)
int_1957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 21), 'int')
int_1958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'int')
slice_1959 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 11, 7), int_1957, int_1958, None)
# Getting the type of 'np' (line 11)
np_1960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 7), 'np')
# Obtaining the member 'mgrid' of a type (line 11)
mgrid_1961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 7), np_1960, 'mgrid')
# Obtaining the member '__getitem__' of a type (line 11)
getitem___1962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 7), mgrid_1961, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 11)
subscript_call_result_1963 = invoke(stypy.reporting.localization.Localization(__file__, 11, 7), getitem___1962, (slice_1956, slice_1959))

# Obtaining the member '__getitem__' of a type (line 11)
getitem___1964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 0), subscript_call_result_1963, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 11)
subscript_call_result_1965 = invoke(stypy.reporting.localization.Localization(__file__, 11, 0), getitem___1964, int_1953)

# Assigning a type to the variable 'tuple_var_assignment_1904' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'tuple_var_assignment_1904', subscript_call_result_1965)

# Assigning a Subscript to a Name (line 11):

# Obtaining the type of the subscript
int_1966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 0), 'int')

# Obtaining the type of the subscript
int_1967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 16), 'int')
int_1968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'int')
slice_1969 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 11, 7), int_1967, int_1968, None)
int_1970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 21), 'int')
int_1971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'int')
slice_1972 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 11, 7), int_1970, int_1971, None)
# Getting the type of 'np' (line 11)
np_1973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 7), 'np')
# Obtaining the member 'mgrid' of a type (line 11)
mgrid_1974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 7), np_1973, 'mgrid')
# Obtaining the member '__getitem__' of a type (line 11)
getitem___1975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 7), mgrid_1974, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 11)
subscript_call_result_1976 = invoke(stypy.reporting.localization.Localization(__file__, 11, 7), getitem___1975, (slice_1969, slice_1972))

# Obtaining the member '__getitem__' of a type (line 11)
getitem___1977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 0), subscript_call_result_1976, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 11)
subscript_call_result_1978 = invoke(stypy.reporting.localization.Localization(__file__, 11, 0), getitem___1977, int_1966)

# Assigning a type to the variable 'tuple_var_assignment_1905' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'tuple_var_assignment_1905', subscript_call_result_1978)

# Assigning a Name to a Name (line 11):
# Getting the type of 'tuple_var_assignment_1904' (line 11)
tuple_var_assignment_1904_1979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'tuple_var_assignment_1904')
# Assigning a type to the variable 'x' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'x', tuple_var_assignment_1904_1979)

# Assigning a Name to a Name (line 11):
# Getting the type of 'tuple_var_assignment_1905' (line 11)
tuple_var_assignment_1905_1980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'tuple_var_assignment_1905')
# Assigning a type to the variable 'y' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 3), 'y', tuple_var_assignment_1905_1980)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
