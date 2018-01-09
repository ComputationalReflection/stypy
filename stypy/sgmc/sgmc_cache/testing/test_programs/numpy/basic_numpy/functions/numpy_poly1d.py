
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.scipy-lectures.org/intro/numpy/numpy.html
2: 
3: import numpy as np
4: 
5: p = np.poly1d([3, 2, -1])
6: r = p(0)
7: 
8: r2 = p.roots
9: r3 = p.order
10: 
11: x = np.linspace(0, 1, 20)
12: y = np.cos(x) + 0.3 * np.random.rand(20)
13: p = np.poly1d(np.polyfit(x, y, 3))
14: 
15: t = np.linspace(0, 1, 200)
16: 
17: p2 = np.polynomial.Polynomial([-1, 2, 3])  # coefs in different order!
18: r4 = p2(0)
19: r5 = p2.roots()
20: # Type error
21: r6 = p2.order
22: x2 = np.linspace(-1, 1, 2000)
23: y2 = np.cos(x) + 0.3 * np.random.rand(20)
24: p3 = np.polynomial.Chebyshev.fit(x, y, 90)
25: 
26: t2 = np.linspace(-1, 1, 200)
27: 
28: # l = globals().copy()
29: # for v in l:
30: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
31: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_1999 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1999) is not StypyTypeError):

    if (import_1999 != 'pyd_module'):
        __import__(import_1999)
        sys_modules_2000 = sys.modules[import_1999]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2000.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1999)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to poly1d(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_2003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_2004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_2003, int_2004)
# Adding element type (line 5)
int_2005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_2003, int_2005)
# Adding element type (line 5)
int_2006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_2003, int_2006)

# Processing the call keyword arguments (line 5)
kwargs_2007 = {}
# Getting the type of 'np' (line 5)
np_2001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'poly1d' of a type (line 5)
poly1d_2002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_2001, 'poly1d')
# Calling poly1d(args, kwargs) (line 5)
poly1d_call_result_2008 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), poly1d_2002, *[list_2003], **kwargs_2007)

# Assigning a type to the variable 'p' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'p', poly1d_call_result_2008)

# Assigning a Call to a Name (line 6):

# Call to p(...): (line 6)
# Processing the call arguments (line 6)
int_2010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 6), 'int')
# Processing the call keyword arguments (line 6)
kwargs_2011 = {}
# Getting the type of 'p' (line 6)
p_2009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'p', False)
# Calling p(args, kwargs) (line 6)
p_call_result_2012 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), p_2009, *[int_2010], **kwargs_2011)

# Assigning a type to the variable 'r' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r', p_call_result_2012)

# Assigning a Attribute to a Name (line 8):
# Getting the type of 'p' (line 8)
p_2013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'p')
# Obtaining the member 'roots' of a type (line 8)
roots_2014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), p_2013, 'roots')
# Assigning a type to the variable 'r2' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r2', roots_2014)

# Assigning a Attribute to a Name (line 9):
# Getting the type of 'p' (line 9)
p_2015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'p')
# Obtaining the member 'order' of a type (line 9)
order_2016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), p_2015, 'order')
# Assigning a type to the variable 'r3' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r3', order_2016)

# Assigning a Call to a Name (line 11):

# Call to linspace(...): (line 11)
# Processing the call arguments (line 11)
int_2019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 16), 'int')
int_2020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 19), 'int')
int_2021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 22), 'int')
# Processing the call keyword arguments (line 11)
kwargs_2022 = {}
# Getting the type of 'np' (line 11)
np_2017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'np', False)
# Obtaining the member 'linspace' of a type (line 11)
linspace_2018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), np_2017, 'linspace')
# Calling linspace(args, kwargs) (line 11)
linspace_call_result_2023 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), linspace_2018, *[int_2019, int_2020, int_2021], **kwargs_2022)

# Assigning a type to the variable 'x' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'x', linspace_call_result_2023)

# Assigning a BinOp to a Name (line 12):

# Call to cos(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'x' (line 12)
x_2026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 11), 'x', False)
# Processing the call keyword arguments (line 12)
kwargs_2027 = {}
# Getting the type of 'np' (line 12)
np_2024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'np', False)
# Obtaining the member 'cos' of a type (line 12)
cos_2025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), np_2024, 'cos')
# Calling cos(args, kwargs) (line 12)
cos_call_result_2028 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), cos_2025, *[x_2026], **kwargs_2027)

float_2029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'float')

# Call to rand(...): (line 12)
# Processing the call arguments (line 12)
int_2033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 37), 'int')
# Processing the call keyword arguments (line 12)
kwargs_2034 = {}
# Getting the type of 'np' (line 12)
np_2030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 22), 'np', False)
# Obtaining the member 'random' of a type (line 12)
random_2031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 22), np_2030, 'random')
# Obtaining the member 'rand' of a type (line 12)
rand_2032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 22), random_2031, 'rand')
# Calling rand(args, kwargs) (line 12)
rand_call_result_2035 = invoke(stypy.reporting.localization.Localization(__file__, 12, 22), rand_2032, *[int_2033], **kwargs_2034)

# Applying the binary operator '*' (line 12)
result_mul_2036 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 16), '*', float_2029, rand_call_result_2035)

# Applying the binary operator '+' (line 12)
result_add_2037 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 4), '+', cos_call_result_2028, result_mul_2036)

# Assigning a type to the variable 'y' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'y', result_add_2037)

# Assigning a Call to a Name (line 13):

# Call to poly1d(...): (line 13)
# Processing the call arguments (line 13)

# Call to polyfit(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'x' (line 13)
x_2042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 25), 'x', False)
# Getting the type of 'y' (line 13)
y_2043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 28), 'y', False)
int_2044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 31), 'int')
# Processing the call keyword arguments (line 13)
kwargs_2045 = {}
# Getting the type of 'np' (line 13)
np_2040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'np', False)
# Obtaining the member 'polyfit' of a type (line 13)
polyfit_2041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 14), np_2040, 'polyfit')
# Calling polyfit(args, kwargs) (line 13)
polyfit_call_result_2046 = invoke(stypy.reporting.localization.Localization(__file__, 13, 14), polyfit_2041, *[x_2042, y_2043, int_2044], **kwargs_2045)

# Processing the call keyword arguments (line 13)
kwargs_2047 = {}
# Getting the type of 'np' (line 13)
np_2038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'np', False)
# Obtaining the member 'poly1d' of a type (line 13)
poly1d_2039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), np_2038, 'poly1d')
# Calling poly1d(args, kwargs) (line 13)
poly1d_call_result_2048 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), poly1d_2039, *[polyfit_call_result_2046], **kwargs_2047)

# Assigning a type to the variable 'p' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'p', poly1d_call_result_2048)

# Assigning a Call to a Name (line 15):

# Call to linspace(...): (line 15)
# Processing the call arguments (line 15)
int_2051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 16), 'int')
int_2052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 19), 'int')
int_2053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 22), 'int')
# Processing the call keyword arguments (line 15)
kwargs_2054 = {}
# Getting the type of 'np' (line 15)
np_2049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'np', False)
# Obtaining the member 'linspace' of a type (line 15)
linspace_2050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), np_2049, 'linspace')
# Calling linspace(args, kwargs) (line 15)
linspace_call_result_2055 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), linspace_2050, *[int_2051, int_2052, int_2053], **kwargs_2054)

# Assigning a type to the variable 't' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 't', linspace_call_result_2055)

# Assigning a Call to a Name (line 17):

# Call to Polynomial(...): (line 17)
# Processing the call arguments (line 17)

# Obtaining an instance of the builtin type 'list' (line 17)
list_2059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 30), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
int_2060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 30), list_2059, int_2060)
# Adding element type (line 17)
int_2061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 30), list_2059, int_2061)
# Adding element type (line 17)
int_2062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 30), list_2059, int_2062)

# Processing the call keyword arguments (line 17)
kwargs_2063 = {}
# Getting the type of 'np' (line 17)
np_2056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'np', False)
# Obtaining the member 'polynomial' of a type (line 17)
polynomial_2057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), np_2056, 'polynomial')
# Obtaining the member 'Polynomial' of a type (line 17)
Polynomial_2058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), polynomial_2057, 'Polynomial')
# Calling Polynomial(args, kwargs) (line 17)
Polynomial_call_result_2064 = invoke(stypy.reporting.localization.Localization(__file__, 17, 5), Polynomial_2058, *[list_2059], **kwargs_2063)

# Assigning a type to the variable 'p2' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'p2', Polynomial_call_result_2064)

# Assigning a Call to a Name (line 18):

# Call to p2(...): (line 18)
# Processing the call arguments (line 18)
int_2066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'int')
# Processing the call keyword arguments (line 18)
kwargs_2067 = {}
# Getting the type of 'p2' (line 18)
p2_2065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'p2', False)
# Calling p2(args, kwargs) (line 18)
p2_call_result_2068 = invoke(stypy.reporting.localization.Localization(__file__, 18, 5), p2_2065, *[int_2066], **kwargs_2067)

# Assigning a type to the variable 'r4' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'r4', p2_call_result_2068)

# Assigning a Call to a Name (line 19):

# Call to roots(...): (line 19)
# Processing the call keyword arguments (line 19)
kwargs_2071 = {}
# Getting the type of 'p2' (line 19)
p2_2069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'p2', False)
# Obtaining the member 'roots' of a type (line 19)
roots_2070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 5), p2_2069, 'roots')
# Calling roots(args, kwargs) (line 19)
roots_call_result_2072 = invoke(stypy.reporting.localization.Localization(__file__, 19, 5), roots_2070, *[], **kwargs_2071)

# Assigning a type to the variable 'r5' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r5', roots_call_result_2072)

# Assigning a Attribute to a Name (line 21):
# Getting the type of 'p2' (line 21)
p2_2073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 5), 'p2')
# Obtaining the member 'order' of a type (line 21)
order_2074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 5), p2_2073, 'order')
# Assigning a type to the variable 'r6' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'r6', order_2074)

# Assigning a Call to a Name (line 22):

# Call to linspace(...): (line 22)
# Processing the call arguments (line 22)
int_2077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'int')
int_2078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'int')
int_2079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 24), 'int')
# Processing the call keyword arguments (line 22)
kwargs_2080 = {}
# Getting the type of 'np' (line 22)
np_2075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 5), 'np', False)
# Obtaining the member 'linspace' of a type (line 22)
linspace_2076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 5), np_2075, 'linspace')
# Calling linspace(args, kwargs) (line 22)
linspace_call_result_2081 = invoke(stypy.reporting.localization.Localization(__file__, 22, 5), linspace_2076, *[int_2077, int_2078, int_2079], **kwargs_2080)

# Assigning a type to the variable 'x2' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'x2', linspace_call_result_2081)

# Assigning a BinOp to a Name (line 23):

# Call to cos(...): (line 23)
# Processing the call arguments (line 23)
# Getting the type of 'x' (line 23)
x_2084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'x', False)
# Processing the call keyword arguments (line 23)
kwargs_2085 = {}
# Getting the type of 'np' (line 23)
np_2082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 5), 'np', False)
# Obtaining the member 'cos' of a type (line 23)
cos_2083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 5), np_2082, 'cos')
# Calling cos(args, kwargs) (line 23)
cos_call_result_2086 = invoke(stypy.reporting.localization.Localization(__file__, 23, 5), cos_2083, *[x_2084], **kwargs_2085)

float_2087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 17), 'float')

# Call to rand(...): (line 23)
# Processing the call arguments (line 23)
int_2091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 38), 'int')
# Processing the call keyword arguments (line 23)
kwargs_2092 = {}
# Getting the type of 'np' (line 23)
np_2088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'np', False)
# Obtaining the member 'random' of a type (line 23)
random_2089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 23), np_2088, 'random')
# Obtaining the member 'rand' of a type (line 23)
rand_2090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 23), random_2089, 'rand')
# Calling rand(args, kwargs) (line 23)
rand_call_result_2093 = invoke(stypy.reporting.localization.Localization(__file__, 23, 23), rand_2090, *[int_2091], **kwargs_2092)

# Applying the binary operator '*' (line 23)
result_mul_2094 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 17), '*', float_2087, rand_call_result_2093)

# Applying the binary operator '+' (line 23)
result_add_2095 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 5), '+', cos_call_result_2086, result_mul_2094)

# Assigning a type to the variable 'y2' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'y2', result_add_2095)

# Assigning a Call to a Name (line 24):

# Call to fit(...): (line 24)
# Processing the call arguments (line 24)
# Getting the type of 'x' (line 24)
x_2100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 33), 'x', False)
# Getting the type of 'y' (line 24)
y_2101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 36), 'y', False)
int_2102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 39), 'int')
# Processing the call keyword arguments (line 24)
kwargs_2103 = {}
# Getting the type of 'np' (line 24)
np_2096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 5), 'np', False)
# Obtaining the member 'polynomial' of a type (line 24)
polynomial_2097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 5), np_2096, 'polynomial')
# Obtaining the member 'Chebyshev' of a type (line 24)
Chebyshev_2098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 5), polynomial_2097, 'Chebyshev')
# Obtaining the member 'fit' of a type (line 24)
fit_2099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 5), Chebyshev_2098, 'fit')
# Calling fit(args, kwargs) (line 24)
fit_call_result_2104 = invoke(stypy.reporting.localization.Localization(__file__, 24, 5), fit_2099, *[x_2100, y_2101, int_2102], **kwargs_2103)

# Assigning a type to the variable 'p3' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'p3', fit_call_result_2104)

# Assigning a Call to a Name (line 26):

# Call to linspace(...): (line 26)
# Processing the call arguments (line 26)
int_2107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 17), 'int')
int_2108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'int')
int_2109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 24), 'int')
# Processing the call keyword arguments (line 26)
kwargs_2110 = {}
# Getting the type of 'np' (line 26)
np_2105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 5), 'np', False)
# Obtaining the member 'linspace' of a type (line 26)
linspace_2106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 5), np_2105, 'linspace')
# Calling linspace(args, kwargs) (line 26)
linspace_call_result_2111 = invoke(stypy.reporting.localization.Localization(__file__, 26, 5), linspace_2106, *[int_2107, int_2108, int_2109], **kwargs_2110)

# Assigning a type to the variable 't2' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 't2', linspace_call_result_2111)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
