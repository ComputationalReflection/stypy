
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # https://docs.scipy.org/doc/numpy/reference/routines.math.html
2: 
3: import numpy as np
4: 
5: x = 2.1
6: 
7: # Trigonometric functions
8: r1 = np.sin(x)  # Trigonometric sine, element-wise.
9: r2 = np.cos(x)  # Cosine element-wise.
10: r3 = np.tan(x)  # Compute tangent element-wise.
11: r4 = np.arcsin(x/10)  # Inverse sine, element-wise.
12: r5 = np.arccos(x/10)  # Trigonometric inverse cosine, element-wise.
13: r6 = np.arctan(x)  # Trigonometric inverse tangent, element-wise.
14: 
15: r7 = np.hypot(3 * np.ones((3, 3)), 4 * np.ones((3, 3)))  # Given the 'legs' of a right triangle, return its hypotenuse.
16: 
17: o1 = np.array([-1, +1, +1, -1])
18: o2 = np.array([-1, -1, +1, +1])
19: r8 = np.arctan2(o2, o1) * 180 / np.pi  # Element-wise arc tangent of x1/x2 choosing the quadrant correctly.
20: 
21: r9 = np.degrees(x)  # Convert angles from radians to degrees.
22: r10 = np.radians(x)  # Convert angles from degrees to radians.
23: 
24: phase = np.linspace(0, np.pi, num=5)
25: phase[3:] += np.pi
26: r11 = np.unwrap(phase)  # Unwrap by changing deltas between values to 2*pi complement.
27: 
28: r12 = np.deg2rad(x)  # Convert angles from degrees to radians.
29: r13 = np.rad2deg(x)  # Convert angles from radians to degrees.
30: 
31: x = [1, 2, 3, 4]
32: x10 = [0.1, 0.2, 0.3, 0.4]
33: 
34: r14 = np.sin(x)  # Trigonometric sine, element-wise.
35: r15 = np.cos(x)  # Cosine element-wise.
36: r16 = np.tan(x)  # Compute tangent element-wise.
37: r17 = np.arcsin(x10)  # Inverse sine, element-wise.
38: r18 = np.arccos(x10)  # Trigonometric inverse cosine, element-wise.
39: r19 = np.arctan(x)  # Trigonometric inverse tangent, element-wise.
40: 
41: r20 = np.degrees(x)  # Convert angles from radians to degrees.
42: r21 = np.radians(x)  # Convert angles from degrees to radians.
43: 
44: r22 = np.deg2rad(x)  # Convert angles from degrees to radians.
45: r23 = np.rad2deg(x)  # Convert angles from radians to degrees.
46: 
47: # l = globals().copy()
48: # for v in l:
49: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
50: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_873 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_873) is not StypyTypeError):

    if (import_873 != 'pyd_module'):
        __import__(import_873)
        sys_modules_874 = sys.modules[import_873]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_874.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_873)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a Num to a Name (line 5):
float_875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 4), 'float')
# Assigning a type to the variable 'x' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'x', float_875)

# Assigning a Call to a Name (line 8):

# Call to sin(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'x' (line 8)
x_878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'x', False)
# Processing the call keyword arguments (line 8)
kwargs_879 = {}
# Getting the type of 'np' (line 8)
np_876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'np', False)
# Obtaining the member 'sin' of a type (line 8)
sin_877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), np_876, 'sin')
# Calling sin(args, kwargs) (line 8)
sin_call_result_880 = invoke(stypy.reporting.localization.Localization(__file__, 8, 5), sin_877, *[x_878], **kwargs_879)

# Assigning a type to the variable 'r1' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r1', sin_call_result_880)

# Assigning a Call to a Name (line 9):

# Call to cos(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'x' (line 9)
x_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'x', False)
# Processing the call keyword arguments (line 9)
kwargs_884 = {}
# Getting the type of 'np' (line 9)
np_881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'np', False)
# Obtaining the member 'cos' of a type (line 9)
cos_882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), np_881, 'cos')
# Calling cos(args, kwargs) (line 9)
cos_call_result_885 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), cos_882, *[x_883], **kwargs_884)

# Assigning a type to the variable 'r2' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r2', cos_call_result_885)

# Assigning a Call to a Name (line 10):

# Call to tan(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'x' (line 10)
x_888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'x', False)
# Processing the call keyword arguments (line 10)
kwargs_889 = {}
# Getting the type of 'np' (line 10)
np_886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'np', False)
# Obtaining the member 'tan' of a type (line 10)
tan_887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), np_886, 'tan')
# Calling tan(args, kwargs) (line 10)
tan_call_result_890 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), tan_887, *[x_888], **kwargs_889)

# Assigning a type to the variable 'r3' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r3', tan_call_result_890)

# Assigning a Call to a Name (line 11):

# Call to arcsin(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'x' (line 11)
x_893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'x', False)
int_894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 17), 'int')
# Applying the binary operator 'div' (line 11)
result_div_895 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 15), 'div', x_893, int_894)

# Processing the call keyword arguments (line 11)
kwargs_896 = {}
# Getting the type of 'np' (line 11)
np_891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'np', False)
# Obtaining the member 'arcsin' of a type (line 11)
arcsin_892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), np_891, 'arcsin')
# Calling arcsin(args, kwargs) (line 11)
arcsin_call_result_897 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), arcsin_892, *[result_div_895], **kwargs_896)

# Assigning a type to the variable 'r4' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r4', arcsin_call_result_897)

# Assigning a Call to a Name (line 12):

# Call to arccos(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'x' (line 12)
x_900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'x', False)
int_901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 17), 'int')
# Applying the binary operator 'div' (line 12)
result_div_902 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 15), 'div', x_900, int_901)

# Processing the call keyword arguments (line 12)
kwargs_903 = {}
# Getting the type of 'np' (line 12)
np_898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'np', False)
# Obtaining the member 'arccos' of a type (line 12)
arccos_899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), np_898, 'arccos')
# Calling arccos(args, kwargs) (line 12)
arccos_call_result_904 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), arccos_899, *[result_div_902], **kwargs_903)

# Assigning a type to the variable 'r5' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r5', arccos_call_result_904)

# Assigning a Call to a Name (line 13):

# Call to arctan(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'x' (line 13)
x_907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 15), 'x', False)
# Processing the call keyword arguments (line 13)
kwargs_908 = {}
# Getting the type of 'np' (line 13)
np_905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'np', False)
# Obtaining the member 'arctan' of a type (line 13)
arctan_906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), np_905, 'arctan')
# Calling arctan(args, kwargs) (line 13)
arctan_call_result_909 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), arctan_906, *[x_907], **kwargs_908)

# Assigning a type to the variable 'r6' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r6', arctan_call_result_909)

# Assigning a Call to a Name (line 15):

# Call to hypot(...): (line 15)
# Processing the call arguments (line 15)
int_912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 14), 'int')

# Call to ones(...): (line 15)
# Processing the call arguments (line 15)

# Obtaining an instance of the builtin type 'tuple' (line 15)
tuple_915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 27), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 15)
# Adding element type (line 15)
int_916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 27), tuple_915, int_916)
# Adding element type (line 15)
int_917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 27), tuple_915, int_917)

# Processing the call keyword arguments (line 15)
kwargs_918 = {}
# Getting the type of 'np' (line 15)
np_913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 18), 'np', False)
# Obtaining the member 'ones' of a type (line 15)
ones_914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 18), np_913, 'ones')
# Calling ones(args, kwargs) (line 15)
ones_call_result_919 = invoke(stypy.reporting.localization.Localization(__file__, 15, 18), ones_914, *[tuple_915], **kwargs_918)

# Applying the binary operator '*' (line 15)
result_mul_920 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 14), '*', int_912, ones_call_result_919)

int_921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 35), 'int')

# Call to ones(...): (line 15)
# Processing the call arguments (line 15)

# Obtaining an instance of the builtin type 'tuple' (line 15)
tuple_924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 48), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 15)
# Adding element type (line 15)
int_925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 48), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 48), tuple_924, int_925)
# Adding element type (line 15)
int_926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 51), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 48), tuple_924, int_926)

# Processing the call keyword arguments (line 15)
kwargs_927 = {}
# Getting the type of 'np' (line 15)
np_922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 39), 'np', False)
# Obtaining the member 'ones' of a type (line 15)
ones_923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 39), np_922, 'ones')
# Calling ones(args, kwargs) (line 15)
ones_call_result_928 = invoke(stypy.reporting.localization.Localization(__file__, 15, 39), ones_923, *[tuple_924], **kwargs_927)

# Applying the binary operator '*' (line 15)
result_mul_929 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 35), '*', int_921, ones_call_result_928)

# Processing the call keyword arguments (line 15)
kwargs_930 = {}
# Getting the type of 'np' (line 15)
np_910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'np', False)
# Obtaining the member 'hypot' of a type (line 15)
hypot_911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), np_910, 'hypot')
# Calling hypot(args, kwargs) (line 15)
hypot_call_result_931 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), hypot_911, *[result_mul_920, result_mul_929], **kwargs_930)

# Assigning a type to the variable 'r7' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r7', hypot_call_result_931)

# Assigning a Call to a Name (line 17):

# Call to array(...): (line 17)
# Processing the call arguments (line 17)

# Obtaining an instance of the builtin type 'list' (line 17)
list_934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
int_935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 14), list_934, int_935)
# Adding element type (line 17)

int_936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 20), 'int')
# Applying the 'uadd' unary operator (line 17)
result___pos___937 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 19), 'uadd', int_936)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 14), list_934, result___pos___937)
# Adding element type (line 17)

int_938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 24), 'int')
# Applying the 'uadd' unary operator (line 17)
result___pos___939 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 23), 'uadd', int_938)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 14), list_934, result___pos___939)
# Adding element type (line 17)
int_940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 14), list_934, int_940)

# Processing the call keyword arguments (line 17)
kwargs_941 = {}
# Getting the type of 'np' (line 17)
np_932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'np', False)
# Obtaining the member 'array' of a type (line 17)
array_933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), np_932, 'array')
# Calling array(args, kwargs) (line 17)
array_call_result_942 = invoke(stypy.reporting.localization.Localization(__file__, 17, 5), array_933, *[list_934], **kwargs_941)

# Assigning a type to the variable 'o1' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'o1', array_call_result_942)

# Assigning a Call to a Name (line 18):

# Call to array(...): (line 18)
# Processing the call arguments (line 18)

# Obtaining an instance of the builtin type 'list' (line 18)
list_945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
int_946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 14), list_945, int_946)
# Adding element type (line 18)
int_947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 14), list_945, int_947)
# Adding element type (line 18)

int_948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 24), 'int')
# Applying the 'uadd' unary operator (line 18)
result___pos___949 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 23), 'uadd', int_948)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 14), list_945, result___pos___949)
# Adding element type (line 18)

int_950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 28), 'int')
# Applying the 'uadd' unary operator (line 18)
result___pos___951 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 27), 'uadd', int_950)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 14), list_945, result___pos___951)

# Processing the call keyword arguments (line 18)
kwargs_952 = {}
# Getting the type of 'np' (line 18)
np_943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'np', False)
# Obtaining the member 'array' of a type (line 18)
array_944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 5), np_943, 'array')
# Calling array(args, kwargs) (line 18)
array_call_result_953 = invoke(stypy.reporting.localization.Localization(__file__, 18, 5), array_944, *[list_945], **kwargs_952)

# Assigning a type to the variable 'o2' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'o2', array_call_result_953)

# Assigning a BinOp to a Name (line 19):

# Call to arctan2(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of 'o2' (line 19)
o2_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'o2', False)
# Getting the type of 'o1' (line 19)
o1_957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'o1', False)
# Processing the call keyword arguments (line 19)
kwargs_958 = {}
# Getting the type of 'np' (line 19)
np_954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'np', False)
# Obtaining the member 'arctan2' of a type (line 19)
arctan2_955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 5), np_954, 'arctan2')
# Calling arctan2(args, kwargs) (line 19)
arctan2_call_result_959 = invoke(stypy.reporting.localization.Localization(__file__, 19, 5), arctan2_955, *[o2_956, o1_957], **kwargs_958)

int_960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'int')
# Applying the binary operator '*' (line 19)
result_mul_961 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 5), '*', arctan2_call_result_959, int_960)

# Getting the type of 'np' (line 19)
np_962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 32), 'np')
# Obtaining the member 'pi' of a type (line 19)
pi_963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 32), np_962, 'pi')
# Applying the binary operator 'div' (line 19)
result_div_964 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 30), 'div', result_mul_961, pi_963)

# Assigning a type to the variable 'r8' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r8', result_div_964)

# Assigning a Call to a Name (line 21):

# Call to degrees(...): (line 21)
# Processing the call arguments (line 21)
# Getting the type of 'x' (line 21)
x_967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'x', False)
# Processing the call keyword arguments (line 21)
kwargs_968 = {}
# Getting the type of 'np' (line 21)
np_965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 5), 'np', False)
# Obtaining the member 'degrees' of a type (line 21)
degrees_966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 5), np_965, 'degrees')
# Calling degrees(args, kwargs) (line 21)
degrees_call_result_969 = invoke(stypy.reporting.localization.Localization(__file__, 21, 5), degrees_966, *[x_967], **kwargs_968)

# Assigning a type to the variable 'r9' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'r9', degrees_call_result_969)

# Assigning a Call to a Name (line 22):

# Call to radians(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of 'x' (line 22)
x_972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'x', False)
# Processing the call keyword arguments (line 22)
kwargs_973 = {}
# Getting the type of 'np' (line 22)
np_970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 6), 'np', False)
# Obtaining the member 'radians' of a type (line 22)
radians_971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 6), np_970, 'radians')
# Calling radians(args, kwargs) (line 22)
radians_call_result_974 = invoke(stypy.reporting.localization.Localization(__file__, 22, 6), radians_971, *[x_972], **kwargs_973)

# Assigning a type to the variable 'r10' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'r10', radians_call_result_974)

# Assigning a Call to a Name (line 24):

# Call to linspace(...): (line 24)
# Processing the call arguments (line 24)
int_977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 20), 'int')
# Getting the type of 'np' (line 24)
np_978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 23), 'np', False)
# Obtaining the member 'pi' of a type (line 24)
pi_979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 23), np_978, 'pi')
# Processing the call keyword arguments (line 24)
int_980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 34), 'int')
keyword_981 = int_980
kwargs_982 = {'num': keyword_981}
# Getting the type of 'np' (line 24)
np_975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'np', False)
# Obtaining the member 'linspace' of a type (line 24)
linspace_976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), np_975, 'linspace')
# Calling linspace(args, kwargs) (line 24)
linspace_call_result_983 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), linspace_976, *[int_977, pi_979], **kwargs_982)

# Assigning a type to the variable 'phase' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'phase', linspace_call_result_983)

# Getting the type of 'phase' (line 25)
phase_984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'phase')

# Obtaining the type of the subscript
int_985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 6), 'int')
slice_986 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 25, 0), int_985, None, None)
# Getting the type of 'phase' (line 25)
phase_987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'phase')
# Obtaining the member '__getitem__' of a type (line 25)
getitem___988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 0), phase_987, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 25)
subscript_call_result_989 = invoke(stypy.reporting.localization.Localization(__file__, 25, 0), getitem___988, slice_986)

# Getting the type of 'np' (line 25)
np_990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 13), 'np')
# Obtaining the member 'pi' of a type (line 25)
pi_991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 13), np_990, 'pi')
# Applying the binary operator '+=' (line 25)
result_iadd_992 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 0), '+=', subscript_call_result_989, pi_991)
# Getting the type of 'phase' (line 25)
phase_993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'phase')
int_994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 6), 'int')
slice_995 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 25, 0), int_994, None, None)
# Storing an element on a container (line 25)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 0), phase_993, (slice_995, result_iadd_992))


# Assigning a Call to a Name (line 26):

# Call to unwrap(...): (line 26)
# Processing the call arguments (line 26)
# Getting the type of 'phase' (line 26)
phase_998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'phase', False)
# Processing the call keyword arguments (line 26)
kwargs_999 = {}
# Getting the type of 'np' (line 26)
np_996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 6), 'np', False)
# Obtaining the member 'unwrap' of a type (line 26)
unwrap_997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 6), np_996, 'unwrap')
# Calling unwrap(args, kwargs) (line 26)
unwrap_call_result_1000 = invoke(stypy.reporting.localization.Localization(__file__, 26, 6), unwrap_997, *[phase_998], **kwargs_999)

# Assigning a type to the variable 'r11' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'r11', unwrap_call_result_1000)

# Assigning a Call to a Name (line 28):

# Call to deg2rad(...): (line 28)
# Processing the call arguments (line 28)
# Getting the type of 'x' (line 28)
x_1003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'x', False)
# Processing the call keyword arguments (line 28)
kwargs_1004 = {}
# Getting the type of 'np' (line 28)
np_1001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 6), 'np', False)
# Obtaining the member 'deg2rad' of a type (line 28)
deg2rad_1002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 6), np_1001, 'deg2rad')
# Calling deg2rad(args, kwargs) (line 28)
deg2rad_call_result_1005 = invoke(stypy.reporting.localization.Localization(__file__, 28, 6), deg2rad_1002, *[x_1003], **kwargs_1004)

# Assigning a type to the variable 'r12' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'r12', deg2rad_call_result_1005)

# Assigning a Call to a Name (line 29):

# Call to rad2deg(...): (line 29)
# Processing the call arguments (line 29)
# Getting the type of 'x' (line 29)
x_1008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'x', False)
# Processing the call keyword arguments (line 29)
kwargs_1009 = {}
# Getting the type of 'np' (line 29)
np_1006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 6), 'np', False)
# Obtaining the member 'rad2deg' of a type (line 29)
rad2deg_1007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 6), np_1006, 'rad2deg')
# Calling rad2deg(args, kwargs) (line 29)
rad2deg_call_result_1010 = invoke(stypy.reporting.localization.Localization(__file__, 29, 6), rad2deg_1007, *[x_1008], **kwargs_1009)

# Assigning a type to the variable 'r13' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'r13', rad2deg_call_result_1010)

# Assigning a List to a Name (line 31):

# Obtaining an instance of the builtin type 'list' (line 31)
list_1011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 31)
# Adding element type (line 31)
int_1012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 4), list_1011, int_1012)
# Adding element type (line 31)
int_1013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 4), list_1011, int_1013)
# Adding element type (line 31)
int_1014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 4), list_1011, int_1014)
# Adding element type (line 31)
int_1015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 4), list_1011, int_1015)

# Assigning a type to the variable 'x' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'x', list_1011)

# Assigning a List to a Name (line 32):

# Obtaining an instance of the builtin type 'list' (line 32)
list_1016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 6), 'list')
# Adding type elements to the builtin type 'list' instance (line 32)
# Adding element type (line 32)
float_1017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 7), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 6), list_1016, float_1017)
# Adding element type (line 32)
float_1018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 12), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 6), list_1016, float_1018)
# Adding element type (line 32)
float_1019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 17), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 6), list_1016, float_1019)
# Adding element type (line 32)
float_1020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 22), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 6), list_1016, float_1020)

# Assigning a type to the variable 'x10' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'x10', list_1016)

# Assigning a Call to a Name (line 34):

# Call to sin(...): (line 34)
# Processing the call arguments (line 34)
# Getting the type of 'x' (line 34)
x_1023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 13), 'x', False)
# Processing the call keyword arguments (line 34)
kwargs_1024 = {}
# Getting the type of 'np' (line 34)
np_1021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 6), 'np', False)
# Obtaining the member 'sin' of a type (line 34)
sin_1022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 6), np_1021, 'sin')
# Calling sin(args, kwargs) (line 34)
sin_call_result_1025 = invoke(stypy.reporting.localization.Localization(__file__, 34, 6), sin_1022, *[x_1023], **kwargs_1024)

# Assigning a type to the variable 'r14' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'r14', sin_call_result_1025)

# Assigning a Call to a Name (line 35):

# Call to cos(...): (line 35)
# Processing the call arguments (line 35)
# Getting the type of 'x' (line 35)
x_1028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 13), 'x', False)
# Processing the call keyword arguments (line 35)
kwargs_1029 = {}
# Getting the type of 'np' (line 35)
np_1026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 6), 'np', False)
# Obtaining the member 'cos' of a type (line 35)
cos_1027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 6), np_1026, 'cos')
# Calling cos(args, kwargs) (line 35)
cos_call_result_1030 = invoke(stypy.reporting.localization.Localization(__file__, 35, 6), cos_1027, *[x_1028], **kwargs_1029)

# Assigning a type to the variable 'r15' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'r15', cos_call_result_1030)

# Assigning a Call to a Name (line 36):

# Call to tan(...): (line 36)
# Processing the call arguments (line 36)
# Getting the type of 'x' (line 36)
x_1033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'x', False)
# Processing the call keyword arguments (line 36)
kwargs_1034 = {}
# Getting the type of 'np' (line 36)
np_1031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 6), 'np', False)
# Obtaining the member 'tan' of a type (line 36)
tan_1032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 6), np_1031, 'tan')
# Calling tan(args, kwargs) (line 36)
tan_call_result_1035 = invoke(stypy.reporting.localization.Localization(__file__, 36, 6), tan_1032, *[x_1033], **kwargs_1034)

# Assigning a type to the variable 'r16' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'r16', tan_call_result_1035)

# Assigning a Call to a Name (line 37):

# Call to arcsin(...): (line 37)
# Processing the call arguments (line 37)
# Getting the type of 'x10' (line 37)
x10_1038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'x10', False)
# Processing the call keyword arguments (line 37)
kwargs_1039 = {}
# Getting the type of 'np' (line 37)
np_1036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 6), 'np', False)
# Obtaining the member 'arcsin' of a type (line 37)
arcsin_1037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 6), np_1036, 'arcsin')
# Calling arcsin(args, kwargs) (line 37)
arcsin_call_result_1040 = invoke(stypy.reporting.localization.Localization(__file__, 37, 6), arcsin_1037, *[x10_1038], **kwargs_1039)

# Assigning a type to the variable 'r17' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'r17', arcsin_call_result_1040)

# Assigning a Call to a Name (line 38):

# Call to arccos(...): (line 38)
# Processing the call arguments (line 38)
# Getting the type of 'x10' (line 38)
x10_1043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'x10', False)
# Processing the call keyword arguments (line 38)
kwargs_1044 = {}
# Getting the type of 'np' (line 38)
np_1041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 6), 'np', False)
# Obtaining the member 'arccos' of a type (line 38)
arccos_1042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 6), np_1041, 'arccos')
# Calling arccos(args, kwargs) (line 38)
arccos_call_result_1045 = invoke(stypy.reporting.localization.Localization(__file__, 38, 6), arccos_1042, *[x10_1043], **kwargs_1044)

# Assigning a type to the variable 'r18' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'r18', arccos_call_result_1045)

# Assigning a Call to a Name (line 39):

# Call to arctan(...): (line 39)
# Processing the call arguments (line 39)
# Getting the type of 'x' (line 39)
x_1048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'x', False)
# Processing the call keyword arguments (line 39)
kwargs_1049 = {}
# Getting the type of 'np' (line 39)
np_1046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 6), 'np', False)
# Obtaining the member 'arctan' of a type (line 39)
arctan_1047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 6), np_1046, 'arctan')
# Calling arctan(args, kwargs) (line 39)
arctan_call_result_1050 = invoke(stypy.reporting.localization.Localization(__file__, 39, 6), arctan_1047, *[x_1048], **kwargs_1049)

# Assigning a type to the variable 'r19' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'r19', arctan_call_result_1050)

# Assigning a Call to a Name (line 41):

# Call to degrees(...): (line 41)
# Processing the call arguments (line 41)
# Getting the type of 'x' (line 41)
x_1053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'x', False)
# Processing the call keyword arguments (line 41)
kwargs_1054 = {}
# Getting the type of 'np' (line 41)
np_1051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 6), 'np', False)
# Obtaining the member 'degrees' of a type (line 41)
degrees_1052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 6), np_1051, 'degrees')
# Calling degrees(args, kwargs) (line 41)
degrees_call_result_1055 = invoke(stypy.reporting.localization.Localization(__file__, 41, 6), degrees_1052, *[x_1053], **kwargs_1054)

# Assigning a type to the variable 'r20' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'r20', degrees_call_result_1055)

# Assigning a Call to a Name (line 42):

# Call to radians(...): (line 42)
# Processing the call arguments (line 42)
# Getting the type of 'x' (line 42)
x_1058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'x', False)
# Processing the call keyword arguments (line 42)
kwargs_1059 = {}
# Getting the type of 'np' (line 42)
np_1056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 6), 'np', False)
# Obtaining the member 'radians' of a type (line 42)
radians_1057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 6), np_1056, 'radians')
# Calling radians(args, kwargs) (line 42)
radians_call_result_1060 = invoke(stypy.reporting.localization.Localization(__file__, 42, 6), radians_1057, *[x_1058], **kwargs_1059)

# Assigning a type to the variable 'r21' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'r21', radians_call_result_1060)

# Assigning a Call to a Name (line 44):

# Call to deg2rad(...): (line 44)
# Processing the call arguments (line 44)
# Getting the type of 'x' (line 44)
x_1063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 17), 'x', False)
# Processing the call keyword arguments (line 44)
kwargs_1064 = {}
# Getting the type of 'np' (line 44)
np_1061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 6), 'np', False)
# Obtaining the member 'deg2rad' of a type (line 44)
deg2rad_1062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 6), np_1061, 'deg2rad')
# Calling deg2rad(args, kwargs) (line 44)
deg2rad_call_result_1065 = invoke(stypy.reporting.localization.Localization(__file__, 44, 6), deg2rad_1062, *[x_1063], **kwargs_1064)

# Assigning a type to the variable 'r22' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'r22', deg2rad_call_result_1065)

# Assigning a Call to a Name (line 45):

# Call to rad2deg(...): (line 45)
# Processing the call arguments (line 45)
# Getting the type of 'x' (line 45)
x_1068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 17), 'x', False)
# Processing the call keyword arguments (line 45)
kwargs_1069 = {}
# Getting the type of 'np' (line 45)
np_1066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 6), 'np', False)
# Obtaining the member 'rad2deg' of a type (line 45)
rad2deg_1067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 6), np_1066, 'rad2deg')
# Calling rad2deg(args, kwargs) (line 45)
rad2deg_call_result_1070 = invoke(stypy.reporting.localization.Localization(__file__, 45, 6), rad2deg_1067, *[x_1068], **kwargs_1069)

# Assigning a type to the variable 'r23' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'r23', rad2deg_call_result_1070)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
