
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://cs231n.github.io/python-numpy-tutorial/
2: 
3: import numpy as np
4: 
5: x = np.array([[1, 2], [3, 4]])
6: y = np.array([[5, 6], [7, 8]])
7: 
8: v = np.array([9, 10])
9: w = np.array([11, 12])
10: 
11: # Inner product of vectors; both produce 219
12: r = v.dot(w)
13: r2 = np.dot(v, w)
14: 
15: # Matrix / vector product; both produce the rank 1 array [29 67]
16: r3 = x.dot(v)
17: r4 = np.dot(x, v)
18: 
19: # Matrix / matrix product; both produce the rank 2 array
20: # [[19 22]
21: #  [43 50]]
22: r5 = x.dot(y)
23: r6 = np.dot(x, y)
24: 
25: # l = globals().copy()
26: # for v in l:
27: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
28: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_853 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_853) is not StypyTypeError):

    if (import_853 != 'pyd_module'):
        __import__(import_853)
        sys_modules_854 = sys.modules[import_853]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_854.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_853)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 5):

# Call to array(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_858, int_859)
# Adding element type (line 5)
int_860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_858, int_860)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_857, list_858)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), list_861, int_862)
# Adding element type (line 5)
int_863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), list_861, int_863)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_857, list_861)

# Processing the call keyword arguments (line 5)
kwargs_864 = {}
# Getting the type of 'np' (line 5)
np_855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'array' of a type (line 5)
array_856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_855, 'array')
# Calling array(args, kwargs) (line 5)
array_call_result_865 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), array_856, *[list_857], **kwargs_864)

# Assigning a type to the variable 'x' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'x', array_call_result_865)

# Assigning a Call to a Name (line 6):

# Call to array(...): (line 6)
# Processing the call arguments (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
int_870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), list_869, int_870)
# Adding element type (line 6)
int_871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), list_869, int_871)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), list_868, list_869)
# Adding element type (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
int_873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 22), list_872, int_873)
# Adding element type (line 6)
int_874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 22), list_872, int_874)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), list_868, list_872)

# Processing the call keyword arguments (line 6)
kwargs_875 = {}
# Getting the type of 'np' (line 6)
np_866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'array' of a type (line 6)
array_867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_866, 'array')
# Calling array(args, kwargs) (line 6)
array_call_result_876 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), array_867, *[list_868], **kwargs_875)

# Assigning a type to the variable 'y' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'y', array_call_result_876)

# Assigning a Call to a Name (line 8):

# Call to array(...): (line 8)
# Processing the call arguments (line 8)

# Obtaining an instance of the builtin type 'list' (line 8)
list_879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
int_880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), list_879, int_880)
# Adding element type (line 8)
int_881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), list_879, int_881)

# Processing the call keyword arguments (line 8)
kwargs_882 = {}
# Getting the type of 'np' (line 8)
np_877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'np', False)
# Obtaining the member 'array' of a type (line 8)
array_878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), np_877, 'array')
# Calling array(args, kwargs) (line 8)
array_call_result_883 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), array_878, *[list_879], **kwargs_882)

# Assigning a type to the variable 'v' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'v', array_call_result_883)

# Assigning a Call to a Name (line 9):

# Call to array(...): (line 9)
# Processing the call arguments (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
int_887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_886, int_887)
# Adding element type (line 9)
int_888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_886, int_888)

# Processing the call keyword arguments (line 9)
kwargs_889 = {}
# Getting the type of 'np' (line 9)
np_884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'np', False)
# Obtaining the member 'array' of a type (line 9)
array_885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), np_884, 'array')
# Calling array(args, kwargs) (line 9)
array_call_result_890 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), array_885, *[list_886], **kwargs_889)

# Assigning a type to the variable 'w' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'w', array_call_result_890)

# Assigning a Call to a Name (line 12):

# Call to dot(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'w' (line 12)
w_893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'w', False)
# Processing the call keyword arguments (line 12)
kwargs_894 = {}
# Getting the type of 'v' (line 12)
v_891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'v', False)
# Obtaining the member 'dot' of a type (line 12)
dot_892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), v_891, 'dot')
# Calling dot(args, kwargs) (line 12)
dot_call_result_895 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), dot_892, *[w_893], **kwargs_894)

# Assigning a type to the variable 'r' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r', dot_call_result_895)

# Assigning a Call to a Name (line 13):

# Call to dot(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'v' (line 13)
v_898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'v', False)
# Getting the type of 'w' (line 13)
w_899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 15), 'w', False)
# Processing the call keyword arguments (line 13)
kwargs_900 = {}
# Getting the type of 'np' (line 13)
np_896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'np', False)
# Obtaining the member 'dot' of a type (line 13)
dot_897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), np_896, 'dot')
# Calling dot(args, kwargs) (line 13)
dot_call_result_901 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), dot_897, *[v_898, w_899], **kwargs_900)

# Assigning a type to the variable 'r2' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r2', dot_call_result_901)

# Assigning a Call to a Name (line 16):

# Call to dot(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of 'v' (line 16)
v_904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'v', False)
# Processing the call keyword arguments (line 16)
kwargs_905 = {}
# Getting the type of 'x' (line 16)
x_902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 5), 'x', False)
# Obtaining the member 'dot' of a type (line 16)
dot_903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 5), x_902, 'dot')
# Calling dot(args, kwargs) (line 16)
dot_call_result_906 = invoke(stypy.reporting.localization.Localization(__file__, 16, 5), dot_903, *[v_904], **kwargs_905)

# Assigning a type to the variable 'r3' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'r3', dot_call_result_906)

# Assigning a Call to a Name (line 17):

# Call to dot(...): (line 17)
# Processing the call arguments (line 17)
# Getting the type of 'x' (line 17)
x_909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'x', False)
# Getting the type of 'v' (line 17)
v_910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'v', False)
# Processing the call keyword arguments (line 17)
kwargs_911 = {}
# Getting the type of 'np' (line 17)
np_907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'np', False)
# Obtaining the member 'dot' of a type (line 17)
dot_908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), np_907, 'dot')
# Calling dot(args, kwargs) (line 17)
dot_call_result_912 = invoke(stypy.reporting.localization.Localization(__file__, 17, 5), dot_908, *[x_909, v_910], **kwargs_911)

# Assigning a type to the variable 'r4' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r4', dot_call_result_912)

# Assigning a Call to a Name (line 22):

# Call to dot(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of 'y' (line 22)
y_915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'y', False)
# Processing the call keyword arguments (line 22)
kwargs_916 = {}
# Getting the type of 'x' (line 22)
x_913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 5), 'x', False)
# Obtaining the member 'dot' of a type (line 22)
dot_914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 5), x_913, 'dot')
# Calling dot(args, kwargs) (line 22)
dot_call_result_917 = invoke(stypy.reporting.localization.Localization(__file__, 22, 5), dot_914, *[y_915], **kwargs_916)

# Assigning a type to the variable 'r5' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'r5', dot_call_result_917)

# Assigning a Call to a Name (line 23):

# Call to dot(...): (line 23)
# Processing the call arguments (line 23)
# Getting the type of 'x' (line 23)
x_920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'x', False)
# Getting the type of 'y' (line 23)
y_921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'y', False)
# Processing the call keyword arguments (line 23)
kwargs_922 = {}
# Getting the type of 'np' (line 23)
np_918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 5), 'np', False)
# Obtaining the member 'dot' of a type (line 23)
dot_919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 5), np_918, 'dot')
# Calling dot(args, kwargs) (line 23)
dot_call_result_923 = invoke(stypy.reporting.localization.Localization(__file__, 23, 5), dot_919, *[x_920, y_921], **kwargs_922)

# Assigning a type to the variable 'r6' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'r6', dot_call_result_923)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
