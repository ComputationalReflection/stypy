
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://cs231n.github.io/python-numpy-tutorial/
2: 
3: import numpy as np
4: 
5: a = np.array([[1, 2], [3, 4], [5, 6]])
6: 
7: bool_idx = (a > 2)  # Find the elements of a that are bigger than 2;
8: # this returns a numpy array of Booleans of the same
9: # shape as a, where each slot of bool_idx tells
10: # whether that element of a is > 2.
11: 
12: r = bool_idx  # Prints "[[False False]
13: #          [ True  True]
14: #          [ True  True]]"
15: 
16: # We use boolean array indexing to construct a rank 1 array
17: # consisting of the elements of a corresponding to the True values
18: # of bool_idx
19: r2 = a[bool_idx]  # Prints "[3 4 5 6]"
20: 
21: # We can do all of the above in a single concise statement:
22: r3 = a[a > 2]  # Prints "[3 4 5 6]"
23: #
24: # l = globals().copy()
25: # for v in l:
26: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
27: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_1058 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1058) is not StypyTypeError):

    if (import_1058 != 'pyd_module'):
        __import__(import_1058)
        sys_modules_1059 = sys.modules[import_1058]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1059.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1058)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 5):

# Call to array(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_1062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_1063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_1064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_1063, int_1064)
# Adding element type (line 5)
int_1065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_1063, int_1065)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_1062, list_1063)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_1066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_1067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), list_1066, int_1067)
# Adding element type (line 5)
int_1068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), list_1066, int_1068)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_1062, list_1066)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_1069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 30), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_1070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 30), list_1069, int_1070)
# Adding element type (line 5)
int_1071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 30), list_1069, int_1071)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_1062, list_1069)

# Processing the call keyword arguments (line 5)
kwargs_1072 = {}
# Getting the type of 'np' (line 5)
np_1060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'array' of a type (line 5)
array_1061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_1060, 'array')
# Calling array(args, kwargs) (line 5)
array_call_result_1073 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), array_1061, *[list_1062], **kwargs_1072)

# Assigning a type to the variable 'a' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'a', array_call_result_1073)

# Assigning a Compare to a Name (line 7):

# Getting the type of 'a' (line 7)
a_1074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'a')
int_1075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 16), 'int')
# Applying the binary operator '>' (line 7)
result_gt_1076 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 12), '>', a_1074, int_1075)

# Assigning a type to the variable 'bool_idx' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'bool_idx', result_gt_1076)

# Assigning a Name to a Name (line 12):
# Getting the type of 'bool_idx' (line 12)
bool_idx_1077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'bool_idx')
# Assigning a type to the variable 'r' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r', bool_idx_1077)

# Assigning a Subscript to a Name (line 19):

# Obtaining the type of the subscript
# Getting the type of 'bool_idx' (line 19)
bool_idx_1078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 7), 'bool_idx')
# Getting the type of 'a' (line 19)
a_1079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'a')
# Obtaining the member '__getitem__' of a type (line 19)
getitem___1080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 5), a_1079, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 19)
subscript_call_result_1081 = invoke(stypy.reporting.localization.Localization(__file__, 19, 5), getitem___1080, bool_idx_1078)

# Assigning a type to the variable 'r2' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r2', subscript_call_result_1081)

# Assigning a Subscript to a Name (line 22):

# Obtaining the type of the subscript

# Getting the type of 'a' (line 22)
a_1082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 7), 'a')
int_1083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'int')
# Applying the binary operator '>' (line 22)
result_gt_1084 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 7), '>', a_1082, int_1083)

# Getting the type of 'a' (line 22)
a_1085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 5), 'a')
# Obtaining the member '__getitem__' of a type (line 22)
getitem___1086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 5), a_1085, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 22)
subscript_call_result_1087 = invoke(stypy.reporting.localization.Localization(__file__, 22, 5), getitem___1086, result_gt_1084)

# Assigning a type to the variable 'r3' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'r3', subscript_call_result_1087)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
