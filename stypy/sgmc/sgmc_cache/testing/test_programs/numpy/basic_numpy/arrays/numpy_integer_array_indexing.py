
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://cs231n.github.io/python-numpy-tutorial/
2: 
3: import numpy as np
4: 
5: a = np.array([[1, 2], [3, 4], [5, 6]])
6: 
7: # An example of integer array indexing.
8: # The returned array will have shape (3,) and
9: r = a[[0, 1, 2], [0, 1, 0]]  # Prints "[1 4 5]"
10: 
11: # The above example of integer array indexing is equivalent to this:
12: r2 = np.array([a[0, 0], a[1, 1], a[2, 0]])  # Prints "[1 4 5]"
13: 
14: # When using integer array indexing, you can reuse the same
15: # element from the source array:
16: r3 = a[[0, 0], [1, 1]]  # Prints "[2 2]"
17: 
18: # Equivalent to the previous integer array indexing example
19: r4 = np.array([a[0, 1], a[0, 1]])  # Prints "[2 2]"
20: 
21: # l = globals().copy()
22: # for v in l:
23: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_1088 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1088) is not StypyTypeError):

    if (import_1088 != 'pyd_module'):
        __import__(import_1088)
        sys_modules_1089 = sys.modules[import_1088]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1089.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1088)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 5):

# Call to array(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_1092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_1093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_1094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_1093, int_1094)
# Adding element type (line 5)
int_1095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_1093, int_1095)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_1092, list_1093)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_1096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_1097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), list_1096, int_1097)
# Adding element type (line 5)
int_1098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), list_1096, int_1098)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_1092, list_1096)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_1099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 30), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_1100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 30), list_1099, int_1100)
# Adding element type (line 5)
int_1101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 30), list_1099, int_1101)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_1092, list_1099)

# Processing the call keyword arguments (line 5)
kwargs_1102 = {}
# Getting the type of 'np' (line 5)
np_1090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'array' of a type (line 5)
array_1091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_1090, 'array')
# Calling array(args, kwargs) (line 5)
array_call_result_1103 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), array_1091, *[list_1092], **kwargs_1102)

# Assigning a type to the variable 'a' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'a', array_call_result_1103)

# Assigning a Subscript to a Name (line 9):

# Obtaining the type of the subscript

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_1104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 6), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_1105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 6), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
int_1106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 7), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 6), list_1105, int_1106)
# Adding element type (line 9)
int_1107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 6), list_1105, int_1107)
# Adding element type (line 9)
int_1108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 6), list_1105, int_1108)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 6), tuple_1104, list_1105)
# Adding element type (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_1109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
int_1110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), list_1109, int_1110)
# Adding element type (line 9)
int_1111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), list_1109, int_1111)
# Adding element type (line 9)
int_1112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), list_1109, int_1112)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 6), tuple_1104, list_1109)

# Getting the type of 'a' (line 9)
a_1113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'a')
# Obtaining the member '__getitem__' of a type (line 9)
getitem___1114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), a_1113, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 9)
subscript_call_result_1115 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), getitem___1114, tuple_1104)

# Assigning a type to the variable 'r' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r', subscript_call_result_1115)

# Assigning a Call to a Name (line 12):

# Call to array(...): (line 12)
# Processing the call arguments (line 12)

# Obtaining an instance of the builtin type 'list' (line 12)
list_1118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)

# Obtaining the type of the subscript

# Obtaining an instance of the builtin type 'tuple' (line 12)
tuple_1119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 12)
# Adding element type (line 12)
int_1120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 17), tuple_1119, int_1120)
# Adding element type (line 12)
int_1121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 17), tuple_1119, int_1121)

# Getting the type of 'a' (line 12)
a_1122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'a', False)
# Obtaining the member '__getitem__' of a type (line 12)
getitem___1123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 15), a_1122, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_1124 = invoke(stypy.reporting.localization.Localization(__file__, 12, 15), getitem___1123, tuple_1119)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), list_1118, subscript_call_result_1124)
# Adding element type (line 12)

# Obtaining the type of the subscript

# Obtaining an instance of the builtin type 'tuple' (line 12)
tuple_1125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 26), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 12)
# Adding element type (line 12)
int_1126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 26), tuple_1125, int_1126)
# Adding element type (line 12)
int_1127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 26), tuple_1125, int_1127)

# Getting the type of 'a' (line 12)
a_1128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 24), 'a', False)
# Obtaining the member '__getitem__' of a type (line 12)
getitem___1129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 24), a_1128, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_1130 = invoke(stypy.reporting.localization.Localization(__file__, 12, 24), getitem___1129, tuple_1125)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), list_1118, subscript_call_result_1130)
# Adding element type (line 12)

# Obtaining the type of the subscript

# Obtaining an instance of the builtin type 'tuple' (line 12)
tuple_1131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 12)
# Adding element type (line 12)
int_1132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 35), tuple_1131, int_1132)
# Adding element type (line 12)
int_1133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 38), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 35), tuple_1131, int_1133)

# Getting the type of 'a' (line 12)
a_1134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 33), 'a', False)
# Obtaining the member '__getitem__' of a type (line 12)
getitem___1135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 33), a_1134, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_1136 = invoke(stypy.reporting.localization.Localization(__file__, 12, 33), getitem___1135, tuple_1131)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), list_1118, subscript_call_result_1136)

# Processing the call keyword arguments (line 12)
kwargs_1137 = {}
# Getting the type of 'np' (line 12)
np_1116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'np', False)
# Obtaining the member 'array' of a type (line 12)
array_1117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), np_1116, 'array')
# Calling array(args, kwargs) (line 12)
array_call_result_1138 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), array_1117, *[list_1118], **kwargs_1137)

# Assigning a type to the variable 'r2' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r2', array_call_result_1138)

# Assigning a Subscript to a Name (line 16):

# Obtaining the type of the subscript

# Obtaining an instance of the builtin type 'tuple' (line 16)
tuple_1139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 7), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 16)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'list' (line 16)
list_1140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 7), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
int_1141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 7), list_1140, int_1141)
# Adding element type (line 16)
int_1142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 7), list_1140, int_1142)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 7), tuple_1139, list_1140)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'list' (line 16)
list_1143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
int_1144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 15), list_1143, int_1144)
# Adding element type (line 16)
int_1145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 15), list_1143, int_1145)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 7), tuple_1139, list_1143)

# Getting the type of 'a' (line 16)
a_1146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 5), 'a')
# Obtaining the member '__getitem__' of a type (line 16)
getitem___1147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 5), a_1146, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 16)
subscript_call_result_1148 = invoke(stypy.reporting.localization.Localization(__file__, 16, 5), getitem___1147, tuple_1139)

# Assigning a type to the variable 'r3' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'r3', subscript_call_result_1148)

# Assigning a Call to a Name (line 19):

# Call to array(...): (line 19)
# Processing the call arguments (line 19)

# Obtaining an instance of the builtin type 'list' (line 19)
list_1151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)

# Obtaining the type of the subscript

# Obtaining an instance of the builtin type 'tuple' (line 19)
tuple_1152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 19)
# Adding element type (line 19)
int_1153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 17), tuple_1152, int_1153)
# Adding element type (line 19)
int_1154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 17), tuple_1152, int_1154)

# Getting the type of 'a' (line 19)
a_1155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'a', False)
# Obtaining the member '__getitem__' of a type (line 19)
getitem___1156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 15), a_1155, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 19)
subscript_call_result_1157 = invoke(stypy.reporting.localization.Localization(__file__, 19, 15), getitem___1156, tuple_1152)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 14), list_1151, subscript_call_result_1157)
# Adding element type (line 19)

# Obtaining the type of the subscript

# Obtaining an instance of the builtin type 'tuple' (line 19)
tuple_1158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 19)
# Adding element type (line 19)
int_1159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 26), tuple_1158, int_1159)
# Adding element type (line 19)
int_1160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 26), tuple_1158, int_1160)

# Getting the type of 'a' (line 19)
a_1161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 24), 'a', False)
# Obtaining the member '__getitem__' of a type (line 19)
getitem___1162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 24), a_1161, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 19)
subscript_call_result_1163 = invoke(stypy.reporting.localization.Localization(__file__, 19, 24), getitem___1162, tuple_1158)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 14), list_1151, subscript_call_result_1163)

# Processing the call keyword arguments (line 19)
kwargs_1164 = {}
# Getting the type of 'np' (line 19)
np_1149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'np', False)
# Obtaining the member 'array' of a type (line 19)
array_1150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 5), np_1149, 'array')
# Calling array(args, kwargs) (line 19)
array_call_result_1165 = invoke(stypy.reporting.localization.Localization(__file__, 19, 5), array_1150, *[list_1151], **kwargs_1164)

# Assigning a type to the variable 'r4' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r4', array_call_result_1165)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
