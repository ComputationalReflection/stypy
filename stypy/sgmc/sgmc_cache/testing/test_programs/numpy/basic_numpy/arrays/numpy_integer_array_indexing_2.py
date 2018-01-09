
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://cs231n.github.io/python-numpy-tutorial/
2: 
3: import numpy as np
4: 
5: # Create a new array from which we will select elements
6: a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
7: 
8: # Create an array of indices
9: b = np.array([0, 2, 0, 1])
10: 
11: # Select one element from each row of a using the indices in b
12: r = a[np.arange(4), b]  # Prints "[ 1  6  7 11]"
13: 
14: # Mutate one element from each row of a using the indices in b
15: a[np.arange(4), b] += 10
16: 
17: # l = globals().copy()
18: # for v in l:
19: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_1166 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1166) is not StypyTypeError):

    if (import_1166 != 'pyd_module'):
        __import__(import_1166)
        sys_modules_1167 = sys.modules[import_1166]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1167.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1166)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 6):

# Call to array(...): (line 6)
# Processing the call arguments (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_1170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_1171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
int_1172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), list_1171, int_1172)
# Adding element type (line 6)
int_1173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), list_1171, int_1173)
# Adding element type (line 6)
int_1174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), list_1171, int_1174)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), list_1170, list_1171)
# Adding element type (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_1175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
int_1176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 25), list_1175, int_1176)
# Adding element type (line 6)
int_1177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 25), list_1175, int_1177)
# Adding element type (line 6)
int_1178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 25), list_1175, int_1178)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), list_1170, list_1175)
# Adding element type (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_1179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 36), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
int_1180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 36), list_1179, int_1180)
# Adding element type (line 6)
int_1181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 36), list_1179, int_1181)
# Adding element type (line 6)
int_1182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 36), list_1179, int_1182)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), list_1170, list_1179)
# Adding element type (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_1183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 47), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
int_1184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 48), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 47), list_1183, int_1184)
# Adding element type (line 6)
int_1185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 52), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 47), list_1183, int_1185)
# Adding element type (line 6)
int_1186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 56), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 47), list_1183, int_1186)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), list_1170, list_1183)

# Processing the call keyword arguments (line 6)
kwargs_1187 = {}
# Getting the type of 'np' (line 6)
np_1168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'array' of a type (line 6)
array_1169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_1168, 'array')
# Calling array(args, kwargs) (line 6)
array_call_result_1188 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), array_1169, *[list_1170], **kwargs_1187)

# Assigning a type to the variable 'a' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'a', array_call_result_1188)

# Assigning a Call to a Name (line 9):

# Call to array(...): (line 9)
# Processing the call arguments (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_1191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
int_1192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_1191, int_1192)
# Adding element type (line 9)
int_1193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_1191, int_1193)
# Adding element type (line 9)
int_1194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_1191, int_1194)
# Adding element type (line 9)
int_1195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 13), list_1191, int_1195)

# Processing the call keyword arguments (line 9)
kwargs_1196 = {}
# Getting the type of 'np' (line 9)
np_1189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'np', False)
# Obtaining the member 'array' of a type (line 9)
array_1190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), np_1189, 'array')
# Calling array(args, kwargs) (line 9)
array_call_result_1197 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), array_1190, *[list_1191], **kwargs_1196)

# Assigning a type to the variable 'b' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'b', array_call_result_1197)

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript

# Obtaining an instance of the builtin type 'tuple' (line 12)
tuple_1198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 6), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 12)
# Adding element type (line 12)

# Call to arange(...): (line 12)
# Processing the call arguments (line 12)
int_1201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'int')
# Processing the call keyword arguments (line 12)
kwargs_1202 = {}
# Getting the type of 'np' (line 12)
np_1199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 6), 'np', False)
# Obtaining the member 'arange' of a type (line 12)
arange_1200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 6), np_1199, 'arange')
# Calling arange(args, kwargs) (line 12)
arange_call_result_1203 = invoke(stypy.reporting.localization.Localization(__file__, 12, 6), arange_1200, *[int_1201], **kwargs_1202)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 6), tuple_1198, arange_call_result_1203)
# Adding element type (line 12)
# Getting the type of 'b' (line 12)
b_1204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 20), 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 6), tuple_1198, b_1204)

# Getting the type of 'a' (line 12)
a_1205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'a')
# Obtaining the member '__getitem__' of a type (line 12)
getitem___1206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), a_1205, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_1207 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), getitem___1206, tuple_1198)

# Assigning a type to the variable 'r' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r', subscript_call_result_1207)

# Getting the type of 'a' (line 15)
a_1208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'a')

# Obtaining the type of the subscript

# Obtaining an instance of the builtin type 'tuple' (line 15)
tuple_1209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 2), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 15)
# Adding element type (line 15)

# Call to arange(...): (line 15)
# Processing the call arguments (line 15)
int_1212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'int')
# Processing the call keyword arguments (line 15)
kwargs_1213 = {}
# Getting the type of 'np' (line 15)
np_1210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 2), 'np', False)
# Obtaining the member 'arange' of a type (line 15)
arange_1211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 2), np_1210, 'arange')
# Calling arange(args, kwargs) (line 15)
arange_call_result_1214 = invoke(stypy.reporting.localization.Localization(__file__, 15, 2), arange_1211, *[int_1212], **kwargs_1213)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 2), tuple_1209, arange_call_result_1214)
# Adding element type (line 15)
# Getting the type of 'b' (line 15)
b_1215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 16), 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 2), tuple_1209, b_1215)

# Getting the type of 'a' (line 15)
a_1216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'a')
# Obtaining the member '__getitem__' of a type (line 15)
getitem___1217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 0), a_1216, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 15)
subscript_call_result_1218 = invoke(stypy.reporting.localization.Localization(__file__, 15, 0), getitem___1217, tuple_1209)

int_1219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 22), 'int')
# Applying the binary operator '+=' (line 15)
result_iadd_1220 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 0), '+=', subscript_call_result_1218, int_1219)
# Getting the type of 'a' (line 15)
a_1221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'a')

# Obtaining an instance of the builtin type 'tuple' (line 15)
tuple_1222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 2), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 15)
# Adding element type (line 15)

# Call to arange(...): (line 15)
# Processing the call arguments (line 15)
int_1225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'int')
# Processing the call keyword arguments (line 15)
kwargs_1226 = {}
# Getting the type of 'np' (line 15)
np_1223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 2), 'np', False)
# Obtaining the member 'arange' of a type (line 15)
arange_1224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 2), np_1223, 'arange')
# Calling arange(args, kwargs) (line 15)
arange_call_result_1227 = invoke(stypy.reporting.localization.Localization(__file__, 15, 2), arange_1224, *[int_1225], **kwargs_1226)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 2), tuple_1222, arange_call_result_1227)
# Adding element type (line 15)
# Getting the type of 'b' (line 15)
b_1228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 16), 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 2), tuple_1222, b_1228)

# Storing an element on a container (line 15)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 0), a_1221, (tuple_1222, result_iadd_1220))


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
