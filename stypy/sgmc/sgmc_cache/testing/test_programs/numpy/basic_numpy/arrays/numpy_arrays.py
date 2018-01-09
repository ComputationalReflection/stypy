
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://cs231n.github.io/python-numpy-tutorial/
2: 
3: import numpy as np
4: 
5: a = np.array([1, 2, 3])  # Create a rank 1 array
6: r = type(a)  # Prints "<type 'numpy.ndarray'>"
7: r2 = a.shape  # Prints "(3,)"
8: r3 = a[0], a[1], a[2]  # Prints "1 2 3"
9: a[0] = 5  # Change an element of the array
10: r4 = a  # Prints "[5, 2, 3]"
11: 
12: b = np.array([[1, 2, 3], [4, 5, 6]])  # Create a rank 2 array
13: r5 = b.shape  # Prints "(2, 3)"
14: r6 = b[0, 0], b[0, 1], b[1, 0]  # Prints "1 2 4"
15: 
16: # l = globals().copy()
17: # for v in l:
18: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_991 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_991) is not StypyTypeError):

    if (import_991 != 'pyd_module'):
        __import__(import_991)
        sys_modules_992 = sys.modules[import_991]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_992.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_991)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 5):

# Call to array(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_995, int_996)
# Adding element type (line 5)
int_997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_995, int_997)
# Adding element type (line 5)
int_998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_995, int_998)

# Processing the call keyword arguments (line 5)
kwargs_999 = {}
# Getting the type of 'np' (line 5)
np_993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'array' of a type (line 5)
array_994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_993, 'array')
# Calling array(args, kwargs) (line 5)
array_call_result_1000 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), array_994, *[list_995], **kwargs_999)

# Assigning a type to the variable 'a' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'a', array_call_result_1000)

# Assigning a Call to a Name (line 6):

# Call to type(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'a' (line 6)
a_1002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 9), 'a', False)
# Processing the call keyword arguments (line 6)
kwargs_1003 = {}
# Getting the type of 'type' (line 6)
type_1001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'type', False)
# Calling type(args, kwargs) (line 6)
type_call_result_1004 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), type_1001, *[a_1002], **kwargs_1003)

# Assigning a type to the variable 'r' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r', type_call_result_1004)

# Assigning a Attribute to a Name (line 7):
# Getting the type of 'a' (line 7)
a_1005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'a')
# Obtaining the member 'shape' of a type (line 7)
shape_1006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 5), a_1005, 'shape')
# Assigning a type to the variable 'r2' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r2', shape_1006)

# Assigning a Tuple to a Name (line 8):

# Obtaining an instance of the builtin type 'tuple' (line 8)
tuple_1007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 8)
# Adding element type (line 8)

# Obtaining the type of the subscript
int_1008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 7), 'int')
# Getting the type of 'a' (line 8)
a_1009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'a')
# Obtaining the member '__getitem__' of a type (line 8)
getitem___1010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), a_1009, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 8)
subscript_call_result_1011 = invoke(stypy.reporting.localization.Localization(__file__, 8, 5), getitem___1010, int_1008)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 5), tuple_1007, subscript_call_result_1011)
# Adding element type (line 8)

# Obtaining the type of the subscript
int_1012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 13), 'int')
# Getting the type of 'a' (line 8)
a_1013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 11), 'a')
# Obtaining the member '__getitem__' of a type (line 8)
getitem___1014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 11), a_1013, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 8)
subscript_call_result_1015 = invoke(stypy.reporting.localization.Localization(__file__, 8, 11), getitem___1014, int_1012)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 5), tuple_1007, subscript_call_result_1015)
# Adding element type (line 8)

# Obtaining the type of the subscript
int_1016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 19), 'int')
# Getting the type of 'a' (line 8)
a_1017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 17), 'a')
# Obtaining the member '__getitem__' of a type (line 8)
getitem___1018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 17), a_1017, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 8)
subscript_call_result_1019 = invoke(stypy.reporting.localization.Localization(__file__, 8, 17), getitem___1018, int_1016)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 5), tuple_1007, subscript_call_result_1019)

# Assigning a type to the variable 'r3' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r3', tuple_1007)

# Assigning a Num to a Subscript (line 9):
int_1020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 7), 'int')
# Getting the type of 'a' (line 9)
a_1021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'a')
int_1022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 2), 'int')
# Storing an element on a container (line 9)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 0), a_1021, (int_1022, int_1020))

# Assigning a Name to a Name (line 10):
# Getting the type of 'a' (line 10)
a_1023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'a')
# Assigning a type to the variable 'r4' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r4', a_1023)

# Assigning a Call to a Name (line 12):

# Call to array(...): (line 12)
# Processing the call arguments (line 12)

# Obtaining an instance of the builtin type 'list' (line 12)
list_1026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)

# Obtaining an instance of the builtin type 'list' (line 12)
list_1027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
int_1028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), list_1027, int_1028)
# Adding element type (line 12)
int_1029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), list_1027, int_1029)
# Adding element type (line 12)
int_1030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), list_1027, int_1030)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 13), list_1026, list_1027)
# Adding element type (line 12)

# Obtaining an instance of the builtin type 'list' (line 12)
list_1031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
int_1032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 25), list_1031, int_1032)
# Adding element type (line 12)
int_1033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 25), list_1031, int_1033)
# Adding element type (line 12)
int_1034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 25), list_1031, int_1034)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 13), list_1026, list_1031)

# Processing the call keyword arguments (line 12)
kwargs_1035 = {}
# Getting the type of 'np' (line 12)
np_1024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'np', False)
# Obtaining the member 'array' of a type (line 12)
array_1025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), np_1024, 'array')
# Calling array(args, kwargs) (line 12)
array_call_result_1036 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), array_1025, *[list_1026], **kwargs_1035)

# Assigning a type to the variable 'b' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'b', array_call_result_1036)

# Assigning a Attribute to a Name (line 13):
# Getting the type of 'b' (line 13)
b_1037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'b')
# Obtaining the member 'shape' of a type (line 13)
shape_1038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), b_1037, 'shape')
# Assigning a type to the variable 'r5' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r5', shape_1038)

# Assigning a Tuple to a Name (line 14):

# Obtaining an instance of the builtin type 'tuple' (line 14)
tuple_1039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 14)
# Adding element type (line 14)

# Obtaining the type of the subscript

# Obtaining an instance of the builtin type 'tuple' (line 14)
tuple_1040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 7), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 14)
# Adding element type (line 14)
int_1041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 7), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 7), tuple_1040, int_1041)
# Adding element type (line 14)
int_1042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 7), tuple_1040, int_1042)

# Getting the type of 'b' (line 14)
b_1043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'b')
# Obtaining the member '__getitem__' of a type (line 14)
getitem___1044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), b_1043, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 14)
subscript_call_result_1045 = invoke(stypy.reporting.localization.Localization(__file__, 14, 5), getitem___1044, tuple_1040)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 5), tuple_1039, subscript_call_result_1045)
# Adding element type (line 14)

# Obtaining the type of the subscript

# Obtaining an instance of the builtin type 'tuple' (line 14)
tuple_1046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 14)
# Adding element type (line 14)
int_1047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), tuple_1046, int_1047)
# Adding element type (line 14)
int_1048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 16), tuple_1046, int_1048)

# Getting the type of 'b' (line 14)
b_1049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'b')
# Obtaining the member '__getitem__' of a type (line 14)
getitem___1050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 14), b_1049, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 14)
subscript_call_result_1051 = invoke(stypy.reporting.localization.Localization(__file__, 14, 14), getitem___1050, tuple_1046)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 5), tuple_1039, subscript_call_result_1051)
# Adding element type (line 14)

# Obtaining the type of the subscript

# Obtaining an instance of the builtin type 'tuple' (line 14)
tuple_1052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 14)
# Adding element type (line 14)
int_1053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 25), tuple_1052, int_1053)
# Adding element type (line 14)
int_1054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 25), tuple_1052, int_1054)

# Getting the type of 'b' (line 14)
b_1055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 23), 'b')
# Obtaining the member '__getitem__' of a type (line 14)
getitem___1056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 23), b_1055, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 14)
subscript_call_result_1057 = invoke(stypy.reporting.localization.Localization(__file__, 14, 23), getitem___1056, tuple_1052)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 5), tuple_1039, subscript_call_result_1057)

# Assigning a type to the variable 'r6' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r6', tuple_1039)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
