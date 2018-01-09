
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: __temp_call_assignment1 = range(1, 6).__getitem__(0)
4: Ident1 = __temp_call_assignment1
5: __temp_call_assignment2 = range(1, 6).__getitem__(1)
6: Ident2 = __temp_call_assignment2
7: __temp_call_assignment3 = range(1, 6).__getitem__(2)
8: Ident3 = __temp_call_assignment3
9: __temp_call_assignment4 = range(1, 6).__getitem__(3)
10: Ident4 = __temp_call_assignment4
11: __temp_call_assignment5 = range(1, 6).__getitem__(4)
12: Ident5 = __temp_call_assignment5
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Call to a Name (line 3):

# Call to __getitem__(...): (line 3)
# Processing the call arguments (line 3)
int_1044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 50), 'int')
# Processing the call keyword arguments (line 3)
kwargs_1045 = {}

# Call to range(...): (line 3)
# Processing the call arguments (line 3)
int_1039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 32), 'int')
int_1040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 35), 'int')
# Processing the call keyword arguments (line 3)
kwargs_1041 = {}
# Getting the type of 'range' (line 3)
range_1038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 26), 'range', False)
# Calling range(args, kwargs) (line 3)
range_call_result_1042 = invoke(stypy.reporting.localization.Localization(__file__, 3, 26), range_1038, *[int_1039, int_1040], **kwargs_1041)

# Obtaining the member '__getitem__' of a type (line 3)
getitem___1043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 3, 26), range_call_result_1042, '__getitem__')
# Calling __getitem__(args, kwargs) (line 3)
getitem___call_result_1046 = invoke(stypy.reporting.localization.Localization(__file__, 3, 26), getitem___1043, *[int_1044], **kwargs_1045)

# Assigning a type to the variable '__temp_call_assignment1' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__temp_call_assignment1', getitem___call_result_1046)

# Assigning a Name to a Name (line 4):
# Getting the type of '__temp_call_assignment1' (line 4)
temp_call_assignment1_1047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 9), '__temp_call_assignment1')
# Assigning a type to the variable 'Ident1' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'Ident1', temp_call_assignment1_1047)

# Assigning a Call to a Name (line 5):

# Call to __getitem__(...): (line 5)
# Processing the call arguments (line 5)
int_1054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 50), 'int')
# Processing the call keyword arguments (line 5)
kwargs_1055 = {}

# Call to range(...): (line 5)
# Processing the call arguments (line 5)
int_1049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 32), 'int')
int_1050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 35), 'int')
# Processing the call keyword arguments (line 5)
kwargs_1051 = {}
# Getting the type of 'range' (line 5)
range_1048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 26), 'range', False)
# Calling range(args, kwargs) (line 5)
range_call_result_1052 = invoke(stypy.reporting.localization.Localization(__file__, 5, 26), range_1048, *[int_1049, int_1050], **kwargs_1051)

# Obtaining the member '__getitem__' of a type (line 5)
getitem___1053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 26), range_call_result_1052, '__getitem__')
# Calling __getitem__(args, kwargs) (line 5)
getitem___call_result_1056 = invoke(stypy.reporting.localization.Localization(__file__, 5, 26), getitem___1053, *[int_1054], **kwargs_1055)

# Assigning a type to the variable '__temp_call_assignment2' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__temp_call_assignment2', getitem___call_result_1056)

# Assigning a Name to a Name (line 6):
# Getting the type of '__temp_call_assignment2' (line 6)
temp_call_assignment2_1057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 9), '__temp_call_assignment2')
# Assigning a type to the variable 'Ident2' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'Ident2', temp_call_assignment2_1057)

# Assigning a Call to a Name (line 7):

# Call to __getitem__(...): (line 7)
# Processing the call arguments (line 7)
int_1064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 50), 'int')
# Processing the call keyword arguments (line 7)
kwargs_1065 = {}

# Call to range(...): (line 7)
# Processing the call arguments (line 7)
int_1059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 32), 'int')
int_1060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 35), 'int')
# Processing the call keyword arguments (line 7)
kwargs_1061 = {}
# Getting the type of 'range' (line 7)
range_1058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 26), 'range', False)
# Calling range(args, kwargs) (line 7)
range_call_result_1062 = invoke(stypy.reporting.localization.Localization(__file__, 7, 26), range_1058, *[int_1059, int_1060], **kwargs_1061)

# Obtaining the member '__getitem__' of a type (line 7)
getitem___1063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 26), range_call_result_1062, '__getitem__')
# Calling __getitem__(args, kwargs) (line 7)
getitem___call_result_1066 = invoke(stypy.reporting.localization.Localization(__file__, 7, 26), getitem___1063, *[int_1064], **kwargs_1065)

# Assigning a type to the variable '__temp_call_assignment3' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__temp_call_assignment3', getitem___call_result_1066)

# Assigning a Name to a Name (line 8):
# Getting the type of '__temp_call_assignment3' (line 8)
temp_call_assignment3_1067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 9), '__temp_call_assignment3')
# Assigning a type to the variable 'Ident3' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'Ident3', temp_call_assignment3_1067)

# Assigning a Call to a Name (line 9):

# Call to __getitem__(...): (line 9)
# Processing the call arguments (line 9)
int_1074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 50), 'int')
# Processing the call keyword arguments (line 9)
kwargs_1075 = {}

# Call to range(...): (line 9)
# Processing the call arguments (line 9)
int_1069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 32), 'int')
int_1070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 35), 'int')
# Processing the call keyword arguments (line 9)
kwargs_1071 = {}
# Getting the type of 'range' (line 9)
range_1068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 26), 'range', False)
# Calling range(args, kwargs) (line 9)
range_call_result_1072 = invoke(stypy.reporting.localization.Localization(__file__, 9, 26), range_1068, *[int_1069, int_1070], **kwargs_1071)

# Obtaining the member '__getitem__' of a type (line 9)
getitem___1073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 26), range_call_result_1072, '__getitem__')
# Calling __getitem__(args, kwargs) (line 9)
getitem___call_result_1076 = invoke(stypy.reporting.localization.Localization(__file__, 9, 26), getitem___1073, *[int_1074], **kwargs_1075)

# Assigning a type to the variable '__temp_call_assignment4' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '__temp_call_assignment4', getitem___call_result_1076)

# Assigning a Name to a Name (line 10):
# Getting the type of '__temp_call_assignment4' (line 10)
temp_call_assignment4_1077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), '__temp_call_assignment4')
# Assigning a type to the variable 'Ident4' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'Ident4', temp_call_assignment4_1077)

# Assigning a Call to a Name (line 11):

# Call to __getitem__(...): (line 11)
# Processing the call arguments (line 11)
int_1084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 50), 'int')
# Processing the call keyword arguments (line 11)
kwargs_1085 = {}

# Call to range(...): (line 11)
# Processing the call arguments (line 11)
int_1079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 32), 'int')
int_1080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 35), 'int')
# Processing the call keyword arguments (line 11)
kwargs_1081 = {}
# Getting the type of 'range' (line 11)
range_1078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 26), 'range', False)
# Calling range(args, kwargs) (line 11)
range_call_result_1082 = invoke(stypy.reporting.localization.Localization(__file__, 11, 26), range_1078, *[int_1079, int_1080], **kwargs_1081)

# Obtaining the member '__getitem__' of a type (line 11)
getitem___1083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 26), range_call_result_1082, '__getitem__')
# Calling __getitem__(args, kwargs) (line 11)
getitem___call_result_1086 = invoke(stypy.reporting.localization.Localization(__file__, 11, 26), getitem___1083, *[int_1084], **kwargs_1085)

# Assigning a type to the variable '__temp_call_assignment5' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), '__temp_call_assignment5', getitem___call_result_1086)

# Assigning a Name to a Name (line 12):
# Getting the type of '__temp_call_assignment5' (line 12)
temp_call_assignment5_1087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), '__temp_call_assignment5')
# Assigning a type to the variable 'Ident5' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'Ident5', temp_call_assignment5_1087)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
