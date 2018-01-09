
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.scipy-lectures.org/intro/numpy/numpy.html
2: 
3: import numpy as np
4: 
5: a = np.exp(2j * np.pi * np.arange(10))
6: fa = np.fft.fft(a)
7: r = np.set_printoptions(suppress=True)  # print small number as 0
8: 
9: a = np.exp(2j * np.pi * np.arange(3))
10: b = a[:, np.newaxis] + a[np.newaxis, :]
11: r2 = np.fft.fftn(b)
12: 
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
import_1064 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1064) is not StypyTypeError):

    if (import_1064 != 'pyd_module'):
        __import__(import_1064)
        sys_modules_1065 = sys.modules[import_1064]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1065.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1064)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to exp(...): (line 5)
# Processing the call arguments (line 5)
complex_1068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 11), 'complex')
# Getting the type of 'np' (line 5)
np_1069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 16), 'np', False)
# Obtaining the member 'pi' of a type (line 5)
pi_1070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 16), np_1069, 'pi')
# Applying the binary operator '*' (line 5)
result_mul_1071 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 11), '*', complex_1068, pi_1070)


# Call to arange(...): (line 5)
# Processing the call arguments (line 5)
int_1074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 34), 'int')
# Processing the call keyword arguments (line 5)
kwargs_1075 = {}
# Getting the type of 'np' (line 5)
np_1072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 24), 'np', False)
# Obtaining the member 'arange' of a type (line 5)
arange_1073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 24), np_1072, 'arange')
# Calling arange(args, kwargs) (line 5)
arange_call_result_1076 = invoke(stypy.reporting.localization.Localization(__file__, 5, 24), arange_1073, *[int_1074], **kwargs_1075)

# Applying the binary operator '*' (line 5)
result_mul_1077 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 22), '*', result_mul_1071, arange_call_result_1076)

# Processing the call keyword arguments (line 5)
kwargs_1078 = {}
# Getting the type of 'np' (line 5)
np_1066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'exp' of a type (line 5)
exp_1067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_1066, 'exp')
# Calling exp(args, kwargs) (line 5)
exp_call_result_1079 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), exp_1067, *[result_mul_1077], **kwargs_1078)

# Assigning a type to the variable 'a' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'a', exp_call_result_1079)

# Assigning a Call to a Name (line 6):

# Call to fft(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'a' (line 6)
a_1083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 16), 'a', False)
# Processing the call keyword arguments (line 6)
kwargs_1084 = {}
# Getting the type of 'np' (line 6)
np_1080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'np', False)
# Obtaining the member 'fft' of a type (line 6)
fft_1081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 5), np_1080, 'fft')
# Obtaining the member 'fft' of a type (line 6)
fft_1082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 5), fft_1081, 'fft')
# Calling fft(args, kwargs) (line 6)
fft_call_result_1085 = invoke(stypy.reporting.localization.Localization(__file__, 6, 5), fft_1082, *[a_1083], **kwargs_1084)

# Assigning a type to the variable 'fa' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'fa', fft_call_result_1085)

# Assigning a Call to a Name (line 7):

# Call to set_printoptions(...): (line 7)
# Processing the call keyword arguments (line 7)
# Getting the type of 'True' (line 7)
True_1088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 33), 'True', False)
keyword_1089 = True_1088
kwargs_1090 = {'suppress': keyword_1089}
# Getting the type of 'np' (line 7)
np_1086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'np', False)
# Obtaining the member 'set_printoptions' of a type (line 7)
set_printoptions_1087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), np_1086, 'set_printoptions')
# Calling set_printoptions(args, kwargs) (line 7)
set_printoptions_call_result_1091 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), set_printoptions_1087, *[], **kwargs_1090)

# Assigning a type to the variable 'r' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r', set_printoptions_call_result_1091)

# Assigning a Call to a Name (line 9):

# Call to exp(...): (line 9)
# Processing the call arguments (line 9)
complex_1094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'complex')
# Getting the type of 'np' (line 9)
np_1095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 16), 'np', False)
# Obtaining the member 'pi' of a type (line 9)
pi_1096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 16), np_1095, 'pi')
# Applying the binary operator '*' (line 9)
result_mul_1097 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 11), '*', complex_1094, pi_1096)


# Call to arange(...): (line 9)
# Processing the call arguments (line 9)
int_1100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 34), 'int')
# Processing the call keyword arguments (line 9)
kwargs_1101 = {}
# Getting the type of 'np' (line 9)
np_1098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 24), 'np', False)
# Obtaining the member 'arange' of a type (line 9)
arange_1099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 24), np_1098, 'arange')
# Calling arange(args, kwargs) (line 9)
arange_call_result_1102 = invoke(stypy.reporting.localization.Localization(__file__, 9, 24), arange_1099, *[int_1100], **kwargs_1101)

# Applying the binary operator '*' (line 9)
result_mul_1103 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 22), '*', result_mul_1097, arange_call_result_1102)

# Processing the call keyword arguments (line 9)
kwargs_1104 = {}
# Getting the type of 'np' (line 9)
np_1092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'np', False)
# Obtaining the member 'exp' of a type (line 9)
exp_1093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), np_1092, 'exp')
# Calling exp(args, kwargs) (line 9)
exp_call_result_1105 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), exp_1093, *[result_mul_1103], **kwargs_1104)

# Assigning a type to the variable 'a' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'a', exp_call_result_1105)

# Assigning a BinOp to a Name (line 10):

# Obtaining the type of the subscript
slice_1106 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 10, 4), None, None, None)
# Getting the type of 'np' (line 10)
np_1107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'np')
# Obtaining the member 'newaxis' of a type (line 10)
newaxis_1108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 9), np_1107, 'newaxis')
# Getting the type of 'a' (line 10)
a_1109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'a')
# Obtaining the member '__getitem__' of a type (line 10)
getitem___1110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), a_1109, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_1111 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), getitem___1110, (slice_1106, newaxis_1108))


# Obtaining the type of the subscript
# Getting the type of 'np' (line 10)
np_1112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 25), 'np')
# Obtaining the member 'newaxis' of a type (line 10)
newaxis_1113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 25), np_1112, 'newaxis')
slice_1114 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 10, 23), None, None, None)
# Getting the type of 'a' (line 10)
a_1115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 23), 'a')
# Obtaining the member '__getitem__' of a type (line 10)
getitem___1116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 23), a_1115, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 10)
subscript_call_result_1117 = invoke(stypy.reporting.localization.Localization(__file__, 10, 23), getitem___1116, (newaxis_1113, slice_1114))

# Applying the binary operator '+' (line 10)
result_add_1118 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 4), '+', subscript_call_result_1111, subscript_call_result_1117)

# Assigning a type to the variable 'b' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'b', result_add_1118)

# Assigning a Call to a Name (line 11):

# Call to fftn(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'b' (line 11)
b_1122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 17), 'b', False)
# Processing the call keyword arguments (line 11)
kwargs_1123 = {}
# Getting the type of 'np' (line 11)
np_1119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'np', False)
# Obtaining the member 'fft' of a type (line 11)
fft_1120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), np_1119, 'fft')
# Obtaining the member 'fftn' of a type (line 11)
fftn_1121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), fft_1120, 'fftn')
# Calling fftn(args, kwargs) (line 11)
fftn_call_result_1124 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), fftn_1121, *[b_1122], **kwargs_1123)

# Assigning a type to the variable 'r2' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r2', fftn_call_result_1124)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
