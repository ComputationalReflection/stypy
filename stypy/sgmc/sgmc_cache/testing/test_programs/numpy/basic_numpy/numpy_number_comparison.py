
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: r1 = 0 * np.nan
6: r2 = np.nan == np.nan
7: r3 = np.inf > np.nan
8: r4 = np.nan - np.nan
9: r5 = 0.3 == 3 * 0.1
10: 
11: Z = np.arange(11)
12: 
13: r6 = Z ** Z
14: r7 = 2 << Z >> 2
15: r8 = Z < - Z
16: r9 = 1j * Z
17: r10 = Z / 1 / 1
18: # Type error
19: #r11 = Z < Z > Z
20: 
21: r12 = np.array(0) // np.array(0)
22: 
23: r13 = np.array(0) // np.array(0.)
24: r14 = np.array(0) / np.array(0)
25: r15 = np.array(0) / np.array(0.)
26: 
27: # l = globals().copy()
28: # for v in l:
29: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
30: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a BinOp to a Name (line 5):
int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 5), 'int')
# Getting the type of 'np' (line 5)
np_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 9), 'np')
# Obtaining the member 'nan' of a type (line 5)
nan_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 9), np_4, 'nan')
# Applying the binary operator '*' (line 5)
result_mul_6 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 5), '*', int_3, nan_5)

# Assigning a type to the variable 'r1' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'r1', result_mul_6)

# Assigning a Compare to a Name (line 6):

# Getting the type of 'np' (line 6)
np_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'np')
# Obtaining the member 'nan' of a type (line 6)
nan_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 5), np_7, 'nan')
# Getting the type of 'np' (line 6)
np_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), 'np')
# Obtaining the member 'nan' of a type (line 6)
nan_10 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 15), np_9, 'nan')
# Applying the binary operator '==' (line 6)
result_eq_11 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 5), '==', nan_8, nan_10)

# Assigning a type to the variable 'r2' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r2', result_eq_11)

# Assigning a Compare to a Name (line 7):

# Getting the type of 'np' (line 7)
np_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'np')
# Obtaining the member 'inf' of a type (line 7)
inf_13 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 5), np_12, 'inf')
# Getting the type of 'np' (line 7)
np_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 14), 'np')
# Obtaining the member 'nan' of a type (line 7)
nan_15 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 14), np_14, 'nan')
# Applying the binary operator '>' (line 7)
result_gt_16 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 5), '>', inf_13, nan_15)

# Assigning a type to the variable 'r3' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r3', result_gt_16)

# Assigning a BinOp to a Name (line 8):
# Getting the type of 'np' (line 8)
np_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'np')
# Obtaining the member 'nan' of a type (line 8)
nan_18 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), np_17, 'nan')
# Getting the type of 'np' (line 8)
np_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 14), 'np')
# Obtaining the member 'nan' of a type (line 8)
nan_20 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 14), np_19, 'nan')
# Applying the binary operator '-' (line 8)
result_sub_21 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 5), '-', nan_18, nan_20)

# Assigning a type to the variable 'r4' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r4', result_sub_21)

# Assigning a Compare to a Name (line 9):

float_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 5), 'float')
int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 12), 'int')
float_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 16), 'float')
# Applying the binary operator '*' (line 9)
result_mul_25 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 12), '*', int_23, float_24)

# Applying the binary operator '==' (line 9)
result_eq_26 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 5), '==', float_22, result_mul_25)

# Assigning a type to the variable 'r5' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r5', result_eq_26)

# Assigning a Call to a Name (line 11):

# Call to arange(...): (line 11)
# Processing the call arguments (line 11)
int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 14), 'int')
# Processing the call keyword arguments (line 11)
kwargs_30 = {}
# Getting the type of 'np' (line 11)
np_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'np', False)
# Obtaining the member 'arange' of a type (line 11)
arange_28 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), np_27, 'arange')
# Calling arange(args, kwargs) (line 11)
arange_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), arange_28, *[int_29], **kwargs_30)

# Assigning a type to the variable 'Z' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'Z', arange_call_result_31)

# Assigning a BinOp to a Name (line 13):
# Getting the type of 'Z' (line 13)
Z_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'Z')
# Getting the type of 'Z' (line 13)
Z_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'Z')
# Applying the binary operator '**' (line 13)
result_pow_34 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 5), '**', Z_32, Z_33)

# Assigning a type to the variable 'r6' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r6', result_pow_34)

# Assigning a BinOp to a Name (line 14):
int_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 5), 'int')
# Getting the type of 'Z' (line 14)
Z_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'Z')
# Applying the binary operator '<<' (line 14)
result_lshift_37 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 5), '<<', int_35, Z_36)

int_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'int')
# Applying the binary operator '>>' (line 14)
result_rshift_39 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 12), '>>', result_lshift_37, int_38)

# Assigning a type to the variable 'r7' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r7', result_rshift_39)

# Assigning a Compare to a Name (line 15):

# Getting the type of 'Z' (line 15)
Z_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'Z')

# Getting the type of 'Z' (line 15)
Z_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'Z')
# Applying the 'usub' unary operator (line 15)
result___neg___42 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 9), 'usub', Z_41)

# Applying the binary operator '<' (line 15)
result_lt_43 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 5), '<', Z_40, result___neg___42)

# Assigning a type to the variable 'r8' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r8', result_lt_43)

# Assigning a BinOp to a Name (line 16):
complex_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 5), 'complex')
# Getting the type of 'Z' (line 16)
Z_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'Z')
# Applying the binary operator '*' (line 16)
result_mul_46 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 5), '*', complex_44, Z_45)

# Assigning a type to the variable 'r9' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'r9', result_mul_46)

# Assigning a BinOp to a Name (line 17):
# Getting the type of 'Z' (line 17)
Z_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 6), 'Z')
int_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 10), 'int')
# Applying the binary operator 'div' (line 17)
result_div_49 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 6), 'div', Z_47, int_48)

int_50 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'int')
# Applying the binary operator 'div' (line 17)
result_div_51 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 12), 'div', result_div_49, int_50)

# Assigning a type to the variable 'r10' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r10', result_div_51)

# Assigning a BinOp to a Name (line 21):

# Call to array(...): (line 21)
# Processing the call arguments (line 21)
int_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 15), 'int')
# Processing the call keyword arguments (line 21)
kwargs_55 = {}
# Getting the type of 'np' (line 21)
np_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 6), 'np', False)
# Obtaining the member 'array' of a type (line 21)
array_53 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 6), np_52, 'array')
# Calling array(args, kwargs) (line 21)
array_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 21, 6), array_53, *[int_54], **kwargs_55)


# Call to array(...): (line 21)
# Processing the call arguments (line 21)
int_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 30), 'int')
# Processing the call keyword arguments (line 21)
kwargs_60 = {}
# Getting the type of 'np' (line 21)
np_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 21), 'np', False)
# Obtaining the member 'array' of a type (line 21)
array_58 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 21), np_57, 'array')
# Calling array(args, kwargs) (line 21)
array_call_result_61 = invoke(stypy.reporting.localization.Localization(__file__, 21, 21), array_58, *[int_59], **kwargs_60)

# Applying the binary operator '//' (line 21)
result_floordiv_62 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 6), '//', array_call_result_56, array_call_result_61)

# Assigning a type to the variable 'r12' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'r12', result_floordiv_62)

# Assigning a BinOp to a Name (line 23):

# Call to array(...): (line 23)
# Processing the call arguments (line 23)
int_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'int')
# Processing the call keyword arguments (line 23)
kwargs_66 = {}
# Getting the type of 'np' (line 23)
np_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 6), 'np', False)
# Obtaining the member 'array' of a type (line 23)
array_64 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 6), np_63, 'array')
# Calling array(args, kwargs) (line 23)
array_call_result_67 = invoke(stypy.reporting.localization.Localization(__file__, 23, 6), array_64, *[int_65], **kwargs_66)


# Call to array(...): (line 23)
# Processing the call arguments (line 23)
float_70 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 30), 'float')
# Processing the call keyword arguments (line 23)
kwargs_71 = {}
# Getting the type of 'np' (line 23)
np_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 21), 'np', False)
# Obtaining the member 'array' of a type (line 23)
array_69 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 21), np_68, 'array')
# Calling array(args, kwargs) (line 23)
array_call_result_72 = invoke(stypy.reporting.localization.Localization(__file__, 23, 21), array_69, *[float_70], **kwargs_71)

# Applying the binary operator '//' (line 23)
result_floordiv_73 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 6), '//', array_call_result_67, array_call_result_72)

# Assigning a type to the variable 'r13' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'r13', result_floordiv_73)

# Assigning a BinOp to a Name (line 24):

# Call to array(...): (line 24)
# Processing the call arguments (line 24)
int_76 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 15), 'int')
# Processing the call keyword arguments (line 24)
kwargs_77 = {}
# Getting the type of 'np' (line 24)
np_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 6), 'np', False)
# Obtaining the member 'array' of a type (line 24)
array_75 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 6), np_74, 'array')
# Calling array(args, kwargs) (line 24)
array_call_result_78 = invoke(stypy.reporting.localization.Localization(__file__, 24, 6), array_75, *[int_76], **kwargs_77)


# Call to array(...): (line 24)
# Processing the call arguments (line 24)
int_81 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 29), 'int')
# Processing the call keyword arguments (line 24)
kwargs_82 = {}
# Getting the type of 'np' (line 24)
np_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'np', False)
# Obtaining the member 'array' of a type (line 24)
array_80 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 20), np_79, 'array')
# Calling array(args, kwargs) (line 24)
array_call_result_83 = invoke(stypy.reporting.localization.Localization(__file__, 24, 20), array_80, *[int_81], **kwargs_82)

# Applying the binary operator 'div' (line 24)
result_div_84 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 6), 'div', array_call_result_78, array_call_result_83)

# Assigning a type to the variable 'r14' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'r14', result_div_84)

# Assigning a BinOp to a Name (line 25):

# Call to array(...): (line 25)
# Processing the call arguments (line 25)
int_87 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'int')
# Processing the call keyword arguments (line 25)
kwargs_88 = {}
# Getting the type of 'np' (line 25)
np_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 6), 'np', False)
# Obtaining the member 'array' of a type (line 25)
array_86 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 6), np_85, 'array')
# Calling array(args, kwargs) (line 25)
array_call_result_89 = invoke(stypy.reporting.localization.Localization(__file__, 25, 6), array_86, *[int_87], **kwargs_88)


# Call to array(...): (line 25)
# Processing the call arguments (line 25)
float_92 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 29), 'float')
# Processing the call keyword arguments (line 25)
kwargs_93 = {}
# Getting the type of 'np' (line 25)
np_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'np', False)
# Obtaining the member 'array' of a type (line 25)
array_91 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 20), np_90, 'array')
# Calling array(args, kwargs) (line 25)
array_call_result_94 = invoke(stypy.reporting.localization.Localization(__file__, 25, 20), array_91, *[float_92], **kwargs_93)

# Applying the binary operator 'div' (line 25)
result_div_95 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 6), 'div', array_call_result_89, array_call_result_94)

# Assigning a type to the variable 'r15' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'r15', result_div_95)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
