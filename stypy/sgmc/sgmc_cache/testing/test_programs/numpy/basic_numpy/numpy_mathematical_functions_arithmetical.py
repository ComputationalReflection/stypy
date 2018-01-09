
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # https://docs.scipy.org/doc/numpy/reference/routines.math.html
2: 
3: 
4: import numpy as np
5: 
6: x1 = 2.1
7: x2 = 3.4
8: x = 5.6
9: 
10: # Arithmetic operations
11: r1 = np.add(x1, x2)  # Add arguments element-wise.
12: r2 = np.reciprocal(x)  # Return the reciprocal of the argument, element-wise.
13: r3 = np.negative(x)  # Numerical negative, element-wise.
14: r4 = np.multiply(x1, x2)  # Multiply arguments element-wise.
15: r5 = np.divide(x1, x2)  # Divide arguments element-wise.
16: r6 = np.power(x1, x2)  # First array elements raised to powers from second array, element-wise.
17: r7 = np.subtract(x1, x2)  # Subtract arguments, element-wise.
18: r8 = np.true_divide(x1, x2)  # Returns a true division of the inputs, element-wise.
19: r9 = np.floor_divide(x1, x2)  # Return the largest integer smaller or equal to the division of the inputs.
20: r10 = np.fmod(x1, x2)  # Return the element-wise remainder of division.
21: r11 = np.mod(x1, x2)  # Return element-wise remainder of division.
22: r12 = np.modf(x)  # Return the fractional and integral parts of an array, element-wise.
23: r13 = np.remainder(x1, x2)  # Return element-wise remainder of division.
24: 
25: x1 = [2.1, 2.2, 2.3]
26: x2 = [3.4, 5.6, 7.8]
27: x = [5.6, 6.5, 7.8]
28: 
29: r14 = np.add(x1, x2)  # Add arguments element-wise.
30: r15 = np.reciprocal(x)  # Return the reciprocal of the argument, element-wise.
31: r16 = np.negative(x)  # Numerical negative, element-wise.
32: r17 = np.multiply(x1, x2)  # Multiply arguments element-wise.
33: r18 = np.divide(x1, x2)  # Divide arguments element-wise.
34: r19 = np.power(x1, x2)  # First array elements raised to powers from second array, element-wise.
35: r20 = np.subtract(x1, x2)  # Subtract arguments, element-wise.
36: r21 = np.true_divide(x1, x2)  # Returns a true division of the inputs, element-wise.
37: r22 = np.floor_divide(x1, x2)  # Return the largest integer smaller or equal to the division of the inputs.
38: r23 = np.fmod(x1, x2)  # Return the element-wise remainder of division.
39: r24 = np.mod(x1, x2)  # Return element-wise remainder of division.
40: r25 = np.modf(x)  # Return the fractional and integral parts of an array, element-wise.
41: r26 = np.remainder(x1, x2)  # Return element-wise remainder of division.
42: 
43: # l = globals().copy()
44: # for v in l:
45: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_9 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_9) is not StypyTypeError):

    if (import_9 != 'pyd_module'):
        __import__(import_9)
        sys_modules_10 = sys.modules[import_9]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_10.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_9)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a Num to a Name (line 6):
float_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 5), 'float')
# Assigning a type to the variable 'x1' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'x1', float_11)

# Assigning a Num to a Name (line 7):
float_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 5), 'float')
# Assigning a type to the variable 'x2' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'x2', float_12)

# Assigning a Num to a Name (line 8):
float_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 4), 'float')
# Assigning a type to the variable 'x' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'x', float_13)

# Assigning a Call to a Name (line 11):

# Call to add(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'x1' (line 11)
x1_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'x1', False)
# Getting the type of 'x2' (line 11)
x2_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'x2', False)
# Processing the call keyword arguments (line 11)
kwargs_18 = {}
# Getting the type of 'np' (line 11)
np_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'np', False)
# Obtaining the member 'add' of a type (line 11)
add_15 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), np_14, 'add')
# Calling add(args, kwargs) (line 11)
add_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), add_15, *[x1_16, x2_17], **kwargs_18)

# Assigning a type to the variable 'r1' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r1', add_call_result_19)

# Assigning a Call to a Name (line 12):

# Call to reciprocal(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'x' (line 12)
x_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'x', False)
# Processing the call keyword arguments (line 12)
kwargs_23 = {}
# Getting the type of 'np' (line 12)
np_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'np', False)
# Obtaining the member 'reciprocal' of a type (line 12)
reciprocal_21 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), np_20, 'reciprocal')
# Calling reciprocal(args, kwargs) (line 12)
reciprocal_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), reciprocal_21, *[x_22], **kwargs_23)

# Assigning a type to the variable 'r2' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r2', reciprocal_call_result_24)

# Assigning a Call to a Name (line 13):

# Call to negative(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'x' (line 13)
x_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 17), 'x', False)
# Processing the call keyword arguments (line 13)
kwargs_28 = {}
# Getting the type of 'np' (line 13)
np_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'np', False)
# Obtaining the member 'negative' of a type (line 13)
negative_26 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), np_25, 'negative')
# Calling negative(args, kwargs) (line 13)
negative_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), negative_26, *[x_27], **kwargs_28)

# Assigning a type to the variable 'r3' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r3', negative_call_result_29)

# Assigning a Call to a Name (line 14):

# Call to multiply(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of 'x1' (line 14)
x1_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 17), 'x1', False)
# Getting the type of 'x2' (line 14)
x2_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 21), 'x2', False)
# Processing the call keyword arguments (line 14)
kwargs_34 = {}
# Getting the type of 'np' (line 14)
np_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'np', False)
# Obtaining the member 'multiply' of a type (line 14)
multiply_31 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), np_30, 'multiply')
# Calling multiply(args, kwargs) (line 14)
multiply_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 14, 5), multiply_31, *[x1_32, x2_33], **kwargs_34)

# Assigning a type to the variable 'r4' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r4', multiply_call_result_35)

# Assigning a Call to a Name (line 15):

# Call to divide(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of 'x1' (line 15)
x1_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'x1', False)
# Getting the type of 'x2' (line 15)
x2_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 19), 'x2', False)
# Processing the call keyword arguments (line 15)
kwargs_40 = {}
# Getting the type of 'np' (line 15)
np_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'np', False)
# Obtaining the member 'divide' of a type (line 15)
divide_37 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), np_36, 'divide')
# Calling divide(args, kwargs) (line 15)
divide_call_result_41 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), divide_37, *[x1_38, x2_39], **kwargs_40)

# Assigning a type to the variable 'r5' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r5', divide_call_result_41)

# Assigning a Call to a Name (line 16):

# Call to power(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of 'x1' (line 16)
x1_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'x1', False)
# Getting the type of 'x2' (line 16)
x2_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 18), 'x2', False)
# Processing the call keyword arguments (line 16)
kwargs_46 = {}
# Getting the type of 'np' (line 16)
np_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 5), 'np', False)
# Obtaining the member 'power' of a type (line 16)
power_43 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 5), np_42, 'power')
# Calling power(args, kwargs) (line 16)
power_call_result_47 = invoke(stypy.reporting.localization.Localization(__file__, 16, 5), power_43, *[x1_44, x2_45], **kwargs_46)

# Assigning a type to the variable 'r6' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'r6', power_call_result_47)

# Assigning a Call to a Name (line 17):

# Call to subtract(...): (line 17)
# Processing the call arguments (line 17)
# Getting the type of 'x1' (line 17)
x1_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 17), 'x1', False)
# Getting the type of 'x2' (line 17)
x2_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 21), 'x2', False)
# Processing the call keyword arguments (line 17)
kwargs_52 = {}
# Getting the type of 'np' (line 17)
np_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'np', False)
# Obtaining the member 'subtract' of a type (line 17)
subtract_49 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), np_48, 'subtract')
# Calling subtract(args, kwargs) (line 17)
subtract_call_result_53 = invoke(stypy.reporting.localization.Localization(__file__, 17, 5), subtract_49, *[x1_50, x2_51], **kwargs_52)

# Assigning a type to the variable 'r7' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r7', subtract_call_result_53)

# Assigning a Call to a Name (line 18):

# Call to true_divide(...): (line 18)
# Processing the call arguments (line 18)
# Getting the type of 'x1' (line 18)
x1_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'x1', False)
# Getting the type of 'x2' (line 18)
x2_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'x2', False)
# Processing the call keyword arguments (line 18)
kwargs_58 = {}
# Getting the type of 'np' (line 18)
np_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'np', False)
# Obtaining the member 'true_divide' of a type (line 18)
true_divide_55 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 5), np_54, 'true_divide')
# Calling true_divide(args, kwargs) (line 18)
true_divide_call_result_59 = invoke(stypy.reporting.localization.Localization(__file__, 18, 5), true_divide_55, *[x1_56, x2_57], **kwargs_58)

# Assigning a type to the variable 'r8' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'r8', true_divide_call_result_59)

# Assigning a Call to a Name (line 19):

# Call to floor_divide(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of 'x1' (line 19)
x1_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 21), 'x1', False)
# Getting the type of 'x2' (line 19)
x2_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), 'x2', False)
# Processing the call keyword arguments (line 19)
kwargs_64 = {}
# Getting the type of 'np' (line 19)
np_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'np', False)
# Obtaining the member 'floor_divide' of a type (line 19)
floor_divide_61 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 5), np_60, 'floor_divide')
# Calling floor_divide(args, kwargs) (line 19)
floor_divide_call_result_65 = invoke(stypy.reporting.localization.Localization(__file__, 19, 5), floor_divide_61, *[x1_62, x2_63], **kwargs_64)

# Assigning a type to the variable 'r9' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r9', floor_divide_call_result_65)

# Assigning a Call to a Name (line 20):

# Call to fmod(...): (line 20)
# Processing the call arguments (line 20)
# Getting the type of 'x1' (line 20)
x1_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 14), 'x1', False)
# Getting the type of 'x2' (line 20)
x2_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 18), 'x2', False)
# Processing the call keyword arguments (line 20)
kwargs_70 = {}
# Getting the type of 'np' (line 20)
np_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 6), 'np', False)
# Obtaining the member 'fmod' of a type (line 20)
fmod_67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 6), np_66, 'fmod')
# Calling fmod(args, kwargs) (line 20)
fmod_call_result_71 = invoke(stypy.reporting.localization.Localization(__file__, 20, 6), fmod_67, *[x1_68, x2_69], **kwargs_70)

# Assigning a type to the variable 'r10' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'r10', fmod_call_result_71)

# Assigning a Call to a Name (line 21):

# Call to mod(...): (line 21)
# Processing the call arguments (line 21)
# Getting the type of 'x1' (line 21)
x1_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 13), 'x1', False)
# Getting the type of 'x2' (line 21)
x2_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'x2', False)
# Processing the call keyword arguments (line 21)
kwargs_76 = {}
# Getting the type of 'np' (line 21)
np_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 6), 'np', False)
# Obtaining the member 'mod' of a type (line 21)
mod_73 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 6), np_72, 'mod')
# Calling mod(args, kwargs) (line 21)
mod_call_result_77 = invoke(stypy.reporting.localization.Localization(__file__, 21, 6), mod_73, *[x1_74, x2_75], **kwargs_76)

# Assigning a type to the variable 'r11' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'r11', mod_call_result_77)

# Assigning a Call to a Name (line 22):

# Call to modf(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of 'x' (line 22)
x_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 14), 'x', False)
# Processing the call keyword arguments (line 22)
kwargs_81 = {}
# Getting the type of 'np' (line 22)
np_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 6), 'np', False)
# Obtaining the member 'modf' of a type (line 22)
modf_79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 6), np_78, 'modf')
# Calling modf(args, kwargs) (line 22)
modf_call_result_82 = invoke(stypy.reporting.localization.Localization(__file__, 22, 6), modf_79, *[x_80], **kwargs_81)

# Assigning a type to the variable 'r12' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'r12', modf_call_result_82)

# Assigning a Call to a Name (line 23):

# Call to remainder(...): (line 23)
# Processing the call arguments (line 23)
# Getting the type of 'x1' (line 23)
x1_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'x1', False)
# Getting the type of 'x2' (line 23)
x2_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'x2', False)
# Processing the call keyword arguments (line 23)
kwargs_87 = {}
# Getting the type of 'np' (line 23)
np_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 6), 'np', False)
# Obtaining the member 'remainder' of a type (line 23)
remainder_84 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 6), np_83, 'remainder')
# Calling remainder(args, kwargs) (line 23)
remainder_call_result_88 = invoke(stypy.reporting.localization.Localization(__file__, 23, 6), remainder_84, *[x1_85, x2_86], **kwargs_87)

# Assigning a type to the variable 'r13' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'r13', remainder_call_result_88)

# Assigning a List to a Name (line 25):

# Obtaining an instance of the builtin type 'list' (line 25)
list_89 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
float_90 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 6), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 5), list_89, float_90)
# Adding element type (line 25)
float_91 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 5), list_89, float_91)
# Adding element type (line 25)
float_92 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 5), list_89, float_92)

# Assigning a type to the variable 'x1' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'x1', list_89)

# Assigning a List to a Name (line 26):

# Obtaining an instance of the builtin type 'list' (line 26)
list_93 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
float_94 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 6), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 5), list_93, float_94)
# Adding element type (line 26)
float_95 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 5), list_93, float_95)
# Adding element type (line 26)
float_96 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 5), list_93, float_96)

# Assigning a type to the variable 'x2' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'x2', list_93)

# Assigning a List to a Name (line 27):

# Obtaining an instance of the builtin type 'list' (line 27)
list_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 27)
# Adding element type (line 27)
float_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 5), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 4), list_97, float_98)
# Adding element type (line 27)
float_99 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 4), list_97, float_99)
# Adding element type (line 27)
float_100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 4), list_97, float_100)

# Assigning a type to the variable 'x' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'x', list_97)

# Assigning a Call to a Name (line 29):

# Call to add(...): (line 29)
# Processing the call arguments (line 29)
# Getting the type of 'x1' (line 29)
x1_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 13), 'x1', False)
# Getting the type of 'x2' (line 29)
x2_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'x2', False)
# Processing the call keyword arguments (line 29)
kwargs_105 = {}
# Getting the type of 'np' (line 29)
np_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 6), 'np', False)
# Obtaining the member 'add' of a type (line 29)
add_102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 6), np_101, 'add')
# Calling add(args, kwargs) (line 29)
add_call_result_106 = invoke(stypy.reporting.localization.Localization(__file__, 29, 6), add_102, *[x1_103, x2_104], **kwargs_105)

# Assigning a type to the variable 'r14' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'r14', add_call_result_106)

# Assigning a Call to a Name (line 30):

# Call to reciprocal(...): (line 30)
# Processing the call arguments (line 30)
# Getting the type of 'x' (line 30)
x_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 20), 'x', False)
# Processing the call keyword arguments (line 30)
kwargs_110 = {}
# Getting the type of 'np' (line 30)
np_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 6), 'np', False)
# Obtaining the member 'reciprocal' of a type (line 30)
reciprocal_108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 6), np_107, 'reciprocal')
# Calling reciprocal(args, kwargs) (line 30)
reciprocal_call_result_111 = invoke(stypy.reporting.localization.Localization(__file__, 30, 6), reciprocal_108, *[x_109], **kwargs_110)

# Assigning a type to the variable 'r15' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'r15', reciprocal_call_result_111)

# Assigning a Call to a Name (line 31):

# Call to negative(...): (line 31)
# Processing the call arguments (line 31)
# Getting the type of 'x' (line 31)
x_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'x', False)
# Processing the call keyword arguments (line 31)
kwargs_115 = {}
# Getting the type of 'np' (line 31)
np_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 6), 'np', False)
# Obtaining the member 'negative' of a type (line 31)
negative_113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 6), np_112, 'negative')
# Calling negative(args, kwargs) (line 31)
negative_call_result_116 = invoke(stypy.reporting.localization.Localization(__file__, 31, 6), negative_113, *[x_114], **kwargs_115)

# Assigning a type to the variable 'r16' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'r16', negative_call_result_116)

# Assigning a Call to a Name (line 32):

# Call to multiply(...): (line 32)
# Processing the call arguments (line 32)
# Getting the type of 'x1' (line 32)
x1_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 18), 'x1', False)
# Getting the type of 'x2' (line 32)
x2_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), 'x2', False)
# Processing the call keyword arguments (line 32)
kwargs_121 = {}
# Getting the type of 'np' (line 32)
np_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 6), 'np', False)
# Obtaining the member 'multiply' of a type (line 32)
multiply_118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 6), np_117, 'multiply')
# Calling multiply(args, kwargs) (line 32)
multiply_call_result_122 = invoke(stypy.reporting.localization.Localization(__file__, 32, 6), multiply_118, *[x1_119, x2_120], **kwargs_121)

# Assigning a type to the variable 'r17' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'r17', multiply_call_result_122)

# Assigning a Call to a Name (line 33):

# Call to divide(...): (line 33)
# Processing the call arguments (line 33)
# Getting the type of 'x1' (line 33)
x1_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'x1', False)
# Getting the type of 'x2' (line 33)
x2_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'x2', False)
# Processing the call keyword arguments (line 33)
kwargs_127 = {}
# Getting the type of 'np' (line 33)
np_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 6), 'np', False)
# Obtaining the member 'divide' of a type (line 33)
divide_124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 6), np_123, 'divide')
# Calling divide(args, kwargs) (line 33)
divide_call_result_128 = invoke(stypy.reporting.localization.Localization(__file__, 33, 6), divide_124, *[x1_125, x2_126], **kwargs_127)

# Assigning a type to the variable 'r18' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'r18', divide_call_result_128)

# Assigning a Call to a Name (line 34):

# Call to power(...): (line 34)
# Processing the call arguments (line 34)
# Getting the type of 'x1' (line 34)
x1_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'x1', False)
# Getting the type of 'x2' (line 34)
x2_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'x2', False)
# Processing the call keyword arguments (line 34)
kwargs_133 = {}
# Getting the type of 'np' (line 34)
np_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 6), 'np', False)
# Obtaining the member 'power' of a type (line 34)
power_130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 6), np_129, 'power')
# Calling power(args, kwargs) (line 34)
power_call_result_134 = invoke(stypy.reporting.localization.Localization(__file__, 34, 6), power_130, *[x1_131, x2_132], **kwargs_133)

# Assigning a type to the variable 'r19' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'r19', power_call_result_134)

# Assigning a Call to a Name (line 35):

# Call to subtract(...): (line 35)
# Processing the call arguments (line 35)
# Getting the type of 'x1' (line 35)
x1_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 18), 'x1', False)
# Getting the type of 'x2' (line 35)
x2_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 22), 'x2', False)
# Processing the call keyword arguments (line 35)
kwargs_139 = {}
# Getting the type of 'np' (line 35)
np_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 6), 'np', False)
# Obtaining the member 'subtract' of a type (line 35)
subtract_136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 6), np_135, 'subtract')
# Calling subtract(args, kwargs) (line 35)
subtract_call_result_140 = invoke(stypy.reporting.localization.Localization(__file__, 35, 6), subtract_136, *[x1_137, x2_138], **kwargs_139)

# Assigning a type to the variable 'r20' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'r20', subtract_call_result_140)

# Assigning a Call to a Name (line 36):

# Call to true_divide(...): (line 36)
# Processing the call arguments (line 36)
# Getting the type of 'x1' (line 36)
x1_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'x1', False)
# Getting the type of 'x2' (line 36)
x2_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 25), 'x2', False)
# Processing the call keyword arguments (line 36)
kwargs_145 = {}
# Getting the type of 'np' (line 36)
np_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 6), 'np', False)
# Obtaining the member 'true_divide' of a type (line 36)
true_divide_142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 6), np_141, 'true_divide')
# Calling true_divide(args, kwargs) (line 36)
true_divide_call_result_146 = invoke(stypy.reporting.localization.Localization(__file__, 36, 6), true_divide_142, *[x1_143, x2_144], **kwargs_145)

# Assigning a type to the variable 'r21' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'r21', true_divide_call_result_146)

# Assigning a Call to a Name (line 37):

# Call to floor_divide(...): (line 37)
# Processing the call arguments (line 37)
# Getting the type of 'x1' (line 37)
x1_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 22), 'x1', False)
# Getting the type of 'x2' (line 37)
x2_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 26), 'x2', False)
# Processing the call keyword arguments (line 37)
kwargs_151 = {}
# Getting the type of 'np' (line 37)
np_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 6), 'np', False)
# Obtaining the member 'floor_divide' of a type (line 37)
floor_divide_148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 6), np_147, 'floor_divide')
# Calling floor_divide(args, kwargs) (line 37)
floor_divide_call_result_152 = invoke(stypy.reporting.localization.Localization(__file__, 37, 6), floor_divide_148, *[x1_149, x2_150], **kwargs_151)

# Assigning a type to the variable 'r22' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'r22', floor_divide_call_result_152)

# Assigning a Call to a Name (line 38):

# Call to fmod(...): (line 38)
# Processing the call arguments (line 38)
# Getting the type of 'x1' (line 38)
x1_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 14), 'x1', False)
# Getting the type of 'x2' (line 38)
x2_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 18), 'x2', False)
# Processing the call keyword arguments (line 38)
kwargs_157 = {}
# Getting the type of 'np' (line 38)
np_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 6), 'np', False)
# Obtaining the member 'fmod' of a type (line 38)
fmod_154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 6), np_153, 'fmod')
# Calling fmod(args, kwargs) (line 38)
fmod_call_result_158 = invoke(stypy.reporting.localization.Localization(__file__, 38, 6), fmod_154, *[x1_155, x2_156], **kwargs_157)

# Assigning a type to the variable 'r23' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'r23', fmod_call_result_158)

# Assigning a Call to a Name (line 39):

# Call to mod(...): (line 39)
# Processing the call arguments (line 39)
# Getting the type of 'x1' (line 39)
x1_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 13), 'x1', False)
# Getting the type of 'x2' (line 39)
x2_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 'x2', False)
# Processing the call keyword arguments (line 39)
kwargs_163 = {}
# Getting the type of 'np' (line 39)
np_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 6), 'np', False)
# Obtaining the member 'mod' of a type (line 39)
mod_160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 6), np_159, 'mod')
# Calling mod(args, kwargs) (line 39)
mod_call_result_164 = invoke(stypy.reporting.localization.Localization(__file__, 39, 6), mod_160, *[x1_161, x2_162], **kwargs_163)

# Assigning a type to the variable 'r24' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'r24', mod_call_result_164)

# Assigning a Call to a Name (line 40):

# Call to modf(...): (line 40)
# Processing the call arguments (line 40)
# Getting the type of 'x' (line 40)
x_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 14), 'x', False)
# Processing the call keyword arguments (line 40)
kwargs_168 = {}
# Getting the type of 'np' (line 40)
np_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 6), 'np', False)
# Obtaining the member 'modf' of a type (line 40)
modf_166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 6), np_165, 'modf')
# Calling modf(args, kwargs) (line 40)
modf_call_result_169 = invoke(stypy.reporting.localization.Localization(__file__, 40, 6), modf_166, *[x_167], **kwargs_168)

# Assigning a type to the variable 'r25' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'r25', modf_call_result_169)

# Assigning a Call to a Name (line 41):

# Call to remainder(...): (line 41)
# Processing the call arguments (line 41)
# Getting the type of 'x1' (line 41)
x1_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 19), 'x1', False)
# Getting the type of 'x2' (line 41)
x2_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'x2', False)
# Processing the call keyword arguments (line 41)
kwargs_174 = {}
# Getting the type of 'np' (line 41)
np_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 6), 'np', False)
# Obtaining the member 'remainder' of a type (line 41)
remainder_171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 6), np_170, 'remainder')
# Calling remainder(args, kwargs) (line 41)
remainder_call_result_175 = invoke(stypy.reporting.localization.Localization(__file__, 41, 6), remainder_171, *[x1_172, x2_173], **kwargs_174)

# Assigning a type to the variable 'r26' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'r26', remainder_call_result_175)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
