
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # https://docs.scipy.org/doc/numpy/reference/routines.math.html
2: 
3: import numpy as np
4: 
5: a = 2.1
6: 
7: # Sums, products, differences
8: r1 = np.prod(a)  # Return the product of array elements over a given axis.
9: r2 = np.sum(a)  # Sum of array elements over a given axis.
10: r3 = np.nansum(a)  # Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
11: r4 = np.cumprod(a)  # Return the cumulative product of elements along a given axis.
12: r5 = np.cumsum(a)  # Return the cumulative sum of the elements along a given axis.
13: # Type error
14: r6 = np.diff(a) 	#Calculate the n-th discrete difference along given axis.
15: r7 = np.ediff1d(a)  # The differences between consecutive elements of an array.
16: # Type error
17: r8 = np.gradient(a) 	#Return the gradient of an N-dimensional array.
18: # Type error
19: r9 = np.cross(a, a) 	#Return the cross product of two (arrays of) vectors.
20: # Type error
21: r10 = np.trapz(a) 	#Integrate along the given axis using the composite trapezoidal rule.
22: 
23: a = [1, 2, 3]
24: b = [8, 7, 5]
25: 
26: r11 = np.prod(a)  # Return the product of array elements over a given axis.
27: r12 = np.sum(a)  # Sum of array elements over a given axis.
28: r13 = np.nansum(a)  # Return the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.
29: r14 = np.cumprod(a)  # Return the cumulative product of elements along a given axis.
30: r15 = np.cumsum(a)  # Return the cumulative sum of the elements along a given axis.
31: r16 = np.diff(a)  # Calculate the n-th discrete difference along given axis.
32: r17 = np.ediff1d(a)  # The differences between consecutive elements of an array.
33: r18 = np.gradient(a)  # Return the gradient of an N-dimensional array.
34: r19 = np.cross(a, b)  # Return the cross product of two (arrays of) vectors.
35: r20 = np.trapz(a)  # Integrate along the given axis using the composite trapezoidal rule.
36: 
37: # l = globals().copy()
38: # for v in l:
39: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
40: 

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


# Assigning a Num to a Name (line 5):
float_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 4), 'float')
# Assigning a type to the variable 'a' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'a', float_3)

# Assigning a Call to a Name (line 8):

# Call to prod(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'a' (line 8)
a_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 13), 'a', False)
# Processing the call keyword arguments (line 8)
kwargs_7 = {}
# Getting the type of 'np' (line 8)
np_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'np', False)
# Obtaining the member 'prod' of a type (line 8)
prod_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), np_4, 'prod')
# Calling prod(args, kwargs) (line 8)
prod_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 8, 5), prod_5, *[a_6], **kwargs_7)

# Assigning a type to the variable 'r1' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r1', prod_call_result_8)

# Assigning a Call to a Name (line 9):

# Call to sum(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'a' (line 9)
a_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'a', False)
# Processing the call keyword arguments (line 9)
kwargs_12 = {}
# Getting the type of 'np' (line 9)
np_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'np', False)
# Obtaining the member 'sum' of a type (line 9)
sum_10 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), np_9, 'sum')
# Calling sum(args, kwargs) (line 9)
sum_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), sum_10, *[a_11], **kwargs_12)

# Assigning a type to the variable 'r2' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r2', sum_call_result_13)

# Assigning a Call to a Name (line 10):

# Call to nansum(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'a' (line 10)
a_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 15), 'a', False)
# Processing the call keyword arguments (line 10)
kwargs_17 = {}
# Getting the type of 'np' (line 10)
np_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'np', False)
# Obtaining the member 'nansum' of a type (line 10)
nansum_15 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), np_14, 'nansum')
# Calling nansum(args, kwargs) (line 10)
nansum_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), nansum_15, *[a_16], **kwargs_17)

# Assigning a type to the variable 'r3' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r3', nansum_call_result_18)

# Assigning a Call to a Name (line 11):

# Call to cumprod(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'a' (line 11)
a_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'a', False)
# Processing the call keyword arguments (line 11)
kwargs_22 = {}
# Getting the type of 'np' (line 11)
np_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'np', False)
# Obtaining the member 'cumprod' of a type (line 11)
cumprod_20 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), np_19, 'cumprod')
# Calling cumprod(args, kwargs) (line 11)
cumprod_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), cumprod_20, *[a_21], **kwargs_22)

# Assigning a type to the variable 'r4' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r4', cumprod_call_result_23)

# Assigning a Call to a Name (line 12):

# Call to cumsum(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'a' (line 12)
a_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'a', False)
# Processing the call keyword arguments (line 12)
kwargs_27 = {}
# Getting the type of 'np' (line 12)
np_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'np', False)
# Obtaining the member 'cumsum' of a type (line 12)
cumsum_25 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), np_24, 'cumsum')
# Calling cumsum(args, kwargs) (line 12)
cumsum_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), cumsum_25, *[a_26], **kwargs_27)

# Assigning a type to the variable 'r5' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r5', cumsum_call_result_28)

# Assigning a Call to a Name (line 14):

# Call to diff(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of 'a' (line 14)
a_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 13), 'a', False)
# Processing the call keyword arguments (line 14)
kwargs_32 = {}
# Getting the type of 'np' (line 14)
np_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'np', False)
# Obtaining the member 'diff' of a type (line 14)
diff_30 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), np_29, 'diff')
# Calling diff(args, kwargs) (line 14)
diff_call_result_33 = invoke(stypy.reporting.localization.Localization(__file__, 14, 5), diff_30, *[a_31], **kwargs_32)

# Assigning a type to the variable 'r6' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r6', diff_call_result_33)

# Assigning a Call to a Name (line 15):

# Call to ediff1d(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of 'a' (line 15)
a_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 16), 'a', False)
# Processing the call keyword arguments (line 15)
kwargs_37 = {}
# Getting the type of 'np' (line 15)
np_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'np', False)
# Obtaining the member 'ediff1d' of a type (line 15)
ediff1d_35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), np_34, 'ediff1d')
# Calling ediff1d(args, kwargs) (line 15)
ediff1d_call_result_38 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), ediff1d_35, *[a_36], **kwargs_37)

# Assigning a type to the variable 'r7' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r7', ediff1d_call_result_38)

# Assigning a Call to a Name (line 17):

# Call to gradient(...): (line 17)
# Processing the call arguments (line 17)
# Getting the type of 'a' (line 17)
a_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 17), 'a', False)
# Processing the call keyword arguments (line 17)
kwargs_42 = {}
# Getting the type of 'np' (line 17)
np_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'np', False)
# Obtaining the member 'gradient' of a type (line 17)
gradient_40 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), np_39, 'gradient')
# Calling gradient(args, kwargs) (line 17)
gradient_call_result_43 = invoke(stypy.reporting.localization.Localization(__file__, 17, 5), gradient_40, *[a_41], **kwargs_42)

# Assigning a type to the variable 'r8' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r8', gradient_call_result_43)

# Assigning a Call to a Name (line 19):

# Call to cross(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of 'a' (line 19)
a_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 14), 'a', False)
# Getting the type of 'a' (line 19)
a_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 17), 'a', False)
# Processing the call keyword arguments (line 19)
kwargs_48 = {}
# Getting the type of 'np' (line 19)
np_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'np', False)
# Obtaining the member 'cross' of a type (line 19)
cross_45 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 5), np_44, 'cross')
# Calling cross(args, kwargs) (line 19)
cross_call_result_49 = invoke(stypy.reporting.localization.Localization(__file__, 19, 5), cross_45, *[a_46, a_47], **kwargs_48)

# Assigning a type to the variable 'r9' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r9', cross_call_result_49)

# Assigning a Call to a Name (line 21):

# Call to trapz(...): (line 21)
# Processing the call arguments (line 21)
# Getting the type of 'a' (line 21)
a_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'a', False)
# Processing the call keyword arguments (line 21)
kwargs_53 = {}
# Getting the type of 'np' (line 21)
np_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 6), 'np', False)
# Obtaining the member 'trapz' of a type (line 21)
trapz_51 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 6), np_50, 'trapz')
# Calling trapz(args, kwargs) (line 21)
trapz_call_result_54 = invoke(stypy.reporting.localization.Localization(__file__, 21, 6), trapz_51, *[a_52], **kwargs_53)

# Assigning a type to the variable 'r10' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'r10', trapz_call_result_54)

# Assigning a List to a Name (line 23):

# Obtaining an instance of the builtin type 'list' (line 23)
list_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
int_56 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 4), list_55, int_56)
# Adding element type (line 23)
int_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 4), list_55, int_57)
# Adding element type (line 23)
int_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 4), list_55, int_58)

# Assigning a type to the variable 'a' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'a', list_55)

# Assigning a List to a Name (line 24):

# Obtaining an instance of the builtin type 'list' (line 24)
list_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)
int_60 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 4), list_59, int_60)
# Adding element type (line 24)
int_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 8), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 4), list_59, int_61)
# Adding element type (line 24)
int_62 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 11), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 4), list_59, int_62)

# Assigning a type to the variable 'b' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'b', list_59)

# Assigning a Call to a Name (line 26):

# Call to prod(...): (line 26)
# Processing the call arguments (line 26)
# Getting the type of 'a' (line 26)
a_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'a', False)
# Processing the call keyword arguments (line 26)
kwargs_66 = {}
# Getting the type of 'np' (line 26)
np_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 6), 'np', False)
# Obtaining the member 'prod' of a type (line 26)
prod_64 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 6), np_63, 'prod')
# Calling prod(args, kwargs) (line 26)
prod_call_result_67 = invoke(stypy.reporting.localization.Localization(__file__, 26, 6), prod_64, *[a_65], **kwargs_66)

# Assigning a type to the variable 'r11' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'r11', prod_call_result_67)

# Assigning a Call to a Name (line 27):

# Call to sum(...): (line 27)
# Processing the call arguments (line 27)
# Getting the type of 'a' (line 27)
a_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 13), 'a', False)
# Processing the call keyword arguments (line 27)
kwargs_71 = {}
# Getting the type of 'np' (line 27)
np_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 6), 'np', False)
# Obtaining the member 'sum' of a type (line 27)
sum_69 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 6), np_68, 'sum')
# Calling sum(args, kwargs) (line 27)
sum_call_result_72 = invoke(stypy.reporting.localization.Localization(__file__, 27, 6), sum_69, *[a_70], **kwargs_71)

# Assigning a type to the variable 'r12' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'r12', sum_call_result_72)

# Assigning a Call to a Name (line 28):

# Call to nansum(...): (line 28)
# Processing the call arguments (line 28)
# Getting the type of 'a' (line 28)
a_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'a', False)
# Processing the call keyword arguments (line 28)
kwargs_76 = {}
# Getting the type of 'np' (line 28)
np_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 6), 'np', False)
# Obtaining the member 'nansum' of a type (line 28)
nansum_74 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 6), np_73, 'nansum')
# Calling nansum(args, kwargs) (line 28)
nansum_call_result_77 = invoke(stypy.reporting.localization.Localization(__file__, 28, 6), nansum_74, *[a_75], **kwargs_76)

# Assigning a type to the variable 'r13' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'r13', nansum_call_result_77)

# Assigning a Call to a Name (line 29):

# Call to cumprod(...): (line 29)
# Processing the call arguments (line 29)
# Getting the type of 'a' (line 29)
a_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'a', False)
# Processing the call keyword arguments (line 29)
kwargs_81 = {}
# Getting the type of 'np' (line 29)
np_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 6), 'np', False)
# Obtaining the member 'cumprod' of a type (line 29)
cumprod_79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 6), np_78, 'cumprod')
# Calling cumprod(args, kwargs) (line 29)
cumprod_call_result_82 = invoke(stypy.reporting.localization.Localization(__file__, 29, 6), cumprod_79, *[a_80], **kwargs_81)

# Assigning a type to the variable 'r14' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'r14', cumprod_call_result_82)

# Assigning a Call to a Name (line 30):

# Call to cumsum(...): (line 30)
# Processing the call arguments (line 30)
# Getting the type of 'a' (line 30)
a_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'a', False)
# Processing the call keyword arguments (line 30)
kwargs_86 = {}
# Getting the type of 'np' (line 30)
np_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 6), 'np', False)
# Obtaining the member 'cumsum' of a type (line 30)
cumsum_84 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 6), np_83, 'cumsum')
# Calling cumsum(args, kwargs) (line 30)
cumsum_call_result_87 = invoke(stypy.reporting.localization.Localization(__file__, 30, 6), cumsum_84, *[a_85], **kwargs_86)

# Assigning a type to the variable 'r15' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'r15', cumsum_call_result_87)

# Assigning a Call to a Name (line 31):

# Call to diff(...): (line 31)
# Processing the call arguments (line 31)
# Getting the type of 'a' (line 31)
a_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'a', False)
# Processing the call keyword arguments (line 31)
kwargs_91 = {}
# Getting the type of 'np' (line 31)
np_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 6), 'np', False)
# Obtaining the member 'diff' of a type (line 31)
diff_89 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 6), np_88, 'diff')
# Calling diff(args, kwargs) (line 31)
diff_call_result_92 = invoke(stypy.reporting.localization.Localization(__file__, 31, 6), diff_89, *[a_90], **kwargs_91)

# Assigning a type to the variable 'r16' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'r16', diff_call_result_92)

# Assigning a Call to a Name (line 32):

# Call to ediff1d(...): (line 32)
# Processing the call arguments (line 32)
# Getting the type of 'a' (line 32)
a_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 17), 'a', False)
# Processing the call keyword arguments (line 32)
kwargs_96 = {}
# Getting the type of 'np' (line 32)
np_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 6), 'np', False)
# Obtaining the member 'ediff1d' of a type (line 32)
ediff1d_94 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 6), np_93, 'ediff1d')
# Calling ediff1d(args, kwargs) (line 32)
ediff1d_call_result_97 = invoke(stypy.reporting.localization.Localization(__file__, 32, 6), ediff1d_94, *[a_95], **kwargs_96)

# Assigning a type to the variable 'r17' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'r17', ediff1d_call_result_97)

# Assigning a Call to a Name (line 33):

# Call to gradient(...): (line 33)
# Processing the call arguments (line 33)
# Getting the type of 'a' (line 33)
a_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 18), 'a', False)
# Processing the call keyword arguments (line 33)
kwargs_101 = {}
# Getting the type of 'np' (line 33)
np_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 6), 'np', False)
# Obtaining the member 'gradient' of a type (line 33)
gradient_99 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 6), np_98, 'gradient')
# Calling gradient(args, kwargs) (line 33)
gradient_call_result_102 = invoke(stypy.reporting.localization.Localization(__file__, 33, 6), gradient_99, *[a_100], **kwargs_101)

# Assigning a type to the variable 'r18' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'r18', gradient_call_result_102)

# Assigning a Call to a Name (line 34):

# Call to cross(...): (line 34)
# Processing the call arguments (line 34)
# Getting the type of 'a' (line 34)
a_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'a', False)
# Getting the type of 'b' (line 34)
b_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'b', False)
# Processing the call keyword arguments (line 34)
kwargs_107 = {}
# Getting the type of 'np' (line 34)
np_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 6), 'np', False)
# Obtaining the member 'cross' of a type (line 34)
cross_104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 6), np_103, 'cross')
# Calling cross(args, kwargs) (line 34)
cross_call_result_108 = invoke(stypy.reporting.localization.Localization(__file__, 34, 6), cross_104, *[a_105, b_106], **kwargs_107)

# Assigning a type to the variable 'r19' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'r19', cross_call_result_108)

# Assigning a Call to a Name (line 35):

# Call to trapz(...): (line 35)
# Processing the call arguments (line 35)
# Getting the type of 'a' (line 35)
a_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'a', False)
# Processing the call keyword arguments (line 35)
kwargs_112 = {}
# Getting the type of 'np' (line 35)
np_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 6), 'np', False)
# Obtaining the member 'trapz' of a type (line 35)
trapz_110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 6), np_109, 'trapz')
# Calling trapz(args, kwargs) (line 35)
trapz_call_result_113 = invoke(stypy.reporting.localization.Localization(__file__, 35, 6), trapz_110, *[a_111], **kwargs_112)

# Assigning a type to the variable 'r20' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'r20', trapz_call_result_113)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
