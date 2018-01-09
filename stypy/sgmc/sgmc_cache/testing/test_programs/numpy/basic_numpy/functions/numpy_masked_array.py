
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.scipy-lectures.org/intro/numpy/numpy.html
2: 
3: import numpy as np
4: 
5: x = np.array([1, 2, 3, -99, 5])
6: 
7: # One way to describe this is to create a masked array:
8: 
9: mx = np.ma.masked_array(x, mask=[0, 0, 0, 1, 0])
10: 
11: # Masked mean ignores masked data:
12: 
13: r = mx.mean()
14: 
15: mx[1] = 9
16: 
17: mx[1] = np.ma.masked
18: 
19: mx[1] = 9
20: 
21: # The mask is also available directly:
22: 
23: r2 = mx.mask
24: 
25: x2 = mx.filled(-1)
26: 
27: # The mask can also be cleared:
28: 
29: mx.mask = np.ma.nomask
30: 
31: r3 = mx
32: 
33: # Domain-aware functions
34: 
35: # The masked array package also contains domain-aware functions:
36: 
37: r4 = np.ma.log(np.array([1, 2, -1, -2, 3, -5]))
38: 
39: # l = globals().copy()
40: # for v in l:
41: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
42: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
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

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to array(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_5, int_6)
# Adding element type (line 5)
int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_5, int_7)
# Adding element type (line 5)
int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_5, int_8)
# Adding element type (line 5)
int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_5, int_9)
# Adding element type (line 5)
int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_5, int_10)

# Processing the call keyword arguments (line 5)
kwargs_11 = {}
# Getting the type of 'np' (line 5)
np_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'array' of a type (line 5)
array_4 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_3, 'array')
# Calling array(args, kwargs) (line 5)
array_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), array_4, *[list_5], **kwargs_11)

# Assigning a type to the variable 'x' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'x', array_call_result_12)

# Assigning a Call to a Name (line 9):

# Call to masked_array(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'x' (line 9)
x_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 24), 'x', False)
# Processing the call keyword arguments (line 9)

# Obtaining an instance of the builtin type 'list' (line 9)
list_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)
# Adding element type (line 9)
int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 33), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 32), list_17, int_18)
# Adding element type (line 9)
int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 32), list_17, int_19)
# Adding element type (line 9)
int_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 32), list_17, int_20)
# Adding element type (line 9)
int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 42), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 32), list_17, int_21)
# Adding element type (line 9)
int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 45), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 32), list_17, int_22)

keyword_23 = list_17
kwargs_24 = {'mask': keyword_23}
# Getting the type of 'np' (line 9)
np_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'np', False)
# Obtaining the member 'ma' of a type (line 9)
ma_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), np_13, 'ma')
# Obtaining the member 'masked_array' of a type (line 9)
masked_array_15 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), ma_14, 'masked_array')
# Calling masked_array(args, kwargs) (line 9)
masked_array_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), masked_array_15, *[x_16], **kwargs_24)

# Assigning a type to the variable 'mx' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'mx', masked_array_call_result_25)

# Assigning a Call to a Name (line 13):

# Call to mean(...): (line 13)
# Processing the call keyword arguments (line 13)
kwargs_28 = {}
# Getting the type of 'mx' (line 13)
mx_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'mx', False)
# Obtaining the member 'mean' of a type (line 13)
mean_27 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), mx_26, 'mean')
# Calling mean(args, kwargs) (line 13)
mean_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), mean_27, *[], **kwargs_28)

# Assigning a type to the variable 'r' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r', mean_call_result_29)

# Assigning a Num to a Subscript (line 15):
int_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 8), 'int')
# Getting the type of 'mx' (line 15)
mx_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'mx')
int_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 3), 'int')
# Storing an element on a container (line 15)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 0), mx_31, (int_32, int_30))

# Assigning a Attribute to a Subscript (line 17):
# Getting the type of 'np' (line 17)
np_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'np')
# Obtaining the member 'ma' of a type (line 17)
ma_34 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), np_33, 'ma')
# Obtaining the member 'masked' of a type (line 17)
masked_35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), ma_34, 'masked')
# Getting the type of 'mx' (line 17)
mx_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'mx')
int_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 3), 'int')
# Storing an element on a container (line 17)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 0), mx_36, (int_37, masked_35))

# Assigning a Num to a Subscript (line 19):
int_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 8), 'int')
# Getting the type of 'mx' (line 19)
mx_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'mx')
int_40 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 3), 'int')
# Storing an element on a container (line 19)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 0), mx_39, (int_40, int_38))

# Assigning a Attribute to a Name (line 23):
# Getting the type of 'mx' (line 23)
mx_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 5), 'mx')
# Obtaining the member 'mask' of a type (line 23)
mask_42 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 5), mx_41, 'mask')
# Assigning a type to the variable 'r2' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'r2', mask_42)

# Assigning a Call to a Name (line 25):

# Call to filled(...): (line 25)
# Processing the call arguments (line 25)
int_45 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'int')
# Processing the call keyword arguments (line 25)
kwargs_46 = {}
# Getting the type of 'mx' (line 25)
mx_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 5), 'mx', False)
# Obtaining the member 'filled' of a type (line 25)
filled_44 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 5), mx_43, 'filled')
# Calling filled(args, kwargs) (line 25)
filled_call_result_47 = invoke(stypy.reporting.localization.Localization(__file__, 25, 5), filled_44, *[int_45], **kwargs_46)

# Assigning a type to the variable 'x2' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'x2', filled_call_result_47)

# Assigning a Attribute to a Attribute (line 29):
# Getting the type of 'np' (line 29)
np_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'np')
# Obtaining the member 'ma' of a type (line 29)
ma_49 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 10), np_48, 'ma')
# Obtaining the member 'nomask' of a type (line 29)
nomask_50 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 10), ma_49, 'nomask')
# Getting the type of 'mx' (line 29)
mx_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'mx')
# Setting the type of the member 'mask' of a type (line 29)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 0), mx_51, 'mask', nomask_50)

# Assigning a Name to a Name (line 31):
# Getting the type of 'mx' (line 31)
mx_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 5), 'mx')
# Assigning a type to the variable 'r3' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'r3', mx_52)

# Assigning a Call to a Name (line 37):

# Call to log(...): (line 37)
# Processing the call arguments (line 37)

# Call to array(...): (line 37)
# Processing the call arguments (line 37)

# Obtaining an instance of the builtin type 'list' (line 37)
list_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 24), 'list')
# Adding type elements to the builtin type 'list' instance (line 37)
# Adding element type (line 37)
int_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 24), list_58, int_59)
# Adding element type (line 37)
int_60 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 24), list_58, int_60)
# Adding element type (line 37)
int_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 24), list_58, int_61)
# Adding element type (line 37)
int_62 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 24), list_58, int_62)
# Adding element type (line 37)
int_63 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 24), list_58, int_63)
# Adding element type (line 37)
int_64 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 42), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 24), list_58, int_64)

# Processing the call keyword arguments (line 37)
kwargs_65 = {}
# Getting the type of 'np' (line 37)
np_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'np', False)
# Obtaining the member 'array' of a type (line 37)
array_57 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 15), np_56, 'array')
# Calling array(args, kwargs) (line 37)
array_call_result_66 = invoke(stypy.reporting.localization.Localization(__file__, 37, 15), array_57, *[list_58], **kwargs_65)

# Processing the call keyword arguments (line 37)
kwargs_67 = {}
# Getting the type of 'np' (line 37)
np_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 5), 'np', False)
# Obtaining the member 'ma' of a type (line 37)
ma_54 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 5), np_53, 'ma')
# Obtaining the member 'log' of a type (line 37)
log_55 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 5), ma_54, 'log')
# Calling log(args, kwargs) (line 37)
log_call_result_68 = invoke(stypy.reporting.localization.Localization(__file__, 37, 5), log_55, *[array_call_result_66], **kwargs_67)

# Assigning a type to the variable 'r4' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'r4', log_call_result_68)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
