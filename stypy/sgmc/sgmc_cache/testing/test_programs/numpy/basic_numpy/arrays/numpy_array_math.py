
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://cs231n.github.io/python-numpy-tutorial/
2: 
3: import numpy as np
4: 
5: x = np.array([[1, 2], [3, 4]], dtype=np.float64)
6: y = np.array([[5, 6], [7, 8]], dtype=np.float64)
7: 
8: # Elementwise sum; both produce the array
9: # [[ 6.0  8.0]
10: #  [10.0 12.0]]
11: r = x + y
12: r2 = np.add(x, y)
13: 
14: # Elementwise difference; both produce the array
15: # [[-4.0 -4.0]
16: #  [-4.0 -4.0]]
17: r3 = x - y
18: r4 = np.subtract(x, y)
19: 
20: # Elementwise product; both produce the array
21: # [[ 5.0 12.0]
22: #  [21.0 32.0]]
23: r5 = x * y
24: r6 = np.multiply(x, y)
25: 
26: # Elementwise division; both produce the array
27: # [[ 0.2         0.33333333]
28: #  [ 0.42857143  0.5       ]]
29: r7 = x / y
30: r8 = np.divide(x, y)
31: 
32: # Elementwise square root; produces the array
33: # [[ 1.          1.41421356]
34: #  [ 1.73205081  2.        ]]
35: r9 = np.sqrt(x)
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
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_718 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_718) is not StypyTypeError):

    if (import_718 != 'pyd_module'):
        __import__(import_718)
        sys_modules_719 = sys.modules[import_718]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_719.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_718)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 5):

# Call to array(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_723, int_724)
# Adding element type (line 5)
int_725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_723, int_725)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_722, list_723)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), list_726, int_727)
# Adding element type (line 5)
int_728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), list_726, int_728)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_722, list_726)

# Processing the call keyword arguments (line 5)
# Getting the type of 'np' (line 5)
np_729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 37), 'np', False)
# Obtaining the member 'float64' of a type (line 5)
float64_730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 37), np_729, 'float64')
keyword_731 = float64_730
kwargs_732 = {'dtype': keyword_731}
# Getting the type of 'np' (line 5)
np_720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'array' of a type (line 5)
array_721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_720, 'array')
# Calling array(args, kwargs) (line 5)
array_call_result_733 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), array_721, *[list_722], **kwargs_732)

# Assigning a type to the variable 'x' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'x', array_call_result_733)

# Assigning a Call to a Name (line 6):

# Call to array(...): (line 6)
# Processing the call arguments (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
int_738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), list_737, int_738)
# Adding element type (line 6)
int_739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 14), list_737, int_739)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), list_736, list_737)
# Adding element type (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
int_741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 22), list_740, int_741)
# Adding element type (line 6)
int_742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 22), list_740, int_742)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), list_736, list_740)

# Processing the call keyword arguments (line 6)
# Getting the type of 'np' (line 6)
np_743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 37), 'np', False)
# Obtaining the member 'float64' of a type (line 6)
float64_744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 37), np_743, 'float64')
keyword_745 = float64_744
kwargs_746 = {'dtype': keyword_745}
# Getting the type of 'np' (line 6)
np_734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'array' of a type (line 6)
array_735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_734, 'array')
# Calling array(args, kwargs) (line 6)
array_call_result_747 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), array_735, *[list_736], **kwargs_746)

# Assigning a type to the variable 'y' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'y', array_call_result_747)

# Assigning a BinOp to a Name (line 11):
# Getting the type of 'x' (line 11)
x_748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'x')
# Getting the type of 'y' (line 11)
y_749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'y')
# Applying the binary operator '+' (line 11)
result_add_750 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 4), '+', x_748, y_749)

# Assigning a type to the variable 'r' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r', result_add_750)

# Assigning a Call to a Name (line 12):

# Call to add(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'x' (line 12)
x_753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'x', False)
# Getting the type of 'y' (line 12)
y_754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'y', False)
# Processing the call keyword arguments (line 12)
kwargs_755 = {}
# Getting the type of 'np' (line 12)
np_751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'np', False)
# Obtaining the member 'add' of a type (line 12)
add_752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), np_751, 'add')
# Calling add(args, kwargs) (line 12)
add_call_result_756 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), add_752, *[x_753, y_754], **kwargs_755)

# Assigning a type to the variable 'r2' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r2', add_call_result_756)

# Assigning a BinOp to a Name (line 17):
# Getting the type of 'x' (line 17)
x_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'x')
# Getting the type of 'y' (line 17)
y_758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 9), 'y')
# Applying the binary operator '-' (line 17)
result_sub_759 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 5), '-', x_757, y_758)

# Assigning a type to the variable 'r3' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r3', result_sub_759)

# Assigning a Call to a Name (line 18):

# Call to subtract(...): (line 18)
# Processing the call arguments (line 18)
# Getting the type of 'x' (line 18)
x_762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'x', False)
# Getting the type of 'y' (line 18)
y_763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'y', False)
# Processing the call keyword arguments (line 18)
kwargs_764 = {}
# Getting the type of 'np' (line 18)
np_760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'np', False)
# Obtaining the member 'subtract' of a type (line 18)
subtract_761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 5), np_760, 'subtract')
# Calling subtract(args, kwargs) (line 18)
subtract_call_result_765 = invoke(stypy.reporting.localization.Localization(__file__, 18, 5), subtract_761, *[x_762, y_763], **kwargs_764)

# Assigning a type to the variable 'r4' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'r4', subtract_call_result_765)

# Assigning a BinOp to a Name (line 23):
# Getting the type of 'x' (line 23)
x_766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 5), 'x')
# Getting the type of 'y' (line 23)
y_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 9), 'y')
# Applying the binary operator '*' (line 23)
result_mul_768 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 5), '*', x_766, y_767)

# Assigning a type to the variable 'r5' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'r5', result_mul_768)

# Assigning a Call to a Name (line 24):

# Call to multiply(...): (line 24)
# Processing the call arguments (line 24)
# Getting the type of 'x' (line 24)
x_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 17), 'x', False)
# Getting the type of 'y' (line 24)
y_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'y', False)
# Processing the call keyword arguments (line 24)
kwargs_773 = {}
# Getting the type of 'np' (line 24)
np_769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 5), 'np', False)
# Obtaining the member 'multiply' of a type (line 24)
multiply_770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 5), np_769, 'multiply')
# Calling multiply(args, kwargs) (line 24)
multiply_call_result_774 = invoke(stypy.reporting.localization.Localization(__file__, 24, 5), multiply_770, *[x_771, y_772], **kwargs_773)

# Assigning a type to the variable 'r6' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'r6', multiply_call_result_774)

# Assigning a BinOp to a Name (line 29):
# Getting the type of 'x' (line 29)
x_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 5), 'x')
# Getting the type of 'y' (line 29)
y_776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 9), 'y')
# Applying the binary operator 'div' (line 29)
result_div_777 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 5), 'div', x_775, y_776)

# Assigning a type to the variable 'r7' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'r7', result_div_777)

# Assigning a Call to a Name (line 30):

# Call to divide(...): (line 30)
# Processing the call arguments (line 30)
# Getting the type of 'x' (line 30)
x_780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'x', False)
# Getting the type of 'y' (line 30)
y_781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 18), 'y', False)
# Processing the call keyword arguments (line 30)
kwargs_782 = {}
# Getting the type of 'np' (line 30)
np_778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 5), 'np', False)
# Obtaining the member 'divide' of a type (line 30)
divide_779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 5), np_778, 'divide')
# Calling divide(args, kwargs) (line 30)
divide_call_result_783 = invoke(stypy.reporting.localization.Localization(__file__, 30, 5), divide_779, *[x_780, y_781], **kwargs_782)

# Assigning a type to the variable 'r8' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'r8', divide_call_result_783)

# Assigning a Call to a Name (line 35):

# Call to sqrt(...): (line 35)
# Processing the call arguments (line 35)
# Getting the type of 'x' (line 35)
x_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 13), 'x', False)
# Processing the call keyword arguments (line 35)
kwargs_787 = {}
# Getting the type of 'np' (line 35)
np_784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 5), 'np', False)
# Obtaining the member 'sqrt' of a type (line 35)
sqrt_785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 5), np_784, 'sqrt')
# Calling sqrt(args, kwargs) (line 35)
sqrt_call_result_788 = invoke(stypy.reporting.localization.Localization(__file__, 35, 5), sqrt_785, *[x_786], **kwargs_787)

# Assigning a type to the variable 'r9' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'r9', sqrt_call_result_788)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
