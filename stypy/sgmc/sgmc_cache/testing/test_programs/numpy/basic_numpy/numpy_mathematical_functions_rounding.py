
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # https://docs.scipy.org/doc/numpy/reference/routines.math.html
2: 
3: 
4: import numpy as np
5: 
6: x = 2.1
7: a = 3.45646
8: 
9: # Rounding
10: r1 = np.around(a)  # Evenly round to the given number of decimals.
11: r2 = np.round_(a)  # Round an array to the given number of decimals.
12: r3 = np.rint(x)  # Round elements of the array to the nearest integer.
13: r4 = np.fix(x)  # Round to nearest integer towards zero.
14: r5 = np.floor(x)  # Return the floor of the input, element-wise.
15: r6 = np.ceil(x)  # Return the ceiling of the input, element-wise.
16: r7 = np.trunc(x)  # Return the truncated value of the input, element-wise.
17: 
18: x = [2.1, 3.4]
19: a = [3.45646, 5.6432564]
20: 
21: r8 = np.around(a)  # Evenly round to the given number of decimals.
22: r9 = np.round_(a)  # Round an array to the given number of decimals.
23: r10 = np.rint(x)  # Round elements of the array to the nearest integer.
24: r11 = np.fix(x)  # Round to nearest integer towards zero.
25: r12 = np.floor(x)  # Return the floor of the input, element-wise.
26: r13 = np.ceil(x)  # Return the ceiling of the input, element-wise.
27: r14 = np.trunc(x)  # Return the truncated value of the input, element-wise.
28: 
29: 
30: # l = globals().copy()
31: # for v in l:
32: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
33: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_680 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_680) is not StypyTypeError):

    if (import_680 != 'pyd_module'):
        __import__(import_680)
        sys_modules_681 = sys.modules[import_680]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_681.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_680)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a Num to a Name (line 6):
float_682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'float')
# Assigning a type to the variable 'x' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'x', float_682)

# Assigning a Num to a Name (line 7):
float_683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 4), 'float')
# Assigning a type to the variable 'a' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'a', float_683)

# Assigning a Call to a Name (line 10):

# Call to around(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'a' (line 10)
a_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 15), 'a', False)
# Processing the call keyword arguments (line 10)
kwargs_687 = {}
# Getting the type of 'np' (line 10)
np_684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'np', False)
# Obtaining the member 'around' of a type (line 10)
around_685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), np_684, 'around')
# Calling around(args, kwargs) (line 10)
around_call_result_688 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), around_685, *[a_686], **kwargs_687)

# Assigning a type to the variable 'r1' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r1', around_call_result_688)

# Assigning a Call to a Name (line 11):

# Call to round_(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'a' (line 11)
a_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'a', False)
# Processing the call keyword arguments (line 11)
kwargs_692 = {}
# Getting the type of 'np' (line 11)
np_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'np', False)
# Obtaining the member 'round_' of a type (line 11)
round__690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), np_689, 'round_')
# Calling round_(args, kwargs) (line 11)
round__call_result_693 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), round__690, *[a_691], **kwargs_692)

# Assigning a type to the variable 'r2' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r2', round__call_result_693)

# Assigning a Call to a Name (line 12):

# Call to rint(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'x' (line 12)
x_696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 13), 'x', False)
# Processing the call keyword arguments (line 12)
kwargs_697 = {}
# Getting the type of 'np' (line 12)
np_694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'np', False)
# Obtaining the member 'rint' of a type (line 12)
rint_695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), np_694, 'rint')
# Calling rint(args, kwargs) (line 12)
rint_call_result_698 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), rint_695, *[x_696], **kwargs_697)

# Assigning a type to the variable 'r3' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r3', rint_call_result_698)

# Assigning a Call to a Name (line 13):

# Call to fix(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'x' (line 13)
x_701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'x', False)
# Processing the call keyword arguments (line 13)
kwargs_702 = {}
# Getting the type of 'np' (line 13)
np_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'np', False)
# Obtaining the member 'fix' of a type (line 13)
fix_700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), np_699, 'fix')
# Calling fix(args, kwargs) (line 13)
fix_call_result_703 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), fix_700, *[x_701], **kwargs_702)

# Assigning a type to the variable 'r4' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r4', fix_call_result_703)

# Assigning a Call to a Name (line 14):

# Call to floor(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of 'x' (line 14)
x_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'x', False)
# Processing the call keyword arguments (line 14)
kwargs_707 = {}
# Getting the type of 'np' (line 14)
np_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'np', False)
# Obtaining the member 'floor' of a type (line 14)
floor_705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), np_704, 'floor')
# Calling floor(args, kwargs) (line 14)
floor_call_result_708 = invoke(stypy.reporting.localization.Localization(__file__, 14, 5), floor_705, *[x_706], **kwargs_707)

# Assigning a type to the variable 'r5' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r5', floor_call_result_708)

# Assigning a Call to a Name (line 15):

# Call to ceil(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of 'x' (line 15)
x_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 13), 'x', False)
# Processing the call keyword arguments (line 15)
kwargs_712 = {}
# Getting the type of 'np' (line 15)
np_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'np', False)
# Obtaining the member 'ceil' of a type (line 15)
ceil_710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), np_709, 'ceil')
# Calling ceil(args, kwargs) (line 15)
ceil_call_result_713 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), ceil_710, *[x_711], **kwargs_712)

# Assigning a type to the variable 'r6' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r6', ceil_call_result_713)

# Assigning a Call to a Name (line 16):

# Call to trunc(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of 'x' (line 16)
x_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'x', False)
# Processing the call keyword arguments (line 16)
kwargs_717 = {}
# Getting the type of 'np' (line 16)
np_714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 5), 'np', False)
# Obtaining the member 'trunc' of a type (line 16)
trunc_715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 5), np_714, 'trunc')
# Calling trunc(args, kwargs) (line 16)
trunc_call_result_718 = invoke(stypy.reporting.localization.Localization(__file__, 16, 5), trunc_715, *[x_716], **kwargs_717)

# Assigning a type to the variable 'r7' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'r7', trunc_call_result_718)

# Assigning a List to a Name (line 18):

# Obtaining an instance of the builtin type 'list' (line 18)
list_719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
float_720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 5), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 4), list_719, float_720)
# Adding element type (line 18)
float_721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 4), list_719, float_721)

# Assigning a type to the variable 'x' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'x', list_719)

# Assigning a List to a Name (line 19):

# Obtaining an instance of the builtin type 'list' (line 19)
list_722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
float_723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 5), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 4), list_722, float_723)
# Adding element type (line 19)
float_724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 14), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 4), list_722, float_724)

# Assigning a type to the variable 'a' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'a', list_722)

# Assigning a Call to a Name (line 21):

# Call to around(...): (line 21)
# Processing the call arguments (line 21)
# Getting the type of 'a' (line 21)
a_727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'a', False)
# Processing the call keyword arguments (line 21)
kwargs_728 = {}
# Getting the type of 'np' (line 21)
np_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 5), 'np', False)
# Obtaining the member 'around' of a type (line 21)
around_726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 5), np_725, 'around')
# Calling around(args, kwargs) (line 21)
around_call_result_729 = invoke(stypy.reporting.localization.Localization(__file__, 21, 5), around_726, *[a_727], **kwargs_728)

# Assigning a type to the variable 'r8' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'r8', around_call_result_729)

# Assigning a Call to a Name (line 22):

# Call to round_(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of 'a' (line 22)
a_732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'a', False)
# Processing the call keyword arguments (line 22)
kwargs_733 = {}
# Getting the type of 'np' (line 22)
np_730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 5), 'np', False)
# Obtaining the member 'round_' of a type (line 22)
round__731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 5), np_730, 'round_')
# Calling round_(args, kwargs) (line 22)
round__call_result_734 = invoke(stypy.reporting.localization.Localization(__file__, 22, 5), round__731, *[a_732], **kwargs_733)

# Assigning a type to the variable 'r9' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'r9', round__call_result_734)

# Assigning a Call to a Name (line 23):

# Call to rint(...): (line 23)
# Processing the call arguments (line 23)
# Getting the type of 'x' (line 23)
x_737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 14), 'x', False)
# Processing the call keyword arguments (line 23)
kwargs_738 = {}
# Getting the type of 'np' (line 23)
np_735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 6), 'np', False)
# Obtaining the member 'rint' of a type (line 23)
rint_736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 6), np_735, 'rint')
# Calling rint(args, kwargs) (line 23)
rint_call_result_739 = invoke(stypy.reporting.localization.Localization(__file__, 23, 6), rint_736, *[x_737], **kwargs_738)

# Assigning a type to the variable 'r10' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'r10', rint_call_result_739)

# Assigning a Call to a Name (line 24):

# Call to fix(...): (line 24)
# Processing the call arguments (line 24)
# Getting the type of 'x' (line 24)
x_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'x', False)
# Processing the call keyword arguments (line 24)
kwargs_743 = {}
# Getting the type of 'np' (line 24)
np_740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 6), 'np', False)
# Obtaining the member 'fix' of a type (line 24)
fix_741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 6), np_740, 'fix')
# Calling fix(args, kwargs) (line 24)
fix_call_result_744 = invoke(stypy.reporting.localization.Localization(__file__, 24, 6), fix_741, *[x_742], **kwargs_743)

# Assigning a type to the variable 'r11' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'r11', fix_call_result_744)

# Assigning a Call to a Name (line 25):

# Call to floor(...): (line 25)
# Processing the call arguments (line 25)
# Getting the type of 'x' (line 25)
x_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'x', False)
# Processing the call keyword arguments (line 25)
kwargs_748 = {}
# Getting the type of 'np' (line 25)
np_745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 6), 'np', False)
# Obtaining the member 'floor' of a type (line 25)
floor_746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 6), np_745, 'floor')
# Calling floor(args, kwargs) (line 25)
floor_call_result_749 = invoke(stypy.reporting.localization.Localization(__file__, 25, 6), floor_746, *[x_747], **kwargs_748)

# Assigning a type to the variable 'r12' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'r12', floor_call_result_749)

# Assigning a Call to a Name (line 26):

# Call to ceil(...): (line 26)
# Processing the call arguments (line 26)
# Getting the type of 'x' (line 26)
x_752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'x', False)
# Processing the call keyword arguments (line 26)
kwargs_753 = {}
# Getting the type of 'np' (line 26)
np_750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 6), 'np', False)
# Obtaining the member 'ceil' of a type (line 26)
ceil_751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 6), np_750, 'ceil')
# Calling ceil(args, kwargs) (line 26)
ceil_call_result_754 = invoke(stypy.reporting.localization.Localization(__file__, 26, 6), ceil_751, *[x_752], **kwargs_753)

# Assigning a type to the variable 'r13' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'r13', ceil_call_result_754)

# Assigning a Call to a Name (line 27):

# Call to trunc(...): (line 27)
# Processing the call arguments (line 27)
# Getting the type of 'x' (line 27)
x_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'x', False)
# Processing the call keyword arguments (line 27)
kwargs_758 = {}
# Getting the type of 'np' (line 27)
np_755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 6), 'np', False)
# Obtaining the member 'trunc' of a type (line 27)
trunc_756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 6), np_755, 'trunc')
# Calling trunc(args, kwargs) (line 27)
trunc_call_result_759 = invoke(stypy.reporting.localization.Localization(__file__, 27, 6), trunc_756, *[x_757], **kwargs_758)

# Assigning a type to the variable 'r14' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'r14', trunc_call_result_759)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
