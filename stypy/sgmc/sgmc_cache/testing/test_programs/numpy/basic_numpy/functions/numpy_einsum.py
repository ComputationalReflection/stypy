
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: a = np.arange(25).reshape(5, 5)
6: b = np.arange(5)
7: c = np.arange(6).reshape(2, 3)
8: 
9: r = np.einsum('ii', a)
10: 
11: r2 = np.einsum(a, [0, 0])
12: 
13: r3 = np.trace(a)
14: 
15: r4 = np.einsum('ii->i', a)
16: 
17: r5 = np.einsum(a, [0, 0], [0])
18: 
19: r6 = np.diag(a)
20: 
21: r7 = np.einsum('ij,j', a, b)
22: 
23: r8 = np.einsum(a, [0, 1], b, [1])
24: 
25: r9 = np.dot(a, b)
26: 
27: r10 = np.einsum('...j,j', a, b)
28: 
29: r11 = np.einsum('ji', c)
30: 
31: r12 = np.einsum(c, [1, 0])
32: 
33: r13 = c.T
34: 
35: r14 = np.einsum('..., ...', 3, c)
36: 
37: r15 = np.einsum(3, [Ellipsis], c, [Ellipsis])
38: 
39: r16 = np.multiply(3, c)
40: 
41: r17 = np.einsum('i,i', b, b)
42: 
43: r18 = np.einsum(b, [0], b, [0])
44: 
45: r19 = np.inner(b, b)
46: 
47: r20 = np.einsum('i,j', np.arange(2) + 1, b)
48: 
49: r21 = np.einsum(np.arange(2) + 1, [0], b, [1])
50: 
51: r22 = np.outer(np.arange(2) + 1, b)
52: 
53: r23 = np.einsum('i...->...', a)
54: 
55: r24 = np.einsum(a, [0, Ellipsis], [Ellipsis])
56: 
57: r25 = np.sum(a, axis=0)
58: 
59: a2 = np.arange(60.).reshape(3, 4, 5)
60: b2 = np.arange(24.).reshape(4, 3, 2)
61: r26 = np.einsum('ijk,jil->kl', a2, b2)
62: 
63: r27 = np.einsum(a2, [0, 1, 2], b2, [1, 0, 3], [2, 3])
64: 
65: r28 = np.tensordot(a2, b2, axes=([1, 0], [0, 1]))
66: 
67: a3 = np.arange(6).reshape((3, 2))
68: b3 = np.arange(12).reshape((4, 3))
69: r29 = np.einsum('ki,jk->ij', a3, b3)
70: 
71: r30 = np.einsum('ki,...k->i...', a3, b3)
72: 
73: r31 = np.einsum('k...,jk', a3, b3)
74: 
75: # since version 1.10.0
76: a4 = np.zeros((3, 3))
77: r32 = np.einsum('ii->i', a4)[:] = 1
78: 
79: # l = globals().copy()
80: # for v in l:
81: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
82: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_701 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_701) is not StypyTypeError):

    if (import_701 != 'pyd_module'):
        __import__(import_701)
        sys_modules_702 = sys.modules[import_701]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_702.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_701)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to reshape(...): (line 5)
# Processing the call arguments (line 5)
int_709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
int_710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 29), 'int')
# Processing the call keyword arguments (line 5)
kwargs_711 = {}

# Call to arange(...): (line 5)
# Processing the call arguments (line 5)
int_705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
# Processing the call keyword arguments (line 5)
kwargs_706 = {}
# Getting the type of 'np' (line 5)
np_703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'arange' of a type (line 5)
arange_704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_703, 'arange')
# Calling arange(args, kwargs) (line 5)
arange_call_result_707 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), arange_704, *[int_705], **kwargs_706)

# Obtaining the member 'reshape' of a type (line 5)
reshape_708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), arange_call_result_707, 'reshape')
# Calling reshape(args, kwargs) (line 5)
reshape_call_result_712 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), reshape_708, *[int_709, int_710], **kwargs_711)

# Assigning a type to the variable 'a' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'a', reshape_call_result_712)

# Assigning a Call to a Name (line 6):

# Call to arange(...): (line 6)
# Processing the call arguments (line 6)
int_715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'int')
# Processing the call keyword arguments (line 6)
kwargs_716 = {}
# Getting the type of 'np' (line 6)
np_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'arange' of a type (line 6)
arange_714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_713, 'arange')
# Calling arange(args, kwargs) (line 6)
arange_call_result_717 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), arange_714, *[int_715], **kwargs_716)

# Assigning a type to the variable 'b' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'b', arange_call_result_717)

# Assigning a Call to a Name (line 7):

# Call to reshape(...): (line 7)
# Processing the call arguments (line 7)
int_724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'int')
int_725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 28), 'int')
# Processing the call keyword arguments (line 7)
kwargs_726 = {}

# Call to arange(...): (line 7)
# Processing the call arguments (line 7)
int_720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'int')
# Processing the call keyword arguments (line 7)
kwargs_721 = {}
# Getting the type of 'np' (line 7)
np_718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'np', False)
# Obtaining the member 'arange' of a type (line 7)
arange_719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), np_718, 'arange')
# Calling arange(args, kwargs) (line 7)
arange_call_result_722 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), arange_719, *[int_720], **kwargs_721)

# Obtaining the member 'reshape' of a type (line 7)
reshape_723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), arange_call_result_722, 'reshape')
# Calling reshape(args, kwargs) (line 7)
reshape_call_result_727 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), reshape_723, *[int_724, int_725], **kwargs_726)

# Assigning a type to the variable 'c' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'c', reshape_call_result_727)

# Assigning a Call to a Name (line 9):

# Call to einsum(...): (line 9)
# Processing the call arguments (line 9)
str_730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 14), 'str', 'ii')
# Getting the type of 'a' (line 9)
a_731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 20), 'a', False)
# Processing the call keyword arguments (line 9)
kwargs_732 = {}
# Getting the type of 'np' (line 9)
np_728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'np', False)
# Obtaining the member 'einsum' of a type (line 9)
einsum_729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), np_728, 'einsum')
# Calling einsum(args, kwargs) (line 9)
einsum_call_result_733 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), einsum_729, *[str_730, a_731], **kwargs_732)

# Assigning a type to the variable 'r' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r', einsum_call_result_733)

# Assigning a Call to a Name (line 11):

# Call to einsum(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'a' (line 11)
a_736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'a', False)

# Obtaining an instance of the builtin type 'list' (line 11)
list_737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)
int_738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 18), list_737, int_738)
# Adding element type (line 11)
int_739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 18), list_737, int_739)

# Processing the call keyword arguments (line 11)
kwargs_740 = {}
# Getting the type of 'np' (line 11)
np_734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'np', False)
# Obtaining the member 'einsum' of a type (line 11)
einsum_735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), np_734, 'einsum')
# Calling einsum(args, kwargs) (line 11)
einsum_call_result_741 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), einsum_735, *[a_736, list_737], **kwargs_740)

# Assigning a type to the variable 'r2' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r2', einsum_call_result_741)

# Assigning a Call to a Name (line 13):

# Call to trace(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'a' (line 13)
a_744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'a', False)
# Processing the call keyword arguments (line 13)
kwargs_745 = {}
# Getting the type of 'np' (line 13)
np_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'np', False)
# Obtaining the member 'trace' of a type (line 13)
trace_743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), np_742, 'trace')
# Calling trace(args, kwargs) (line 13)
trace_call_result_746 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), trace_743, *[a_744], **kwargs_745)

# Assigning a type to the variable 'r3' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r3', trace_call_result_746)

# Assigning a Call to a Name (line 15):

# Call to einsum(...): (line 15)
# Processing the call arguments (line 15)
str_749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'ii->i')
# Getting the type of 'a' (line 15)
a_750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 24), 'a', False)
# Processing the call keyword arguments (line 15)
kwargs_751 = {}
# Getting the type of 'np' (line 15)
np_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'np', False)
# Obtaining the member 'einsum' of a type (line 15)
einsum_748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), np_747, 'einsum')
# Calling einsum(args, kwargs) (line 15)
einsum_call_result_752 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), einsum_748, *[str_749, a_750], **kwargs_751)

# Assigning a type to the variable 'r4' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r4', einsum_call_result_752)

# Assigning a Call to a Name (line 17):

# Call to einsum(...): (line 17)
# Processing the call arguments (line 17)
# Getting the type of 'a' (line 17)
a_755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'a', False)

# Obtaining an instance of the builtin type 'list' (line 17)
list_756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
int_757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 18), list_756, int_757)
# Adding element type (line 17)
int_758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 18), list_756, int_758)


# Obtaining an instance of the builtin type 'list' (line 17)
list_759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
int_760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 26), list_759, int_760)

# Processing the call keyword arguments (line 17)
kwargs_761 = {}
# Getting the type of 'np' (line 17)
np_753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'np', False)
# Obtaining the member 'einsum' of a type (line 17)
einsum_754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), np_753, 'einsum')
# Calling einsum(args, kwargs) (line 17)
einsum_call_result_762 = invoke(stypy.reporting.localization.Localization(__file__, 17, 5), einsum_754, *[a_755, list_756, list_759], **kwargs_761)

# Assigning a type to the variable 'r5' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r5', einsum_call_result_762)

# Assigning a Call to a Name (line 19):

# Call to diag(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of 'a' (line 19)
a_765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), 'a', False)
# Processing the call keyword arguments (line 19)
kwargs_766 = {}
# Getting the type of 'np' (line 19)
np_763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'np', False)
# Obtaining the member 'diag' of a type (line 19)
diag_764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 5), np_763, 'diag')
# Calling diag(args, kwargs) (line 19)
diag_call_result_767 = invoke(stypy.reporting.localization.Localization(__file__, 19, 5), diag_764, *[a_765], **kwargs_766)

# Assigning a type to the variable 'r6' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r6', diag_call_result_767)

# Assigning a Call to a Name (line 21):

# Call to einsum(...): (line 21)
# Processing the call arguments (line 21)
str_770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 15), 'str', 'ij,j')
# Getting the type of 'a' (line 21)
a_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 23), 'a', False)
# Getting the type of 'b' (line 21)
b_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 26), 'b', False)
# Processing the call keyword arguments (line 21)
kwargs_773 = {}
# Getting the type of 'np' (line 21)
np_768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 5), 'np', False)
# Obtaining the member 'einsum' of a type (line 21)
einsum_769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 5), np_768, 'einsum')
# Calling einsum(args, kwargs) (line 21)
einsum_call_result_774 = invoke(stypy.reporting.localization.Localization(__file__, 21, 5), einsum_769, *[str_770, a_771, b_772], **kwargs_773)

# Assigning a type to the variable 'r7' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'r7', einsum_call_result_774)

# Assigning a Call to a Name (line 23):

# Call to einsum(...): (line 23)
# Processing the call arguments (line 23)
# Getting the type of 'a' (line 23)
a_777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'a', False)

# Obtaining an instance of the builtin type 'list' (line 23)
list_778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
int_779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 18), list_778, int_779)
# Adding element type (line 23)
int_780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 18), list_778, int_780)

# Getting the type of 'b' (line 23)
b_781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 26), 'b', False)

# Obtaining an instance of the builtin type 'list' (line 23)
list_782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 29), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
int_783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 29), list_782, int_783)

# Processing the call keyword arguments (line 23)
kwargs_784 = {}
# Getting the type of 'np' (line 23)
np_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 5), 'np', False)
# Obtaining the member 'einsum' of a type (line 23)
einsum_776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 5), np_775, 'einsum')
# Calling einsum(args, kwargs) (line 23)
einsum_call_result_785 = invoke(stypy.reporting.localization.Localization(__file__, 23, 5), einsum_776, *[a_777, list_778, b_781, list_782], **kwargs_784)

# Assigning a type to the variable 'r8' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'r8', einsum_call_result_785)

# Assigning a Call to a Name (line 25):

# Call to dot(...): (line 25)
# Processing the call arguments (line 25)
# Getting the type of 'a' (line 25)
a_788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'a', False)
# Getting the type of 'b' (line 25)
b_789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'b', False)
# Processing the call keyword arguments (line 25)
kwargs_790 = {}
# Getting the type of 'np' (line 25)
np_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 5), 'np', False)
# Obtaining the member 'dot' of a type (line 25)
dot_787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 5), np_786, 'dot')
# Calling dot(args, kwargs) (line 25)
dot_call_result_791 = invoke(stypy.reporting.localization.Localization(__file__, 25, 5), dot_787, *[a_788, b_789], **kwargs_790)

# Assigning a type to the variable 'r9' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'r9', dot_call_result_791)

# Assigning a Call to a Name (line 27):

# Call to einsum(...): (line 27)
# Processing the call arguments (line 27)
str_794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'str', '...j,j')
# Getting the type of 'a' (line 27)
a_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 26), 'a', False)
# Getting the type of 'b' (line 27)
b_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 29), 'b', False)
# Processing the call keyword arguments (line 27)
kwargs_797 = {}
# Getting the type of 'np' (line 27)
np_792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 27)
einsum_793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 6), np_792, 'einsum')
# Calling einsum(args, kwargs) (line 27)
einsum_call_result_798 = invoke(stypy.reporting.localization.Localization(__file__, 27, 6), einsum_793, *[str_794, a_795, b_796], **kwargs_797)

# Assigning a type to the variable 'r10' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'r10', einsum_call_result_798)

# Assigning a Call to a Name (line 29):

# Call to einsum(...): (line 29)
# Processing the call arguments (line 29)
str_801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 16), 'str', 'ji')
# Getting the type of 'c' (line 29)
c_802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'c', False)
# Processing the call keyword arguments (line 29)
kwargs_803 = {}
# Getting the type of 'np' (line 29)
np_799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 29)
einsum_800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 6), np_799, 'einsum')
# Calling einsum(args, kwargs) (line 29)
einsum_call_result_804 = invoke(stypy.reporting.localization.Localization(__file__, 29, 6), einsum_800, *[str_801, c_802], **kwargs_803)

# Assigning a type to the variable 'r11' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'r11', einsum_call_result_804)

# Assigning a Call to a Name (line 31):

# Call to einsum(...): (line 31)
# Processing the call arguments (line 31)
# Getting the type of 'c' (line 31)
c_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'c', False)

# Obtaining an instance of the builtin type 'list' (line 31)
list_808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 31)
# Adding element type (line 31)
int_809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), list_808, int_809)
# Adding element type (line 31)
int_810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), list_808, int_810)

# Processing the call keyword arguments (line 31)
kwargs_811 = {}
# Getting the type of 'np' (line 31)
np_805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 31)
einsum_806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 6), np_805, 'einsum')
# Calling einsum(args, kwargs) (line 31)
einsum_call_result_812 = invoke(stypy.reporting.localization.Localization(__file__, 31, 6), einsum_806, *[c_807, list_808], **kwargs_811)

# Assigning a type to the variable 'r12' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'r12', einsum_call_result_812)

# Assigning a Attribute to a Name (line 33):
# Getting the type of 'c' (line 33)
c_813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 6), 'c')
# Obtaining the member 'T' of a type (line 33)
T_814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 6), c_813, 'T')
# Assigning a type to the variable 'r13' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'r13', T_814)

# Assigning a Call to a Name (line 35):

# Call to einsum(...): (line 35)
# Processing the call arguments (line 35)
str_817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 16), 'str', '..., ...')
int_818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 28), 'int')
# Getting the type of 'c' (line 35)
c_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 31), 'c', False)
# Processing the call keyword arguments (line 35)
kwargs_820 = {}
# Getting the type of 'np' (line 35)
np_815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 35)
einsum_816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 6), np_815, 'einsum')
# Calling einsum(args, kwargs) (line 35)
einsum_call_result_821 = invoke(stypy.reporting.localization.Localization(__file__, 35, 6), einsum_816, *[str_817, int_818, c_819], **kwargs_820)

# Assigning a type to the variable 'r14' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'r14', einsum_call_result_821)

# Assigning a Call to a Name (line 37):

# Call to einsum(...): (line 37)
# Processing the call arguments (line 37)
int_824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 16), 'int')

# Obtaining an instance of the builtin type 'list' (line 37)
list_825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 37)
# Adding element type (line 37)
# Getting the type of 'Ellipsis' (line 37)
Ellipsis_826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'Ellipsis', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 19), list_825, Ellipsis_826)

# Getting the type of 'c' (line 37)
c_827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), 'c', False)

# Obtaining an instance of the builtin type 'list' (line 37)
list_828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 34), 'list')
# Adding type elements to the builtin type 'list' instance (line 37)
# Adding element type (line 37)
# Getting the type of 'Ellipsis' (line 37)
Ellipsis_829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 35), 'Ellipsis', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 34), list_828, Ellipsis_829)

# Processing the call keyword arguments (line 37)
kwargs_830 = {}
# Getting the type of 'np' (line 37)
np_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 37)
einsum_823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 6), np_822, 'einsum')
# Calling einsum(args, kwargs) (line 37)
einsum_call_result_831 = invoke(stypy.reporting.localization.Localization(__file__, 37, 6), einsum_823, *[int_824, list_825, c_827, list_828], **kwargs_830)

# Assigning a type to the variable 'r15' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'r15', einsum_call_result_831)

# Assigning a Call to a Name (line 39):

# Call to multiply(...): (line 39)
# Processing the call arguments (line 39)
int_834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 18), 'int')
# Getting the type of 'c' (line 39)
c_835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 21), 'c', False)
# Processing the call keyword arguments (line 39)
kwargs_836 = {}
# Getting the type of 'np' (line 39)
np_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 6), 'np', False)
# Obtaining the member 'multiply' of a type (line 39)
multiply_833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 6), np_832, 'multiply')
# Calling multiply(args, kwargs) (line 39)
multiply_call_result_837 = invoke(stypy.reporting.localization.Localization(__file__, 39, 6), multiply_833, *[int_834, c_835], **kwargs_836)

# Assigning a type to the variable 'r16' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'r16', multiply_call_result_837)

# Assigning a Call to a Name (line 41):

# Call to einsum(...): (line 41)
# Processing the call arguments (line 41)
str_840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 16), 'str', 'i,i')
# Getting the type of 'b' (line 41)
b_841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'b', False)
# Getting the type of 'b' (line 41)
b_842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 26), 'b', False)
# Processing the call keyword arguments (line 41)
kwargs_843 = {}
# Getting the type of 'np' (line 41)
np_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 41)
einsum_839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 6), np_838, 'einsum')
# Calling einsum(args, kwargs) (line 41)
einsum_call_result_844 = invoke(stypy.reporting.localization.Localization(__file__, 41, 6), einsum_839, *[str_840, b_841, b_842], **kwargs_843)

# Assigning a type to the variable 'r17' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'r17', einsum_call_result_844)

# Assigning a Call to a Name (line 43):

# Call to einsum(...): (line 43)
# Processing the call arguments (line 43)
# Getting the type of 'b' (line 43)
b_847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'b', False)

# Obtaining an instance of the builtin type 'list' (line 43)
list_848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 43)
# Adding element type (line 43)
int_849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 19), list_848, int_849)

# Getting the type of 'b' (line 43)
b_850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'b', False)

# Obtaining an instance of the builtin type 'list' (line 43)
list_851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 43)
# Adding element type (line 43)
int_852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 27), list_851, int_852)

# Processing the call keyword arguments (line 43)
kwargs_853 = {}
# Getting the type of 'np' (line 43)
np_845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 43)
einsum_846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 6), np_845, 'einsum')
# Calling einsum(args, kwargs) (line 43)
einsum_call_result_854 = invoke(stypy.reporting.localization.Localization(__file__, 43, 6), einsum_846, *[b_847, list_848, b_850, list_851], **kwargs_853)

# Assigning a type to the variable 'r18' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'r18', einsum_call_result_854)

# Assigning a Call to a Name (line 45):

# Call to inner(...): (line 45)
# Processing the call arguments (line 45)
# Getting the type of 'b' (line 45)
b_857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'b', False)
# Getting the type of 'b' (line 45)
b_858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 18), 'b', False)
# Processing the call keyword arguments (line 45)
kwargs_859 = {}
# Getting the type of 'np' (line 45)
np_855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 6), 'np', False)
# Obtaining the member 'inner' of a type (line 45)
inner_856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 6), np_855, 'inner')
# Calling inner(args, kwargs) (line 45)
inner_call_result_860 = invoke(stypy.reporting.localization.Localization(__file__, 45, 6), inner_856, *[b_857, b_858], **kwargs_859)

# Assigning a type to the variable 'r19' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'r19', inner_call_result_860)

# Assigning a Call to a Name (line 47):

# Call to einsum(...): (line 47)
# Processing the call arguments (line 47)
str_863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 16), 'str', 'i,j')

# Call to arange(...): (line 47)
# Processing the call arguments (line 47)
int_866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 33), 'int')
# Processing the call keyword arguments (line 47)
kwargs_867 = {}
# Getting the type of 'np' (line 47)
np_864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 23), 'np', False)
# Obtaining the member 'arange' of a type (line 47)
arange_865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 23), np_864, 'arange')
# Calling arange(args, kwargs) (line 47)
arange_call_result_868 = invoke(stypy.reporting.localization.Localization(__file__, 47, 23), arange_865, *[int_866], **kwargs_867)

int_869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 38), 'int')
# Applying the binary operator '+' (line 47)
result_add_870 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 23), '+', arange_call_result_868, int_869)

# Getting the type of 'b' (line 47)
b_871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 41), 'b', False)
# Processing the call keyword arguments (line 47)
kwargs_872 = {}
# Getting the type of 'np' (line 47)
np_861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 47)
einsum_862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 6), np_861, 'einsum')
# Calling einsum(args, kwargs) (line 47)
einsum_call_result_873 = invoke(stypy.reporting.localization.Localization(__file__, 47, 6), einsum_862, *[str_863, result_add_870, b_871], **kwargs_872)

# Assigning a type to the variable 'r20' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'r20', einsum_call_result_873)

# Assigning a Call to a Name (line 49):

# Call to einsum(...): (line 49)
# Processing the call arguments (line 49)

# Call to arange(...): (line 49)
# Processing the call arguments (line 49)
int_878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 26), 'int')
# Processing the call keyword arguments (line 49)
kwargs_879 = {}
# Getting the type of 'np' (line 49)
np_876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'np', False)
# Obtaining the member 'arange' of a type (line 49)
arange_877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 16), np_876, 'arange')
# Calling arange(args, kwargs) (line 49)
arange_call_result_880 = invoke(stypy.reporting.localization.Localization(__file__, 49, 16), arange_877, *[int_878], **kwargs_879)

int_881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 31), 'int')
# Applying the binary operator '+' (line 49)
result_add_882 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 16), '+', arange_call_result_880, int_881)


# Obtaining an instance of the builtin type 'list' (line 49)
list_883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 34), 'list')
# Adding type elements to the builtin type 'list' instance (line 49)
# Adding element type (line 49)
int_884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 34), list_883, int_884)

# Getting the type of 'b' (line 49)
b_885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 39), 'b', False)

# Obtaining an instance of the builtin type 'list' (line 49)
list_886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 42), 'list')
# Adding type elements to the builtin type 'list' instance (line 49)
# Adding element type (line 49)
int_887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 42), list_886, int_887)

# Processing the call keyword arguments (line 49)
kwargs_888 = {}
# Getting the type of 'np' (line 49)
np_874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 49)
einsum_875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 6), np_874, 'einsum')
# Calling einsum(args, kwargs) (line 49)
einsum_call_result_889 = invoke(stypy.reporting.localization.Localization(__file__, 49, 6), einsum_875, *[result_add_882, list_883, b_885, list_886], **kwargs_888)

# Assigning a type to the variable 'r21' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'r21', einsum_call_result_889)

# Assigning a Call to a Name (line 51):

# Call to outer(...): (line 51)
# Processing the call arguments (line 51)

# Call to arange(...): (line 51)
# Processing the call arguments (line 51)
int_894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 25), 'int')
# Processing the call keyword arguments (line 51)
kwargs_895 = {}
# Getting the type of 'np' (line 51)
np_892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'np', False)
# Obtaining the member 'arange' of a type (line 51)
arange_893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 15), np_892, 'arange')
# Calling arange(args, kwargs) (line 51)
arange_call_result_896 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), arange_893, *[int_894], **kwargs_895)

int_897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 30), 'int')
# Applying the binary operator '+' (line 51)
result_add_898 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 15), '+', arange_call_result_896, int_897)

# Getting the type of 'b' (line 51)
b_899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 33), 'b', False)
# Processing the call keyword arguments (line 51)
kwargs_900 = {}
# Getting the type of 'np' (line 51)
np_890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 6), 'np', False)
# Obtaining the member 'outer' of a type (line 51)
outer_891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 6), np_890, 'outer')
# Calling outer(args, kwargs) (line 51)
outer_call_result_901 = invoke(stypy.reporting.localization.Localization(__file__, 51, 6), outer_891, *[result_add_898, b_899], **kwargs_900)

# Assigning a type to the variable 'r22' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'r22', outer_call_result_901)

# Assigning a Call to a Name (line 53):

# Call to einsum(...): (line 53)
# Processing the call arguments (line 53)
str_904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 16), 'str', 'i...->...')
# Getting the type of 'a' (line 53)
a_905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 29), 'a', False)
# Processing the call keyword arguments (line 53)
kwargs_906 = {}
# Getting the type of 'np' (line 53)
np_902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 53)
einsum_903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 6), np_902, 'einsum')
# Calling einsum(args, kwargs) (line 53)
einsum_call_result_907 = invoke(stypy.reporting.localization.Localization(__file__, 53, 6), einsum_903, *[str_904, a_905], **kwargs_906)

# Assigning a type to the variable 'r23' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'r23', einsum_call_result_907)

# Assigning a Call to a Name (line 55):

# Call to einsum(...): (line 55)
# Processing the call arguments (line 55)
# Getting the type of 'a' (line 55)
a_910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'a', False)

# Obtaining an instance of the builtin type 'list' (line 55)
list_911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 55)
# Adding element type (line 55)
int_912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 19), list_911, int_912)
# Adding element type (line 55)
# Getting the type of 'Ellipsis' (line 55)
Ellipsis_913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'Ellipsis', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 19), list_911, Ellipsis_913)


# Obtaining an instance of the builtin type 'list' (line 55)
list_914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 34), 'list')
# Adding type elements to the builtin type 'list' instance (line 55)
# Adding element type (line 55)
# Getting the type of 'Ellipsis' (line 55)
Ellipsis_915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'Ellipsis', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 34), list_914, Ellipsis_915)

# Processing the call keyword arguments (line 55)
kwargs_916 = {}
# Getting the type of 'np' (line 55)
np_908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 55)
einsum_909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 6), np_908, 'einsum')
# Calling einsum(args, kwargs) (line 55)
einsum_call_result_917 = invoke(stypy.reporting.localization.Localization(__file__, 55, 6), einsum_909, *[a_910, list_911, list_914], **kwargs_916)

# Assigning a type to the variable 'r24' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'r24', einsum_call_result_917)

# Assigning a Call to a Name (line 57):

# Call to sum(...): (line 57)
# Processing the call arguments (line 57)
# Getting the type of 'a' (line 57)
a_920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 13), 'a', False)
# Processing the call keyword arguments (line 57)
int_921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 21), 'int')
keyword_922 = int_921
kwargs_923 = {'axis': keyword_922}
# Getting the type of 'np' (line 57)
np_918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 6), 'np', False)
# Obtaining the member 'sum' of a type (line 57)
sum_919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 6), np_918, 'sum')
# Calling sum(args, kwargs) (line 57)
sum_call_result_924 = invoke(stypy.reporting.localization.Localization(__file__, 57, 6), sum_919, *[a_920], **kwargs_923)

# Assigning a type to the variable 'r25' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'r25', sum_call_result_924)

# Assigning a Call to a Name (line 59):

# Call to reshape(...): (line 59)
# Processing the call arguments (line 59)
int_931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 28), 'int')
int_932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 31), 'int')
int_933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 34), 'int')
# Processing the call keyword arguments (line 59)
kwargs_934 = {}

# Call to arange(...): (line 59)
# Processing the call arguments (line 59)
float_927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 15), 'float')
# Processing the call keyword arguments (line 59)
kwargs_928 = {}
# Getting the type of 'np' (line 59)
np_925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 5), 'np', False)
# Obtaining the member 'arange' of a type (line 59)
arange_926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 5), np_925, 'arange')
# Calling arange(args, kwargs) (line 59)
arange_call_result_929 = invoke(stypy.reporting.localization.Localization(__file__, 59, 5), arange_926, *[float_927], **kwargs_928)

# Obtaining the member 'reshape' of a type (line 59)
reshape_930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 5), arange_call_result_929, 'reshape')
# Calling reshape(args, kwargs) (line 59)
reshape_call_result_935 = invoke(stypy.reporting.localization.Localization(__file__, 59, 5), reshape_930, *[int_931, int_932, int_933], **kwargs_934)

# Assigning a type to the variable 'a2' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'a2', reshape_call_result_935)

# Assigning a Call to a Name (line 60):

# Call to reshape(...): (line 60)
# Processing the call arguments (line 60)
int_942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 28), 'int')
int_943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 31), 'int')
int_944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 34), 'int')
# Processing the call keyword arguments (line 60)
kwargs_945 = {}

# Call to arange(...): (line 60)
# Processing the call arguments (line 60)
float_938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 15), 'float')
# Processing the call keyword arguments (line 60)
kwargs_939 = {}
# Getting the type of 'np' (line 60)
np_936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 5), 'np', False)
# Obtaining the member 'arange' of a type (line 60)
arange_937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 5), np_936, 'arange')
# Calling arange(args, kwargs) (line 60)
arange_call_result_940 = invoke(stypy.reporting.localization.Localization(__file__, 60, 5), arange_937, *[float_938], **kwargs_939)

# Obtaining the member 'reshape' of a type (line 60)
reshape_941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 5), arange_call_result_940, 'reshape')
# Calling reshape(args, kwargs) (line 60)
reshape_call_result_946 = invoke(stypy.reporting.localization.Localization(__file__, 60, 5), reshape_941, *[int_942, int_943, int_944], **kwargs_945)

# Assigning a type to the variable 'b2' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'b2', reshape_call_result_946)

# Assigning a Call to a Name (line 61):

# Call to einsum(...): (line 61)
# Processing the call arguments (line 61)
str_949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 16), 'str', 'ijk,jil->kl')
# Getting the type of 'a2' (line 61)
a2_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 31), 'a2', False)
# Getting the type of 'b2' (line 61)
b2_951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 35), 'b2', False)
# Processing the call keyword arguments (line 61)
kwargs_952 = {}
# Getting the type of 'np' (line 61)
np_947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 61)
einsum_948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 6), np_947, 'einsum')
# Calling einsum(args, kwargs) (line 61)
einsum_call_result_953 = invoke(stypy.reporting.localization.Localization(__file__, 61, 6), einsum_948, *[str_949, a2_950, b2_951], **kwargs_952)

# Assigning a type to the variable 'r26' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'r26', einsum_call_result_953)

# Assigning a Call to a Name (line 63):

# Call to einsum(...): (line 63)
# Processing the call arguments (line 63)
# Getting the type of 'a2' (line 63)
a2_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'a2', False)

# Obtaining an instance of the builtin type 'list' (line 63)
list_957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 63)
# Adding element type (line 63)
int_958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 20), list_957, int_958)
# Adding element type (line 63)
int_959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 20), list_957, int_959)
# Adding element type (line 63)
int_960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 20), list_957, int_960)

# Getting the type of 'b2' (line 63)
b2_961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 31), 'b2', False)

# Obtaining an instance of the builtin type 'list' (line 63)
list_962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 35), 'list')
# Adding type elements to the builtin type 'list' instance (line 63)
# Adding element type (line 63)
int_963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 36), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 35), list_962, int_963)
# Adding element type (line 63)
int_964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 39), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 35), list_962, int_964)
# Adding element type (line 63)
int_965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 42), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 35), list_962, int_965)


# Obtaining an instance of the builtin type 'list' (line 63)
list_966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 46), 'list')
# Adding type elements to the builtin type 'list' instance (line 63)
# Adding element type (line 63)
int_967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 47), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 46), list_966, int_967)
# Adding element type (line 63)
int_968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 50), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 46), list_966, int_968)

# Processing the call keyword arguments (line 63)
kwargs_969 = {}
# Getting the type of 'np' (line 63)
np_954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 63)
einsum_955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 6), np_954, 'einsum')
# Calling einsum(args, kwargs) (line 63)
einsum_call_result_970 = invoke(stypy.reporting.localization.Localization(__file__, 63, 6), einsum_955, *[a2_956, list_957, b2_961, list_962, list_966], **kwargs_969)

# Assigning a type to the variable 'r27' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'r27', einsum_call_result_970)

# Assigning a Call to a Name (line 65):

# Call to tensordot(...): (line 65)
# Processing the call arguments (line 65)
# Getting the type of 'a2' (line 65)
a2_973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'a2', False)
# Getting the type of 'b2' (line 65)
b2_974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 23), 'b2', False)
# Processing the call keyword arguments (line 65)

# Obtaining an instance of the builtin type 'tuple' (line 65)
tuple_975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 65)
# Adding element type (line 65)

# Obtaining an instance of the builtin type 'list' (line 65)
list_976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 33), 'list')
# Adding type elements to the builtin type 'list' instance (line 65)
# Adding element type (line 65)
int_977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 33), list_976, int_977)
# Adding element type (line 65)
int_978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 33), list_976, int_978)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 33), tuple_975, list_976)
# Adding element type (line 65)

# Obtaining an instance of the builtin type 'list' (line 65)
list_979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 41), 'list')
# Adding type elements to the builtin type 'list' instance (line 65)
# Adding element type (line 65)
int_980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 42), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 41), list_979, int_980)
# Adding element type (line 65)
int_981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 45), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 41), list_979, int_981)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 33), tuple_975, list_979)

keyword_982 = tuple_975
kwargs_983 = {'axes': keyword_982}
# Getting the type of 'np' (line 65)
np_971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 6), 'np', False)
# Obtaining the member 'tensordot' of a type (line 65)
tensordot_972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 6), np_971, 'tensordot')
# Calling tensordot(args, kwargs) (line 65)
tensordot_call_result_984 = invoke(stypy.reporting.localization.Localization(__file__, 65, 6), tensordot_972, *[a2_973, b2_974], **kwargs_983)

# Assigning a type to the variable 'r28' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'r28', tensordot_call_result_984)

# Assigning a Call to a Name (line 67):

# Call to reshape(...): (line 67)
# Processing the call arguments (line 67)

# Obtaining an instance of the builtin type 'tuple' (line 67)
tuple_991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 27), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 67)
# Adding element type (line 67)
int_992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 27), tuple_991, int_992)
# Adding element type (line 67)
int_993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 30), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 27), tuple_991, int_993)

# Processing the call keyword arguments (line 67)
kwargs_994 = {}

# Call to arange(...): (line 67)
# Processing the call arguments (line 67)
int_987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 15), 'int')
# Processing the call keyword arguments (line 67)
kwargs_988 = {}
# Getting the type of 'np' (line 67)
np_985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 5), 'np', False)
# Obtaining the member 'arange' of a type (line 67)
arange_986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 5), np_985, 'arange')
# Calling arange(args, kwargs) (line 67)
arange_call_result_989 = invoke(stypy.reporting.localization.Localization(__file__, 67, 5), arange_986, *[int_987], **kwargs_988)

# Obtaining the member 'reshape' of a type (line 67)
reshape_990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 5), arange_call_result_989, 'reshape')
# Calling reshape(args, kwargs) (line 67)
reshape_call_result_995 = invoke(stypy.reporting.localization.Localization(__file__, 67, 5), reshape_990, *[tuple_991], **kwargs_994)

# Assigning a type to the variable 'a3' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'a3', reshape_call_result_995)

# Assigning a Call to a Name (line 68):

# Call to reshape(...): (line 68)
# Processing the call arguments (line 68)

# Obtaining an instance of the builtin type 'tuple' (line 68)
tuple_1002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 68)
# Adding element type (line 68)
int_1003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 28), tuple_1002, int_1003)
# Adding element type (line 68)
int_1004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 28), tuple_1002, int_1004)

# Processing the call keyword arguments (line 68)
kwargs_1005 = {}

# Call to arange(...): (line 68)
# Processing the call arguments (line 68)
int_998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 15), 'int')
# Processing the call keyword arguments (line 68)
kwargs_999 = {}
# Getting the type of 'np' (line 68)
np_996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 5), 'np', False)
# Obtaining the member 'arange' of a type (line 68)
arange_997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 5), np_996, 'arange')
# Calling arange(args, kwargs) (line 68)
arange_call_result_1000 = invoke(stypy.reporting.localization.Localization(__file__, 68, 5), arange_997, *[int_998], **kwargs_999)

# Obtaining the member 'reshape' of a type (line 68)
reshape_1001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 5), arange_call_result_1000, 'reshape')
# Calling reshape(args, kwargs) (line 68)
reshape_call_result_1006 = invoke(stypy.reporting.localization.Localization(__file__, 68, 5), reshape_1001, *[tuple_1002], **kwargs_1005)

# Assigning a type to the variable 'b3' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'b3', reshape_call_result_1006)

# Assigning a Call to a Name (line 69):

# Call to einsum(...): (line 69)
# Processing the call arguments (line 69)
str_1009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 16), 'str', 'ki,jk->ij')
# Getting the type of 'a3' (line 69)
a3_1010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 29), 'a3', False)
# Getting the type of 'b3' (line 69)
b3_1011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 33), 'b3', False)
# Processing the call keyword arguments (line 69)
kwargs_1012 = {}
# Getting the type of 'np' (line 69)
np_1007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 69)
einsum_1008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 6), np_1007, 'einsum')
# Calling einsum(args, kwargs) (line 69)
einsum_call_result_1013 = invoke(stypy.reporting.localization.Localization(__file__, 69, 6), einsum_1008, *[str_1009, a3_1010, b3_1011], **kwargs_1012)

# Assigning a type to the variable 'r29' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'r29', einsum_call_result_1013)

# Assigning a Call to a Name (line 71):

# Call to einsum(...): (line 71)
# Processing the call arguments (line 71)
str_1016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 16), 'str', 'ki,...k->i...')
# Getting the type of 'a3' (line 71)
a3_1017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 33), 'a3', False)
# Getting the type of 'b3' (line 71)
b3_1018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 37), 'b3', False)
# Processing the call keyword arguments (line 71)
kwargs_1019 = {}
# Getting the type of 'np' (line 71)
np_1014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 71)
einsum_1015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 6), np_1014, 'einsum')
# Calling einsum(args, kwargs) (line 71)
einsum_call_result_1020 = invoke(stypy.reporting.localization.Localization(__file__, 71, 6), einsum_1015, *[str_1016, a3_1017, b3_1018], **kwargs_1019)

# Assigning a type to the variable 'r30' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'r30', einsum_call_result_1020)

# Assigning a Call to a Name (line 73):

# Call to einsum(...): (line 73)
# Processing the call arguments (line 73)
str_1023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 16), 'str', 'k...,jk')
# Getting the type of 'a3' (line 73)
a3_1024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 27), 'a3', False)
# Getting the type of 'b3' (line 73)
b3_1025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 31), 'b3', False)
# Processing the call keyword arguments (line 73)
kwargs_1026 = {}
# Getting the type of 'np' (line 73)
np_1021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 73)
einsum_1022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 6), np_1021, 'einsum')
# Calling einsum(args, kwargs) (line 73)
einsum_call_result_1027 = invoke(stypy.reporting.localization.Localization(__file__, 73, 6), einsum_1022, *[str_1023, a3_1024, b3_1025], **kwargs_1026)

# Assigning a type to the variable 'r31' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'r31', einsum_call_result_1027)

# Assigning a Call to a Name (line 76):

# Call to zeros(...): (line 76)
# Processing the call arguments (line 76)

# Obtaining an instance of the builtin type 'tuple' (line 76)
tuple_1030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 76)
# Adding element type (line 76)
int_1031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 15), tuple_1030, int_1031)
# Adding element type (line 76)
int_1032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 15), tuple_1030, int_1032)

# Processing the call keyword arguments (line 76)
kwargs_1033 = {}
# Getting the type of 'np' (line 76)
np_1028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 5), 'np', False)
# Obtaining the member 'zeros' of a type (line 76)
zeros_1029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 5), np_1028, 'zeros')
# Calling zeros(args, kwargs) (line 76)
zeros_call_result_1034 = invoke(stypy.reporting.localization.Localization(__file__, 76, 5), zeros_1029, *[tuple_1030], **kwargs_1033)

# Assigning a type to the variable 'a4' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'a4', zeros_call_result_1034)

# Multiple assignment of 2 elements.
int_1035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 34), 'int')

# Call to einsum(...): (line 77)
# Processing the call arguments (line 77)
str_1038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 16), 'str', 'ii->i')
# Getting the type of 'a4' (line 77)
a4_1039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'a4', False)
# Processing the call keyword arguments (line 77)
kwargs_1040 = {}
# Getting the type of 'np' (line 77)
np_1036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 77)
einsum_1037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 6), np_1036, 'einsum')
# Calling einsum(args, kwargs) (line 77)
einsum_call_result_1041 = invoke(stypy.reporting.localization.Localization(__file__, 77, 6), einsum_1037, *[str_1038, a4_1039], **kwargs_1040)

slice_1042 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 77, 6), None, None, None)
# Storing an element on a container (line 77)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 6), einsum_call_result_1041, (slice_1042, int_1035))

# Obtaining the type of the subscript
slice_1043 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 77, 6), None, None, None)

# Call to einsum(...): (line 77)
# Processing the call arguments (line 77)
str_1046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 16), 'str', 'ii->i')
# Getting the type of 'a4' (line 77)
a4_1047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'a4', False)
# Processing the call keyword arguments (line 77)
kwargs_1048 = {}
# Getting the type of 'np' (line 77)
np_1044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 6), 'np', False)
# Obtaining the member 'einsum' of a type (line 77)
einsum_1045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 6), np_1044, 'einsum')
# Calling einsum(args, kwargs) (line 77)
einsum_call_result_1049 = invoke(stypy.reporting.localization.Localization(__file__, 77, 6), einsum_1045, *[str_1046, a4_1047], **kwargs_1048)

# Obtaining the member '__getitem__' of a type (line 77)
getitem___1050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 6), einsum_call_result_1049, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 77)
subscript_call_result_1051 = invoke(stypy.reporting.localization.Localization(__file__, 77, 6), getitem___1050, slice_1043)

# Assigning a type to the variable 'r32' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'r32', subscript_call_result_1051)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
