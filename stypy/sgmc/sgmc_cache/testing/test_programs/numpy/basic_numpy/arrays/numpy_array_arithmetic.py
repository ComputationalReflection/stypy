
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.scipy-lectures.org/intro/numpy/numpy.html
2: 
3: import numpy as np
4: 
5: a = np.array([1, 2, 3, 4])
6: r = a + 1
7: 
8: r2 = 2 ** a
9: 
10: b = np.ones(4) + 1
11: r3 = a - b
12: 
13: r4 = a * b
14: 
15: j = np.arange(5)
16: r5 = 2 ** (j + 1) - j
17: 
18: c = np.ones((3, 3))
19: r6 = c * c  # NOT matrix multiplication!
20: 
21: # Matrix multiplication:
22: 
23: r = c.dot(c)
24: 
25: a = np.array([1, 2, 3, 4])
26: b = np.array([4, 2, 2, 4])
27: r7 = a == b
28: 
29: r8 = a > b
30: 
31: # Logical operations:
32: 
33: a = np.array([1, 1, 0, 0], dtype=bool)
34: b = np.array([1, 0, 1, 0], dtype=bool)
35: r9 = np.logical_or(a, b)
36: 
37: r10 = np.logical_and(a, b)
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
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_60 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_60) is not StypyTypeError):

    if (import_60 != 'pyd_module'):
        __import__(import_60)
        sys_modules_61 = sys.modules[import_60]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_61.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_60)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 5):

# Call to array(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_64 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_64, int_65)
# Adding element type (line 5)
int_66 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_64, int_66)
# Adding element type (line 5)
int_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_64, int_67)
# Adding element type (line 5)
int_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_64, int_68)

# Processing the call keyword arguments (line 5)
kwargs_69 = {}
# Getting the type of 'np' (line 5)
np_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'array' of a type (line 5)
array_63 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_62, 'array')
# Calling array(args, kwargs) (line 5)
array_call_result_70 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), array_63, *[list_64], **kwargs_69)

# Assigning a type to the variable 'a' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'a', array_call_result_70)

# Assigning a BinOp to a Name (line 6):
# Getting the type of 'a' (line 6)
a_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'a')
int_72 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 8), 'int')
# Applying the binary operator '+' (line 6)
result_add_73 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 4), '+', a_71, int_72)

# Assigning a type to the variable 'r' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r', result_add_73)

# Assigning a BinOp to a Name (line 8):
int_74 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 5), 'int')
# Getting the type of 'a' (line 8)
a_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 10), 'a')
# Applying the binary operator '**' (line 8)
result_pow_76 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 5), '**', int_74, a_75)

# Assigning a type to the variable 'r2' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r2', result_pow_76)

# Assigning a BinOp to a Name (line 10):

# Call to ones(...): (line 10)
# Processing the call arguments (line 10)
int_79 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 12), 'int')
# Processing the call keyword arguments (line 10)
kwargs_80 = {}
# Getting the type of 'np' (line 10)
np_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'np', False)
# Obtaining the member 'ones' of a type (line 10)
ones_78 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), np_77, 'ones')
# Calling ones(args, kwargs) (line 10)
ones_call_result_81 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), ones_78, *[int_79], **kwargs_80)

int_82 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 17), 'int')
# Applying the binary operator '+' (line 10)
result_add_83 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 4), '+', ones_call_result_81, int_82)

# Assigning a type to the variable 'b' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'b', result_add_83)

# Assigning a BinOp to a Name (line 11):
# Getting the type of 'a' (line 11)
a_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'a')
# Getting the type of 'b' (line 11)
b_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 9), 'b')
# Applying the binary operator '-' (line 11)
result_sub_86 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 5), '-', a_84, b_85)

# Assigning a type to the variable 'r3' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r3', result_sub_86)

# Assigning a BinOp to a Name (line 13):
# Getting the type of 'a' (line 13)
a_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'a')
# Getting the type of 'b' (line 13)
b_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 9), 'b')
# Applying the binary operator '*' (line 13)
result_mul_89 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 5), '*', a_87, b_88)

# Assigning a type to the variable 'r4' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r4', result_mul_89)

# Assigning a Call to a Name (line 15):

# Call to arange(...): (line 15)
# Processing the call arguments (line 15)
int_92 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 14), 'int')
# Processing the call keyword arguments (line 15)
kwargs_93 = {}
# Getting the type of 'np' (line 15)
np_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'np', False)
# Obtaining the member 'arange' of a type (line 15)
arange_91 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), np_90, 'arange')
# Calling arange(args, kwargs) (line 15)
arange_call_result_94 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), arange_91, *[int_92], **kwargs_93)

# Assigning a type to the variable 'j' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'j', arange_call_result_94)

# Assigning a BinOp to a Name (line 16):
int_95 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 5), 'int')
# Getting the type of 'j' (line 16)
j_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'j')
int_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 15), 'int')
# Applying the binary operator '+' (line 16)
result_add_98 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 11), '+', j_96, int_97)

# Applying the binary operator '**' (line 16)
result_pow_99 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 5), '**', int_95, result_add_98)

# Getting the type of 'j' (line 16)
j_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 20), 'j')
# Applying the binary operator '-' (line 16)
result_sub_101 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 5), '-', result_pow_99, j_100)

# Assigning a type to the variable 'r5' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'r5', result_sub_101)

# Assigning a Call to a Name (line 18):

# Call to ones(...): (line 18)
# Processing the call arguments (line 18)

# Obtaining an instance of the builtin type 'tuple' (line 18)
tuple_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 18)
# Adding element type (line 18)
int_105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 13), tuple_104, int_105)
# Adding element type (line 18)
int_106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 13), tuple_104, int_106)

# Processing the call keyword arguments (line 18)
kwargs_107 = {}
# Getting the type of 'np' (line 18)
np_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'np', False)
# Obtaining the member 'ones' of a type (line 18)
ones_103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 4), np_102, 'ones')
# Calling ones(args, kwargs) (line 18)
ones_call_result_108 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), ones_103, *[tuple_104], **kwargs_107)

# Assigning a type to the variable 'c' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'c', ones_call_result_108)

# Assigning a BinOp to a Name (line 19):
# Getting the type of 'c' (line 19)
c_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'c')
# Getting the type of 'c' (line 19)
c_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 9), 'c')
# Applying the binary operator '*' (line 19)
result_mul_111 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 5), '*', c_109, c_110)

# Assigning a type to the variable 'r6' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r6', result_mul_111)

# Assigning a Call to a Name (line 23):

# Call to dot(...): (line 23)
# Processing the call arguments (line 23)
# Getting the type of 'c' (line 23)
c_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'c', False)
# Processing the call keyword arguments (line 23)
kwargs_115 = {}
# Getting the type of 'c' (line 23)
c_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'c', False)
# Obtaining the member 'dot' of a type (line 23)
dot_113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 4), c_112, 'dot')
# Calling dot(args, kwargs) (line 23)
dot_call_result_116 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), dot_113, *[c_114], **kwargs_115)

# Assigning a type to the variable 'r' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'r', dot_call_result_116)

# Assigning a Call to a Name (line 25):

# Call to array(...): (line 25)
# Processing the call arguments (line 25)

# Obtaining an instance of the builtin type 'list' (line 25)
list_119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
int_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 13), list_119, int_120)
# Adding element type (line 25)
int_121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 13), list_119, int_121)
# Adding element type (line 25)
int_122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 13), list_119, int_122)
# Adding element type (line 25)
int_123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 13), list_119, int_123)

# Processing the call keyword arguments (line 25)
kwargs_124 = {}
# Getting the type of 'np' (line 25)
np_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'np', False)
# Obtaining the member 'array' of a type (line 25)
array_118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), np_117, 'array')
# Calling array(args, kwargs) (line 25)
array_call_result_125 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), array_118, *[list_119], **kwargs_124)

# Assigning a type to the variable 'a' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'a', array_call_result_125)

# Assigning a Call to a Name (line 26):

# Call to array(...): (line 26)
# Processing the call arguments (line 26)

# Obtaining an instance of the builtin type 'list' (line 26)
list_128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
int_129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 13), list_128, int_129)
# Adding element type (line 26)
int_130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 13), list_128, int_130)
# Adding element type (line 26)
int_131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 13), list_128, int_131)
# Adding element type (line 26)
int_132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 13), list_128, int_132)

# Processing the call keyword arguments (line 26)
kwargs_133 = {}
# Getting the type of 'np' (line 26)
np_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'np', False)
# Obtaining the member 'array' of a type (line 26)
array_127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 4), np_126, 'array')
# Calling array(args, kwargs) (line 26)
array_call_result_134 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), array_127, *[list_128], **kwargs_133)

# Assigning a type to the variable 'b' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'b', array_call_result_134)

# Assigning a Compare to a Name (line 27):

# Getting the type of 'a' (line 27)
a_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 5), 'a')
# Getting the type of 'b' (line 27)
b_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 10), 'b')
# Applying the binary operator '==' (line 27)
result_eq_137 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 5), '==', a_135, b_136)

# Assigning a type to the variable 'r7' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'r7', result_eq_137)

# Assigning a Compare to a Name (line 29):

# Getting the type of 'a' (line 29)
a_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 5), 'a')
# Getting the type of 'b' (line 29)
b_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 9), 'b')
# Applying the binary operator '>' (line 29)
result_gt_140 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 5), '>', a_138, b_139)

# Assigning a type to the variable 'r8' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'r8', result_gt_140)

# Assigning a Call to a Name (line 33):

# Call to array(...): (line 33)
# Processing the call arguments (line 33)

# Obtaining an instance of the builtin type 'list' (line 33)
list_143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 33)
# Adding element type (line 33)
int_144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 13), list_143, int_144)
# Adding element type (line 33)
int_145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 13), list_143, int_145)
# Adding element type (line 33)
int_146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 13), list_143, int_146)
# Adding element type (line 33)
int_147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 13), list_143, int_147)

# Processing the call keyword arguments (line 33)
# Getting the type of 'bool' (line 33)
bool_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 33), 'bool', False)
keyword_149 = bool_148
kwargs_150 = {'dtype': keyword_149}
# Getting the type of 'np' (line 33)
np_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'np', False)
# Obtaining the member 'array' of a type (line 33)
array_142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 4), np_141, 'array')
# Calling array(args, kwargs) (line 33)
array_call_result_151 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), array_142, *[list_143], **kwargs_150)

# Assigning a type to the variable 'a' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'a', array_call_result_151)

# Assigning a Call to a Name (line 34):

# Call to array(...): (line 34)
# Processing the call arguments (line 34)

# Obtaining an instance of the builtin type 'list' (line 34)
list_154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 34)
# Adding element type (line 34)
int_155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 13), list_154, int_155)
# Adding element type (line 34)
int_156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 13), list_154, int_156)
# Adding element type (line 34)
int_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 13), list_154, int_157)
# Adding element type (line 34)
int_158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 13), list_154, int_158)

# Processing the call keyword arguments (line 34)
# Getting the type of 'bool' (line 34)
bool_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 33), 'bool', False)
keyword_160 = bool_159
kwargs_161 = {'dtype': keyword_160}
# Getting the type of 'np' (line 34)
np_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'np', False)
# Obtaining the member 'array' of a type (line 34)
array_153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 4), np_152, 'array')
# Calling array(args, kwargs) (line 34)
array_call_result_162 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), array_153, *[list_154], **kwargs_161)

# Assigning a type to the variable 'b' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'b', array_call_result_162)

# Assigning a Call to a Name (line 35):

# Call to logical_or(...): (line 35)
# Processing the call arguments (line 35)
# Getting the type of 'a' (line 35)
a_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'a', False)
# Getting the type of 'b' (line 35)
b_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 22), 'b', False)
# Processing the call keyword arguments (line 35)
kwargs_167 = {}
# Getting the type of 'np' (line 35)
np_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 5), 'np', False)
# Obtaining the member 'logical_or' of a type (line 35)
logical_or_164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 5), np_163, 'logical_or')
# Calling logical_or(args, kwargs) (line 35)
logical_or_call_result_168 = invoke(stypy.reporting.localization.Localization(__file__, 35, 5), logical_or_164, *[a_165, b_166], **kwargs_167)

# Assigning a type to the variable 'r9' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'r9', logical_or_call_result_168)

# Assigning a Call to a Name (line 37):

# Call to logical_and(...): (line 37)
# Processing the call arguments (line 37)
# Getting the type of 'a' (line 37)
a_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 21), 'a', False)
# Getting the type of 'b' (line 37)
b_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 24), 'b', False)
# Processing the call keyword arguments (line 37)
kwargs_173 = {}
# Getting the type of 'np' (line 37)
np_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 6), 'np', False)
# Obtaining the member 'logical_and' of a type (line 37)
logical_and_170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 6), np_169, 'logical_and')
# Calling logical_and(args, kwargs) (line 37)
logical_and_call_result_174 = invoke(stypy.reporting.localization.Localization(__file__, 37, 6), logical_and_170, *[a_171, b_172], **kwargs_173)

# Assigning a type to the variable 'r10' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'r10', logical_and_call_result_174)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
