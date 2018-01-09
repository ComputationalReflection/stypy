
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: a = (3,4)
3: 
4: (b, c) = a
5: 
6: 
7: (test1, (Type1, instance1, _)), (test2, (Type2, instance2, _)) = [('a', (3, 4, 5)), ('b', (10.1, 11.2, 12.3))]
8: 
9: def getlist():
10:     return [('a', (3, 4, 5)), ('b', (10.1, 11.2, 12.3))]
11: 
12: (test1b, (Type1b, instance1b, _)), (test2b, (Type2b, instance2b, _)) = getlist()
13: 
14: 
15: def func():
16:     (test1b, (Type1b, instance1b, _)), (test2b, (Type2b, instance2b, _)) = getlist()
17: 
18: func()
19: 
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Tuple to a Name (line 2):

# Assigning a Tuple to a Name (line 2):

# Assigning a Tuple to a Name (line 2):

# Assigning a Tuple to a Name (line 2):

# Obtaining an instance of the builtin type 'tuple' (line 2)
tuple_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 2)
# Adding element type (line 2)
int_158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 5), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 5), tuple_157, int_158)
# Adding element type (line 2)
int_159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 7), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 2, 5), tuple_157, int_159)

# Assigning a type to the variable 'a' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'a', tuple_157)

# Assigning a Name to a Tuple (line 4):

# Assigning a Subscript to a Name (line 4):

# Assigning a Subscript to a Name (line 4):

# Assigning a Subscript to a Name (line 4):

# Obtaining the type of the subscript
int_160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 0), 'int')
# Getting the type of 'a' (line 4)
a_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 9), 'a')
# Obtaining the member '__getitem__' of a type (line 4)
getitem___162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 0), a_161, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 4)
subscript_call_result_163 = invoke(stypy.reporting.localization.Localization(__file__, 4, 0), getitem___162, int_160)

# Assigning a type to the variable 'tuple_var_assignment_119' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'tuple_var_assignment_119', subscript_call_result_163)

# Assigning a Subscript to a Name (line 4):

# Assigning a Subscript to a Name (line 4):

# Assigning a Subscript to a Name (line 4):

# Obtaining the type of the subscript
int_164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 0), 'int')
# Getting the type of 'a' (line 4)
a_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 9), 'a')
# Obtaining the member '__getitem__' of a type (line 4)
getitem___166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 0), a_165, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 4)
subscript_call_result_167 = invoke(stypy.reporting.localization.Localization(__file__, 4, 0), getitem___166, int_164)

# Assigning a type to the variable 'tuple_var_assignment_120' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'tuple_var_assignment_120', subscript_call_result_167)

# Assigning a Name to a Name (line 4):

# Assigning a Name to a Name (line 4):

# Assigning a Name to a Name (line 4):
# Getting the type of 'tuple_var_assignment_119' (line 4)
tuple_var_assignment_119_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'tuple_var_assignment_119')
# Assigning a type to the variable 'b' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 1), 'b', tuple_var_assignment_119_168)

# Assigning a Name to a Name (line 4):

# Assigning a Name to a Name (line 4):

# Assigning a Name to a Name (line 4):
# Getting the type of 'tuple_var_assignment_120' (line 4)
tuple_var_assignment_120_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'tuple_var_assignment_120')
# Assigning a type to the variable 'c' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'c', tuple_var_assignment_120_169)

# Assigning a List to a Tuple (line 7):

# Assigning a Tuple to a Name (line 7):

# Assigning a Tuple to a Name (line 7):

# Assigning a Tuple to a Name (line 7):

# Obtaining an instance of the builtin type 'tuple' (line 7)
tuple_170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 67), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7)
# Adding element type (line 7)
str_171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 67), 'str', 'a')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 67), tuple_170, str_171)
# Adding element type (line 7)

# Obtaining an instance of the builtin type 'tuple' (line 7)
tuple_172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 73), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7)
# Adding element type (line 7)
int_173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 73), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 73), tuple_172, int_173)
# Adding element type (line 7)
int_174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 76), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 73), tuple_172, int_174)
# Adding element type (line 7)
int_175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 79), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 73), tuple_172, int_175)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 67), tuple_170, tuple_172)

# Assigning a type to the variable 'tuple_assignment_121' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_121', tuple_170)

# Assigning a Tuple to a Name (line 7):

# Assigning a Tuple to a Name (line 7):

# Assigning a Tuple to a Name (line 7):

# Obtaining an instance of the builtin type 'tuple' (line 7)
tuple_176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 85), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7)
# Adding element type (line 7)
str_177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 85), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 85), tuple_176, str_177)
# Adding element type (line 7)

# Obtaining an instance of the builtin type 'tuple' (line 7)
tuple_178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 91), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7)
# Adding element type (line 7)
float_179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 91), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 91), tuple_178, float_179)
# Adding element type (line 7)
float_180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 97), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 91), tuple_178, float_180)
# Adding element type (line 7)
float_181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 103), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 91), tuple_178, float_181)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 85), tuple_176, tuple_178)

# Assigning a type to the variable 'tuple_assignment_122' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_122', tuple_176)

# Assigning a Name to a Tuple (line 7):

# Assigning a Subscript to a Name (line 7):

# Assigning a Subscript to a Name (line 7):

# Obtaining the type of the subscript
int_182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 0), 'int')
# Getting the type of 'tuple_assignment_121' (line 7)
tuple_assignment_121_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_121')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 0), tuple_assignment_121_183, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_185 = invoke(stypy.reporting.localization.Localization(__file__, 7, 0), getitem___184, int_182)

# Assigning a type to the variable 'tuple_var_assignment_127' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_127', subscript_call_result_185)

# Assigning a Subscript to a Name (line 7):

# Assigning a Subscript to a Name (line 7):

# Obtaining the type of the subscript
int_186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 0), 'int')
# Getting the type of 'tuple_assignment_121' (line 7)
tuple_assignment_121_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_121')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 0), tuple_assignment_121_187, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_189 = invoke(stypy.reporting.localization.Localization(__file__, 7, 0), getitem___188, int_186)

# Assigning a type to the variable 'tuple_var_assignment_128' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_128', subscript_call_result_189)

# Assigning a Name to a Name (line 7):

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_var_assignment_127' (line 7)
tuple_var_assignment_127_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_127')
# Assigning a type to the variable 'test1' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 1), 'test1', tuple_var_assignment_127_190)

# Assigning a Name to a Tuple (line 7):

# Assigning a Subscript to a Name (line 7):

# Obtaining the type of the subscript
int_191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 0), 'int')
# Getting the type of 'tuple_var_assignment_128' (line 7)
tuple_var_assignment_128_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_128')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 0), tuple_var_assignment_128_192, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_194 = invoke(stypy.reporting.localization.Localization(__file__, 7, 0), getitem___193, int_191)

# Assigning a type to the variable 'tuple_var_assignment_139' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_139', subscript_call_result_194)

# Assigning a Subscript to a Name (line 7):

# Obtaining the type of the subscript
int_195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 0), 'int')
# Getting the type of 'tuple_var_assignment_128' (line 7)
tuple_var_assignment_128_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_128')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 0), tuple_var_assignment_128_196, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_198 = invoke(stypy.reporting.localization.Localization(__file__, 7, 0), getitem___197, int_195)

# Assigning a type to the variable 'tuple_var_assignment_140' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_140', subscript_call_result_198)

# Assigning a Subscript to a Name (line 7):

# Obtaining the type of the subscript
int_199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 0), 'int')
# Getting the type of 'tuple_var_assignment_128' (line 7)
tuple_var_assignment_128_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_128')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 0), tuple_var_assignment_128_200, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_202 = invoke(stypy.reporting.localization.Localization(__file__, 7, 0), getitem___201, int_199)

# Assigning a type to the variable 'tuple_var_assignment_141' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_141', subscript_call_result_202)

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_var_assignment_139' (line 7)
tuple_var_assignment_139_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_139')
# Assigning a type to the variable 'Type1' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'Type1', tuple_var_assignment_139_203)

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_var_assignment_140' (line 7)
tuple_var_assignment_140_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_140')
# Assigning a type to the variable 'instance1' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 16), 'instance1', tuple_var_assignment_140_204)

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_var_assignment_141' (line 7)
tuple_var_assignment_141_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_141')
# Assigning a type to the variable '_' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 27), '_', tuple_var_assignment_141_205)

# Assigning a Name to a Tuple (line 7):

# Assigning a Subscript to a Name (line 7):

# Assigning a Subscript to a Name (line 7):

# Obtaining the type of the subscript
int_206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 0), 'int')
# Getting the type of 'tuple_assignment_122' (line 7)
tuple_assignment_122_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_122')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 0), tuple_assignment_122_207, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_209 = invoke(stypy.reporting.localization.Localization(__file__, 7, 0), getitem___208, int_206)

# Assigning a type to the variable 'tuple_var_assignment_129' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_129', subscript_call_result_209)

# Assigning a Subscript to a Name (line 7):

# Assigning a Subscript to a Name (line 7):

# Obtaining the type of the subscript
int_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 0), 'int')
# Getting the type of 'tuple_assignment_122' (line 7)
tuple_assignment_122_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_assignment_122')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 0), tuple_assignment_122_211, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 7, 0), getitem___212, int_210)

# Assigning a type to the variable 'tuple_var_assignment_130' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_130', subscript_call_result_213)

# Assigning a Name to a Name (line 7):

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_var_assignment_129' (line 7)
tuple_var_assignment_129_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_129')
# Assigning a type to the variable 'test2' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 33), 'test2', tuple_var_assignment_129_214)

# Assigning a Name to a Tuple (line 7):

# Assigning a Subscript to a Name (line 7):

# Obtaining the type of the subscript
int_215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 0), 'int')
# Getting the type of 'tuple_var_assignment_130' (line 7)
tuple_var_assignment_130_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_130')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 0), tuple_var_assignment_130_216, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_218 = invoke(stypy.reporting.localization.Localization(__file__, 7, 0), getitem___217, int_215)

# Assigning a type to the variable 'tuple_var_assignment_142' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_142', subscript_call_result_218)

# Assigning a Subscript to a Name (line 7):

# Obtaining the type of the subscript
int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 0), 'int')
# Getting the type of 'tuple_var_assignment_130' (line 7)
tuple_var_assignment_130_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_130')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 0), tuple_var_assignment_130_220, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_222 = invoke(stypy.reporting.localization.Localization(__file__, 7, 0), getitem___221, int_219)

# Assigning a type to the variable 'tuple_var_assignment_143' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_143', subscript_call_result_222)

# Assigning a Subscript to a Name (line 7):

# Obtaining the type of the subscript
int_223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 0), 'int')
# Getting the type of 'tuple_var_assignment_130' (line 7)
tuple_var_assignment_130_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_130')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 0), tuple_var_assignment_130_224, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_226 = invoke(stypy.reporting.localization.Localization(__file__, 7, 0), getitem___225, int_223)

# Assigning a type to the variable 'tuple_var_assignment_144' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_144', subscript_call_result_226)

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_var_assignment_142' (line 7)
tuple_var_assignment_142_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_142')
# Assigning a type to the variable 'Type2' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 41), 'Type2', tuple_var_assignment_142_227)

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_var_assignment_143' (line 7)
tuple_var_assignment_143_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_143')
# Assigning a type to the variable 'instance2' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 48), 'instance2', tuple_var_assignment_143_228)

# Assigning a Name to a Name (line 7):
# Getting the type of 'tuple_var_assignment_144' (line 7)
tuple_var_assignment_144_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'tuple_var_assignment_144')
# Assigning a type to the variable '_' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 59), '_', tuple_var_assignment_144_229)

@norecursion
def getlist(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getlist'
    module_type_store = module_type_store.open_function_context('getlist', 9, 0, False)
    
    # Passed parameters checking function
    getlist.stypy_localization = localization
    getlist.stypy_type_of_self = None
    getlist.stypy_type_store = module_type_store
    getlist.stypy_function_name = 'getlist'
    getlist.stypy_param_names_list = []
    getlist.stypy_varargs_param_name = None
    getlist.stypy_kwargs_param_name = None
    getlist.stypy_call_defaults = defaults
    getlist.stypy_call_varargs = varargs
    getlist.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getlist', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getlist', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getlist(...)' code ##################

    
    # Obtaining an instance of the builtin type 'list' (line 10)
    list_230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 10)
    # Adding element type (line 10)
    
    # Obtaining an instance of the builtin type 'tuple' (line 10)
    tuple_231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 10)
    # Adding element type (line 10)
    str_232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 13), 'str', 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 13), tuple_231, str_232)
    # Adding element type (line 10)
    
    # Obtaining an instance of the builtin type 'tuple' (line 10)
    tuple_233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 10)
    # Adding element type (line 10)
    int_234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 19), tuple_233, int_234)
    # Adding element type (line 10)
    int_235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 19), tuple_233, int_235)
    # Adding element type (line 10)
    int_236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 19), tuple_233, int_236)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 13), tuple_231, tuple_233)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 11), list_230, tuple_231)
    # Adding element type (line 10)
    
    # Obtaining an instance of the builtin type 'tuple' (line 10)
    tuple_237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 10)
    # Adding element type (line 10)
    str_238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 31), 'str', 'b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 31), tuple_237, str_238)
    # Adding element type (line 10)
    
    # Obtaining an instance of the builtin type 'tuple' (line 10)
    tuple_239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 10)
    # Adding element type (line 10)
    float_240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 37), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 37), tuple_239, float_240)
    # Adding element type (line 10)
    float_241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 43), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 37), tuple_239, float_241)
    # Adding element type (line 10)
    float_242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 49), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 37), tuple_239, float_242)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 31), tuple_237, tuple_239)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 11), list_230, tuple_237)
    
    # Assigning a type to the variable 'stypy_return_type' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type', list_230)
    
    # ################# End of 'getlist(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getlist' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_243)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getlist'
    return stypy_return_type_243

# Assigning a type to the variable 'getlist' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'getlist', getlist)

# Assigning a Call to a Tuple (line 12):

# Assigning a Subscript to a Name (line 12):

# Assigning a Subscript to a Name (line 12):

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
int_244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 0), 'int')

# Call to getlist(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_246 = {}
# Getting the type of 'getlist' (line 12)
getlist_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 71), 'getlist', False)
# Calling getlist(args, kwargs) (line 12)
getlist_call_result_247 = invoke(stypy.reporting.localization.Localization(__file__, 12, 71), getlist_245, *[], **kwargs_246)

# Obtaining the member '__getitem__' of a type (line 12)
getitem___248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 0), getlist_call_result_247, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_249 = invoke(stypy.reporting.localization.Localization(__file__, 12, 0), getitem___248, int_244)

# Assigning a type to the variable 'tuple_var_assignment_123' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_123', subscript_call_result_249)

# Assigning a Subscript to a Name (line 12):

# Assigning a Subscript to a Name (line 12):

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
int_250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 0), 'int')

# Call to getlist(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_252 = {}
# Getting the type of 'getlist' (line 12)
getlist_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 71), 'getlist', False)
# Calling getlist(args, kwargs) (line 12)
getlist_call_result_253 = invoke(stypy.reporting.localization.Localization(__file__, 12, 71), getlist_251, *[], **kwargs_252)

# Obtaining the member '__getitem__' of a type (line 12)
getitem___254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 0), getlist_call_result_253, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 12, 0), getitem___254, int_250)

# Assigning a type to the variable 'tuple_var_assignment_124' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_124', subscript_call_result_255)

# Assigning a Name to a Tuple (line 12):

# Assigning a Subscript to a Name (line 12):

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
int_256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 0), 'int')
# Getting the type of 'tuple_var_assignment_123' (line 12)
tuple_var_assignment_123_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_123')
# Obtaining the member '__getitem__' of a type (line 12)
getitem___258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 0), tuple_var_assignment_123_257, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_259 = invoke(stypy.reporting.localization.Localization(__file__, 12, 0), getitem___258, int_256)

# Assigning a type to the variable 'tuple_var_assignment_131' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_131', subscript_call_result_259)

# Assigning a Subscript to a Name (line 12):

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
int_260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 0), 'int')
# Getting the type of 'tuple_var_assignment_123' (line 12)
tuple_var_assignment_123_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_123')
# Obtaining the member '__getitem__' of a type (line 12)
getitem___262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 0), tuple_var_assignment_123_261, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_263 = invoke(stypy.reporting.localization.Localization(__file__, 12, 0), getitem___262, int_260)

# Assigning a type to the variable 'tuple_var_assignment_132' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_132', subscript_call_result_263)

# Assigning a Name to a Name (line 12):

# Assigning a Name to a Name (line 12):
# Getting the type of 'tuple_var_assignment_131' (line 12)
tuple_var_assignment_131_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_131')
# Assigning a type to the variable 'test1b' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 1), 'test1b', tuple_var_assignment_131_264)

# Assigning a Name to a Tuple (line 12):

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
int_265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 0), 'int')
# Getting the type of 'tuple_var_assignment_132' (line 12)
tuple_var_assignment_132_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_132')
# Obtaining the member '__getitem__' of a type (line 12)
getitem___267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 0), tuple_var_assignment_132_266, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_268 = invoke(stypy.reporting.localization.Localization(__file__, 12, 0), getitem___267, int_265)

# Assigning a type to the variable 'tuple_var_assignment_145' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_145', subscript_call_result_268)

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
int_269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 0), 'int')
# Getting the type of 'tuple_var_assignment_132' (line 12)
tuple_var_assignment_132_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_132')
# Obtaining the member '__getitem__' of a type (line 12)
getitem___271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 0), tuple_var_assignment_132_270, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_272 = invoke(stypy.reporting.localization.Localization(__file__, 12, 0), getitem___271, int_269)

# Assigning a type to the variable 'tuple_var_assignment_146' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_146', subscript_call_result_272)

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
int_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 0), 'int')
# Getting the type of 'tuple_var_assignment_132' (line 12)
tuple_var_assignment_132_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_132')
# Obtaining the member '__getitem__' of a type (line 12)
getitem___275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 0), tuple_var_assignment_132_274, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_276 = invoke(stypy.reporting.localization.Localization(__file__, 12, 0), getitem___275, int_273)

# Assigning a type to the variable 'tuple_var_assignment_147' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_147', subscript_call_result_276)

# Assigning a Name to a Name (line 12):
# Getting the type of 'tuple_var_assignment_145' (line 12)
tuple_var_assignment_145_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_145')
# Assigning a type to the variable 'Type1b' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'Type1b', tuple_var_assignment_145_277)

# Assigning a Name to a Name (line 12):
# Getting the type of 'tuple_var_assignment_146' (line 12)
tuple_var_assignment_146_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_146')
# Assigning a type to the variable 'instance1b' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 18), 'instance1b', tuple_var_assignment_146_278)

# Assigning a Name to a Name (line 12):
# Getting the type of 'tuple_var_assignment_147' (line 12)
tuple_var_assignment_147_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_147')
# Assigning a type to the variable '_' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 30), '_', tuple_var_assignment_147_279)

# Assigning a Name to a Tuple (line 12):

# Assigning a Subscript to a Name (line 12):

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 0), 'int')
# Getting the type of 'tuple_var_assignment_124' (line 12)
tuple_var_assignment_124_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_124')
# Obtaining the member '__getitem__' of a type (line 12)
getitem___282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 0), tuple_var_assignment_124_281, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_283 = invoke(stypy.reporting.localization.Localization(__file__, 12, 0), getitem___282, int_280)

# Assigning a type to the variable 'tuple_var_assignment_133' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_133', subscript_call_result_283)

# Assigning a Subscript to a Name (line 12):

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
int_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 0), 'int')
# Getting the type of 'tuple_var_assignment_124' (line 12)
tuple_var_assignment_124_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_124')
# Obtaining the member '__getitem__' of a type (line 12)
getitem___286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 0), tuple_var_assignment_124_285, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_287 = invoke(stypy.reporting.localization.Localization(__file__, 12, 0), getitem___286, int_284)

# Assigning a type to the variable 'tuple_var_assignment_134' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_134', subscript_call_result_287)

# Assigning a Name to a Name (line 12):

# Assigning a Name to a Name (line 12):
# Getting the type of 'tuple_var_assignment_133' (line 12)
tuple_var_assignment_133_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_133')
# Assigning a type to the variable 'test2b' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 36), 'test2b', tuple_var_assignment_133_288)

# Assigning a Name to a Tuple (line 12):

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
int_289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 0), 'int')
# Getting the type of 'tuple_var_assignment_134' (line 12)
tuple_var_assignment_134_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_134')
# Obtaining the member '__getitem__' of a type (line 12)
getitem___291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 0), tuple_var_assignment_134_290, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_292 = invoke(stypy.reporting.localization.Localization(__file__, 12, 0), getitem___291, int_289)

# Assigning a type to the variable 'tuple_var_assignment_148' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_148', subscript_call_result_292)

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
int_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 0), 'int')
# Getting the type of 'tuple_var_assignment_134' (line 12)
tuple_var_assignment_134_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_134')
# Obtaining the member '__getitem__' of a type (line 12)
getitem___295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 0), tuple_var_assignment_134_294, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_296 = invoke(stypy.reporting.localization.Localization(__file__, 12, 0), getitem___295, int_293)

# Assigning a type to the variable 'tuple_var_assignment_149' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_149', subscript_call_result_296)

# Assigning a Subscript to a Name (line 12):

# Obtaining the type of the subscript
int_297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 0), 'int')
# Getting the type of 'tuple_var_assignment_134' (line 12)
tuple_var_assignment_134_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_134')
# Obtaining the member '__getitem__' of a type (line 12)
getitem___299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 0), tuple_var_assignment_134_298, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_300 = invoke(stypy.reporting.localization.Localization(__file__, 12, 0), getitem___299, int_297)

# Assigning a type to the variable 'tuple_var_assignment_150' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_150', subscript_call_result_300)

# Assigning a Name to a Name (line 12):
# Getting the type of 'tuple_var_assignment_148' (line 12)
tuple_var_assignment_148_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_148')
# Assigning a type to the variable 'Type2b' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 45), 'Type2b', tuple_var_assignment_148_301)

# Assigning a Name to a Name (line 12):
# Getting the type of 'tuple_var_assignment_149' (line 12)
tuple_var_assignment_149_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_149')
# Assigning a type to the variable 'instance2b' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 53), 'instance2b', tuple_var_assignment_149_302)

# Assigning a Name to a Name (line 12):
# Getting the type of 'tuple_var_assignment_150' (line 12)
tuple_var_assignment_150_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'tuple_var_assignment_150')
# Assigning a type to the variable '_' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 65), '_', tuple_var_assignment_150_303)

@norecursion
def func(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'func'
    module_type_store = module_type_store.open_function_context('func', 15, 0, False)
    
    # Passed parameters checking function
    func.stypy_localization = localization
    func.stypy_type_of_self = None
    func.stypy_type_store = module_type_store
    func.stypy_function_name = 'func'
    func.stypy_param_names_list = []
    func.stypy_varargs_param_name = None
    func.stypy_kwargs_param_name = None
    func.stypy_call_defaults = defaults
    func.stypy_call_varargs = varargs
    func.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'func', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'func', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'func(...)' code ##################

    
    # Assigning a Call to a Tuple (line 16):
    
    # Assigning a Subscript to a Name (line 16):
    
    # Assigning a Subscript to a Name (line 16):
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    int_304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'int')
    
    # Call to getlist(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_306 = {}
    # Getting the type of 'getlist' (line 16)
    getlist_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 75), 'getlist', False)
    # Calling getlist(args, kwargs) (line 16)
    getlist_call_result_307 = invoke(stypy.reporting.localization.Localization(__file__, 16, 75), getlist_305, *[], **kwargs_306)
    
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), getlist_call_result_307, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_309 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), getitem___308, int_304)
    
    # Assigning a type to the variable 'tuple_var_assignment_125' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_125', subscript_call_result_309)
    
    # Assigning a Subscript to a Name (line 16):
    
    # Assigning a Subscript to a Name (line 16):
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    int_310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'int')
    
    # Call to getlist(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_312 = {}
    # Getting the type of 'getlist' (line 16)
    getlist_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 75), 'getlist', False)
    # Calling getlist(args, kwargs) (line 16)
    getlist_call_result_313 = invoke(stypy.reporting.localization.Localization(__file__, 16, 75), getlist_311, *[], **kwargs_312)
    
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), getlist_call_result_313, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_315 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), getitem___314, int_310)
    
    # Assigning a type to the variable 'tuple_var_assignment_126' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_126', subscript_call_result_315)
    
    # Assigning a Name to a Tuple (line 16):
    
    # Assigning a Subscript to a Name (line 16):
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    int_316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'int')
    # Getting the type of 'tuple_var_assignment_125' (line 16)
    tuple_var_assignment_125_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_125')
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), tuple_var_assignment_125_317, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_319 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), getitem___318, int_316)
    
    # Assigning a type to the variable 'tuple_var_assignment_135' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_135', subscript_call_result_319)
    
    # Assigning a Subscript to a Name (line 16):
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    int_320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'int')
    # Getting the type of 'tuple_var_assignment_125' (line 16)
    tuple_var_assignment_125_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_125')
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), tuple_var_assignment_125_321, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_323 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), getitem___322, int_320)
    
    # Assigning a type to the variable 'tuple_var_assignment_136' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_136', subscript_call_result_323)
    
    # Assigning a Name to a Name (line 16):
    
    # Assigning a Name to a Name (line 16):
    # Getting the type of 'tuple_var_assignment_135' (line 16)
    tuple_var_assignment_135_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_135')
    # Assigning a type to the variable 'test1b' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 5), 'test1b', tuple_var_assignment_135_324)
    
    # Assigning a Name to a Tuple (line 16):
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    int_325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'int')
    # Getting the type of 'tuple_var_assignment_136' (line 16)
    tuple_var_assignment_136_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_136')
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), tuple_var_assignment_136_326, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_328 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), getitem___327, int_325)
    
    # Assigning a type to the variable 'tuple_var_assignment_151' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_151', subscript_call_result_328)
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    int_329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'int')
    # Getting the type of 'tuple_var_assignment_136' (line 16)
    tuple_var_assignment_136_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_136')
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), tuple_var_assignment_136_330, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_332 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), getitem___331, int_329)
    
    # Assigning a type to the variable 'tuple_var_assignment_152' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_152', subscript_call_result_332)
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    int_333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'int')
    # Getting the type of 'tuple_var_assignment_136' (line 16)
    tuple_var_assignment_136_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_136')
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), tuple_var_assignment_136_334, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_336 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), getitem___335, int_333)
    
    # Assigning a type to the variable 'tuple_var_assignment_153' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_153', subscript_call_result_336)
    
    # Assigning a Name to a Name (line 16):
    # Getting the type of 'tuple_var_assignment_151' (line 16)
    tuple_var_assignment_151_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_151')
    # Assigning a type to the variable 'Type1b' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'Type1b', tuple_var_assignment_151_337)
    
    # Assigning a Name to a Name (line 16):
    # Getting the type of 'tuple_var_assignment_152' (line 16)
    tuple_var_assignment_152_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_152')
    # Assigning a type to the variable 'instance1b' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 22), 'instance1b', tuple_var_assignment_152_338)
    
    # Assigning a Name to a Name (line 16):
    # Getting the type of 'tuple_var_assignment_153' (line 16)
    tuple_var_assignment_153_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_153')
    # Assigning a type to the variable '_' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 34), '_', tuple_var_assignment_153_339)
    
    # Assigning a Name to a Tuple (line 16):
    
    # Assigning a Subscript to a Name (line 16):
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    int_340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'int')
    # Getting the type of 'tuple_var_assignment_126' (line 16)
    tuple_var_assignment_126_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_126')
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), tuple_var_assignment_126_341, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_343 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), getitem___342, int_340)
    
    # Assigning a type to the variable 'tuple_var_assignment_137' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_137', subscript_call_result_343)
    
    # Assigning a Subscript to a Name (line 16):
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    int_344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'int')
    # Getting the type of 'tuple_var_assignment_126' (line 16)
    tuple_var_assignment_126_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_126')
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), tuple_var_assignment_126_345, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_347 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), getitem___346, int_344)
    
    # Assigning a type to the variable 'tuple_var_assignment_138' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_138', subscript_call_result_347)
    
    # Assigning a Name to a Name (line 16):
    
    # Assigning a Name to a Name (line 16):
    # Getting the type of 'tuple_var_assignment_137' (line 16)
    tuple_var_assignment_137_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_137')
    # Assigning a type to the variable 'test2b' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 40), 'test2b', tuple_var_assignment_137_348)
    
    # Assigning a Name to a Tuple (line 16):
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    int_349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'int')
    # Getting the type of 'tuple_var_assignment_138' (line 16)
    tuple_var_assignment_138_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_138')
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), tuple_var_assignment_138_350, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_352 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), getitem___351, int_349)
    
    # Assigning a type to the variable 'tuple_var_assignment_154' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_154', subscript_call_result_352)
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    int_353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'int')
    # Getting the type of 'tuple_var_assignment_138' (line 16)
    tuple_var_assignment_138_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_138')
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), tuple_var_assignment_138_354, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_356 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), getitem___355, int_353)
    
    # Assigning a type to the variable 'tuple_var_assignment_155' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_155', subscript_call_result_356)
    
    # Assigning a Subscript to a Name (line 16):
    
    # Obtaining the type of the subscript
    int_357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'int')
    # Getting the type of 'tuple_var_assignment_138' (line 16)
    tuple_var_assignment_138_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_138')
    # Obtaining the member '__getitem__' of a type (line 16)
    getitem___359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), tuple_var_assignment_138_358, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 16)
    subscript_call_result_360 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), getitem___359, int_357)
    
    # Assigning a type to the variable 'tuple_var_assignment_156' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_156', subscript_call_result_360)
    
    # Assigning a Name to a Name (line 16):
    # Getting the type of 'tuple_var_assignment_154' (line 16)
    tuple_var_assignment_154_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_154')
    # Assigning a type to the variable 'Type2b' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 49), 'Type2b', tuple_var_assignment_154_361)
    
    # Assigning a Name to a Name (line 16):
    # Getting the type of 'tuple_var_assignment_155' (line 16)
    tuple_var_assignment_155_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_155')
    # Assigning a type to the variable 'instance2b' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 57), 'instance2b', tuple_var_assignment_155_362)
    
    # Assigning a Name to a Name (line 16):
    # Getting the type of 'tuple_var_assignment_156' (line 16)
    tuple_var_assignment_156_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'tuple_var_assignment_156')
    # Assigning a type to the variable '_' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 69), '_', tuple_var_assignment_156_363)
    
    # ################# End of 'func(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'func' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_364)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'func'
    return stypy_return_type_364

# Assigning a type to the variable 'func' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'func', func)

# Call to func(...): (line 18)
# Processing the call keyword arguments (line 18)
kwargs_366 = {}
# Getting the type of 'func' (line 18)
func_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'func', False)
# Calling func(args, kwargs) (line 18)
func_call_result_367 = invoke(stypy.reporting.localization.Localization(__file__, 18, 0), func_365, *[], **kwargs_366)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
