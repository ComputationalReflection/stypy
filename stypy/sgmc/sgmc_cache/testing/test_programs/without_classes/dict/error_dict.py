
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: dict1 = {'a': 1, 'b': 2}
2: r0 = dict1[3]  # Detected
3: 
4: dict2 = {'a': 1, 'b': 2, 3: 'hola'}
5: 
6: r1 = dict2[list]  # More than one type of keys -> everything is possible
7: 
8: if True:
9:     dict3 = {'a': 1, 'b': 2, 3: 'hola'}
10: else:
11:     dict3 = {'a': 1, 'b': 2, 3: 'hola'}
12: 
13: r2 = dict3[list]  # Not detected again
14: 
15: S = [x ** 2 for x in range(10)]
16: d4 = {k: k for k in S}
17: r3 = d4["hi"]  # Comprehension-generated dicts are not analyzed
18: 
19: noprimes = [j for i in range(2, 8) for j in range(i * 2, 50, i)]
20: primes = [x for x in range(2, 50) if x not in noprimes]
21: d5 = {k: k for k in primes}
22: r4 = d5["hi"]  # Not detected again
23: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Dict to a Name (line 1):

# Obtaining an instance of the builtin type 'dict' (line 1)
dict_7208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 8), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 1)
# Adding element type (key, value) (line 1)
str_7209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 9), 'str', 'a')
int_7210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 14), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 8), dict_7208, (str_7209, int_7210))
# Adding element type (key, value) (line 1)
str_7211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 17), 'str', 'b')
int_7212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 22), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1, 8), dict_7208, (str_7211, int_7212))

# Assigning a type to the variable 'dict1' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'dict1', dict_7208)

# Assigning a Subscript to a Name (line 2):

# Obtaining the type of the subscript
int_7213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'int')
# Getting the type of 'dict1' (line 2)
dict1_7214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 5), 'dict1')
# Obtaining the member '__getitem__' of a type (line 2)
getitem___7215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2, 5), dict1_7214, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 2)
subscript_call_result_7216 = invoke(stypy.reporting.localization.Localization(__file__, 2, 5), getitem___7215, int_7213)

# Assigning a type to the variable 'r0' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'r0', subscript_call_result_7216)

# Assigning a Dict to a Name (line 4):

# Obtaining an instance of the builtin type 'dict' (line 4)
dict_7217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 8), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 4)
# Adding element type (key, value) (line 4)
str_7218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 9), 'str', 'a')
int_7219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 14), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 8), dict_7217, (str_7218, int_7219))
# Adding element type (key, value) (line 4)
str_7220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 17), 'str', 'b')
int_7221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 22), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 8), dict_7217, (str_7220, int_7221))
# Adding element type (key, value) (line 4)
int_7222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 25), 'int')
str_7223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 28), 'str', 'hola')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 8), dict_7217, (int_7222, str_7223))

# Assigning a type to the variable 'dict2' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'dict2', dict_7217)

# Assigning a Subscript to a Name (line 6):

# Obtaining the type of the subscript
# Getting the type of 'list' (line 6)
list_7224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 11), 'list')
# Getting the type of 'dict2' (line 6)
dict2_7225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'dict2')
# Obtaining the member '__getitem__' of a type (line 6)
getitem___7226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 5), dict2_7225, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 6)
subscript_call_result_7227 = invoke(stypy.reporting.localization.Localization(__file__, 6, 5), getitem___7226, list_7224)

# Assigning a type to the variable 'r1' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r1', subscript_call_result_7227)

# Getting the type of 'True' (line 8)
True_7228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 3), 'True')
# Testing the type of an if condition (line 8)
if_condition_7229 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 8, 0), True_7228)
# Assigning a type to the variable 'if_condition_7229' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'if_condition_7229', if_condition_7229)
# SSA begins for if statement (line 8)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Dict to a Name (line 9):

# Obtaining an instance of the builtin type 'dict' (line 9)
dict_7230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 9)
# Adding element type (key, value) (line 9)
str_7231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'str', 'a')
int_7232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 18), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 12), dict_7230, (str_7231, int_7232))
# Adding element type (key, value) (line 9)
str_7233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 21), 'str', 'b')
int_7234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 26), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 12), dict_7230, (str_7233, int_7234))
# Adding element type (key, value) (line 9)
int_7235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 29), 'int')
str_7236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 32), 'str', 'hola')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 12), dict_7230, (int_7235, str_7236))

# Assigning a type to the variable 'dict3' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'dict3', dict_7230)
# SSA branch for the else part of an if statement (line 8)
module_type_store.open_ssa_branch('else')

# Assigning a Dict to a Name (line 11):

# Obtaining an instance of the builtin type 'dict' (line 11)
dict_7237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 11)
# Adding element type (key, value) (line 11)
str_7238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 13), 'str', 'a')
int_7239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 12), dict_7237, (str_7238, int_7239))
# Adding element type (key, value) (line 11)
str_7240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 21), 'str', 'b')
int_7241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 12), dict_7237, (str_7240, int_7241))
# Adding element type (key, value) (line 11)
int_7242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 29), 'int')
str_7243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 32), 'str', 'hola')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 12), dict_7237, (int_7242, str_7243))

# Assigning a type to the variable 'dict3' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'dict3', dict_7237)
# SSA join for if statement (line 8)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Subscript to a Name (line 13):

# Obtaining the type of the subscript
# Getting the type of 'list' (line 13)
list_7244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'list')
# Getting the type of 'dict3' (line 13)
dict3_7245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'dict3')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___7246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), dict3_7245, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_7247 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), getitem___7246, list_7244)

# Assigning a type to the variable 'r2' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r2', subscript_call_result_7247)

# Assigning a ListComp to a Name (line 15):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 15)
# Processing the call arguments (line 15)
int_7252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 27), 'int')
# Processing the call keyword arguments (line 15)
kwargs_7253 = {}
# Getting the type of 'range' (line 15)
range_7251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 21), 'range', False)
# Calling range(args, kwargs) (line 15)
range_call_result_7254 = invoke(stypy.reporting.localization.Localization(__file__, 15, 21), range_7251, *[int_7252], **kwargs_7253)

comprehension_7255 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 5), range_call_result_7254)
# Assigning a type to the variable 'x' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'x', comprehension_7255)
# Getting the type of 'x' (line 15)
x_7248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'x')
int_7249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'int')
# Applying the binary operator '**' (line 15)
result_pow_7250 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 5), '**', x_7248, int_7249)

list_7256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 5), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 5), list_7256, result_pow_7250)
# Assigning a type to the variable 'S' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'S', list_7256)

# Assigning a DictComp to a Name (line 16):
# Calculating dict comprehension
module_type_store = module_type_store.open_function_context('dict comprehension expression', 16, 6, True)
# Calculating comprehension expression
# Getting the type of 'S' (line 16)
S_7259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 20), 'S')
comprehension_7260 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 6), S_7259)
# Assigning a type to the variable 'k' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 6), 'k', comprehension_7260)
# Getting the type of 'k' (line 16)
k_7257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 6), 'k')
# Getting the type of 'k' (line 16)
k_7258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 9), 'k')
dict_7261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 6), 'dict')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 6), dict_7261, (k_7257, k_7258))
# Assigning a type to the variable 'd4' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'd4', dict_7261)

# Assigning a Subscript to a Name (line 17):

# Obtaining the type of the subscript
str_7262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'str', 'hi')
# Getting the type of 'd4' (line 17)
d4_7263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'd4')
# Obtaining the member '__getitem__' of a type (line 17)
getitem___7264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), d4_7263, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 17)
subscript_call_result_7265 = invoke(stypy.reporting.localization.Localization(__file__, 17, 5), getitem___7264, str_7262)

# Assigning a type to the variable 'r3' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r3', subscript_call_result_7265)

# Assigning a ListComp to a Name (line 19):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 19)
# Processing the call arguments (line 19)
int_7268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 29), 'int')
int_7269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 32), 'int')
# Processing the call keyword arguments (line 19)
kwargs_7270 = {}
# Getting the type of 'range' (line 19)
range_7267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'range', False)
# Calling range(args, kwargs) (line 19)
range_call_result_7271 = invoke(stypy.reporting.localization.Localization(__file__, 19, 23), range_7267, *[int_7268, int_7269], **kwargs_7270)

comprehension_7272 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 12), range_call_result_7271)
# Assigning a type to the variable 'i' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'i', comprehension_7272)
# Calculating comprehension expression

# Call to range(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of 'i' (line 19)
i_7274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 50), 'i', False)
int_7275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 54), 'int')
# Applying the binary operator '*' (line 19)
result_mul_7276 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 50), '*', i_7274, int_7275)

int_7277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 57), 'int')
# Getting the type of 'i' (line 19)
i_7278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 61), 'i', False)
# Processing the call keyword arguments (line 19)
kwargs_7279 = {}
# Getting the type of 'range' (line 19)
range_7273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 44), 'range', False)
# Calling range(args, kwargs) (line 19)
range_call_result_7280 = invoke(stypy.reporting.localization.Localization(__file__, 19, 44), range_7273, *[result_mul_7276, int_7277, i_7278], **kwargs_7279)

comprehension_7281 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 12), range_call_result_7280)
# Assigning a type to the variable 'j' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'j', comprehension_7281)
# Getting the type of 'j' (line 19)
j_7266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'j')
list_7282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 12), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 12), list_7282, j_7266)
# Assigning a type to the variable 'noprimes' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'noprimes', list_7282)

# Assigning a ListComp to a Name (line 20):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 20)
# Processing the call arguments (line 20)
int_7288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 27), 'int')
int_7289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 30), 'int')
# Processing the call keyword arguments (line 20)
kwargs_7290 = {}
# Getting the type of 'range' (line 20)
range_7287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'range', False)
# Calling range(args, kwargs) (line 20)
range_call_result_7291 = invoke(stypy.reporting.localization.Localization(__file__, 20, 21), range_7287, *[int_7288, int_7289], **kwargs_7290)

comprehension_7292 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), range_call_result_7291)
# Assigning a type to the variable 'x' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'x', comprehension_7292)

# Getting the type of 'x' (line 20)
x_7284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 37), 'x')
# Getting the type of 'noprimes' (line 20)
noprimes_7285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 46), 'noprimes')
# Applying the binary operator 'notin' (line 20)
result_contains_7286 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 37), 'notin', x_7284, noprimes_7285)

# Getting the type of 'x' (line 20)
x_7283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'x')
list_7293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 10), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 10), list_7293, x_7283)
# Assigning a type to the variable 'primes' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'primes', list_7293)

# Assigning a DictComp to a Name (line 21):
# Calculating dict comprehension
module_type_store = module_type_store.open_function_context('dict comprehension expression', 21, 6, True)
# Calculating comprehension expression
# Getting the type of 'primes' (line 21)
primes_7296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'primes')
comprehension_7297 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 6), primes_7296)
# Assigning a type to the variable 'k' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 6), 'k', comprehension_7297)
# Getting the type of 'k' (line 21)
k_7294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 6), 'k')
# Getting the type of 'k' (line 21)
k_7295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 9), 'k')
dict_7298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 6), 'dict')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 6), dict_7298, (k_7294, k_7295))
# Assigning a type to the variable 'd5' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'd5', dict_7298)

# Assigning a Subscript to a Name (line 22):

# Obtaining the type of the subscript
str_7299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 8), 'str', 'hi')
# Getting the type of 'd5' (line 22)
d5_7300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 5), 'd5')
# Obtaining the member '__getitem__' of a type (line 22)
getitem___7301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 5), d5_7300, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 22)
subscript_call_result_7302 = invoke(stypy.reporting.localization.Localization(__file__, 22, 5), getitem___7301, str_7299)

# Assigning a type to the variable 'r4' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'r4', subscript_call_result_7302)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
