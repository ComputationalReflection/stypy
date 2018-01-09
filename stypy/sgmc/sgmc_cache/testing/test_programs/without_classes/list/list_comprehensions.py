
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: S = [x**2 for x in range(10)]
4: V = [str(i) for i in range(13)]
5: M = [x for x in S if x % 2 == 0]
6: 
7: noprimes = [j for i in range(2, 8) for j in range(i*2, 50, i)]
8: primes = [x for x in range(2, 50) if x not in noprimes]
9: 
10: words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
11: stuff = [[w.upper(), w.lower(), len(w)] for w in words]

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a ListComp to a Name (line 3):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 3)
# Processing the call arguments (line 3)
int_5228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 25), 'int')
# Processing the call keyword arguments (line 3)
kwargs_5229 = {}
# Getting the type of 'range' (line 3)
range_5227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 19), 'range', False)
# Calling range(args, kwargs) (line 3)
range_call_result_5230 = invoke(stypy.reporting.localization.Localization(__file__, 3, 19), range_5227, *[int_5228], **kwargs_5229)

comprehension_5231 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 5), range_call_result_5230)
# Assigning a type to the variable 'x' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 5), 'x', comprehension_5231)
# Getting the type of 'x' (line 3)
x_5224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 5), 'x')
int_5225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 8), 'int')
# Applying the binary operator '**' (line 3)
result_pow_5226 = python_operator(stypy.reporting.localization.Localization(__file__, 3, 5), '**', x_5224, int_5225)

list_5232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 5), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 5), list_5232, result_pow_5226)
# Assigning a type to the variable 'S' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'S', list_5232)

# Assigning a ListComp to a Name (line 4):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 4)
# Processing the call arguments (line 4)
int_5238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 27), 'int')
# Processing the call keyword arguments (line 4)
kwargs_5239 = {}
# Getting the type of 'range' (line 4)
range_5237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 21), 'range', False)
# Calling range(args, kwargs) (line 4)
range_call_result_5240 = invoke(stypy.reporting.localization.Localization(__file__, 4, 21), range_5237, *[int_5238], **kwargs_5239)

comprehension_5241 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 5), range_call_result_5240)
# Assigning a type to the variable 'i' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 5), 'i', comprehension_5241)

# Call to str(...): (line 4)
# Processing the call arguments (line 4)
# Getting the type of 'i' (line 4)
i_5234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 9), 'i', False)
# Processing the call keyword arguments (line 4)
kwargs_5235 = {}
# Getting the type of 'str' (line 4)
str_5233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 5), 'str', False)
# Calling str(args, kwargs) (line 4)
str_call_result_5236 = invoke(stypy.reporting.localization.Localization(__file__, 4, 5), str_5233, *[i_5234], **kwargs_5235)

list_5242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 5), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 5), list_5242, str_call_result_5236)
# Assigning a type to the variable 'V' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'V', list_5242)

# Assigning a ListComp to a Name (line 5):
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'S' (line 5)
S_5249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 16), 'S')
comprehension_5250 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 5), S_5249)
# Assigning a type to the variable 'x' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 5), 'x', comprehension_5250)

# Getting the type of 'x' (line 5)
x_5244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 21), 'x')
int_5245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 25), 'int')
# Applying the binary operator '%' (line 5)
result_mod_5246 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 21), '%', x_5244, int_5245)

int_5247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 30), 'int')
# Applying the binary operator '==' (line 5)
result_eq_5248 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 21), '==', result_mod_5246, int_5247)

# Getting the type of 'x' (line 5)
x_5243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 5), 'x')
list_5251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 5), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 5), list_5251, x_5243)
# Assigning a type to the variable 'M' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'M', list_5251)

# Assigning a ListComp to a Name (line 7):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 7)
# Processing the call arguments (line 7)
int_5254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 29), 'int')
int_5255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 32), 'int')
# Processing the call keyword arguments (line 7)
kwargs_5256 = {}
# Getting the type of 'range' (line 7)
range_5253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 23), 'range', False)
# Calling range(args, kwargs) (line 7)
range_call_result_5257 = invoke(stypy.reporting.localization.Localization(__file__, 7, 23), range_5253, *[int_5254, int_5255], **kwargs_5256)

comprehension_5258 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 12), range_call_result_5257)
# Assigning a type to the variable 'i' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'i', comprehension_5258)
# Calculating comprehension expression

# Call to range(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'i' (line 7)
i_5260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 50), 'i', False)
int_5261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 52), 'int')
# Applying the binary operator '*' (line 7)
result_mul_5262 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 50), '*', i_5260, int_5261)

int_5263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 55), 'int')
# Getting the type of 'i' (line 7)
i_5264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 59), 'i', False)
# Processing the call keyword arguments (line 7)
kwargs_5265 = {}
# Getting the type of 'range' (line 7)
range_5259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 44), 'range', False)
# Calling range(args, kwargs) (line 7)
range_call_result_5266 = invoke(stypy.reporting.localization.Localization(__file__, 7, 44), range_5259, *[result_mul_5262, int_5263, i_5264], **kwargs_5265)

comprehension_5267 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 12), range_call_result_5266)
# Assigning a type to the variable 'j' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'j', comprehension_5267)
# Getting the type of 'j' (line 7)
j_5252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'j')
list_5268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 12), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 12), list_5268, j_5252)
# Assigning a type to the variable 'noprimes' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'noprimes', list_5268)

# Assigning a ListComp to a Name (line 8):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 8)
# Processing the call arguments (line 8)
int_5274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 27), 'int')
int_5275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 30), 'int')
# Processing the call keyword arguments (line 8)
kwargs_5276 = {}
# Getting the type of 'range' (line 8)
range_5273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 21), 'range', False)
# Calling range(args, kwargs) (line 8)
range_call_result_5277 = invoke(stypy.reporting.localization.Localization(__file__, 8, 21), range_5273, *[int_5274, int_5275], **kwargs_5276)

comprehension_5278 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), range_call_result_5277)
# Assigning a type to the variable 'x' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 10), 'x', comprehension_5278)

# Getting the type of 'x' (line 8)
x_5270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 37), 'x')
# Getting the type of 'noprimes' (line 8)
noprimes_5271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 46), 'noprimes')
# Applying the binary operator 'notin' (line 8)
result_contains_5272 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 37), 'notin', x_5270, noprimes_5271)

# Getting the type of 'x' (line 8)
x_5269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 10), 'x')
list_5279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_5279, x_5269)
# Assigning a type to the variable 'primes' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'primes', list_5279)

# Assigning a List to a Name (line 10):

# Obtaining an instance of the builtin type 'list' (line 10)
list_5280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
str_5281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 9), 'str', 'The')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), list_5280, str_5281)
# Adding element type (line 10)
str_5282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 16), 'str', 'quick')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), list_5280, str_5282)
# Adding element type (line 10)
str_5283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 25), 'str', 'brown')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), list_5280, str_5283)
# Adding element type (line 10)
str_5284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 34), 'str', 'fox')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), list_5280, str_5284)
# Adding element type (line 10)
str_5285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 41), 'str', 'jumps')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), list_5280, str_5285)
# Adding element type (line 10)
str_5286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 50), 'str', 'over')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), list_5280, str_5286)
# Adding element type (line 10)
str_5287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 58), 'str', 'the')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), list_5280, str_5287)
# Adding element type (line 10)
str_5288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 65), 'str', 'lazy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), list_5280, str_5288)
# Adding element type (line 10)
str_5289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 73), 'str', 'dog')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), list_5280, str_5289)

# Assigning a type to the variable 'words' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'words', list_5280)

# Assigning a ListComp to a Name (line 11):
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'words' (line 11)
words_5303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 49), 'words')
comprehension_5304 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 9), words_5303)
# Assigning a type to the variable 'w' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 9), 'w', comprehension_5304)

# Obtaining an instance of the builtin type 'list' (line 11)
list_5290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 9), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)

# Call to upper(...): (line 11)
# Processing the call keyword arguments (line 11)
kwargs_5293 = {}
# Getting the type of 'w' (line 11)
w_5291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'w', False)
# Obtaining the member 'upper' of a type (line 11)
upper_5292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 10), w_5291, 'upper')
# Calling upper(args, kwargs) (line 11)
upper_call_result_5294 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), upper_5292, *[], **kwargs_5293)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 9), list_5290, upper_call_result_5294)
# Adding element type (line 11)

# Call to lower(...): (line 11)
# Processing the call keyword arguments (line 11)
kwargs_5297 = {}
# Getting the type of 'w' (line 11)
w_5295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 21), 'w', False)
# Obtaining the member 'lower' of a type (line 11)
lower_5296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 21), w_5295, 'lower')
# Calling lower(args, kwargs) (line 11)
lower_call_result_5298 = invoke(stypy.reporting.localization.Localization(__file__, 11, 21), lower_5296, *[], **kwargs_5297)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 9), list_5290, lower_call_result_5298)
# Adding element type (line 11)

# Call to len(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'w' (line 11)
w_5300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 36), 'w', False)
# Processing the call keyword arguments (line 11)
kwargs_5301 = {}
# Getting the type of 'len' (line 11)
len_5299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 32), 'len', False)
# Calling len(args, kwargs) (line 11)
len_call_result_5302 = invoke(stypy.reporting.localization.Localization(__file__, 11, 32), len_5299, *[w_5300], **kwargs_5301)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 9), list_5290, len_call_result_5302)

list_5305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 9), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 9), list_5305, list_5290)
# Assigning a type to the variable 'stuff' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stuff', list_5305)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
