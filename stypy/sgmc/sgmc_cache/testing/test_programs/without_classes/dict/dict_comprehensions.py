
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: d = {n: n**2 for n in range(5)}
4: print d
5: 
6: d2 = {n: True for n in range(5)}
7: print d2
8: 
9: d3 = {k: k for k in range(10)}
10: print d3
11: 
12: old_dict = {'a': 1, 'c': 3, 'b': 2}
13: print old_dict
14: 
15: new_dict = {key: 'your value here' for key in old_dict.keys()}
16: print new_dict
17: 
18: S = [x**2 for x in range(10)]
19: d4 = {k: k for k in S}
20: print d4
21: 
22: 
23: noprimes = [j for i in range(2, 8) for j in range(i*2, 50, i)]
24: primes = [x for x in range(2, 50) if x not in noprimes]
25: d5 = {k: k for k in primes}
26: print d5
27: 
28: words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
29: d6 = {k: len(k) for k in words}
30: print d6
31: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a DictComp to a Name (line 3):
# Calculating dict comprehension
module_type_store = module_type_store.open_function_context('dict comprehension expression', 3, 5, True)
# Calculating comprehension expression

# Call to range(...): (line 3)
# Processing the call arguments (line 3)
int_2524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 28), 'int')
# Processing the call keyword arguments (line 3)
kwargs_2525 = {}
# Getting the type of 'range' (line 3)
range_2523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 22), 'range', False)
# Calling range(args, kwargs) (line 3)
range_call_result_2526 = invoke(stypy.reporting.localization.Localization(__file__, 3, 22), range_2523, *[int_2524], **kwargs_2525)

comprehension_2527 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 5), range_call_result_2526)
# Assigning a type to the variable 'n' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 5), 'n', comprehension_2527)
# Getting the type of 'n' (line 3)
n_2519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 5), 'n')
# Getting the type of 'n' (line 3)
n_2520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'n')
int_2521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'int')
# Applying the binary operator '**' (line 3)
result_pow_2522 = python_operator(stypy.reporting.localization.Localization(__file__, 3, 8), '**', n_2520, int_2521)

dict_2528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 5), 'dict')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 5), dict_2528, (n_2519, result_pow_2522))
# Assigning a type to the variable 'd' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'd', dict_2528)
# Getting the type of 'd' (line 4)
d_2529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 6), 'd')

# Assigning a DictComp to a Name (line 6):
# Calculating dict comprehension
module_type_store = module_type_store.open_function_context('dict comprehension expression', 6, 6, True)
# Calculating comprehension expression

# Call to range(...): (line 6)
# Processing the call arguments (line 6)
int_2533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 29), 'int')
# Processing the call keyword arguments (line 6)
kwargs_2534 = {}
# Getting the type of 'range' (line 6)
range_2532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 23), 'range', False)
# Calling range(args, kwargs) (line 6)
range_call_result_2535 = invoke(stypy.reporting.localization.Localization(__file__, 6, 23), range_2532, *[int_2533], **kwargs_2534)

comprehension_2536 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 6), range_call_result_2535)
# Assigning a type to the variable 'n' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 6), 'n', comprehension_2536)
# Getting the type of 'n' (line 6)
n_2530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 6), 'n')
# Getting the type of 'True' (line 6)
True_2531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 9), 'True')
dict_2537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 6), 'dict')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 6), dict_2537, (n_2530, True_2531))
# Assigning a type to the variable 'd2' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'd2', dict_2537)
# Getting the type of 'd2' (line 7)
d2_2538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 6), 'd2')

# Assigning a DictComp to a Name (line 9):
# Calculating dict comprehension
module_type_store = module_type_store.open_function_context('dict comprehension expression', 9, 6, True)
# Calculating comprehension expression

# Call to range(...): (line 9)
# Processing the call arguments (line 9)
int_2542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 26), 'int')
# Processing the call keyword arguments (line 9)
kwargs_2543 = {}
# Getting the type of 'range' (line 9)
range_2541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 20), 'range', False)
# Calling range(args, kwargs) (line 9)
range_call_result_2544 = invoke(stypy.reporting.localization.Localization(__file__, 9, 20), range_2541, *[int_2542], **kwargs_2543)

comprehension_2545 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 6), range_call_result_2544)
# Assigning a type to the variable 'k' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 6), 'k', comprehension_2545)
# Getting the type of 'k' (line 9)
k_2539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 6), 'k')
# Getting the type of 'k' (line 9)
k_2540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 9), 'k')
dict_2546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 6), 'dict')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 6), dict_2546, (k_2539, k_2540))
# Assigning a type to the variable 'd3' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'd3', dict_2546)
# Getting the type of 'd3' (line 10)
d3_2547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 6), 'd3')

# Assigning a Dict to a Name (line 12):

# Obtaining an instance of the builtin type 'dict' (line 12)
dict_2548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 12)
# Adding element type (key, value) (line 12)
str_2549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 12), 'str', 'a')
int_2550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 17), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 11), dict_2548, (str_2549, int_2550))
# Adding element type (key, value) (line 12)
str_2551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 20), 'str', 'c')
int_2552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 25), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 11), dict_2548, (str_2551, int_2552))
# Adding element type (key, value) (line 12)
str_2553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 28), 'str', 'b')
int_2554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 33), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 11), dict_2548, (str_2553, int_2554))

# Assigning a type to the variable 'old_dict' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'old_dict', dict_2548)
# Getting the type of 'old_dict' (line 13)
old_dict_2555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 6), 'old_dict')

# Assigning a DictComp to a Name (line 15):
# Calculating dict comprehension
module_type_store = module_type_store.open_function_context('dict comprehension expression', 15, 12, True)
# Calculating comprehension expression

# Call to keys(...): (line 15)
# Processing the call keyword arguments (line 15)
kwargs_2560 = {}
# Getting the type of 'old_dict' (line 15)
old_dict_2558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 46), 'old_dict', False)
# Obtaining the member 'keys' of a type (line 15)
keys_2559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 46), old_dict_2558, 'keys')
# Calling keys(args, kwargs) (line 15)
keys_call_result_2561 = invoke(stypy.reporting.localization.Localization(__file__, 15, 46), keys_2559, *[], **kwargs_2560)

comprehension_2562 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 12), keys_call_result_2561)
# Assigning a type to the variable 'key' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'key', comprehension_2562)
# Getting the type of 'key' (line 15)
key_2556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'key')
str_2557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 17), 'str', 'your value here')
dict_2563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'dict')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 12), dict_2563, (key_2556, str_2557))
# Assigning a type to the variable 'new_dict' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'new_dict', dict_2563)
# Getting the type of 'new_dict' (line 16)
new_dict_2564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 6), 'new_dict')

# Assigning a ListComp to a Name (line 18):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 18)
# Processing the call arguments (line 18)
int_2569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'int')
# Processing the call keyword arguments (line 18)
kwargs_2570 = {}
# Getting the type of 'range' (line 18)
range_2568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 19), 'range', False)
# Calling range(args, kwargs) (line 18)
range_call_result_2571 = invoke(stypy.reporting.localization.Localization(__file__, 18, 19), range_2568, *[int_2569], **kwargs_2570)

comprehension_2572 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 5), range_call_result_2571)
# Assigning a type to the variable 'x' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'x', comprehension_2572)
# Getting the type of 'x' (line 18)
x_2565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'x')
int_2566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'int')
# Applying the binary operator '**' (line 18)
result_pow_2567 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 5), '**', x_2565, int_2566)

list_2573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 5), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 5), list_2573, result_pow_2567)
# Assigning a type to the variable 'S' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'S', list_2573)

# Assigning a DictComp to a Name (line 19):
# Calculating dict comprehension
module_type_store = module_type_store.open_function_context('dict comprehension expression', 19, 6, True)
# Calculating comprehension expression
# Getting the type of 'S' (line 19)
S_2576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'S')
comprehension_2577 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 6), S_2576)
# Assigning a type to the variable 'k' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 6), 'k', comprehension_2577)
# Getting the type of 'k' (line 19)
k_2574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 6), 'k')
# Getting the type of 'k' (line 19)
k_2575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 9), 'k')
dict_2578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 6), 'dict')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 6), dict_2578, (k_2574, k_2575))
# Assigning a type to the variable 'd4' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'd4', dict_2578)
# Getting the type of 'd4' (line 20)
d4_2579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 6), 'd4')

# Assigning a ListComp to a Name (line 23):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 23)
# Processing the call arguments (line 23)
int_2582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 29), 'int')
int_2583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 32), 'int')
# Processing the call keyword arguments (line 23)
kwargs_2584 = {}
# Getting the type of 'range' (line 23)
range_2581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'range', False)
# Calling range(args, kwargs) (line 23)
range_call_result_2585 = invoke(stypy.reporting.localization.Localization(__file__, 23, 23), range_2581, *[int_2582, int_2583], **kwargs_2584)

comprehension_2586 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 12), range_call_result_2585)
# Assigning a type to the variable 'i' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'i', comprehension_2586)
# Calculating comprehension expression

# Call to range(...): (line 23)
# Processing the call arguments (line 23)
# Getting the type of 'i' (line 23)
i_2588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 50), 'i', False)
int_2589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 52), 'int')
# Applying the binary operator '*' (line 23)
result_mul_2590 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 50), '*', i_2588, int_2589)

int_2591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 55), 'int')
# Getting the type of 'i' (line 23)
i_2592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 59), 'i', False)
# Processing the call keyword arguments (line 23)
kwargs_2593 = {}
# Getting the type of 'range' (line 23)
range_2587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 44), 'range', False)
# Calling range(args, kwargs) (line 23)
range_call_result_2594 = invoke(stypy.reporting.localization.Localization(__file__, 23, 44), range_2587, *[result_mul_2590, int_2591, i_2592], **kwargs_2593)

comprehension_2595 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 12), range_call_result_2594)
# Assigning a type to the variable 'j' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'j', comprehension_2595)
# Getting the type of 'j' (line 23)
j_2580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'j')
list_2596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 12), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 12), list_2596, j_2580)
# Assigning a type to the variable 'noprimes' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'noprimes', list_2596)

# Assigning a ListComp to a Name (line 24):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 24)
# Processing the call arguments (line 24)
int_2602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 27), 'int')
int_2603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 30), 'int')
# Processing the call keyword arguments (line 24)
kwargs_2604 = {}
# Getting the type of 'range' (line 24)
range_2601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 21), 'range', False)
# Calling range(args, kwargs) (line 24)
range_call_result_2605 = invoke(stypy.reporting.localization.Localization(__file__, 24, 21), range_2601, *[int_2602, int_2603], **kwargs_2604)

comprehension_2606 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), range_call_result_2605)
# Assigning a type to the variable 'x' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 10), 'x', comprehension_2606)

# Getting the type of 'x' (line 24)
x_2598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 37), 'x')
# Getting the type of 'noprimes' (line 24)
noprimes_2599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 46), 'noprimes')
# Applying the binary operator 'notin' (line 24)
result_contains_2600 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 37), 'notin', x_2598, noprimes_2599)

# Getting the type of 'x' (line 24)
x_2597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 10), 'x')
list_2607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 10), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 10), list_2607, x_2597)
# Assigning a type to the variable 'primes' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'primes', list_2607)

# Assigning a DictComp to a Name (line 25):
# Calculating dict comprehension
module_type_store = module_type_store.open_function_context('dict comprehension expression', 25, 6, True)
# Calculating comprehension expression
# Getting the type of 'primes' (line 25)
primes_2610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'primes')
comprehension_2611 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 6), primes_2610)
# Assigning a type to the variable 'k' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 6), 'k', comprehension_2611)
# Getting the type of 'k' (line 25)
k_2608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 6), 'k')
# Getting the type of 'k' (line 25)
k_2609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 9), 'k')
dict_2612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 6), 'dict')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 6), dict_2612, (k_2608, k_2609))
# Assigning a type to the variable 'd5' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'd5', dict_2612)
# Getting the type of 'd5' (line 26)
d5_2613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 6), 'd5')

# Assigning a List to a Name (line 28):

# Obtaining an instance of the builtin type 'list' (line 28)
list_2614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'list')
# Adding type elements to the builtin type 'list' instance (line 28)
# Adding element type (line 28)
str_2615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 9), 'str', 'The')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), list_2614, str_2615)
# Adding element type (line 28)
str_2616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 16), 'str', 'quick')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), list_2614, str_2616)
# Adding element type (line 28)
str_2617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'str', 'brown')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), list_2614, str_2617)
# Adding element type (line 28)
str_2618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 34), 'str', 'fox')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), list_2614, str_2618)
# Adding element type (line 28)
str_2619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 41), 'str', 'jumps')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), list_2614, str_2619)
# Adding element type (line 28)
str_2620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 50), 'str', 'over')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), list_2614, str_2620)
# Adding element type (line 28)
str_2621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 58), 'str', 'the')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), list_2614, str_2621)
# Adding element type (line 28)
str_2622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 65), 'str', 'lazy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), list_2614, str_2622)
# Adding element type (line 28)
str_2623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 73), 'str', 'dog')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), list_2614, str_2623)

# Assigning a type to the variable 'words' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'words', list_2614)

# Assigning a DictComp to a Name (line 29):
# Calculating dict comprehension
module_type_store = module_type_store.open_function_context('dict comprehension expression', 29, 6, True)
# Calculating comprehension expression
# Getting the type of 'words' (line 29)
words_2629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 25), 'words')
comprehension_2630 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 6), words_2629)
# Assigning a type to the variable 'k' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 6), 'k', comprehension_2630)
# Getting the type of 'k' (line 29)
k_2624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 6), 'k')

# Call to len(...): (line 29)
# Processing the call arguments (line 29)
# Getting the type of 'k' (line 29)
k_2626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 13), 'k', False)
# Processing the call keyword arguments (line 29)
kwargs_2627 = {}
# Getting the type of 'len' (line 29)
len_2625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 9), 'len', False)
# Calling len(args, kwargs) (line 29)
len_call_result_2628 = invoke(stypy.reporting.localization.Localization(__file__, 29, 9), len_2625, *[k_2626], **kwargs_2627)

dict_2631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 6), 'dict')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 6), dict_2631, (k_2624, len_call_result_2628))
# Assigning a type to the variable 'd6' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'd6', dict_2631)
# Getting the type of 'd6' (line 30)
d6_2632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 6), 'd6')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
