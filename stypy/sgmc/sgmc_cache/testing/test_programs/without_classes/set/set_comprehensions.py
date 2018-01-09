
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: S = {x**2 for x in range(10)}
4: V = {str(i) for i in range(13)}
5: M = {x for x in S if x % 2 == 0}
6: 
7: noprimes = {j for i in range(2, 8) for j in range(i*2, 50, i)}
8: primes = {x for x in range(2, 50) if x not in noprimes}
9: 
10: words = {'The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'}
11: stuff = [[w.upper(), w.lower(), len(w)] for w in words]
12: 
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a SetComp to a Name (line 3):
# Calculating set comprehension
module_type_store = module_type_store.open_function_context('set comprehension expression', 3, 5, True)
# Calculating comprehension expression

# Call to range(...): (line 3)
# Processing the call arguments (line 3)
int_6525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 25), 'int')
# Processing the call keyword arguments (line 3)
kwargs_6526 = {}
# Getting the type of 'range' (line 3)
range_6524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 19), 'range', False)
# Calling range(args, kwargs) (line 3)
range_call_result_6527 = invoke(stypy.reporting.localization.Localization(__file__, 3, 19), range_6524, *[int_6525], **kwargs_6526)

comprehension_6528 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 5), range_call_result_6527)
# Assigning a type to the variable 'x' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 5), 'x', comprehension_6528)
# Getting the type of 'x' (line 3)
x_6521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 5), 'x')
int_6522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 8), 'int')
# Applying the binary operator '**' (line 3)
result_pow_6523 = python_operator(stypy.reporting.localization.Localization(__file__, 3, 5), '**', x_6521, int_6522)

set_6529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 5), 'set')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 5), set_6529, result_pow_6523)
# Assigning a type to the variable 'S' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'S', set_6529)

# Assigning a SetComp to a Name (line 4):
# Calculating set comprehension
module_type_store = module_type_store.open_function_context('set comprehension expression', 4, 5, True)
# Calculating comprehension expression

# Call to range(...): (line 4)
# Processing the call arguments (line 4)
int_6535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 27), 'int')
# Processing the call keyword arguments (line 4)
kwargs_6536 = {}
# Getting the type of 'range' (line 4)
range_6534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 21), 'range', False)
# Calling range(args, kwargs) (line 4)
range_call_result_6537 = invoke(stypy.reporting.localization.Localization(__file__, 4, 21), range_6534, *[int_6535], **kwargs_6536)

comprehension_6538 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 5), range_call_result_6537)
# Assigning a type to the variable 'i' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 5), 'i', comprehension_6538)

# Call to str(...): (line 4)
# Processing the call arguments (line 4)
# Getting the type of 'i' (line 4)
i_6531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 9), 'i', False)
# Processing the call keyword arguments (line 4)
kwargs_6532 = {}
# Getting the type of 'str' (line 4)
str_6530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 5), 'str', False)
# Calling str(args, kwargs) (line 4)
str_call_result_6533 = invoke(stypy.reporting.localization.Localization(__file__, 4, 5), str_6530, *[i_6531], **kwargs_6532)

set_6539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 5), 'set')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 5), set_6539, str_call_result_6533)
# Assigning a type to the variable 'V' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'V', set_6539)

# Assigning a SetComp to a Name (line 5):
# Calculating set comprehension
module_type_store = module_type_store.open_function_context('set comprehension expression', 5, 5, True)
# Calculating comprehension expression
# Getting the type of 'S' (line 5)
S_6546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 16), 'S')
comprehension_6547 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 5), S_6546)
# Assigning a type to the variable 'x' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 5), 'x', comprehension_6547)

# Getting the type of 'x' (line 5)
x_6541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 21), 'x')
int_6542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 25), 'int')
# Applying the binary operator '%' (line 5)
result_mod_6543 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 21), '%', x_6541, int_6542)

int_6544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 30), 'int')
# Applying the binary operator '==' (line 5)
result_eq_6545 = python_operator(stypy.reporting.localization.Localization(__file__, 5, 21), '==', result_mod_6543, int_6544)

# Getting the type of 'x' (line 5)
x_6540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 5), 'x')
set_6548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 5), 'set')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 5), set_6548, x_6540)
# Assigning a type to the variable 'M' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'M', set_6548)

# Assigning a SetComp to a Name (line 7):
# Calculating set comprehension
module_type_store = module_type_store.open_function_context('set comprehension expression', 7, 12, True)
# Calculating comprehension expression

# Call to range(...): (line 7)
# Processing the call arguments (line 7)
int_6551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 29), 'int')
int_6552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 32), 'int')
# Processing the call keyword arguments (line 7)
kwargs_6553 = {}
# Getting the type of 'range' (line 7)
range_6550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 23), 'range', False)
# Calling range(args, kwargs) (line 7)
range_call_result_6554 = invoke(stypy.reporting.localization.Localization(__file__, 7, 23), range_6550, *[int_6551, int_6552], **kwargs_6553)

comprehension_6555 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 12), range_call_result_6554)
# Assigning a type to the variable 'i' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'i', comprehension_6555)
# Calculating comprehension expression

# Call to range(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'i' (line 7)
i_6557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 50), 'i', False)
int_6558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 52), 'int')
# Applying the binary operator '*' (line 7)
result_mul_6559 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 50), '*', i_6557, int_6558)

int_6560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 55), 'int')
# Getting the type of 'i' (line 7)
i_6561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 59), 'i', False)
# Processing the call keyword arguments (line 7)
kwargs_6562 = {}
# Getting the type of 'range' (line 7)
range_6556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 44), 'range', False)
# Calling range(args, kwargs) (line 7)
range_call_result_6563 = invoke(stypy.reporting.localization.Localization(__file__, 7, 44), range_6556, *[result_mul_6559, int_6560, i_6561], **kwargs_6562)

comprehension_6564 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 12), range_call_result_6563)
# Assigning a type to the variable 'j' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'j', comprehension_6564)
# Getting the type of 'j' (line 7)
j_6549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'j')
set_6565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 12), 'set')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 12), set_6565, j_6549)
# Assigning a type to the variable 'noprimes' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'noprimes', set_6565)

# Assigning a SetComp to a Name (line 8):
# Calculating set comprehension
module_type_store = module_type_store.open_function_context('set comprehension expression', 8, 10, True)
# Calculating comprehension expression

# Call to range(...): (line 8)
# Processing the call arguments (line 8)
int_6571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 27), 'int')
int_6572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 30), 'int')
# Processing the call keyword arguments (line 8)
kwargs_6573 = {}
# Getting the type of 'range' (line 8)
range_6570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 21), 'range', False)
# Calling range(args, kwargs) (line 8)
range_call_result_6574 = invoke(stypy.reporting.localization.Localization(__file__, 8, 21), range_6570, *[int_6571, int_6572], **kwargs_6573)

comprehension_6575 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), range_call_result_6574)
# Assigning a type to the variable 'x' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 10), 'x', comprehension_6575)

# Getting the type of 'x' (line 8)
x_6567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 37), 'x')
# Getting the type of 'noprimes' (line 8)
noprimes_6568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 46), 'noprimes')
# Applying the binary operator 'notin' (line 8)
result_contains_6569 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 37), 'notin', x_6567, noprimes_6568)

# Getting the type of 'x' (line 8)
x_6566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 10), 'x')
set_6576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'set')
# Destroy the current context
module_type_store = module_type_store.close_function_context()
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), set_6576, x_6566)
# Assigning a type to the variable 'primes' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'primes', set_6576)

# Assigning a Set to a Name (line 10):

# Obtaining an instance of the builtin type 'set' (line 10)
set_6577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'set')
# Adding type elements to the builtin type 'set' instance (line 10)
# Adding element type (line 10)
str_6578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 9), 'str', 'The')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), set_6577, str_6578)
# Adding element type (line 10)
str_6579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 16), 'str', 'quick')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), set_6577, str_6579)
# Adding element type (line 10)
str_6580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 25), 'str', 'brown')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), set_6577, str_6580)
# Adding element type (line 10)
str_6581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 34), 'str', 'fox')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), set_6577, str_6581)
# Adding element type (line 10)
str_6582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 41), 'str', 'jumps')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), set_6577, str_6582)
# Adding element type (line 10)
str_6583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 50), 'str', 'over')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), set_6577, str_6583)
# Adding element type (line 10)
str_6584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 58), 'str', 'the')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), set_6577, str_6584)
# Adding element type (line 10)
str_6585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 65), 'str', 'lazy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), set_6577, str_6585)
# Adding element type (line 10)
str_6586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 73), 'str', 'dog')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), set_6577, str_6586)

# Assigning a type to the variable 'words' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'words', set_6577)

# Assigning a ListComp to a Name (line 11):
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'words' (line 11)
words_6600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 49), 'words')
comprehension_6601 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 9), words_6600)
# Assigning a type to the variable 'w' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 9), 'w', comprehension_6601)

# Obtaining an instance of the builtin type 'list' (line 11)
list_6587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 9), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)

# Call to upper(...): (line 11)
# Processing the call keyword arguments (line 11)
kwargs_6590 = {}
# Getting the type of 'w' (line 11)
w_6588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'w', False)
# Obtaining the member 'upper' of a type (line 11)
upper_6589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 10), w_6588, 'upper')
# Calling upper(args, kwargs) (line 11)
upper_call_result_6591 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), upper_6589, *[], **kwargs_6590)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 9), list_6587, upper_call_result_6591)
# Adding element type (line 11)

# Call to lower(...): (line 11)
# Processing the call keyword arguments (line 11)
kwargs_6594 = {}
# Getting the type of 'w' (line 11)
w_6592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 21), 'w', False)
# Obtaining the member 'lower' of a type (line 11)
lower_6593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 21), w_6592, 'lower')
# Calling lower(args, kwargs) (line 11)
lower_call_result_6595 = invoke(stypy.reporting.localization.Localization(__file__, 11, 21), lower_6593, *[], **kwargs_6594)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 9), list_6587, lower_call_result_6595)
# Adding element type (line 11)

# Call to len(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'w' (line 11)
w_6597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 36), 'w', False)
# Processing the call keyword arguments (line 11)
kwargs_6598 = {}
# Getting the type of 'len' (line 11)
len_6596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 32), 'len', False)
# Calling len(args, kwargs) (line 11)
len_call_result_6599 = invoke(stypy.reporting.localization.Localization(__file__, 11, 32), len_6596, *[w_6597], **kwargs_6598)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 9), list_6587, len_call_result_6599)

list_6602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 9), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 9), list_6602, list_6587)
# Assigning a type to the variable 'stuff' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stuff', list_6602)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
