
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def fun1(l):
2:     return "aaa" + l[0]
3: 
4: 
5: def fun2(l):
6:     return 3 / l[0]
7: 
8: 
9: S = [x ** 2 for x in range(10)]
10: V = [str(i) for i in range(13)]
11: 
12: normal_list = [1, 2, 3]
13: tuple_ = (1, 2, 3)
14: 
15: r1 = fun1(S[0])  # The error is reported on callsite instead on the actual code
16: r2 = fun1(S)  # No error reported. Runtime crash
17: r3 = fun1(normal_list)  # No error reported
18: r4 = fun1(tuple_)  # No error reported
19: r5 = fun2(V)  # No error reported. Runtime crash
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def fun1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fun1'
    module_type_store = module_type_store.open_function_context('fun1', 1, 0, False)
    
    # Passed parameters checking function
    fun1.stypy_localization = localization
    fun1.stypy_type_of_self = None
    fun1.stypy_type_store = module_type_store
    fun1.stypy_function_name = 'fun1'
    fun1.stypy_param_names_list = ['l']
    fun1.stypy_varargs_param_name = None
    fun1.stypy_kwargs_param_name = None
    fun1.stypy_call_defaults = defaults
    fun1.stypy_call_varargs = varargs
    fun1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fun1', ['l'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fun1', localization, ['l'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fun1(...)' code ##################

    str_7474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 11), 'str', 'aaa')
    
    # Obtaining the type of the subscript
    int_7475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 21), 'int')
    # Getting the type of 'l' (line 2)
    l_7476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 19), 'l')
    # Obtaining the member '__getitem__' of a type (line 2)
    getitem___7477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2, 19), l_7476, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 2)
    subscript_call_result_7478 = invoke(stypy.reporting.localization.Localization(__file__, 2, 19), getitem___7477, int_7475)
    
    # Applying the binary operator '+' (line 2)
    result_add_7479 = python_operator(stypy.reporting.localization.Localization(__file__, 2, 11), '+', str_7474, subscript_call_result_7478)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type', result_add_7479)
    
    # ################# End of 'fun1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fun1' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_7480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7480)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fun1'
    return stypy_return_type_7480

# Assigning a type to the variable 'fun1' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'fun1', fun1)

@norecursion
def fun2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fun2'
    module_type_store = module_type_store.open_function_context('fun2', 5, 0, False)
    
    # Passed parameters checking function
    fun2.stypy_localization = localization
    fun2.stypy_type_of_self = None
    fun2.stypy_type_store = module_type_store
    fun2.stypy_function_name = 'fun2'
    fun2.stypy_param_names_list = ['l']
    fun2.stypy_varargs_param_name = None
    fun2.stypy_kwargs_param_name = None
    fun2.stypy_call_defaults = defaults
    fun2.stypy_call_varargs = varargs
    fun2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fun2', ['l'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fun2', localization, ['l'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fun2(...)' code ##################

    int_7481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 11), 'int')
    
    # Obtaining the type of the subscript
    int_7482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 17), 'int')
    # Getting the type of 'l' (line 6)
    l_7483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), 'l')
    # Obtaining the member '__getitem__' of a type (line 6)
    getitem___7484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 15), l_7483, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 6)
    subscript_call_result_7485 = invoke(stypy.reporting.localization.Localization(__file__, 6, 15), getitem___7484, int_7482)
    
    # Applying the binary operator 'div' (line 6)
    result_div_7486 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 11), 'div', int_7481, subscript_call_result_7485)
    
    # Assigning a type to the variable 'stypy_return_type' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type', result_div_7486)
    
    # ################# End of 'fun2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fun2' in the type store
    # Getting the type of 'stypy_return_type' (line 5)
    stypy_return_type_7487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7487)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fun2'
    return stypy_return_type_7487

# Assigning a type to the variable 'fun2' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'fun2', fun2)

# Assigning a ListComp to a Name (line 9):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 9)
# Processing the call arguments (line 9)
int_7492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 27), 'int')
# Processing the call keyword arguments (line 9)
kwargs_7493 = {}
# Getting the type of 'range' (line 9)
range_7491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 21), 'range', False)
# Calling range(args, kwargs) (line 9)
range_call_result_7494 = invoke(stypy.reporting.localization.Localization(__file__, 9, 21), range_7491, *[int_7492], **kwargs_7493)

comprehension_7495 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 5), range_call_result_7494)
# Assigning a type to the variable 'x' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'x', comprehension_7495)
# Getting the type of 'x' (line 9)
x_7488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'x')
int_7489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'int')
# Applying the binary operator '**' (line 9)
result_pow_7490 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 5), '**', x_7488, int_7489)

list_7496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 5), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 5), list_7496, result_pow_7490)
# Assigning a type to the variable 'S' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'S', list_7496)

# Assigning a ListComp to a Name (line 10):
# Calculating list comprehension
# Calculating comprehension expression

# Call to range(...): (line 10)
# Processing the call arguments (line 10)
int_7502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 27), 'int')
# Processing the call keyword arguments (line 10)
kwargs_7503 = {}
# Getting the type of 'range' (line 10)
range_7501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 21), 'range', False)
# Calling range(args, kwargs) (line 10)
range_call_result_7504 = invoke(stypy.reporting.localization.Localization(__file__, 10, 21), range_7501, *[int_7502], **kwargs_7503)

comprehension_7505 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 5), range_call_result_7504)
# Assigning a type to the variable 'i' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'i', comprehension_7505)

# Call to str(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'i' (line 10)
i_7498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 9), 'i', False)
# Processing the call keyword arguments (line 10)
kwargs_7499 = {}
# Getting the type of 'str' (line 10)
str_7497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'str', False)
# Calling str(args, kwargs) (line 10)
str_call_result_7500 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), str_7497, *[i_7498], **kwargs_7499)

list_7506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 5), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 5), list_7506, str_call_result_7500)
# Assigning a type to the variable 'V' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'V', list_7506)

# Assigning a List to a Name (line 12):

# Obtaining an instance of the builtin type 'list' (line 12)
list_7507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
int_7508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), list_7507, int_7508)
# Adding element type (line 12)
int_7509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), list_7507, int_7509)
# Adding element type (line 12)
int_7510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 14), list_7507, int_7510)

# Assigning a type to the variable 'normal_list' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'normal_list', list_7507)

# Assigning a Tuple to a Name (line 13):

# Obtaining an instance of the builtin type 'tuple' (line 13)
tuple_7511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 13)
# Adding element type (line 13)
int_7512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), tuple_7511, int_7512)
# Adding element type (line 13)
int_7513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), tuple_7511, int_7513)
# Adding element type (line 13)
int_7514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), tuple_7511, int_7514)

# Assigning a type to the variable 'tuple_' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'tuple_', tuple_7511)

# Assigning a Call to a Name (line 15):

# Call to fun1(...): (line 15)
# Processing the call arguments (line 15)

# Obtaining the type of the subscript
int_7516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'int')
# Getting the type of 'S' (line 15)
S_7517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'S', False)
# Obtaining the member '__getitem__' of a type (line 15)
getitem___7518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 10), S_7517, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 15)
subscript_call_result_7519 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), getitem___7518, int_7516)

# Processing the call keyword arguments (line 15)
kwargs_7520 = {}
# Getting the type of 'fun1' (line 15)
fun1_7515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'fun1', False)
# Calling fun1(args, kwargs) (line 15)
fun1_call_result_7521 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), fun1_7515, *[subscript_call_result_7519], **kwargs_7520)

# Assigning a type to the variable 'r1' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r1', fun1_call_result_7521)

# Assigning a Call to a Name (line 16):

# Call to fun1(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of 'S' (line 16)
S_7523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'S', False)
# Processing the call keyword arguments (line 16)
kwargs_7524 = {}
# Getting the type of 'fun1' (line 16)
fun1_7522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 5), 'fun1', False)
# Calling fun1(args, kwargs) (line 16)
fun1_call_result_7525 = invoke(stypy.reporting.localization.Localization(__file__, 16, 5), fun1_7522, *[S_7523], **kwargs_7524)

# Assigning a type to the variable 'r2' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'r2', fun1_call_result_7525)

# Assigning a Call to a Name (line 17):

# Call to fun1(...): (line 17)
# Processing the call arguments (line 17)
# Getting the type of 'normal_list' (line 17)
normal_list_7527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'normal_list', False)
# Processing the call keyword arguments (line 17)
kwargs_7528 = {}
# Getting the type of 'fun1' (line 17)
fun1_7526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'fun1', False)
# Calling fun1(args, kwargs) (line 17)
fun1_call_result_7529 = invoke(stypy.reporting.localization.Localization(__file__, 17, 5), fun1_7526, *[normal_list_7527], **kwargs_7528)

# Assigning a type to the variable 'r3' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r3', fun1_call_result_7529)

# Assigning a Call to a Name (line 18):

# Call to fun1(...): (line 18)
# Processing the call arguments (line 18)
# Getting the type of 'tuple_' (line 18)
tuple__7531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'tuple_', False)
# Processing the call keyword arguments (line 18)
kwargs_7532 = {}
# Getting the type of 'fun1' (line 18)
fun1_7530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'fun1', False)
# Calling fun1(args, kwargs) (line 18)
fun1_call_result_7533 = invoke(stypy.reporting.localization.Localization(__file__, 18, 5), fun1_7530, *[tuple__7531], **kwargs_7532)

# Assigning a type to the variable 'r4' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'r4', fun1_call_result_7533)

# Assigning a Call to a Name (line 19):

# Call to fun2(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of 'V' (line 19)
V_7535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'V', False)
# Processing the call keyword arguments (line 19)
kwargs_7536 = {}
# Getting the type of 'fun2' (line 19)
fun2_7534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'fun2', False)
# Calling fun2(args, kwargs) (line 19)
fun2_call_result_7537 = invoke(stypy.reporting.localization.Localization(__file__, 19, 5), fun2_7534, *[V_7535], **kwargs_7536)

# Assigning a type to the variable 'r5' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r5', fun2_call_result_7537)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
