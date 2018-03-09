
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: def f():
3:     return True, 0.0
4: 
5: 
6: d = {1: "one",
7:      2: "two"}
8: 
9: d2 = {(1,1): "pair of ones",
10:       (2,2): "pair of twos"}
11: 
12: for (k, v) in enumerate(d.values()):
13:     print str(k) + ", " + str(v)
14: 
15: for (k2, v2) in d.items():
16:     print str(k2) + ", " + str(v2)
17: 
18: for (k3, v3) in d2.keys():
19:     print str(k3) + ", " + str(v3)
20: 
21: k4, v4 = (True, 1)
22: print str(k4) + ", " + str(v4)
23: 
24: k5, v5 = f()
25: print str(k5) + ", " + str(v5)
26: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def f(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f'
    module_type_store = module_type_store.open_function_context('f', 2, 0, False)
    
    # Passed parameters checking function
    f.stypy_localization = localization
    f.stypy_type_of_self = None
    f.stypy_type_store = module_type_store
    f.stypy_function_name = 'f'
    f.stypy_param_names_list = []
    f.stypy_varargs_param_name = None
    f.stypy_kwargs_param_name = None
    f.stypy_call_defaults = defaults
    f.stypy_call_varargs = varargs
    f.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f(...)' code ##################

    
    # Obtaining an instance of the builtin type 'tuple' (line 3)
    tuple_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 3)
    # Adding element type (line 3)
    # Getting the type of 'True' (line 3)
    True_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 11), 'True')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 11), tuple_6, True_7)
    # Adding element type (line 3)
    float_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 17), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 3, 11), tuple_6, float_8)
    
    # Assigning a type to the variable 'stypy_return_type' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'stypy_return_type', tuple_6)
    
    # ################# End of 'f(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f' in the type store
    # Getting the type of 'stypy_return_type' (line 2)
    stypy_return_type_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f'
    return stypy_return_type_9

# Assigning a type to the variable 'f' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'f', f)

# Assigning a Dict to a Name (line 6):

# Assigning a Dict to a Name (line 6):

# Obtaining an instance of the builtin type 'dict' (line 6)
dict_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 6)
# Adding element type (key, value) (line 6)
int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 5), 'int')
str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 8), 'str', 'one')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 4), dict_10, (int_11, str_12))
# Adding element type (key, value) (line 6)
int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 5), 'int')
str_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 8), 'str', 'two')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 4), dict_10, (int_13, str_14))

# Assigning a type to the variable 'd' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'd', dict_10)

# Assigning a Dict to a Name (line 9):

# Assigning a Dict to a Name (line 9):

# Obtaining an instance of the builtin type 'dict' (line 9)
dict_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 5), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 9)
# Adding element type (key, value) (line 9)

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 7), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 7), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 7), tuple_16, int_17)
# Adding element type (line 9)
int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 7), tuple_16, int_18)

str_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'str', 'pair of ones')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 5), dict_15, (tuple_16, str_19))
# Adding element type (key, value) (line 9)

# Obtaining an instance of the builtin type 'tuple' (line 10)
tuple_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 7), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 10)
# Adding element type (line 10)
int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 7), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 7), tuple_20, int_21)
# Adding element type (line 10)
int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 9), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 7), tuple_20, int_22)

str_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 13), 'str', 'pair of twos')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 5), dict_15, (tuple_20, str_23))

# Assigning a type to the variable 'd2' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'd2', dict_15)


# Call to enumerate(...): (line 12)
# Processing the call arguments (line 12)

# Call to values(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_27 = {}
# Getting the type of 'd' (line 12)
d_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 24), 'd', False)
# Obtaining the member 'values' of a type (line 12)
values_26 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 24), d_25, 'values')
# Calling values(args, kwargs) (line 12)
values_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 12, 24), values_26, *[], **kwargs_27)

# Processing the call keyword arguments (line 12)
kwargs_29 = {}
# Getting the type of 'enumerate' (line 12)
enumerate_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'enumerate', False)
# Calling enumerate(args, kwargs) (line 12)
enumerate_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 12, 14), enumerate_24, *[values_call_result_28], **kwargs_29)

# Assigning a type to the variable 'enumerate_call_result_30' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'enumerate_call_result_30', enumerate_call_result_30)
# Testing if the for loop is going to be iterated (line 12)
# Testing the type of a for loop iterable (line 12)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 12, 0), enumerate_call_result_30)

if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 12, 0), enumerate_call_result_30):
    # Getting the type of the for loop variable (line 12)
    for_loop_var_31 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 12, 0), enumerate_call_result_30)
    # Assigning a type to the variable 'k' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 0), for_loop_var_31, 2, 0))
    # Assigning a type to the variable 'v' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 0), for_loop_var_31, 2, 1))
    # SSA begins for a for statement (line 12)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to str(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'k' (line 13)
    k_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'k', False)
    # Processing the call keyword arguments (line 13)
    kwargs_34 = {}
    # Getting the type of 'str' (line 13)
    str_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'str', False)
    # Calling str(args, kwargs) (line 13)
    str_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), str_32, *[k_33], **kwargs_34)
    
    str_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 19), 'str', ', ')
    # Applying the binary operator '+' (line 13)
    result_add_37 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 10), '+', str_call_result_35, str_36)
    
    
    # Call to str(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'v' (line 13)
    v_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 30), 'v', False)
    # Processing the call keyword arguments (line 13)
    kwargs_40 = {}
    # Getting the type of 'str' (line 13)
    str_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 26), 'str', False)
    # Calling str(args, kwargs) (line 13)
    str_call_result_41 = invoke(stypy.reporting.localization.Localization(__file__, 13, 26), str_38, *[v_39], **kwargs_40)
    
    # Applying the binary operator '+' (line 13)
    result_add_42 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 24), '+', result_add_37, str_call_result_41)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()




# Call to items(...): (line 15)
# Processing the call keyword arguments (line 15)
kwargs_45 = {}
# Getting the type of 'd' (line 15)
d_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 16), 'd', False)
# Obtaining the member 'items' of a type (line 15)
items_44 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 16), d_43, 'items')
# Calling items(args, kwargs) (line 15)
items_call_result_46 = invoke(stypy.reporting.localization.Localization(__file__, 15, 16), items_44, *[], **kwargs_45)

# Assigning a type to the variable 'items_call_result_46' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'items_call_result_46', items_call_result_46)
# Testing if the for loop is going to be iterated (line 15)
# Testing the type of a for loop iterable (line 15)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 15, 0), items_call_result_46)

if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 15, 0), items_call_result_46):
    # Getting the type of the for loop variable (line 15)
    for_loop_var_47 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 15, 0), items_call_result_46)
    # Assigning a type to the variable 'k2' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'k2', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 0), for_loop_var_47, 2, 0))
    # Assigning a type to the variable 'v2' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'v2', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 0), for_loop_var_47, 2, 1))
    # SSA begins for a for statement (line 15)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to str(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'k2' (line 16)
    k2_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'k2', False)
    # Processing the call keyword arguments (line 16)
    kwargs_50 = {}
    # Getting the type of 'str' (line 16)
    str_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'str', False)
    # Calling str(args, kwargs) (line 16)
    str_call_result_51 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), str_48, *[k2_49], **kwargs_50)
    
    str_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 20), 'str', ', ')
    # Applying the binary operator '+' (line 16)
    result_add_53 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 10), '+', str_call_result_51, str_52)
    
    
    # Call to str(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'v2' (line 16)
    v2_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 31), 'v2', False)
    # Processing the call keyword arguments (line 16)
    kwargs_56 = {}
    # Getting the type of 'str' (line 16)
    str_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 27), 'str', False)
    # Calling str(args, kwargs) (line 16)
    str_call_result_57 = invoke(stypy.reporting.localization.Localization(__file__, 16, 27), str_54, *[v2_55], **kwargs_56)
    
    # Applying the binary operator '+' (line 16)
    result_add_58 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 25), '+', result_add_53, str_call_result_57)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()




# Call to keys(...): (line 18)
# Processing the call keyword arguments (line 18)
kwargs_61 = {}
# Getting the type of 'd2' (line 18)
d2_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'd2', False)
# Obtaining the member 'keys' of a type (line 18)
keys_60 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 16), d2_59, 'keys')
# Calling keys(args, kwargs) (line 18)
keys_call_result_62 = invoke(stypy.reporting.localization.Localization(__file__, 18, 16), keys_60, *[], **kwargs_61)

# Assigning a type to the variable 'keys_call_result_62' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'keys_call_result_62', keys_call_result_62)
# Testing if the for loop is going to be iterated (line 18)
# Testing the type of a for loop iterable (line 18)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 18, 0), keys_call_result_62)

if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 18, 0), keys_call_result_62):
    # Getting the type of the for loop variable (line 18)
    for_loop_var_63 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 18, 0), keys_call_result_62)
    # Assigning a type to the variable 'k3' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'k3', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 0), for_loop_var_63, 2, 0))
    # Assigning a type to the variable 'v3' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'v3', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 0), for_loop_var_63, 2, 1))
    # SSA begins for a for statement (line 18)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to str(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'k3' (line 19)
    k3_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 14), 'k3', False)
    # Processing the call keyword arguments (line 19)
    kwargs_66 = {}
    # Getting the type of 'str' (line 19)
    str_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'str', False)
    # Calling str(args, kwargs) (line 19)
    str_call_result_67 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), str_64, *[k3_65], **kwargs_66)
    
    str_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'str', ', ')
    # Applying the binary operator '+' (line 19)
    result_add_69 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 10), '+', str_call_result_67, str_68)
    
    
    # Call to str(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'v3' (line 19)
    v3_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 31), 'v3', False)
    # Processing the call keyword arguments (line 19)
    kwargs_72 = {}
    # Getting the type of 'str' (line 19)
    str_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 27), 'str', False)
    # Calling str(args, kwargs) (line 19)
    str_call_result_73 = invoke(stypy.reporting.localization.Localization(__file__, 19, 27), str_70, *[v3_71], **kwargs_72)
    
    # Applying the binary operator '+' (line 19)
    result_add_74 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 25), '+', result_add_69, str_call_result_73)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()



# Assigning a Tuple to a Tuple (line 21):

# Assigning a Name to a Name (line 21):
# Getting the type of 'True' (line 21)
True_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'True')
# Assigning a type to the variable 'tuple_assignment_1' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'tuple_assignment_1', True_75)

# Assigning a Num to a Name (line 21):
int_76 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 16), 'int')
# Assigning a type to the variable 'tuple_assignment_2' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'tuple_assignment_2', int_76)

# Assigning a Name to a Name (line 21):
# Getting the type of 'tuple_assignment_1' (line 21)
tuple_assignment_1_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'tuple_assignment_1')
# Assigning a type to the variable 'k4' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'k4', tuple_assignment_1_77)

# Assigning a Name to a Name (line 21):
# Getting the type of 'tuple_assignment_2' (line 21)
tuple_assignment_2_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'tuple_assignment_2')
# Assigning a type to the variable 'v4' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'v4', tuple_assignment_2_78)

# Call to str(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of 'k4' (line 22)
k4_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 10), 'k4', False)
# Processing the call keyword arguments (line 22)
kwargs_81 = {}
# Getting the type of 'str' (line 22)
str_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 6), 'str', False)
# Calling str(args, kwargs) (line 22)
str_call_result_82 = invoke(stypy.reporting.localization.Localization(__file__, 22, 6), str_79, *[k4_80], **kwargs_81)

str_83 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'str', ', ')
# Applying the binary operator '+' (line 22)
result_add_84 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 6), '+', str_call_result_82, str_83)


# Call to str(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of 'v4' (line 22)
v4_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 27), 'v4', False)
# Processing the call keyword arguments (line 22)
kwargs_87 = {}
# Getting the type of 'str' (line 22)
str_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 23), 'str', False)
# Calling str(args, kwargs) (line 22)
str_call_result_88 = invoke(stypy.reporting.localization.Localization(__file__, 22, 23), str_85, *[v4_86], **kwargs_87)

# Applying the binary operator '+' (line 22)
result_add_89 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 21), '+', result_add_84, str_call_result_88)


# Assigning a Call to a Tuple (line 24):

# Assigning a Call to a Name:

# Call to f(...): (line 24)
# Processing the call keyword arguments (line 24)
kwargs_91 = {}
# Getting the type of 'f' (line 24)
f_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 9), 'f', False)
# Calling f(args, kwargs) (line 24)
f_call_result_92 = invoke(stypy.reporting.localization.Localization(__file__, 24, 9), f_90, *[], **kwargs_91)

# Assigning a type to the variable 'call_assignment_3' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'call_assignment_3', f_call_result_92)

# Assigning a Call to a Name (line 24):

# Call to stypy_get_value_from_tuple(...):
# Processing the call arguments
# Getting the type of 'call_assignment_3' (line 24)
call_assignment_3_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'call_assignment_3', False)
# Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
stypy_get_value_from_tuple_call_result_94 = stypy_get_value_from_tuple(call_assignment_3_93, 2, 0)

# Assigning a type to the variable 'call_assignment_4' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'call_assignment_4', stypy_get_value_from_tuple_call_result_94)

# Assigning a Name to a Name (line 24):
# Getting the type of 'call_assignment_4' (line 24)
call_assignment_4_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'call_assignment_4')
# Assigning a type to the variable 'k5' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'k5', call_assignment_4_95)

# Assigning a Call to a Name (line 24):

# Call to stypy_get_value_from_tuple(...):
# Processing the call arguments
# Getting the type of 'call_assignment_3' (line 24)
call_assignment_3_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'call_assignment_3', False)
# Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
stypy_get_value_from_tuple_call_result_97 = stypy_get_value_from_tuple(call_assignment_3_96, 2, 1)

# Assigning a type to the variable 'call_assignment_5' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'call_assignment_5', stypy_get_value_from_tuple_call_result_97)

# Assigning a Name to a Name (line 24):
# Getting the type of 'call_assignment_5' (line 24)
call_assignment_5_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'call_assignment_5')
# Assigning a type to the variable 'v5' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'v5', call_assignment_5_98)

# Call to str(...): (line 25)
# Processing the call arguments (line 25)
# Getting the type of 'k5' (line 25)
k5_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'k5', False)
# Processing the call keyword arguments (line 25)
kwargs_101 = {}
# Getting the type of 'str' (line 25)
str_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 6), 'str', False)
# Calling str(args, kwargs) (line 25)
str_call_result_102 = invoke(stypy.reporting.localization.Localization(__file__, 25, 6), str_99, *[k5_100], **kwargs_101)

str_103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'str', ', ')
# Applying the binary operator '+' (line 25)
result_add_104 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 6), '+', str_call_result_102, str_103)


# Call to str(...): (line 25)
# Processing the call arguments (line 25)
# Getting the type of 'v5' (line 25)
v5_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 27), 'v5', False)
# Processing the call keyword arguments (line 25)
kwargs_107 = {}
# Getting the type of 'str' (line 25)
str_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 23), 'str', False)
# Calling str(args, kwargs) (line 25)
str_call_result_108 = invoke(stypy.reporting.localization.Localization(__file__, 25, 23), str_105, *[v5_106], **kwargs_107)

# Applying the binary operator '+' (line 25)
result_add_109 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 21), '+', result_add_104, str_call_result_108)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
