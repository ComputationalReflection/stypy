
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: 
4: def Proc8(Array1Par, Array2Par, IntParI1, IntParI2):
5:     global IntGlob
6: 
7:     IntLoc = (IntParI1 + 5)
8:     Array1Par[IntLoc] = IntParI2
9:     Array1Par[(IntLoc + 1)] = Array1Par[IntLoc]
10:     Array1Par[(IntLoc + 30)] = IntLoc
11: 
12:     for IntIndex in range(IntLoc, (IntLoc + 2)):
13:          Array2Par[IntLoc][IntIndex] = IntLoc
14:     Array2Par[IntLoc][(IntLoc - 1)] = (Array2Par[IntLoc][(IntLoc - 1)] + 1)
15:     Array2Par[(IntLoc + 20)][IntLoc] = Array1Par[IntLoc]
16: 
17:     IntGlob = 5
18: 
19: 
20: Array1Glob = ([0] * 51)
21: Array2Glob = map((lambda x: x[:]), ([Array1Glob] * 51))
22: IntLoc1 = 1
23: IntLoc3 = 3
24: 
25: Proc8(Array1Glob, Array2Glob, IntLoc1, IntLoc3)

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def Proc8(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Proc8'
    module_type_store = module_type_store.open_function_context('Proc8', 4, 0, False)
    
    # Passed parameters checking function
    Proc8.stypy_localization = localization
    Proc8.stypy_type_of_self = None
    Proc8.stypy_type_store = module_type_store
    Proc8.stypy_function_name = 'Proc8'
    Proc8.stypy_param_names_list = ['Array1Par', 'Array2Par', 'IntParI1', 'IntParI2']
    Proc8.stypy_varargs_param_name = None
    Proc8.stypy_kwargs_param_name = None
    Proc8.stypy_call_defaults = defaults
    Proc8.stypy_call_varargs = varargs
    Proc8.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Proc8', ['Array1Par', 'Array2Par', 'IntParI1', 'IntParI2'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Proc8', localization, ['Array1Par', 'Array2Par', 'IntParI1', 'IntParI2'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Proc8(...)' code ##################

    # Marking variables as global (line 5)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 5, 4), 'IntGlob')
    
    # Assigning a BinOp to a Name (line 7):
    # Getting the type of 'IntParI1' (line 7)
    IntParI1_1 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 14), 'IntParI1')
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'int')
    # Applying the binary operator '+' (line 7)
    result_add_3 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 14), '+', IntParI1_1, int_2)
    
    # Assigning a type to the variable 'IntLoc' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'IntLoc', result_add_3)
    
    # Assigning a Name to a Subscript (line 8):
    # Getting the type of 'IntParI2' (line 8)
    IntParI2_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 24), 'IntParI2')
    # Getting the type of 'Array1Par' (line 8)
    Array1Par_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'Array1Par')
    # Getting the type of 'IntLoc' (line 8)
    IntLoc_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 14), 'IntLoc')
    # Storing an element on a container (line 8)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 4), Array1Par_5, (IntLoc_6, IntParI2_4))
    
    # Assigning a Subscript to a Subscript (line 9):
    
    # Obtaining the type of the subscript
    # Getting the type of 'IntLoc' (line 9)
    IntLoc_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 40), 'IntLoc')
    # Getting the type of 'Array1Par' (line 9)
    Array1Par_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 30), 'Array1Par')
    # Obtaining the member '__getitem__' of a type (line 9)
    getitem___9 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 30), Array1Par_8, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 9)
    subscript_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 9, 30), getitem___9, IntLoc_7)
    
    # Getting the type of 'Array1Par' (line 9)
    Array1Par_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'Array1Par')
    # Getting the type of 'IntLoc' (line 9)
    IntLoc_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 15), 'IntLoc')
    int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 24), 'int')
    # Applying the binary operator '+' (line 9)
    result_add_14 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 15), '+', IntLoc_12, int_13)
    
    # Storing an element on a container (line 9)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 4), Array1Par_11, (result_add_14, subscript_call_result_10))
    
    # Assigning a Name to a Subscript (line 10):
    # Getting the type of 'IntLoc' (line 10)
    IntLoc_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 31), 'IntLoc')
    # Getting the type of 'Array1Par' (line 10)
    Array1Par_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'Array1Par')
    # Getting the type of 'IntLoc' (line 10)
    IntLoc_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 15), 'IntLoc')
    int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 24), 'int')
    # Applying the binary operator '+' (line 10)
    result_add_19 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 15), '+', IntLoc_17, int_18)
    
    # Storing an element on a container (line 10)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 4), Array1Par_16, (result_add_19, IntLoc_15))
    
    
    # Call to range(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'IntLoc' (line 12)
    IntLoc_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 26), 'IntLoc', False)
    # Getting the type of 'IntLoc' (line 12)
    IntLoc_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 35), 'IntLoc', False)
    int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 44), 'int')
    # Applying the binary operator '+' (line 12)
    result_add_24 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 35), '+', IntLoc_22, int_23)
    
    # Processing the call keyword arguments (line 12)
    kwargs_25 = {}
    # Getting the type of 'range' (line 12)
    range_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 20), 'range', False)
    # Calling range(args, kwargs) (line 12)
    range_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 12, 20), range_20, *[IntLoc_21, result_add_24], **kwargs_25)
    
    # Testing the type of a for loop iterable (line 12)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 12, 4), range_call_result_26)
    # Getting the type of the for loop variable (line 12)
    for_loop_var_27 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 12, 4), range_call_result_26)
    # Assigning a type to the variable 'IntIndex' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'IntIndex', for_loop_var_27)
    # SSA begins for a for statement (line 12)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Subscript (line 13):
    # Getting the type of 'IntLoc' (line 13)
    IntLoc_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 39), 'IntLoc')
    
    # Obtaining the type of the subscript
    # Getting the type of 'IntLoc' (line 13)
    IntLoc_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 19), 'IntLoc')
    # Getting the type of 'Array2Par' (line 13)
    Array2Par_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 9), 'Array2Par')
    # Obtaining the member '__getitem__' of a type (line 13)
    getitem___31 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 9), Array2Par_30, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 13)
    subscript_call_result_32 = invoke(stypy.reporting.localization.Localization(__file__, 13, 9), getitem___31, IntLoc_29)
    
    # Getting the type of 'IntIndex' (line 13)
    IntIndex_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 27), 'IntIndex')
    # Storing an element on a container (line 13)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 9), subscript_call_result_32, (IntIndex_33, IntLoc_28))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Subscript (line 14):
    
    # Obtaining the type of the subscript
    # Getting the type of 'IntLoc' (line 14)
    IntLoc_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 58), 'IntLoc')
    int_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 67), 'int')
    # Applying the binary operator '-' (line 14)
    result_sub_36 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 58), '-', IntLoc_34, int_35)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'IntLoc' (line 14)
    IntLoc_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 49), 'IntLoc')
    # Getting the type of 'Array2Par' (line 14)
    Array2Par_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 39), 'Array2Par')
    # Obtaining the member '__getitem__' of a type (line 14)
    getitem___39 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 39), Array2Par_38, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 14)
    subscript_call_result_40 = invoke(stypy.reporting.localization.Localization(__file__, 14, 39), getitem___39, IntLoc_37)
    
    # Obtaining the member '__getitem__' of a type (line 14)
    getitem___41 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 39), subscript_call_result_40, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 14)
    subscript_call_result_42 = invoke(stypy.reporting.localization.Localization(__file__, 14, 39), getitem___41, result_sub_36)
    
    int_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 73), 'int')
    # Applying the binary operator '+' (line 14)
    result_add_44 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 39), '+', subscript_call_result_42, int_43)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'IntLoc' (line 14)
    IntLoc_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'IntLoc')
    # Getting the type of 'Array2Par' (line 14)
    Array2Par_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'Array2Par')
    # Obtaining the member '__getitem__' of a type (line 14)
    getitem___47 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), Array2Par_46, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 14)
    subscript_call_result_48 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), getitem___47, IntLoc_45)
    
    # Getting the type of 'IntLoc' (line 14)
    IntLoc_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 23), 'IntLoc')
    int_50 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 32), 'int')
    # Applying the binary operator '-' (line 14)
    result_sub_51 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 23), '-', IntLoc_49, int_50)
    
    # Storing an element on a container (line 14)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 4), subscript_call_result_48, (result_sub_51, result_add_44))
    
    # Assigning a Subscript to a Subscript (line 15):
    
    # Obtaining the type of the subscript
    # Getting the type of 'IntLoc' (line 15)
    IntLoc_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 49), 'IntLoc')
    # Getting the type of 'Array1Par' (line 15)
    Array1Par_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 39), 'Array1Par')
    # Obtaining the member '__getitem__' of a type (line 15)
    getitem___54 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 39), Array1Par_53, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 15)
    subscript_call_result_55 = invoke(stypy.reporting.localization.Localization(__file__, 15, 39), getitem___54, IntLoc_52)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'IntLoc' (line 15)
    IntLoc_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'IntLoc')
    int_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 24), 'int')
    # Applying the binary operator '+' (line 15)
    result_add_58 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 15), '+', IntLoc_56, int_57)
    
    # Getting the type of 'Array2Par' (line 15)
    Array2Par_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'Array2Par')
    # Obtaining the member '__getitem__' of a type (line 15)
    getitem___60 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), Array2Par_59, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 15)
    subscript_call_result_61 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), getitem___60, result_add_58)
    
    # Getting the type of 'IntLoc' (line 15)
    IntLoc_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 29), 'IntLoc')
    # Storing an element on a container (line 15)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 4), subscript_call_result_61, (IntLoc_62, subscript_call_result_55))
    
    # Assigning a Num to a Name (line 17):
    int_63 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'int')
    # Assigning a type to the variable 'IntGlob' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'IntGlob', int_63)
    
    # ################# End of 'Proc8(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Proc8' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_64)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Proc8'
    return stypy_return_type_64

# Assigning a type to the variable 'Proc8' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'Proc8', Proc8)

# Assigning a BinOp to a Name (line 20):

# Obtaining an instance of the builtin type 'list' (line 20)
list_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
int_66 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 14), list_65, int_66)

int_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'int')
# Applying the binary operator '*' (line 20)
result_mul_68 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 14), '*', list_65, int_67)

# Assigning a type to the variable 'Array1Glob' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'Array1Glob', result_mul_68)

# Assigning a Call to a Name (line 21):

# Call to map(...): (line 21)
# Processing the call arguments (line 21)

@norecursion
def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_1'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 21, 18, True)
    # Passed parameters checking function
    _stypy_temp_lambda_1.stypy_localization = localization
    _stypy_temp_lambda_1.stypy_type_of_self = None
    _stypy_temp_lambda_1.stypy_type_store = module_type_store
    _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
    _stypy_temp_lambda_1.stypy_param_names_list = ['x']
    _stypy_temp_lambda_1.stypy_varargs_param_name = None
    _stypy_temp_lambda_1.stypy_kwargs_param_name = None
    _stypy_temp_lambda_1.stypy_call_defaults = defaults
    _stypy_temp_lambda_1.stypy_call_varargs = varargs
    _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_1', ['x'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Obtaining the type of the subscript
    slice_70 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 21, 28), None, None, None)
    # Getting the type of 'x' (line 21)
    x_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 28), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 21)
    getitem___72 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 28), x_71, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 21)
    subscript_call_result_73 = invoke(stypy.reporting.localization.Localization(__file__, 21, 28), getitem___72, slice_70)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 18), 'stypy_return_type', subscript_call_result_73)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_1' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 18), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_74)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_1'
    return stypy_return_type_74

# Assigning a type to the variable '_stypy_temp_lambda_1' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 18), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
# Getting the type of '_stypy_temp_lambda_1' (line 21)
_stypy_temp_lambda_1_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 18), '_stypy_temp_lambda_1')

# Obtaining an instance of the builtin type 'list' (line 21)
list_76 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 36), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
# Getting the type of 'Array1Glob' (line 21)
Array1Glob_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 37), 'Array1Glob', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 36), list_76, Array1Glob_77)

int_78 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 51), 'int')
# Applying the binary operator '*' (line 21)
result_mul_79 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 36), '*', list_76, int_78)

# Processing the call keyword arguments (line 21)
kwargs_80 = {}
# Getting the type of 'map' (line 21)
map_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 13), 'map', False)
# Calling map(args, kwargs) (line 21)
map_call_result_81 = invoke(stypy.reporting.localization.Localization(__file__, 21, 13), map_69, *[_stypy_temp_lambda_1_75, result_mul_79], **kwargs_80)

# Assigning a type to the variable 'Array2Glob' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'Array2Glob', map_call_result_81)

# Assigning a Num to a Name (line 22):
int_82 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 10), 'int')
# Assigning a type to the variable 'IntLoc1' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'IntLoc1', int_82)

# Assigning a Num to a Name (line 23):
int_83 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 10), 'int')
# Assigning a type to the variable 'IntLoc3' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'IntLoc3', int_83)

# Call to Proc8(...): (line 25)
# Processing the call arguments (line 25)
# Getting the type of 'Array1Glob' (line 25)
Array1Glob_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 6), 'Array1Glob', False)
# Getting the type of 'Array2Glob' (line 25)
Array2Glob_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 18), 'Array2Glob', False)
# Getting the type of 'IntLoc1' (line 25)
IntLoc1_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 30), 'IntLoc1', False)
# Getting the type of 'IntLoc3' (line 25)
IntLoc3_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 39), 'IntLoc3', False)
# Processing the call keyword arguments (line 25)
kwargs_89 = {}
# Getting the type of 'Proc8' (line 25)
Proc8_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'Proc8', False)
# Calling Proc8(args, kwargs) (line 25)
Proc8_call_result_90 = invoke(stypy.reporting.localization.Localization(__file__, 25, 0), Proc8_84, *[Array1Glob_85, Array2Glob_86, IntLoc1_87, IntLoc3_88], **kwargs_89)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
