
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: if True:
3:     x = None
4: else:
5:     x = 3
6: 
7: 
8: def f():
9:     if x is None:
10:         raise Exception("x is None")
11:     else:
12:         pass
13: 
14:     return x / 3
15: 
16: r = f()

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Getting the type of 'True' (line 2)
True_1 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 3), 'True')
# Testing if the type of an if condition is none (line 2)

if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 2, 0), True_1):
    
    # Assigning a Num to a Name (line 5):
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 8), 'int')
    # Assigning a type to the variable 'x' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'x', int_4)
else:
    
    # Testing the type of an if condition (line 2)
    if_condition_2 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 2, 0), True_1)
    # Assigning a type to the variable 'if_condition_2' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'if_condition_2', if_condition_2)
    # SSA begins for if statement (line 2)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 3):
    # Getting the type of 'None' (line 3)
    None_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'None')
    # Assigning a type to the variable 'x' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'x', None_3)
    # SSA branch for the else part of an if statement (line 2)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 5):
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 8), 'int')
    # Assigning a type to the variable 'x' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'x', int_4)
    # SSA join for if statement (line 2)
    module_type_store = module_type_store.join_ssa_context()
    


@norecursion
def f(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f'
    module_type_store = module_type_store.open_function_context('f', 8, 0, False)
    
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

    
    # Type idiom detected: calculating its left and rigth part (line 9)
    # Getting the type of 'x' (line 9)
    x_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 7), 'x')
    # Getting the type of 'None' (line 9)
    None_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'None')
    
    (may_be_7, more_types_in_union_8) = may_be_none(x_5, None_6)

    if may_be_7:

        if more_types_in_union_8:
            # Runtime conditional SSA (line 9)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to Exception(...): (line 10)
        # Processing the call arguments (line 10)
        str_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 24), 'str', 'x is None')
        # Processing the call keyword arguments (line 10)
        kwargs_11 = {}
        # Getting the type of 'Exception' (line 10)
        Exception_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 14), 'Exception', False)
        # Calling Exception(args, kwargs) (line 10)
        Exception_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 10, 14), Exception_9, *[str_10], **kwargs_11)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 10, 8), Exception_call_result_12, 'raise parameter', BaseException)

        if more_types_in_union_8:
            # Runtime conditional SSA for else branch (line 9)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_7) or more_types_in_union_8):
        pass

        if (may_be_7 and more_types_in_union_8):
            # SSA join for if statement (line 9)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'x' (line 9)
    x_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'x')
    # Assigning a type to the variable 'x' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'x', remove_type_from_union(x_13, types.NoneType))
    # Getting the type of 'x' (line 14)
    x_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'x')
    int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'int')
    # Applying the binary operator 'div' (line 14)
    result_div_16 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 11), 'div', x_14, int_15)
    
    # Assigning a type to the variable 'stypy_return_type' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type', result_div_16)
    
    # ################# End of 'f(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f'
    return stypy_return_type_17

# Assigning a type to the variable 'f' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'f', f)

# Assigning a Call to a Name (line 16):

# Call to f(...): (line 16)
# Processing the call keyword arguments (line 16)
kwargs_19 = {}
# Getting the type of 'f' (line 16)
f_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'f', False)
# Calling f(args, kwargs) (line 16)
f_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), f_18, *[], **kwargs_19)

# Assigning a type to the variable 'r' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'r', f_call_result_20)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
