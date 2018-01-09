
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "At least one (but not all) execution paths has an execution flow free of type errors"
3: 
4: if __name__ == '__main__':
5: 
6:     def functionb(x):
7:         if True:
8:             # Type warning
9:             x = x / 2
10:         return x
11: 
12: 
13:     def functionb2(x):
14:         if True:
15:             # Type warning
16:             x = x / 2
17:         return x
18: 
19: 
20:     y = functionb("a")
21:     y = functionb2(range(5))
22: 
23: 
24:     def functionc(x):
25:         for i in range(5):
26:             # Type warning
27:             x /= 2
28:         return x
29: 
30: 
31:     def functionc2(x):
32:         for i in range(5):
33:             # Type warning
34:             x /= 2
35:         return x
36: 
37: 
38:     y = functionc("a")
39:     y = functionc2(range(5))
40: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'At least one (but not all) execution paths has an execution flow free of type errors')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def functionb(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'functionb'
        module_type_store = module_type_store.open_function_context('functionb', 6, 4, False)
        
        # Passed parameters checking function
        functionb.stypy_localization = localization
        functionb.stypy_type_of_self = None
        functionb.stypy_type_store = module_type_store
        functionb.stypy_function_name = 'functionb'
        functionb.stypy_param_names_list = ['x']
        functionb.stypy_varargs_param_name = None
        functionb.stypy_kwargs_param_name = None
        functionb.stypy_call_defaults = defaults
        functionb.stypy_call_varargs = varargs
        functionb.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'functionb', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'functionb', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'functionb(...)' code ##################

        
        # Getting the type of 'True' (line 7)
        True_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'True')
        # Testing the type of an if condition (line 7)
        if_condition_3 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 7, 8), True_2)
        # Assigning a type to the variable 'if_condition_3' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'if_condition_3', if_condition_3)
        # SSA begins for if statement (line 7)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 9):
        # Getting the type of 'x' (line 9)
        x_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 16), 'x')
        int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 20), 'int')
        # Applying the binary operator 'div' (line 9)
        result_div_6 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 16), 'div', x_4, int_5)
        
        # Assigning a type to the variable 'x' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'x', result_div_6)
        # SSA join for if statement (line 7)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'x' (line 10)
        x_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'stypy_return_type', x_7)
        
        # ################# End of 'functionb(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'functionb' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'functionb'
        return stypy_return_type_8

    # Assigning a type to the variable 'functionb' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'functionb', functionb)

    @norecursion
    def functionb2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'functionb2'
        module_type_store = module_type_store.open_function_context('functionb2', 13, 4, False)
        
        # Passed parameters checking function
        functionb2.stypy_localization = localization
        functionb2.stypy_type_of_self = None
        functionb2.stypy_type_store = module_type_store
        functionb2.stypy_function_name = 'functionb2'
        functionb2.stypy_param_names_list = ['x']
        functionb2.stypy_varargs_param_name = None
        functionb2.stypy_kwargs_param_name = None
        functionb2.stypy_call_defaults = defaults
        functionb2.stypy_call_varargs = varargs
        functionb2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'functionb2', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'functionb2', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'functionb2(...)' code ##################

        
        # Getting the type of 'True' (line 14)
        True_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'True')
        # Testing the type of an if condition (line 14)
        if_condition_10 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 8), True_9)
        # Assigning a type to the variable 'if_condition_10' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'if_condition_10', if_condition_10)
        # SSA begins for if statement (line 14)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 16):
        # Getting the type of 'x' (line 16)
        x_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'x')
        int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 20), 'int')
        # Applying the binary operator 'div' (line 16)
        result_div_13 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 16), 'div', x_11, int_12)
        
        # Assigning a type to the variable 'x' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'x', result_div_13)
        # SSA join for if statement (line 14)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'x' (line 17)
        x_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'stypy_return_type', x_14)
        
        # ################# End of 'functionb2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'functionb2' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'functionb2'
        return stypy_return_type_15

    # Assigning a type to the variable 'functionb2' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'functionb2', functionb2)
    
    # Assigning a Call to a Name (line 20):
    
    # Call to functionb(...): (line 20)
    # Processing the call arguments (line 20)
    str_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 18), 'str', 'a')
    # Processing the call keyword arguments (line 20)
    kwargs_18 = {}
    # Getting the type of 'functionb' (line 20)
    functionb_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'functionb', False)
    # Calling functionb(args, kwargs) (line 20)
    functionb_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), functionb_16, *[str_17], **kwargs_18)
    
    # Assigning a type to the variable 'y' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'y', functionb_call_result_19)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to functionb2(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Call to range(...): (line 21)
    # Processing the call arguments (line 21)
    int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'int')
    # Processing the call keyword arguments (line 21)
    kwargs_23 = {}
    # Getting the type of 'range' (line 21)
    range_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'range', False)
    # Calling range(args, kwargs) (line 21)
    range_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 21, 19), range_21, *[int_22], **kwargs_23)
    
    # Processing the call keyword arguments (line 21)
    kwargs_25 = {}
    # Getting the type of 'functionb2' (line 21)
    functionb2_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'functionb2', False)
    # Calling functionb2(args, kwargs) (line 21)
    functionb2_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), functionb2_20, *[range_call_result_24], **kwargs_25)
    
    # Assigning a type to the variable 'y' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'y', functionb2_call_result_26)

    @norecursion
    def functionc(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'functionc'
        module_type_store = module_type_store.open_function_context('functionc', 24, 4, False)
        
        # Passed parameters checking function
        functionc.stypy_localization = localization
        functionc.stypy_type_of_self = None
        functionc.stypy_type_store = module_type_store
        functionc.stypy_function_name = 'functionc'
        functionc.stypy_param_names_list = ['x']
        functionc.stypy_varargs_param_name = None
        functionc.stypy_kwargs_param_name = None
        functionc.stypy_call_defaults = defaults
        functionc.stypy_call_varargs = varargs
        functionc.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'functionc', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'functionc', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'functionc(...)' code ##################

        
        
        # Call to range(...): (line 25)
        # Processing the call arguments (line 25)
        int_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 23), 'int')
        # Processing the call keyword arguments (line 25)
        kwargs_29 = {}
        # Getting the type of 'range' (line 25)
        range_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'range', False)
        # Calling range(args, kwargs) (line 25)
        range_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 25, 17), range_27, *[int_28], **kwargs_29)
        
        # Testing the type of a for loop iterable (line 25)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 25, 8), range_call_result_30)
        # Getting the type of the for loop variable (line 25)
        for_loop_var_31 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 25, 8), range_call_result_30)
        # Assigning a type to the variable 'i' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'i', for_loop_var_31)
        # SSA begins for a for statement (line 25)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'x' (line 27)
        x_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'x')
        int_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 17), 'int')
        # Applying the binary operator 'div=' (line 27)
        result_div_34 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 12), 'div=', x_32, int_33)
        # Assigning a type to the variable 'x' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'x', result_div_34)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'x' (line 28)
        x_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type', x_35)
        
        # ################# End of 'functionc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'functionc' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_36)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'functionc'
        return stypy_return_type_36

    # Assigning a type to the variable 'functionc' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'functionc', functionc)

    @norecursion
    def functionc2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'functionc2'
        module_type_store = module_type_store.open_function_context('functionc2', 31, 4, False)
        
        # Passed parameters checking function
        functionc2.stypy_localization = localization
        functionc2.stypy_type_of_self = None
        functionc2.stypy_type_store = module_type_store
        functionc2.stypy_function_name = 'functionc2'
        functionc2.stypy_param_names_list = ['x']
        functionc2.stypy_varargs_param_name = None
        functionc2.stypy_kwargs_param_name = None
        functionc2.stypy_call_defaults = defaults
        functionc2.stypy_call_varargs = varargs
        functionc2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'functionc2', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'functionc2', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'functionc2(...)' code ##################

        
        
        # Call to range(...): (line 32)
        # Processing the call arguments (line 32)
        int_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'int')
        # Processing the call keyword arguments (line 32)
        kwargs_39 = {}
        # Getting the type of 'range' (line 32)
        range_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 17), 'range', False)
        # Calling range(args, kwargs) (line 32)
        range_call_result_40 = invoke(stypy.reporting.localization.Localization(__file__, 32, 17), range_37, *[int_38], **kwargs_39)
        
        # Testing the type of a for loop iterable (line 32)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 32, 8), range_call_result_40)
        # Getting the type of the for loop variable (line 32)
        for_loop_var_41 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 32, 8), range_call_result_40)
        # Assigning a type to the variable 'i' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'i', for_loop_var_41)
        # SSA begins for a for statement (line 32)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'x' (line 34)
        x_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'x')
        int_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 17), 'int')
        # Applying the binary operator 'div=' (line 34)
        result_div_44 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 12), 'div=', x_42, int_43)
        # Assigning a type to the variable 'x' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'x', result_div_44)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'x' (line 35)
        x_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type', x_45)
        
        # ################# End of 'functionc2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'functionc2' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_46)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'functionc2'
        return stypy_return_type_46

    # Assigning a type to the variable 'functionc2' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'functionc2', functionc2)
    
    # Assigning a Call to a Name (line 38):
    
    # Call to functionc(...): (line 38)
    # Processing the call arguments (line 38)
    str_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 18), 'str', 'a')
    # Processing the call keyword arguments (line 38)
    kwargs_49 = {}
    # Getting the type of 'functionc' (line 38)
    functionc_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'functionc', False)
    # Calling functionc(args, kwargs) (line 38)
    functionc_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), functionc_47, *[str_48], **kwargs_49)
    
    # Assigning a type to the variable 'y' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'y', functionc_call_result_50)
    
    # Assigning a Call to a Name (line 39):
    
    # Call to functionc2(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Call to range(...): (line 39)
    # Processing the call arguments (line 39)
    int_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 25), 'int')
    # Processing the call keyword arguments (line 39)
    kwargs_54 = {}
    # Getting the type of 'range' (line 39)
    range_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'range', False)
    # Calling range(args, kwargs) (line 39)
    range_call_result_55 = invoke(stypy.reporting.localization.Localization(__file__, 39, 19), range_52, *[int_53], **kwargs_54)
    
    # Processing the call keyword arguments (line 39)
    kwargs_56 = {}
    # Getting the type of 'functionc2' (line 39)
    functionc2_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'functionc2', False)
    # Calling functionc2(args, kwargs) (line 39)
    functionc2_call_result_57 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), functionc2_51, *[range_call_result_55], **kwargs_56)
    
    # Assigning a type to the variable 'y' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'y', functionc2_call_result_57)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
