
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Incorrect types passed on function calls, but erroneous code is unreachable"
4: 
5: if __name__ == '__main__':
6: 
7:     def functionb(x):
8:         if type(x) is str or type(x) is list:
9:             return 3
10:         # Type warning
11:         x = x / 2
12:         return x
13: 
14: 
15:     def functionb2(x):
16:         if type(x) is str or type(x) is list:
17:             return 3
18:         # Type warning
19:         x = x / 2
20:         return x
21: 
22: 
23:     y = functionb("a")
24:     y = functionb2(range(5))
25: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Incorrect types passed on function calls, but erroneous code is unreachable')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def functionb(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'functionb'
        module_type_store = module_type_store.open_function_context('functionb', 7, 4, False)
        
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

        
        
        # Evaluating a boolean operation
        
        
        # Call to type(...): (line 8)
        # Processing the call arguments (line 8)
        # Getting the type of 'x' (line 8)
        x_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 16), 'x', False)
        # Processing the call keyword arguments (line 8)
        kwargs_4 = {}
        # Getting the type of 'type' (line 8)
        type_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 11), 'type', False)
        # Calling type(args, kwargs) (line 8)
        type_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 8, 11), type_2, *[x_3], **kwargs_4)
        
        # Getting the type of 'str' (line 8)
        str_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 22), 'str')
        # Applying the binary operator 'is' (line 8)
        result_is__7 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 11), 'is', type_call_result_5, str_6)
        
        
        
        # Call to type(...): (line 8)
        # Processing the call arguments (line 8)
        # Getting the type of 'x' (line 8)
        x_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 34), 'x', False)
        # Processing the call keyword arguments (line 8)
        kwargs_10 = {}
        # Getting the type of 'type' (line 8)
        type_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 29), 'type', False)
        # Calling type(args, kwargs) (line 8)
        type_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 8, 29), type_8, *[x_9], **kwargs_10)
        
        # Getting the type of 'list' (line 8)
        list_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 40), 'list')
        # Applying the binary operator 'is' (line 8)
        result_is__13 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 29), 'is', type_call_result_11, list_12)
        
        # Applying the binary operator 'or' (line 8)
        result_or_keyword_14 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 11), 'or', result_is__7, result_is__13)
        
        # Testing the type of an if condition (line 8)
        if_condition_15 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 8, 8), result_or_keyword_14)
        # Assigning a type to the variable 'if_condition_15' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'if_condition_15', if_condition_15)
        # SSA begins for if statement (line 8)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'stypy_return_type', int_16)
        # SSA join for if statement (line 8)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 11):
        # Getting the type of 'x' (line 11)
        x_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'x')
        int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 16), 'int')
        # Applying the binary operator 'div' (line 11)
        result_div_19 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 12), 'div', x_17, int_18)
        
        # Assigning a type to the variable 'x' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'x', result_div_19)
        # Getting the type of 'x' (line 12)
        x_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'stypy_return_type', x_20)
        
        # ################# End of 'functionb(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'functionb' in the type store
        # Getting the type of 'stypy_return_type' (line 7)
        stypy_return_type_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'functionb'
        return stypy_return_type_21

    # Assigning a type to the variable 'functionb' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'functionb', functionb)

    @norecursion
    def functionb2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'functionb2'
        module_type_store = module_type_store.open_function_context('functionb2', 15, 4, False)
        
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

        
        
        # Evaluating a boolean operation
        
        
        # Call to type(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'x' (line 16)
        x_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'x', False)
        # Processing the call keyword arguments (line 16)
        kwargs_24 = {}
        # Getting the type of 'type' (line 16)
        type_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'type', False)
        # Calling type(args, kwargs) (line 16)
        type_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 16, 11), type_22, *[x_23], **kwargs_24)
        
        # Getting the type of 'str' (line 16)
        str_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 22), 'str')
        # Applying the binary operator 'is' (line 16)
        result_is__27 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 11), 'is', type_call_result_25, str_26)
        
        
        
        # Call to type(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'x' (line 16)
        x_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 34), 'x', False)
        # Processing the call keyword arguments (line 16)
        kwargs_30 = {}
        # Getting the type of 'type' (line 16)
        type_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 29), 'type', False)
        # Calling type(args, kwargs) (line 16)
        type_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 16, 29), type_28, *[x_29], **kwargs_30)
        
        # Getting the type of 'list' (line 16)
        list_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 40), 'list')
        # Applying the binary operator 'is' (line 16)
        result_is__33 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 29), 'is', type_call_result_31, list_32)
        
        # Applying the binary operator 'or' (line 16)
        result_or_keyword_34 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 11), 'or', result_is__27, result_is__33)
        
        # Testing the type of an if condition (line 16)
        if_condition_35 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 16, 8), result_or_keyword_34)
        # Assigning a type to the variable 'if_condition_35' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'if_condition_35', if_condition_35)
        # SSA begins for if statement (line 16)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'stypy_return_type', int_36)
        # SSA join for if statement (line 16)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 19):
        # Getting the type of 'x' (line 19)
        x_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'x')
        int_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 16), 'int')
        # Applying the binary operator 'div' (line 19)
        result_div_39 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 12), 'div', x_37, int_38)
        
        # Assigning a type to the variable 'x' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'x', result_div_39)
        # Getting the type of 'x' (line 20)
        x_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'stypy_return_type', x_40)
        
        # ################# End of 'functionb2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'functionb2' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_41)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'functionb2'
        return stypy_return_type_41

    # Assigning a type to the variable 'functionb2' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'functionb2', functionb2)
    
    # Assigning a Call to a Name (line 23):
    
    # Call to functionb(...): (line 23)
    # Processing the call arguments (line 23)
    str_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 18), 'str', 'a')
    # Processing the call keyword arguments (line 23)
    kwargs_44 = {}
    # Getting the type of 'functionb' (line 23)
    functionb_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'functionb', False)
    # Calling functionb(args, kwargs) (line 23)
    functionb_call_result_45 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), functionb_42, *[str_43], **kwargs_44)
    
    # Assigning a type to the variable 'y' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'y', functionb_call_result_45)
    
    # Assigning a Call to a Name (line 24):
    
    # Call to functionb2(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Call to range(...): (line 24)
    # Processing the call arguments (line 24)
    int_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'int')
    # Processing the call keyword arguments (line 24)
    kwargs_49 = {}
    # Getting the type of 'range' (line 24)
    range_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 19), 'range', False)
    # Calling range(args, kwargs) (line 24)
    range_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 24, 19), range_47, *[int_48], **kwargs_49)
    
    # Processing the call keyword arguments (line 24)
    kwargs_51 = {}
    # Getting the type of 'functionb2' (line 24)
    functionb2_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'functionb2', False)
    # Calling functionb2(args, kwargs) (line 24)
    functionb2_call_result_52 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), functionb2_46, *[range_call_result_50], **kwargs_51)
    
    # Assigning a type to the variable 'y' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'y', functionb2_call_result_52)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
