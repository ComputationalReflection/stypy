
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "No execution path has an execution flow free of type errors"
3: 
4: if __name__ == '__main__':
5: 
6:     def functionb(x):
7:         if True:
8:             # Type error
9:             x /= 2
10:         else:
11:             # Type error
12:             x -= 2
13:         return 3
14: 
15: 
16:     def functionb2(x):
17:         if True:
18:             # Type error
19:             x /= 2
20:         else:
21:             # Type error
22:             x -= 2
23:         return 3
24: 
25: 
26:     y = functionb("a")
27:     y = functionb2(range(5))
28: 
29: 
30:     def functionc(x, **kwargs):
31:         if True:
32:             return int(x)
33:         else:
34:             # Type warning
35:             return kwargs[0]
36: 
37: 
38:     y = functionc(3, val="hi")
39:     # Type error
40:     y = y.thisdonotexist()
41: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'No execution path has an execution flow free of type errors')
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
        
        # Getting the type of 'x' (line 9)
        x_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'x')
        int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 17), 'int')
        # Applying the binary operator 'div=' (line 9)
        result_div_6 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 12), 'div=', x_4, int_5)
        # Assigning a type to the variable 'x' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'x', result_div_6)
        
        # SSA branch for the else part of an if statement (line 7)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'x' (line 12)
        x_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'x')
        int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 17), 'int')
        # Applying the binary operator '-=' (line 12)
        result_isub_9 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 12), '-=', x_7, int_8)
        # Assigning a type to the variable 'x' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'x', result_isub_9)
        
        # SSA join for if statement (line 7)
        module_type_store = module_type_store.join_ssa_context()
        
        int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', int_10)
        
        # ################# End of 'functionb(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'functionb' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'functionb'
        return stypy_return_type_11

    # Assigning a type to the variable 'functionb' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'functionb', functionb)

    @norecursion
    def functionb2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'functionb2'
        module_type_store = module_type_store.open_function_context('functionb2', 16, 4, False)
        
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

        
        # Getting the type of 'True' (line 17)
        True_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'True')
        # Testing the type of an if condition (line 17)
        if_condition_13 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 8), True_12)
        # Assigning a type to the variable 'if_condition_13' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'if_condition_13', if_condition_13)
        # SSA begins for if statement (line 17)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'x' (line 19)
        x_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'x')
        int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'int')
        # Applying the binary operator 'div=' (line 19)
        result_div_16 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 12), 'div=', x_14, int_15)
        # Assigning a type to the variable 'x' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'x', result_div_16)
        
        # SSA branch for the else part of an if statement (line 17)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'x' (line 22)
        x_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'x')
        int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'int')
        # Applying the binary operator '-=' (line 22)
        result_isub_19 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 12), '-=', x_17, int_18)
        # Assigning a type to the variable 'x' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'x', result_isub_19)
        
        # SSA join for if statement (line 17)
        module_type_store = module_type_store.join_ssa_context()
        
        int_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type', int_20)
        
        # ################# End of 'functionb2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'functionb2' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'functionb2'
        return stypy_return_type_21

    # Assigning a type to the variable 'functionb2' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'functionb2', functionb2)
    
    # Assigning a Call to a Name (line 26):
    
    # Call to functionb(...): (line 26)
    # Processing the call arguments (line 26)
    str_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 18), 'str', 'a')
    # Processing the call keyword arguments (line 26)
    kwargs_24 = {}
    # Getting the type of 'functionb' (line 26)
    functionb_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'functionb', False)
    # Calling functionb(args, kwargs) (line 26)
    functionb_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), functionb_22, *[str_23], **kwargs_24)
    
    # Assigning a type to the variable 'y' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'y', functionb_call_result_25)
    
    # Assigning a Call to a Name (line 27):
    
    # Call to functionb2(...): (line 27)
    # Processing the call arguments (line 27)
    
    # Call to range(...): (line 27)
    # Processing the call arguments (line 27)
    int_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'int')
    # Processing the call keyword arguments (line 27)
    kwargs_29 = {}
    # Getting the type of 'range' (line 27)
    range_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 'range', False)
    # Calling range(args, kwargs) (line 27)
    range_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 27, 19), range_27, *[int_28], **kwargs_29)
    
    # Processing the call keyword arguments (line 27)
    kwargs_31 = {}
    # Getting the type of 'functionb2' (line 27)
    functionb2_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'functionb2', False)
    # Calling functionb2(args, kwargs) (line 27)
    functionb2_call_result_32 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), functionb2_26, *[range_call_result_30], **kwargs_31)
    
    # Assigning a type to the variable 'y' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'y', functionb2_call_result_32)

    @norecursion
    def functionc(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'functionc'
        module_type_store = module_type_store.open_function_context('functionc', 30, 4, False)
        
        # Passed parameters checking function
        functionc.stypy_localization = localization
        functionc.stypy_type_of_self = None
        functionc.stypy_type_store = module_type_store
        functionc.stypy_function_name = 'functionc'
        functionc.stypy_param_names_list = ['x']
        functionc.stypy_varargs_param_name = None
        functionc.stypy_kwargs_param_name = 'kwargs'
        functionc.stypy_call_defaults = defaults
        functionc.stypy_call_varargs = varargs
        functionc.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'functionc', ['x'], None, 'kwargs', defaults, varargs, kwargs)

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

        
        # Getting the type of 'True' (line 31)
        True_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'True')
        # Testing the type of an if condition (line 31)
        if_condition_34 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 8), True_33)
        # Assigning a type to the variable 'if_condition_34' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'if_condition_34', if_condition_34)
        # SSA begins for if statement (line 31)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to int(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'x' (line 32)
        x_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'x', False)
        # Processing the call keyword arguments (line 32)
        kwargs_37 = {}
        # Getting the type of 'int' (line 32)
        int_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'int', False)
        # Calling int(args, kwargs) (line 32)
        int_call_result_38 = invoke(stypy.reporting.localization.Localization(__file__, 32, 19), int_35, *[x_36], **kwargs_37)
        
        # Assigning a type to the variable 'stypy_return_type' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'stypy_return_type', int_call_result_38)
        # SSA branch for the else part of an if statement (line 31)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining the type of the subscript
        int_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 26), 'int')
        # Getting the type of 'kwargs' (line 35)
        kwargs_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'kwargs')
        # Obtaining the member '__getitem__' of a type (line 35)
        getitem___41 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 19), kwargs_40, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 35)
        subscript_call_result_42 = invoke(stypy.reporting.localization.Localization(__file__, 35, 19), getitem___41, int_39)
        
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'stypy_return_type', subscript_call_result_42)
        # SSA join for if statement (line 31)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'functionc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'functionc' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'functionc'
        return stypy_return_type_43

    # Assigning a type to the variable 'functionc' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'functionc', functionc)
    
    # Assigning a Call to a Name (line 38):
    
    # Call to functionc(...): (line 38)
    # Processing the call arguments (line 38)
    int_45 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 18), 'int')
    # Processing the call keyword arguments (line 38)
    str_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 25), 'str', 'hi')
    keyword_47 = str_46
    kwargs_48 = {'val': keyword_47}
    # Getting the type of 'functionc' (line 38)
    functionc_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'functionc', False)
    # Calling functionc(args, kwargs) (line 38)
    functionc_call_result_49 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), functionc_44, *[int_45], **kwargs_48)
    
    # Assigning a type to the variable 'y' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'y', functionc_call_result_49)
    
    # Assigning a Call to a Name (line 40):
    
    # Call to thisdonotexist(...): (line 40)
    # Processing the call keyword arguments (line 40)
    kwargs_52 = {}
    # Getting the type of 'y' (line 40)
    y_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'y', False)
    # Obtaining the member 'thisdonotexist' of a type (line 40)
    thisdonotexist_51 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), y_50, 'thisdonotexist')
    # Calling thisdonotexist(args, kwargs) (line 40)
    thisdonotexist_call_result_53 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), thisdonotexist_51, *[], **kwargs_52)
    
    # Assigning a type to the variable 'y' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'y', thisdonotexist_call_result_53)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
