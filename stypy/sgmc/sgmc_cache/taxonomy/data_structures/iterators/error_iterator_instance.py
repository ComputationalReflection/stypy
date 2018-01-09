
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Use an iterator type instead of an iterator instance"
4: 
5: if __name__ == '__main__':
6:     it_list = type(iter(range(5)))
7: 
8:     # Type error
9:     for i in it_list:
10:         print i
11: 
12: 
13:     def fun1(l):
14:         # Type error
15:         return "aaa" + l.next()
16: 
17: 
18:     def fun1b(l):
19:         # Type error
20:         return "aaa" + l.next()
21: 
22: 
23:     print fun1(it_list)
24:     print fun1b(iter)
25: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Use an iterator type instead of an iterator instance')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 6):
    
    # Call to type(...): (line 6)
    # Processing the call arguments (line 6)
    
    # Call to iter(...): (line 6)
    # Processing the call arguments (line 6)
    
    # Call to range(...): (line 6)
    # Processing the call arguments (line 6)
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 30), 'int')
    # Processing the call keyword arguments (line 6)
    kwargs_6 = {}
    # Getting the type of 'range' (line 6)
    range_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 24), 'range', False)
    # Calling range(args, kwargs) (line 6)
    range_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 6, 24), range_4, *[int_5], **kwargs_6)
    
    # Processing the call keyword arguments (line 6)
    kwargs_8 = {}
    # Getting the type of 'iter' (line 6)
    iter_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 19), 'iter', False)
    # Calling iter(args, kwargs) (line 6)
    iter_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 6, 19), iter_3, *[range_call_result_7], **kwargs_8)
    
    # Processing the call keyword arguments (line 6)
    kwargs_10 = {}
    # Getting the type of 'type' (line 6)
    type_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 14), 'type', False)
    # Calling type(args, kwargs) (line 6)
    type_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 6, 14), type_2, *[iter_call_result_9], **kwargs_10)
    
    # Assigning a type to the variable 'it_list' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'it_list', type_call_result_11)
    
    # Getting the type of 'it_list' (line 9)
    it_list_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 13), 'it_list')
    # Testing the type of a for loop iterable (line 9)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 9, 4), it_list_12)
    # Getting the type of the for loop variable (line 9)
    for_loop_var_13 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 9, 4), it_list_12)
    # Assigning a type to the variable 'i' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'i', for_loop_var_13)
    # SSA begins for a for statement (line 9)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    # Getting the type of 'i' (line 10)
    i_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 14), 'i')
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def fun1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun1'
        module_type_store = module_type_store.open_function_context('fun1', 13, 4, False)
        
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

        str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'aaa')
        
        # Call to next(...): (line 15)
        # Processing the call keyword arguments (line 15)
        kwargs_18 = {}
        # Getting the type of 'l' (line 15)
        l_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 23), 'l', False)
        # Obtaining the member 'next' of a type (line 15)
        next_17 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 23), l_16, 'next')
        # Calling next(args, kwargs) (line 15)
        next_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 15, 23), next_17, *[], **kwargs_18)
        
        # Applying the binary operator '+' (line 15)
        result_add_20 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 15), '+', str_15, next_call_result_19)
        
        # Assigning a type to the variable 'stypy_return_type' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', result_add_20)
        
        # ################# End of 'fun1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun1' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun1'
        return stypy_return_type_21

    # Assigning a type to the variable 'fun1' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'fun1', fun1)

    @norecursion
    def fun1b(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun1b'
        module_type_store = module_type_store.open_function_context('fun1b', 18, 4, False)
        
        # Passed parameters checking function
        fun1b.stypy_localization = localization
        fun1b.stypy_type_of_self = None
        fun1b.stypy_type_store = module_type_store
        fun1b.stypy_function_name = 'fun1b'
        fun1b.stypy_param_names_list = ['l']
        fun1b.stypy_varargs_param_name = None
        fun1b.stypy_kwargs_param_name = None
        fun1b.stypy_call_defaults = defaults
        fun1b.stypy_call_varargs = varargs
        fun1b.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'fun1b', ['l'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fun1b', localization, ['l'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fun1b(...)' code ##################

        str_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'str', 'aaa')
        
        # Call to next(...): (line 20)
        # Processing the call keyword arguments (line 20)
        kwargs_25 = {}
        # Getting the type of 'l' (line 20)
        l_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 23), 'l', False)
        # Obtaining the member 'next' of a type (line 20)
        next_24 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 23), l_23, 'next')
        # Calling next(args, kwargs) (line 20)
        next_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 20, 23), next_24, *[], **kwargs_25)
        
        # Applying the binary operator '+' (line 20)
        result_add_27 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 15), '+', str_22, next_call_result_26)
        
        # Assigning a type to the variable 'stypy_return_type' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'stypy_return_type', result_add_27)
        
        # ################# End of 'fun1b(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun1b' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun1b'
        return stypy_return_type_28

    # Assigning a type to the variable 'fun1b' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'fun1b', fun1b)
    
    # Call to fun1(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'it_list' (line 23)
    it_list_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'it_list', False)
    # Processing the call keyword arguments (line 23)
    kwargs_31 = {}
    # Getting the type of 'fun1' (line 23)
    fun1_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'fun1', False)
    # Calling fun1(args, kwargs) (line 23)
    fun1_call_result_32 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), fun1_29, *[it_list_30], **kwargs_31)
    
    
    # Call to fun1b(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'iter' (line 24)
    iter_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'iter', False)
    # Processing the call keyword arguments (line 24)
    kwargs_35 = {}
    # Getting the type of 'fun1b' (line 24)
    fun1b_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 10), 'fun1b', False)
    # Calling fun1b(args, kwargs) (line 24)
    fun1b_call_result_36 = invoke(stypy.reporting.localization.Localization(__file__, 24, 10), fun1b_33, *[iter_34], **kwargs_35)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
