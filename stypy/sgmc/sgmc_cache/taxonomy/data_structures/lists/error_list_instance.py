
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Use an tuple type instead of a tuple instance"
4: 
5: if __name__ == '__main__':
6:     # Type error
7:     for i in tuple:
8:         print i
9: 
10: 
11:     def fun1(l):
12:         # Type error
13:         return "aaa" + l[0]
14: 
15: 
16:     print fun1(tuple)
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Use an tuple type instead of a tuple instance')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Getting the type of 'tuple' (line 7)
    tuple_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 13), 'tuple')
    # Testing the type of a for loop iterable (line 7)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 7, 4), tuple_2)
    # Getting the type of the for loop variable (line 7)
    for_loop_var_3 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 7, 4), tuple_2)
    # Assigning a type to the variable 'i' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'i', for_loop_var_3)
    # SSA begins for a for statement (line 7)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    # Getting the type of 'i' (line 8)
    i_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 14), 'i')
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def fun1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'fun1'
        module_type_store = module_type_store.open_function_context('fun1', 11, 4, False)
        
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

        str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'aaa')
        
        # Obtaining the type of the subscript
        int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'int')
        # Getting the type of 'l' (line 13)
        l_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 23), 'l')
        # Obtaining the member '__getitem__' of a type (line 13)
        getitem___8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 23), l_7, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 13)
        subscript_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 13, 23), getitem___8, int_6)
        
        # Applying the binary operator '+' (line 13)
        result_add_10 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 15), '+', str_5, subscript_call_result_9)
        
        # Assigning a type to the variable 'stypy_return_type' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', result_add_10)
        
        # ################# End of 'fun1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fun1' in the type store
        # Getting the type of 'stypy_return_type' (line 11)
        stypy_return_type_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fun1'
        return stypy_return_type_11

    # Assigning a type to the variable 'fun1' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'fun1', fun1)
    
    # Call to fun1(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'tuple' (line 16)
    tuple_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'tuple', False)
    # Processing the call keyword arguments (line 16)
    kwargs_14 = {}
    # Getting the type of 'fun1' (line 16)
    fun1_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'fun1', False)
    # Calling fun1(args, kwargs) (line 16)
    fun1_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), fun1_12, *[tuple_13], **kwargs_14)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
