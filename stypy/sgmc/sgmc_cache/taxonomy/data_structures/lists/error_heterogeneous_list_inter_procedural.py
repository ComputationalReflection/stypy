
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Inferring the type of inter-procedural heterogeneous lists"
4: 
5: if __name__ == '__main__':
6: 
7:     def create_list():
8:         ret = []
9:         for i in range(10):
10:             if i % 2 == 0:
11:                 ret.append("str")
12:             else:
13:                 ret.append(i)
14:         return ret
15: 
16: 
17:     l = create_list()
18: 
19:     for elem in l:
20:         # Type warning
21:         print "|" + elem + "|"
22: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Inferring the type of inter-procedural heterogeneous lists')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def create_list(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_list'
        module_type_store = module_type_store.open_function_context('create_list', 7, 4, False)
        
        # Passed parameters checking function
        create_list.stypy_localization = localization
        create_list.stypy_type_of_self = None
        create_list.stypy_type_store = module_type_store
        create_list.stypy_function_name = 'create_list'
        create_list.stypy_param_names_list = []
        create_list.stypy_varargs_param_name = None
        create_list.stypy_kwargs_param_name = None
        create_list.stypy_call_defaults = defaults
        create_list.stypy_call_varargs = varargs
        create_list.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'create_list', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_list', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_list(...)' code ##################

        
        # Assigning a List to a Name (line 8):
        
        # Obtaining an instance of the builtin type 'list' (line 8)
        list_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 8)
        
        # Assigning a type to the variable 'ret' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'ret', list_2)
        
        
        # Call to range(...): (line 9)
        # Processing the call arguments (line 9)
        int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 23), 'int')
        # Processing the call keyword arguments (line 9)
        kwargs_5 = {}
        # Getting the type of 'range' (line 9)
        range_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 17), 'range', False)
        # Calling range(args, kwargs) (line 9)
        range_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 9, 17), range_3, *[int_4], **kwargs_5)
        
        # Testing the type of a for loop iterable (line 9)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 9, 8), range_call_result_6)
        # Getting the type of the for loop variable (line 9)
        for_loop_var_7 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 9, 8), range_call_result_6)
        # Assigning a type to the variable 'i' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'i', for_loop_var_7)
        # SSA begins for a for statement (line 9)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'i' (line 10)
        i_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 15), 'i')
        int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 19), 'int')
        # Applying the binary operator '%' (line 10)
        result_mod_10 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 15), '%', i_8, int_9)
        
        int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 24), 'int')
        # Applying the binary operator '==' (line 10)
        result_eq_12 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 15), '==', result_mod_10, int_11)
        
        # Testing the type of an if condition (line 10)
        if_condition_13 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 10, 12), result_eq_12)
        # Assigning a type to the variable 'if_condition_13' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'if_condition_13', if_condition_13)
        # SSA begins for if statement (line 10)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 11)
        # Processing the call arguments (line 11)
        str_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 27), 'str', 'str')
        # Processing the call keyword arguments (line 11)
        kwargs_17 = {}
        # Getting the type of 'ret' (line 11)
        ret_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'ret', False)
        # Obtaining the member 'append' of a type (line 11)
        append_15 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 16), ret_14, 'append')
        # Calling append(args, kwargs) (line 11)
        append_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 11, 16), append_15, *[str_16], **kwargs_17)
        
        # SSA branch for the else part of an if statement (line 10)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 13)
        # Processing the call arguments (line 13)
        # Getting the type of 'i' (line 13)
        i_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 27), 'i', False)
        # Processing the call keyword arguments (line 13)
        kwargs_22 = {}
        # Getting the type of 'ret' (line 13)
        ret_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'ret', False)
        # Obtaining the member 'append' of a type (line 13)
        append_20 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 16), ret_19, 'append')
        # Calling append(args, kwargs) (line 13)
        append_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 13, 16), append_20, *[i_21], **kwargs_22)
        
        # SSA join for if statement (line 10)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'ret' (line 14)
        ret_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'stypy_return_type', ret_24)
        
        # ################# End of 'create_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_list' in the type store
        # Getting the type of 'stypy_return_type' (line 7)
        stypy_return_type_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_list'
        return stypy_return_type_25

    # Assigning a type to the variable 'create_list' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'create_list', create_list)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to create_list(...): (line 17)
    # Processing the call keyword arguments (line 17)
    kwargs_27 = {}
    # Getting the type of 'create_list' (line 17)
    create_list_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'create_list', False)
    # Calling create_list(args, kwargs) (line 17)
    create_list_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), create_list_26, *[], **kwargs_27)
    
    # Assigning a type to the variable 'l' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'l', create_list_call_result_28)
    
    # Getting the type of 'l' (line 19)
    l_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'l')
    # Testing the type of a for loop iterable (line 19)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 19, 4), l_29)
    # Getting the type of the for loop variable (line 19)
    for_loop_var_30 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 19, 4), l_29)
    # Assigning a type to the variable 'elem' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'elem', for_loop_var_30)
    # SSA begins for a for statement (line 19)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    str_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 14), 'str', '|')
    # Getting the type of 'elem' (line 21)
    elem_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'elem')
    # Applying the binary operator '+' (line 21)
    result_add_33 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 14), '+', str_31, elem_32)
    
    str_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 27), 'str', '|')
    # Applying the binary operator '+' (line 21)
    result_add_35 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 25), '+', result_add_33, str_34)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
