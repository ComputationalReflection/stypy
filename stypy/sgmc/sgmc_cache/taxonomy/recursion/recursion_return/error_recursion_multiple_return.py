
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Inferring the type returned by a recursive function when returning multiple possible types"
3: 
4: if __name__ == '__main__':
5:     call_count = 10
6: 
7: 
8:     def recursion_multiple_return():
9:         global call_count
10:         if call_count > 0:
11:             call_count -= 1
12:             return recursion_multiple_return()
13:         else:
14:             if call_count == 0:
15:                 return call_count
16:             else:
17:                 return str(call_count)
18: 
19: 
20:     # Type warning
21:     print recursion_multiple_return() + "str"
22: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Inferring the type returned by a recursive function when returning multiple possible types')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Num to a Name (line 5):
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'int')
    # Assigning a type to the variable 'call_count' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'call_count', int_2)

    @norecursion
    def recursion_multiple_return(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'recursion_multiple_return'
        module_type_store = module_type_store.open_function_context('recursion_multiple_return', 8, 4, False)
        
        # Passed parameters checking function
        recursion_multiple_return.stypy_localization = localization
        recursion_multiple_return.stypy_type_of_self = None
        recursion_multiple_return.stypy_type_store = module_type_store
        recursion_multiple_return.stypy_function_name = 'recursion_multiple_return'
        recursion_multiple_return.stypy_param_names_list = []
        recursion_multiple_return.stypy_varargs_param_name = None
        recursion_multiple_return.stypy_kwargs_param_name = None
        recursion_multiple_return.stypy_call_defaults = defaults
        recursion_multiple_return.stypy_call_varargs = varargs
        recursion_multiple_return.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'recursion_multiple_return', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'recursion_multiple_return', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'recursion_multiple_return(...)' code ##################

        # Marking variables as global (line 9)
        module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 9, 8), 'call_count')
        
        
        # Getting the type of 'call_count' (line 10)
        call_count_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'call_count')
        int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 24), 'int')
        # Applying the binary operator '>' (line 10)
        result_gt_5 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 11), '>', call_count_3, int_4)
        
        # Testing the type of an if condition (line 10)
        if_condition_6 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 10, 8), result_gt_5)
        # Assigning a type to the variable 'if_condition_6' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'if_condition_6', if_condition_6)
        # SSA begins for if statement (line 10)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'call_count' (line 11)
        call_count_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'call_count')
        int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'int')
        # Applying the binary operator '-=' (line 11)
        result_isub_9 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 12), '-=', call_count_7, int_8)
        # Assigning a type to the variable 'call_count' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'call_count', result_isub_9)
        
        
        # Call to recursion_multiple_return(...): (line 12)
        # Processing the call keyword arguments (line 12)
        kwargs_11 = {}
        # Getting the type of 'recursion_multiple_return' (line 12)
        recursion_multiple_return_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'recursion_multiple_return', False)
        # Calling recursion_multiple_return(args, kwargs) (line 12)
        recursion_multiple_return_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 12, 19), recursion_multiple_return_10, *[], **kwargs_11)
        
        # Assigning a type to the variable 'stypy_return_type' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'stypy_return_type', recursion_multiple_return_call_result_12)
        # SSA branch for the else part of an if statement (line 10)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'call_count' (line 14)
        call_count_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'call_count')
        int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 29), 'int')
        # Applying the binary operator '==' (line 14)
        result_eq_15 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 15), '==', call_count_13, int_14)
        
        # Testing the type of an if condition (line 14)
        if_condition_16 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 12), result_eq_15)
        # Assigning a type to the variable 'if_condition_16' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'if_condition_16', if_condition_16)
        # SSA begins for if statement (line 14)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'call_count' (line 15)
        call_count_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 23), 'call_count')
        # Assigning a type to the variable 'stypy_return_type' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 16), 'stypy_return_type', call_count_17)
        # SSA branch for the else part of an if statement (line 14)
        module_type_store.open_ssa_branch('else')
        
        # Call to str(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'call_count' (line 17)
        call_count_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 27), 'call_count', False)
        # Processing the call keyword arguments (line 17)
        kwargs_20 = {}
        # Getting the type of 'str' (line 17)
        str_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 23), 'str', False)
        # Calling str(args, kwargs) (line 17)
        str_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 17, 23), str_18, *[call_count_19], **kwargs_20)
        
        # Assigning a type to the variable 'stypy_return_type' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), 'stypy_return_type', str_call_result_21)
        # SSA join for if statement (line 14)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 10)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'recursion_multiple_return(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'recursion_multiple_return' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_22)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'recursion_multiple_return'
        return stypy_return_type_22

    # Assigning a type to the variable 'recursion_multiple_return' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'recursion_multiple_return', recursion_multiple_return)
    
    # Call to recursion_multiple_return(...): (line 21)
    # Processing the call keyword arguments (line 21)
    kwargs_24 = {}
    # Getting the type of 'recursion_multiple_return' (line 21)
    recursion_multiple_return_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'recursion_multiple_return', False)
    # Calling recursion_multiple_return(args, kwargs) (line 21)
    recursion_multiple_return_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), recursion_multiple_return_23, *[], **kwargs_24)
    
    str_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 40), 'str', 'str')
    # Applying the binary operator '+' (line 21)
    result_add_27 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 10), '+', recursion_multiple_return_call_result_25, str_26)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
