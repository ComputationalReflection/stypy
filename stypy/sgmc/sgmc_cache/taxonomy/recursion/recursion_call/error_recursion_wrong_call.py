
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Wrong recursive call"
3: 
4: if __name__ == '__main__':
5:     call_count = 10
6: 
7: 
8:     def recursion_wrong_call():
9:         global call_count
10:         if call_count > 0:
11:             call_count -= 1
12:             # Type warning
13:             return recursion_wrong_call(call_count)
14:         else:
15:             return call_count
16: 
17: 
18:     print recursion_wrong_call()
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Wrong recursive call')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Num to a Name (line 5):
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'int')
    # Assigning a type to the variable 'call_count' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'call_count', int_2)

    @norecursion
    def recursion_wrong_call(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'recursion_wrong_call'
        module_type_store = module_type_store.open_function_context('recursion_wrong_call', 8, 4, False)
        
        # Passed parameters checking function
        recursion_wrong_call.stypy_localization = localization
        recursion_wrong_call.stypy_type_of_self = None
        recursion_wrong_call.stypy_type_store = module_type_store
        recursion_wrong_call.stypy_function_name = 'recursion_wrong_call'
        recursion_wrong_call.stypy_param_names_list = []
        recursion_wrong_call.stypy_varargs_param_name = None
        recursion_wrong_call.stypy_kwargs_param_name = None
        recursion_wrong_call.stypy_call_defaults = defaults
        recursion_wrong_call.stypy_call_varargs = varargs
        recursion_wrong_call.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'recursion_wrong_call', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'recursion_wrong_call', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'recursion_wrong_call(...)' code ##################

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
        
        
        # Call to recursion_wrong_call(...): (line 13)
        # Processing the call arguments (line 13)
        # Getting the type of 'call_count' (line 13)
        call_count_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 40), 'call_count', False)
        # Processing the call keyword arguments (line 13)
        kwargs_12 = {}
        # Getting the type of 'recursion_wrong_call' (line 13)
        recursion_wrong_call_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 19), 'recursion_wrong_call', False)
        # Calling recursion_wrong_call(args, kwargs) (line 13)
        recursion_wrong_call_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 13, 19), recursion_wrong_call_10, *[call_count_11], **kwargs_12)
        
        # Assigning a type to the variable 'stypy_return_type' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'stypy_return_type', recursion_wrong_call_call_result_13)
        # SSA branch for the else part of an if statement (line 10)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'call_count' (line 15)
        call_count_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 19), 'call_count')
        # Assigning a type to the variable 'stypy_return_type' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'stypy_return_type', call_count_14)
        # SSA join for if statement (line 10)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'recursion_wrong_call(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'recursion_wrong_call' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'recursion_wrong_call'
        return stypy_return_type_15

    # Assigning a type to the variable 'recursion_wrong_call' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'recursion_wrong_call', recursion_wrong_call)
    
    # Call to recursion_wrong_call(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_17 = {}
    # Getting the type of 'recursion_wrong_call' (line 18)
    recursion_wrong_call_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'recursion_wrong_call', False)
    # Calling recursion_wrong_call(args, kwargs) (line 18)
    recursion_wrong_call_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), recursion_wrong_call_16, *[], **kwargs_17)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
