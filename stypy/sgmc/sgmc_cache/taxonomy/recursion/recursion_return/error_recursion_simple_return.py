
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Inferring the type returned by a recursive function"
3: 
4: if __name__ == '__main__':
5:     call_count = 10
6: 
7: 
8:     def recursion():
9:         global call_count
10:         if call_count > 0:
11:             call_count -= 1
12:             return recursion()
13:         else:
14:             return call_count
15: 
16: 
17:     # Type error
18:     print recursion() + "str"
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Inferring the type returned by a recursive function')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Num to a Name (line 5):
    int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'int')
    # Assigning a type to the variable 'call_count' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'call_count', int_2)

    @norecursion
    def recursion(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'recursion'
        module_type_store = module_type_store.open_function_context('recursion', 8, 4, False)
        
        # Passed parameters checking function
        recursion.stypy_localization = localization
        recursion.stypy_type_of_self = None
        recursion.stypy_type_store = module_type_store
        recursion.stypy_function_name = 'recursion'
        recursion.stypy_param_names_list = []
        recursion.stypy_varargs_param_name = None
        recursion.stypy_kwargs_param_name = None
        recursion.stypy_call_defaults = defaults
        recursion.stypy_call_varargs = varargs
        recursion.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'recursion', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'recursion', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'recursion(...)' code ##################

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
        
        
        # Call to recursion(...): (line 12)
        # Processing the call keyword arguments (line 12)
        kwargs_11 = {}
        # Getting the type of 'recursion' (line 12)
        recursion_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'recursion', False)
        # Calling recursion(args, kwargs) (line 12)
        recursion_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 12, 19), recursion_10, *[], **kwargs_11)
        
        # Assigning a type to the variable 'stypy_return_type' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'stypy_return_type', recursion_call_result_12)
        # SSA branch for the else part of an if statement (line 10)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'call_count' (line 14)
        call_count_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'call_count')
        # Assigning a type to the variable 'stypy_return_type' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'stypy_return_type', call_count_13)
        # SSA join for if statement (line 10)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'recursion(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'recursion' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'recursion'
        return stypy_return_type_14

    # Assigning a type to the variable 'recursion' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'recursion', recursion)
    
    # Call to recursion(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_16 = {}
    # Getting the type of 'recursion' (line 18)
    recursion_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'recursion', False)
    # Calling recursion(args, kwargs) (line 18)
    recursion_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), recursion_15, *[], **kwargs_16)
    
    str_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 24), 'str', 'str')
    # Applying the binary operator '+' (line 18)
    result_add_19 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 10), '+', recursion_call_result_17, str_18)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
