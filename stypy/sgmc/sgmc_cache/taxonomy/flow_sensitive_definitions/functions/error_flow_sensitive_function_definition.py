
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Flow-sensitive function definition"
3: 
4: if __name__ == '__main__':
5:     if True:
6:         def funcIf():
7:             return 0
8: 
9: 
10:         c = funcIf
11:     else:
12:         def funcElse():
13:             return "str"
14: 
15: 
16:         c = funcElse
17: 
18:     # Type warning
19:     print c() + "str"
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Flow-sensitive function definition')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Getting the type of 'True' (line 5)
    True_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 7), 'True')
    # Testing the type of an if condition (line 5)
    if_condition_3 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 5, 4), True_2)
    # Assigning a type to the variable 'if_condition_3' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'if_condition_3', if_condition_3)
    # SSA begins for if statement (line 5)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

    @norecursion
    def funcIf(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'funcIf'
        module_type_store = module_type_store.open_function_context('funcIf', 6, 8, False)
        
        # Passed parameters checking function
        funcIf.stypy_localization = localization
        funcIf.stypy_type_of_self = None
        funcIf.stypy_type_store = module_type_store
        funcIf.stypy_function_name = 'funcIf'
        funcIf.stypy_param_names_list = []
        funcIf.stypy_varargs_param_name = None
        funcIf.stypy_kwargs_param_name = None
        funcIf.stypy_call_defaults = defaults
        funcIf.stypy_call_varargs = varargs
        funcIf.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'funcIf', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'funcIf', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'funcIf(...)' code ##################

        int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'stypy_return_type', int_4)
        
        # ################# End of 'funcIf(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'funcIf' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'funcIf'
        return stypy_return_type_5

    # Assigning a type to the variable 'funcIf' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'funcIf', funcIf)
    
    # Assigning a Name to a Name (line 10):
    # Getting the type of 'funcIf' (line 10)
    funcIf_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'funcIf')
    # Assigning a type to the variable 'c' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'c', funcIf_6)
    # SSA branch for the else part of an if statement (line 5)
    module_type_store.open_ssa_branch('else')

    @norecursion
    def funcElse(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'funcElse'
        module_type_store = module_type_store.open_function_context('funcElse', 12, 8, False)
        
        # Passed parameters checking function
        funcElse.stypy_localization = localization
        funcElse.stypy_type_of_self = None
        funcElse.stypy_type_store = module_type_store
        funcElse.stypy_function_name = 'funcElse'
        funcElse.stypy_param_names_list = []
        funcElse.stypy_varargs_param_name = None
        funcElse.stypy_kwargs_param_name = None
        funcElse.stypy_call_defaults = defaults
        funcElse.stypy_call_varargs = varargs
        funcElse.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'funcElse', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'funcElse', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'funcElse(...)' code ##################

        str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 19), 'str', 'str')
        # Assigning a type to the variable 'stypy_return_type' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'stypy_return_type', str_7)
        
        # ################# End of 'funcElse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'funcElse' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'funcElse'
        return stypy_return_type_8

    # Assigning a type to the variable 'funcElse' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'funcElse', funcElse)
    
    # Assigning a Name to a Name (line 16):
    # Getting the type of 'funcElse' (line 16)
    funcElse_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'funcElse')
    # Assigning a type to the variable 'c' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'c', funcElse_9)
    # SSA join for if statement (line 5)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to c(...): (line 19)
    # Processing the call keyword arguments (line 19)
    kwargs_11 = {}
    # Getting the type of 'c' (line 19)
    c_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'c', False)
    # Calling c(args, kwargs) (line 19)
    c_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), c_10, *[], **kwargs_11)
    
    str_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 16), 'str', 'str')
    # Applying the binary operator '+' (line 19)
    result_add_14 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 10), '+', c_call_result_12, str_13)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
