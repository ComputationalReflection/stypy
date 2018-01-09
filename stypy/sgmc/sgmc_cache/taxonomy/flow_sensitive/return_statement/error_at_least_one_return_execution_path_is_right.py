
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "At least one (but not all) execution paths has an execution flow free of type errors"
3: 
4: if __name__ == '__main__':
5: 
6:     def get_str():
7:         if True:
8:             return "hi"
9:         else:
10:             return 2
11: 
12: 
13:     # Type warning
14:     a = 4 + get_str()
15: 

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
    def get_str(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_str'
        module_type_store = module_type_store.open_function_context('get_str', 6, 4, False)
        
        # Passed parameters checking function
        get_str.stypy_localization = localization
        get_str.stypy_type_of_self = None
        get_str.stypy_type_store = module_type_store
        get_str.stypy_function_name = 'get_str'
        get_str.stypy_param_names_list = []
        get_str.stypy_varargs_param_name = None
        get_str.stypy_kwargs_param_name = None
        get_str.stypy_call_defaults = defaults
        get_str.stypy_call_varargs = varargs
        get_str.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'get_str', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_str', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_str(...)' code ##################

        
        # Getting the type of 'True' (line 7)
        True_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'True')
        # Testing the type of an if condition (line 7)
        if_condition_3 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 7, 8), True_2)
        # Assigning a type to the variable 'if_condition_3' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'if_condition_3', if_condition_3)
        # SSA begins for if statement (line 7)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 19), 'str', 'hi')
        # Assigning a type to the variable 'stypy_return_type' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'stypy_return_type', str_4)
        # SSA branch for the else part of an if statement (line 7)
        module_type_store.open_ssa_branch('else')
        int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'stypy_return_type', int_5)
        # SSA join for if statement (line 7)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_str(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_str' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_str'
        return stypy_return_type_6

    # Assigning a type to the variable 'get_str' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'get_str', get_str)
    
    # Assigning a BinOp to a Name (line 14):
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 8), 'int')
    
    # Call to get_str(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_9 = {}
    # Getting the type of 'get_str' (line 14)
    get_str_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'get_str', False)
    # Calling get_str(args, kwargs) (line 14)
    get_str_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 14, 12), get_str_8, *[], **kwargs_9)
    
    # Applying the binary operator '+' (line 14)
    result_add_11 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 8), '+', int_7, get_str_call_result_10)
    
    # Assigning a type to the variable 'a' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'a', result_add_11)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
