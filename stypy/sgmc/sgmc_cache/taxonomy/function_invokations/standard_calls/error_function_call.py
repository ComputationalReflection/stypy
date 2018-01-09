
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Incorrect types passed on function calls"
4: 
5: if __name__ == '__main__':
6:     def functionb(x):
7:         # Type error
8:         return x / 2
9: 
10: 
11:     def functionb2(x):
12:         # Type error
13:         return x / 2
14: 
15: 
16:     def functionb3(x):
17:         return x / 2
18: 
19: 
20:     y = functionb("a")
21:     y = functionb2(range(5))
22:     y = functionb3(4)
23: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Incorrect types passed on function calls')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

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

        # Getting the type of 'x' (line 8)
        x_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 15), 'x')
        int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 19), 'int')
        # Applying the binary operator 'div' (line 8)
        result_div_4 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 15), 'div', x_2, int_3)
        
        # Assigning a type to the variable 'stypy_return_type' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type', result_div_4)
        
        # ################# End of 'functionb(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'functionb' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'functionb'
        return stypy_return_type_5

    # Assigning a type to the variable 'functionb' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'functionb', functionb)

    @norecursion
    def functionb2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'functionb2'
        module_type_store = module_type_store.open_function_context('functionb2', 11, 4, False)
        
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

        # Getting the type of 'x' (line 13)
        x_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 15), 'x')
        int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 19), 'int')
        # Applying the binary operator 'div' (line 13)
        result_div_8 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 15), 'div', x_6, int_7)
        
        # Assigning a type to the variable 'stypy_return_type' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', result_div_8)
        
        # ################# End of 'functionb2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'functionb2' in the type store
        # Getting the type of 'stypy_return_type' (line 11)
        stypy_return_type_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'functionb2'
        return stypy_return_type_9

    # Assigning a type to the variable 'functionb2' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'functionb2', functionb2)

    @norecursion
    def functionb3(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'functionb3'
        module_type_store = module_type_store.open_function_context('functionb3', 16, 4, False)
        
        # Passed parameters checking function
        functionb3.stypy_localization = localization
        functionb3.stypy_type_of_self = None
        functionb3.stypy_type_store = module_type_store
        functionb3.stypy_function_name = 'functionb3'
        functionb3.stypy_param_names_list = ['x']
        functionb3.stypy_varargs_param_name = None
        functionb3.stypy_kwargs_param_name = None
        functionb3.stypy_call_defaults = defaults
        functionb3.stypy_call_varargs = varargs
        functionb3.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'functionb3', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'functionb3', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'functionb3(...)' code ##################

        # Getting the type of 'x' (line 17)
        x_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'x')
        int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 19), 'int')
        # Applying the binary operator 'div' (line 17)
        result_div_12 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 15), 'div', x_10, int_11)
        
        # Assigning a type to the variable 'stypy_return_type' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'stypy_return_type', result_div_12)
        
        # ################# End of 'functionb3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'functionb3' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'functionb3'
        return stypy_return_type_13

    # Assigning a type to the variable 'functionb3' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'functionb3', functionb3)
    
    # Assigning a Call to a Name (line 20):
    
    # Call to functionb(...): (line 20)
    # Processing the call arguments (line 20)
    str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 18), 'str', 'a')
    # Processing the call keyword arguments (line 20)
    kwargs_16 = {}
    # Getting the type of 'functionb' (line 20)
    functionb_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'functionb', False)
    # Calling functionb(args, kwargs) (line 20)
    functionb_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), functionb_14, *[str_15], **kwargs_16)
    
    # Assigning a type to the variable 'y' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'y', functionb_call_result_17)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to functionb2(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Call to range(...): (line 21)
    # Processing the call arguments (line 21)
    int_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'int')
    # Processing the call keyword arguments (line 21)
    kwargs_21 = {}
    # Getting the type of 'range' (line 21)
    range_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'range', False)
    # Calling range(args, kwargs) (line 21)
    range_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 21, 19), range_19, *[int_20], **kwargs_21)
    
    # Processing the call keyword arguments (line 21)
    kwargs_23 = {}
    # Getting the type of 'functionb2' (line 21)
    functionb2_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'functionb2', False)
    # Calling functionb2(args, kwargs) (line 21)
    functionb2_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), functionb2_18, *[range_call_result_22], **kwargs_23)
    
    # Assigning a type to the variable 'y' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'y', functionb2_call_result_24)
    
    # Assigning a Call to a Name (line 22):
    
    # Call to functionb3(...): (line 22)
    # Processing the call arguments (line 22)
    int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 19), 'int')
    # Processing the call keyword arguments (line 22)
    kwargs_27 = {}
    # Getting the type of 'functionb3' (line 22)
    functionb3_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'functionb3', False)
    # Calling functionb3(args, kwargs) (line 22)
    functionb3_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), functionb3_25, *[int_26], **kwargs_27)
    
    # Assigning a type to the variable 'y' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'y', functionb3_call_result_28)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
