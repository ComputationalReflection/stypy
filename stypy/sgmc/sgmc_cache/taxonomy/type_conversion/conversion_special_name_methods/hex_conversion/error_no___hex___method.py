
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "No special __hex__ method is defined"
4: 
5: if __name__ == '__main__':
6:     class DefinesMethod:
7:         def __hex__(self):
8:             return "1"
9: 
10: 
11:     class Empty:
12:         pass
13: 
14: 
15:     print hex(DefinesMethod())
16: 
17:     # Type error #
18:     print hex(Empty())
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'No special __hex__ method is defined')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'DefinesMethod' class

    class DefinesMethod:

        @norecursion
        def __hex__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__hex__'
            module_type_store = module_type_store.open_function_context('__hex__', 7, 8, False)
            # Assigning a type to the variable 'self' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            DefinesMethod.__hex__.__dict__.__setitem__('stypy_localization', localization)
            DefinesMethod.__hex__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            DefinesMethod.__hex__.__dict__.__setitem__('stypy_type_store', module_type_store)
            DefinesMethod.__hex__.__dict__.__setitem__('stypy_function_name', 'DefinesMethod.__hex__')
            DefinesMethod.__hex__.__dict__.__setitem__('stypy_param_names_list', [])
            DefinesMethod.__hex__.__dict__.__setitem__('stypy_varargs_param_name', None)
            DefinesMethod.__hex__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            DefinesMethod.__hex__.__dict__.__setitem__('stypy_call_defaults', defaults)
            DefinesMethod.__hex__.__dict__.__setitem__('stypy_call_varargs', varargs)
            DefinesMethod.__hex__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            DefinesMethod.__hex__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DefinesMethod.__hex__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__hex__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__hex__(...)' code ##################

            str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 19), 'str', '1')
            # Assigning a type to the variable 'stypy_return_type' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'stypy_return_type', str_2)
            
            # ################# End of '__hex__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__hex__' in the type store
            # Getting the type of 'stypy_return_type' (line 7)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__hex__'
            return stypy_return_type_3


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 6, 4, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DefinesMethod.__init__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            pass
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()

    
    # Assigning a type to the variable 'DefinesMethod' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'DefinesMethod', DefinesMethod)
    # Declaration of the 'Empty' class

    class Empty:
        pass

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 11, 4, False)
            # Assigning a type to the variable 'self' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Empty.__init__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            pass
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()

    
    # Assigning a type to the variable 'Empty' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'Empty', Empty)
    
    # Call to hex(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to DefinesMethod(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_6 = {}
    # Getting the type of 'DefinesMethod' (line 15)
    DefinesMethod_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 14), 'DefinesMethod', False)
    # Calling DefinesMethod(args, kwargs) (line 15)
    DefinesMethod_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 15, 14), DefinesMethod_5, *[], **kwargs_6)
    
    # Processing the call keyword arguments (line 15)
    kwargs_8 = {}
    # Getting the type of 'hex' (line 15)
    hex_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'hex', False)
    # Calling hex(args, kwargs) (line 15)
    hex_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), hex_4, *[DefinesMethod_call_result_7], **kwargs_8)
    
    
    # Call to hex(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Call to Empty(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_12 = {}
    # Getting the type of 'Empty' (line 18)
    Empty_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'Empty', False)
    # Calling Empty(args, kwargs) (line 18)
    Empty_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 18, 14), Empty_11, *[], **kwargs_12)
    
    # Processing the call keyword arguments (line 18)
    kwargs_14 = {}
    # Getting the type of 'hex' (line 18)
    hex_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'hex', False)
    # Calling hex(args, kwargs) (line 18)
    hex_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), hex_10, *[Empty_call_result_13], **kwargs_14)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
