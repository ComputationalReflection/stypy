
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: import math
3: 
4: __doc__ = "No special __float__ method is defined"
5: 
6: if __name__ == '__main__':
7:     class DefinesFloat:
8:         def __float__(self):
9:             return 1.0
10: 
11: 
12:     class Empty:
13:         pass
14: 
15: 
16:     print math.pow(DefinesFloat(), 3)
17: 
18:     # Type error #
19:     print math.pow(Empty(), 3)
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import math' statement (line 2)
import math

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'math', math, module_type_store)


# Assigning a Str to a Name (line 4):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'str', 'No special __float__ method is defined')
# Assigning a type to the variable '__doc__' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'DefinesFloat' class

    class DefinesFloat:

        @norecursion
        def __float__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__float__'
            module_type_store = module_type_store.open_function_context('__float__', 8, 8, False)
            # Assigning a type to the variable 'self' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            DefinesFloat.__float__.__dict__.__setitem__('stypy_localization', localization)
            DefinesFloat.__float__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            DefinesFloat.__float__.__dict__.__setitem__('stypy_type_store', module_type_store)
            DefinesFloat.__float__.__dict__.__setitem__('stypy_function_name', 'DefinesFloat.__float__')
            DefinesFloat.__float__.__dict__.__setitem__('stypy_param_names_list', [])
            DefinesFloat.__float__.__dict__.__setitem__('stypy_varargs_param_name', None)
            DefinesFloat.__float__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            DefinesFloat.__float__.__dict__.__setitem__('stypy_call_defaults', defaults)
            DefinesFloat.__float__.__dict__.__setitem__('stypy_call_varargs', varargs)
            DefinesFloat.__float__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            DefinesFloat.__float__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DefinesFloat.__float__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__float__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__float__(...)' code ##################

            float_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'stypy_return_type', float_2)
            
            # ################# End of '__float__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__float__' in the type store
            # Getting the type of 'stypy_return_type' (line 8)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__float__'
            return stypy_return_type_3


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 7, 4, False)
            # Assigning a type to the variable 'self' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'DefinesFloat.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'DefinesFloat' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'DefinesFloat', DefinesFloat)
    # Declaration of the 'Empty' class

    class Empty:
        pass

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 12, 4, False)
            # Assigning a type to the variable 'self' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Empty' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'Empty', Empty)
    
    # Call to pow(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Call to DefinesFloat(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_7 = {}
    # Getting the type of 'DefinesFloat' (line 16)
    DefinesFloat_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 19), 'DefinesFloat', False)
    # Calling DefinesFloat(args, kwargs) (line 16)
    DefinesFloat_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 16, 19), DefinesFloat_6, *[], **kwargs_7)
    
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 35), 'int')
    # Processing the call keyword arguments (line 16)
    kwargs_10 = {}
    # Getting the type of 'math' (line 16)
    math_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'math', False)
    # Obtaining the member 'pow' of a type (line 16)
    pow_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 10), math_4, 'pow')
    # Calling pow(args, kwargs) (line 16)
    pow_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), pow_5, *[DefinesFloat_call_result_8, int_9], **kwargs_10)
    
    
    # Call to pow(...): (line 19)
    # Processing the call arguments (line 19)
    
    # Call to Empty(...): (line 19)
    # Processing the call keyword arguments (line 19)
    kwargs_15 = {}
    # Getting the type of 'Empty' (line 19)
    Empty_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'Empty', False)
    # Calling Empty(args, kwargs) (line 19)
    Empty_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 19, 19), Empty_14, *[], **kwargs_15)
    
    int_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 28), 'int')
    # Processing the call keyword arguments (line 19)
    kwargs_18 = {}
    # Getting the type of 'math' (line 19)
    math_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'math', False)
    # Obtaining the member 'pow' of a type (line 19)
    pow_13 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 10), math_12, 'pow')
    # Calling pow(args, kwargs) (line 19)
    pow_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), pow_13, *[Empty_call_result_16, int_17], **kwargs_18)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
