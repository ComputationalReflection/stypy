
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: import math
3: 
4: __doc__ = "No __trunc__ method is present"
5: 
6: if __name__ == '__main__':
7:     class Correct:
8:         def __trunc__(self):
9:             return 1
10: 
11: 
12:     print math.trunc(Correct())
13: 
14: 
15:     class Sample:
16:         pass
17: 
18: 
19:     # Type error
20:     print math.trunc(Sample())
21: 

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
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'str', 'No __trunc__ method is present')
# Assigning a type to the variable '__doc__' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Correct' class

    class Correct:

        @norecursion
        def __trunc__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__trunc__'
            module_type_store = module_type_store.open_function_context('__trunc__', 8, 8, False)
            # Assigning a type to the variable 'self' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Correct.__trunc__.__dict__.__setitem__('stypy_localization', localization)
            Correct.__trunc__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Correct.__trunc__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Correct.__trunc__.__dict__.__setitem__('stypy_function_name', 'Correct.__trunc__')
            Correct.__trunc__.__dict__.__setitem__('stypy_param_names_list', [])
            Correct.__trunc__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Correct.__trunc__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Correct.__trunc__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Correct.__trunc__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Correct.__trunc__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Correct.__trunc__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Correct.__trunc__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__trunc__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__trunc__(...)' code ##################

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'stypy_return_type', int_2)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 8)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__trunc__'
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
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Correct.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Correct' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'Correct', Correct)
    
    # Call to trunc(...): (line 12)
    # Processing the call arguments (line 12)
    
    # Call to Correct(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_7 = {}
    # Getting the type of 'Correct' (line 12)
    Correct_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 21), 'Correct', False)
    # Calling Correct(args, kwargs) (line 12)
    Correct_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 12, 21), Correct_6, *[], **kwargs_7)
    
    # Processing the call keyword arguments (line 12)
    kwargs_9 = {}
    # Getting the type of 'math' (line 12)
    math_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 10), 'math', False)
    # Obtaining the member 'trunc' of a type (line 12)
    trunc_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 10), math_4, 'trunc')
    # Calling trunc(args, kwargs) (line 12)
    trunc_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 12, 10), trunc_5, *[Correct_call_result_8], **kwargs_9)
    
    # Declaration of the 'Sample' class

    class Sample:
        pass

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 15, 4, False)
            # Assigning a type to the variable 'self' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Sample' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'Sample', Sample)
    
    # Call to trunc(...): (line 20)
    # Processing the call arguments (line 20)
    
    # Call to Sample(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_14 = {}
    # Getting the type of 'Sample' (line 20)
    Sample_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'Sample', False)
    # Calling Sample(args, kwargs) (line 20)
    Sample_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 20, 21), Sample_13, *[], **kwargs_14)
    
    # Processing the call keyword arguments (line 20)
    kwargs_16 = {}
    # Getting the type of 'math' (line 20)
    math_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'math', False)
    # Obtaining the member 'trunc' of a type (line 20)
    trunc_12 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 10), math_11, 'trunc')
    # Calling trunc(args, kwargs) (line 20)
    trunc_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 20, 10), trunc_12, *[Sample_call_result_15], **kwargs_16)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
