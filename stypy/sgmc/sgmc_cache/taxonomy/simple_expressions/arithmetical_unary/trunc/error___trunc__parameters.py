
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: import math
3: 
4: __doc__ = "__trunc__ method is present, but is declared with a wrong number of parameters"
5: 
6: if __name__ == '__main__':
7:     class Sample:
8:         def __trunc__(self, other, another):
9:             return 1
10: 
11: 
12:     # Type error
13:     print math.trunc(Sample())
14: 
15: 
16:     class OtherSample:
17:         def __trunc__(self, other):
18:             return 1
19: 
20: 
21:     # Type error
22:     print math.trunc(OtherSample())
23: 

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
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'str', '__trunc__ method is present, but is declared with a wrong number of parameters')
# Assigning a type to the variable '__doc__' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

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
            Sample.__trunc__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__trunc__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__trunc__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__trunc__.__dict__.__setitem__('stypy_function_name', 'Sample.__trunc__')
            Sample.__trunc__.__dict__.__setitem__('stypy_param_names_list', ['other', 'another'])
            Sample.__trunc__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__trunc__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__trunc__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__trunc__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__trunc__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__trunc__.__dict__.__setitem__('stypy_declared_arg_number', 3)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__trunc__', ['other', 'another'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__trunc__', localization, ['other', 'another'], arguments)
            
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

    
    # Assigning a type to the variable 'Sample' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'Sample', Sample)
    
    # Call to trunc(...): (line 13)
    # Processing the call arguments (line 13)
    
    # Call to Sample(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_7 = {}
    # Getting the type of 'Sample' (line 13)
    Sample_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 21), 'Sample', False)
    # Calling Sample(args, kwargs) (line 13)
    Sample_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 13, 21), Sample_6, *[], **kwargs_7)
    
    # Processing the call keyword arguments (line 13)
    kwargs_9 = {}
    # Getting the type of 'math' (line 13)
    math_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'math', False)
    # Obtaining the member 'trunc' of a type (line 13)
    trunc_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 10), math_4, 'trunc')
    # Calling trunc(args, kwargs) (line 13)
    trunc_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), trunc_5, *[Sample_call_result_8], **kwargs_9)
    
    # Declaration of the 'OtherSample' class

    class OtherSample:

        @norecursion
        def __trunc__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__trunc__'
            module_type_store = module_type_store.open_function_context('__trunc__', 17, 8, False)
            # Assigning a type to the variable 'self' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            OtherSample.__trunc__.__dict__.__setitem__('stypy_localization', localization)
            OtherSample.__trunc__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            OtherSample.__trunc__.__dict__.__setitem__('stypy_type_store', module_type_store)
            OtherSample.__trunc__.__dict__.__setitem__('stypy_function_name', 'OtherSample.__trunc__')
            OtherSample.__trunc__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            OtherSample.__trunc__.__dict__.__setitem__('stypy_varargs_param_name', None)
            OtherSample.__trunc__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            OtherSample.__trunc__.__dict__.__setitem__('stypy_call_defaults', defaults)
            OtherSample.__trunc__.__dict__.__setitem__('stypy_call_varargs', varargs)
            OtherSample.__trunc__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            OtherSample.__trunc__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'OtherSample.__trunc__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__trunc__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__trunc__(...)' code ##################

            int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'stypy_return_type', int_11)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 17)
            stypy_return_type_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_12)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__trunc__'
            return stypy_return_type_12


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 16, 4, False)
            # Assigning a type to the variable 'self' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'OtherSample.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'OtherSample' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'OtherSample', OtherSample)
    
    # Call to trunc(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Call to OtherSample(...): (line 22)
    # Processing the call keyword arguments (line 22)
    kwargs_16 = {}
    # Getting the type of 'OtherSample' (line 22)
    OtherSample_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 21), 'OtherSample', False)
    # Calling OtherSample(args, kwargs) (line 22)
    OtherSample_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 22, 21), OtherSample_15, *[], **kwargs_16)
    
    # Processing the call keyword arguments (line 22)
    kwargs_18 = {}
    # Getting the type of 'math' (line 22)
    math_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 10), 'math', False)
    # Obtaining the member 'trunc' of a type (line 22)
    trunc_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 10), math_13, 'trunc')
    # Calling trunc(args, kwargs) (line 22)
    trunc_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 22, 10), trunc_14, *[OtherSample_call_result_17], **kwargs_18)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
