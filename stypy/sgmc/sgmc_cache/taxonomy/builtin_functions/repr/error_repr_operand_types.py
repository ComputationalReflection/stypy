
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "repr builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Has__repr__) -> <type 'str'>
7: 
8:     class Sample:
9:         def __repr__(self):
10:             return "str"
11: 
12: 
13:     class Wrong1:
14:         def __repr__(self, x):
15:             return x
16: 
17: 
18:     class Wrong2:
19:         def __repr__(self):
20:             return 4
21: 
22: 
23:     # Call the builtin with correct parameters
24:     ret = repr(Sample())
25: 
26:     # Call the builtin with incorrect types of parameters
27:     # Type error
28:     ret = repr(Wrong1())
29:     # Type error
30:     ret = repr(Wrong2())
31: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'repr builtin is invoked, but incorrect parameter types are passed')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__repr__'
            module_type_store = module_type_store.open_function_context('__repr__', 9, 8, False)
            # Assigning a type to the variable 'self' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
            Sample.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Sample.__repr__')
            Sample.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
            Sample.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__repr__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__repr__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__repr__(...)' code ##################

            str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 19), 'str', 'str')
            # Assigning a type to the variable 'stypy_return_type' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'stypy_return_type', str_2)
            
            # ################# End of '__repr__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__repr__' in the type store
            # Getting the type of 'stypy_return_type' (line 9)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__repr__'
            return stypy_return_type_3


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 8, 4, False)
            # Assigning a type to the variable 'self' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Sample' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'Sample', Sample)
    # Declaration of the 'Wrong1' class

    class Wrong1:

        @norecursion
        def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__repr__'
            module_type_store = module_type_store.open_function_context('__repr__', 14, 8, False)
            # Assigning a type to the variable 'self' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong1.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
            Wrong1.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong1.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong1.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Wrong1.__repr__')
            Wrong1.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', ['x'])
            Wrong1.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong1.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong1.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong1.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong1.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong1.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong1.__repr__', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__repr__', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__repr__(...)' code ##################

            # Getting the type of 'x' (line 15)
            x_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 19), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'stypy_return_type', x_4)
            
            # ################# End of '__repr__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__repr__' in the type store
            # Getting the type of 'stypy_return_type' (line 14)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__repr__'
            return stypy_return_type_5


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 13, 4, False)
            # Assigning a type to the variable 'self' (line 14)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong1.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Wrong1' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'Wrong1', Wrong1)
    # Declaration of the 'Wrong2' class

    class Wrong2:

        @norecursion
        def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__repr__'
            module_type_store = module_type_store.open_function_context('__repr__', 19, 8, False)
            # Assigning a type to the variable 'self' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong2.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
            Wrong2.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong2.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong2.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Wrong2.__repr__')
            Wrong2.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
            Wrong2.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong2.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong2.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong2.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong2.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong2.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong2.__repr__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__repr__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__repr__(...)' code ##################

            int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'stypy_return_type', int_6)
            
            # ################# End of '__repr__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__repr__' in the type store
            # Getting the type of 'stypy_return_type' (line 19)
            stypy_return_type_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_7)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__repr__'
            return stypy_return_type_7


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 18, 4, False)
            # Assigning a type to the variable 'self' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong2.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Wrong2' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'Wrong2', Wrong2)
    
    # Assigning a Call to a Name (line 24):
    
    # Call to repr(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Call to Sample(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_10 = {}
    # Getting the type of 'Sample' (line 24)
    Sample_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 15), 'Sample', False)
    # Calling Sample(args, kwargs) (line 24)
    Sample_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 24, 15), Sample_9, *[], **kwargs_10)
    
    # Processing the call keyword arguments (line 24)
    kwargs_12 = {}
    # Getting the type of 'repr' (line 24)
    repr_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 10), 'repr', False)
    # Calling repr(args, kwargs) (line 24)
    repr_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 24, 10), repr_8, *[Sample_call_result_11], **kwargs_12)
    
    # Assigning a type to the variable 'ret' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'ret', repr_call_result_13)
    
    # Assigning a Call to a Name (line 28):
    
    # Call to repr(...): (line 28)
    # Processing the call arguments (line 28)
    
    # Call to Wrong1(...): (line 28)
    # Processing the call keyword arguments (line 28)
    kwargs_16 = {}
    # Getting the type of 'Wrong1' (line 28)
    Wrong1_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'Wrong1', False)
    # Calling Wrong1(args, kwargs) (line 28)
    Wrong1_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 28, 15), Wrong1_15, *[], **kwargs_16)
    
    # Processing the call keyword arguments (line 28)
    kwargs_18 = {}
    # Getting the type of 'repr' (line 28)
    repr_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 10), 'repr', False)
    # Calling repr(args, kwargs) (line 28)
    repr_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 28, 10), repr_14, *[Wrong1_call_result_17], **kwargs_18)
    
    # Assigning a type to the variable 'ret' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'ret', repr_call_result_19)
    
    # Assigning a Call to a Name (line 30):
    
    # Call to repr(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Call to Wrong2(...): (line 30)
    # Processing the call keyword arguments (line 30)
    kwargs_22 = {}
    # Getting the type of 'Wrong2' (line 30)
    Wrong2_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'Wrong2', False)
    # Calling Wrong2(args, kwargs) (line 30)
    Wrong2_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 30, 15), Wrong2_21, *[], **kwargs_22)
    
    # Processing the call keyword arguments (line 30)
    kwargs_24 = {}
    # Getting the type of 'repr' (line 30)
    repr_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 10), 'repr', False)
    # Calling repr(args, kwargs) (line 30)
    repr_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 30, 10), repr_20, *[Wrong2_call_result_23], **kwargs_24)
    
    # Assigning a type to the variable 'ret' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'ret', repr_call_result_25)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
