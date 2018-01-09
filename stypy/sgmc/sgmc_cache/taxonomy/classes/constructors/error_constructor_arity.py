
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Check the arity of the constructors"
3: 
4: if __name__ == '__main__':
5:     class Foo:
6:         def __init__(self, x, y):
7:             pass
8: 
9:         def method(self, x):
10:             return x
11: 
12: 
13:     # Type error
14:     f = Foo(3, 4, 5)
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Check the arity of the constructors')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Foo' class

    class Foo:

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.__init__', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['x', 'y'], arguments)
            
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


        @norecursion
        def method(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'method'
            module_type_store = module_type_store.open_function_context('method', 9, 8, False)
            # Assigning a type to the variable 'self' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Foo.method.__dict__.__setitem__('stypy_localization', localization)
            Foo.method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Foo.method.__dict__.__setitem__('stypy_type_store', module_type_store)
            Foo.method.__dict__.__setitem__('stypy_function_name', 'Foo.method')
            Foo.method.__dict__.__setitem__('stypy_param_names_list', ['x'])
            Foo.method.__dict__.__setitem__('stypy_varargs_param_name', None)
            Foo.method.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Foo.method.__dict__.__setitem__('stypy_call_defaults', defaults)
            Foo.method.__dict__.__setitem__('stypy_call_varargs', varargs)
            Foo.method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Foo.method.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.method', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'method', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'method(...)' code ##################

            # Getting the type of 'x' (line 10)
            x_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 19), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'stypy_return_type', x_2)
            
            # ################# End of 'method(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'method' in the type store
            # Getting the type of 'stypy_return_type' (line 9)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'method'
            return stypy_return_type_3

    
    # Assigning a type to the variable 'Foo' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'Foo', Foo)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to Foo(...): (line 14)
    # Processing the call arguments (line 14)
    int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'int')
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'int')
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'int')
    # Processing the call keyword arguments (line 14)
    kwargs_8 = {}
    # Getting the type of 'Foo' (line 14)
    Foo_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'Foo', False)
    # Calling Foo(args, kwargs) (line 14)
    Foo_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), Foo_4, *[int_5, int_6, int_7], **kwargs_8)
    
    # Assigning a type to the variable 'f' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'f', Foo_call_result_9)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
