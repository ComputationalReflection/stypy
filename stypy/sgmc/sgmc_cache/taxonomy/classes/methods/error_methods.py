
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Check the existence of methods defined in the class"
3: 
4: if __name__ == '__main__':
5:     class Foo:
6:         def method(self, x):
7:             return x
8: 
9: 
10:     f = Foo()
11:     # Type error
12:     f.method_alternative(3, 4, 5)
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Check the existence of methods defined in the class')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Foo' class

    class Foo:

        @norecursion
        def method(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'method'
            module_type_store = module_type_store.open_function_context('method', 6, 8, False)
            # Assigning a type to the variable 'self' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'self', type_of_self)
            
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

            # Getting the type of 'x' (line 7)
            x_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 19), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 7)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'stypy_return_type', x_2)
            
            # ################# End of 'method(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'method' in the type store
            # Getting the type of 'stypy_return_type' (line 6)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'method'
            return stypy_return_type_3


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 5, 4, False)
            # Assigning a type to the variable 'self' (line 6)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Foo' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'Foo', Foo)
    
    # Assigning a Call to a Name (line 10):
    
    # Call to Foo(...): (line 10)
    # Processing the call keyword arguments (line 10)
    kwargs_5 = {}
    # Getting the type of 'Foo' (line 10)
    Foo_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'Foo', False)
    # Calling Foo(args, kwargs) (line 10)
    Foo_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 10, 8), Foo_4, *[], **kwargs_5)
    
    # Assigning a type to the variable 'f' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'f', Foo_call_result_6)
    
    # Call to method_alternative(...): (line 12)
    # Processing the call arguments (line 12)
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 25), 'int')
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 28), 'int')
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 31), 'int')
    # Processing the call keyword arguments (line 12)
    kwargs_12 = {}
    # Getting the type of 'f' (line 12)
    f_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'f', False)
    # Obtaining the member 'method_alternative' of a type (line 12)
    method_alternative_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), f_7, 'method_alternative')
    # Calling method_alternative(args, kwargs) (line 12)
    method_alternative_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), method_alternative_8, *[int_9, int_10, int_11], **kwargs_12)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
