
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Check the parameters of the constructors"
3: 
4: if __name__ == '__main__':
5:     import math
6: 
7: 
8:     class Foo:
9:         def __init__(self, x, y, z):
10:             # Type error
11:             self.a = len(x)
12:             self.b = len(y)
13:             # Type error
14:             self.c = math.cos(z)
15: 
16:         def method(self, x):
17:             return x
18: 
19: 
20:     f = Foo(3, list(), "5")
21: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Check the parameters of the constructors')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 4))
    
    # 'import math' statement (line 5)
    import math

    import_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'math', math, module_type_store)
    
    # Declaration of the 'Foo' class

    class Foo:

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 9, 8, False)
            # Assigning a type to the variable 'self' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.__init__', ['x', 'y', 'z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, ['x', 'y', 'z'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            
            # Assigning a Call to a Attribute (line 11):
            
            # Call to len(...): (line 11)
            # Processing the call arguments (line 11)
            # Getting the type of 'x' (line 11)
            x_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 25), 'x', False)
            # Processing the call keyword arguments (line 11)
            kwargs_4 = {}
            # Getting the type of 'len' (line 11)
            len_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 21), 'len', False)
            # Calling len(args, kwargs) (line 11)
            len_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 11, 21), len_2, *[x_3], **kwargs_4)
            
            # Getting the type of 'self' (line 11)
            self_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'self')
            # Setting the type of the member 'a' of a type (line 11)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 12), self_6, 'a', len_call_result_5)
            
            # Assigning a Call to a Attribute (line 12):
            
            # Call to len(...): (line 12)
            # Processing the call arguments (line 12)
            # Getting the type of 'y' (line 12)
            y_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 25), 'y', False)
            # Processing the call keyword arguments (line 12)
            kwargs_9 = {}
            # Getting the type of 'len' (line 12)
            len_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 21), 'len', False)
            # Calling len(args, kwargs) (line 12)
            len_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 12, 21), len_7, *[y_8], **kwargs_9)
            
            # Getting the type of 'self' (line 12)
            self_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'self')
            # Setting the type of the member 'b' of a type (line 12)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 12), self_11, 'b', len_call_result_10)
            
            # Assigning a Call to a Attribute (line 14):
            
            # Call to cos(...): (line 14)
            # Processing the call arguments (line 14)
            # Getting the type of 'z' (line 14)
            z_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 30), 'z', False)
            # Processing the call keyword arguments (line 14)
            kwargs_15 = {}
            # Getting the type of 'math' (line 14)
            math_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 21), 'math', False)
            # Obtaining the member 'cos' of a type (line 14)
            cos_13 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 21), math_12, 'cos')
            # Calling cos(args, kwargs) (line 14)
            cos_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 14, 21), cos_13, *[z_14], **kwargs_15)
            
            # Getting the type of 'self' (line 14)
            self_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'self')
            # Setting the type of the member 'c' of a type (line 14)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 12), self_17, 'c', cos_call_result_16)
            
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
            module_type_store = module_type_store.open_function_context('method', 16, 8, False)
            # Assigning a type to the variable 'self' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self', type_of_self)
            
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

            # Getting the type of 'x' (line 17)
            x_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 19), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'stypy_return_type', x_18)
            
            # ################# End of 'method(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'method' in the type store
            # Getting the type of 'stypy_return_type' (line 16)
            stypy_return_type_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_19)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'method'
            return stypy_return_type_19

    
    # Assigning a type to the variable 'Foo' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'Foo', Foo)
    
    # Assigning a Call to a Name (line 20):
    
    # Call to Foo(...): (line 20)
    # Processing the call arguments (line 20)
    int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 12), 'int')
    
    # Call to list(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_23 = {}
    # Getting the type of 'list' (line 20)
    list_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'list', False)
    # Calling list(args, kwargs) (line 20)
    list_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 20, 15), list_22, *[], **kwargs_23)
    
    str_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'str', '5')
    # Processing the call keyword arguments (line 20)
    kwargs_26 = {}
    # Getting the type of 'Foo' (line 20)
    Foo_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'Foo', False)
    # Calling Foo(args, kwargs) (line 20)
    Foo_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), Foo_20, *[int_21, list_call_result_24, str_25], **kwargs_26)
    
    # Assigning a type to the variable 'f' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'f', Foo_call_result_27)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
