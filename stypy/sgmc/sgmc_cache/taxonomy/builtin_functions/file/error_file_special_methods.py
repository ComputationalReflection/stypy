
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "file builtin is invoked, but classes and instances with special name methods are passed "
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Str) -> <type 'file'>
7:     # (Str, Str) -> <type 'file'>
8:     # (Str, Str, Integer) -> <type 'file'>
9:     # (Str, Str, Overloads__trunc__) -> <type 'file'>
10: 
11:     class Empty:
12:         pass
13: 
14: 
15:     class Sample:
16:         def __trunc__(self):
17:             return 4
18: 
19: 
20:     class Wrong1:
21:         def __trunc__(self, x):
22:             return x
23: 
24: 
25:     class Wrong2:
26:         def __trunc__(self):
27:             return "str"
28: 
29: 
30:     # Call the builtin with correct parameters
31:     # No error
32:     ret = file("f.py", "r", Sample())
33: 
34:     # Call the builtin with incorrect types of parameters
35:     # Type error
36:     ret = file("f.py", "r", Wrong1())
37:     # Type error
38:     ret = file("f.py", "r", Wrong2())
39:     # Type error
40:     ret = file("f.py", "r", Empty())
41: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'file builtin is invoked, but classes and instances with special name methods are passed ')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
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
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __trunc__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__trunc__'
            module_type_store = module_type_store.open_function_context('__trunc__', 16, 8, False)
            # Assigning a type to the variable 'self' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__trunc__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__trunc__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__trunc__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__trunc__.__dict__.__setitem__('stypy_function_name', 'Sample.__trunc__')
            Sample.__trunc__.__dict__.__setitem__('stypy_param_names_list', [])
            Sample.__trunc__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__trunc__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__trunc__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__trunc__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__trunc__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__trunc__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__trunc__', [], None, None, defaults, varargs, kwargs)

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

            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'stypy_return_type', int_2)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 16)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'stypy_return_type')
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
    # Declaration of the 'Wrong1' class

    class Wrong1:

        @norecursion
        def __trunc__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__trunc__'
            module_type_store = module_type_store.open_function_context('__trunc__', 21, 8, False)
            # Assigning a type to the variable 'self' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong1.__trunc__.__dict__.__setitem__('stypy_localization', localization)
            Wrong1.__trunc__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong1.__trunc__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong1.__trunc__.__dict__.__setitem__('stypy_function_name', 'Wrong1.__trunc__')
            Wrong1.__trunc__.__dict__.__setitem__('stypy_param_names_list', ['x'])
            Wrong1.__trunc__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong1.__trunc__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong1.__trunc__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong1.__trunc__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong1.__trunc__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong1.__trunc__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong1.__trunc__', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__trunc__', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__trunc__(...)' code ##################

            # Getting the type of 'x' (line 22)
            x_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'stypy_return_type', x_4)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 21)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__trunc__'
            return stypy_return_type_5


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 20, 4, False)
            # Assigning a type to the variable 'self' (line 21)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Wrong1' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'Wrong1', Wrong1)
    # Declaration of the 'Wrong2' class

    class Wrong2:

        @norecursion
        def __trunc__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__trunc__'
            module_type_store = module_type_store.open_function_context('__trunc__', 26, 8, False)
            # Assigning a type to the variable 'self' (line 27)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong2.__trunc__.__dict__.__setitem__('stypy_localization', localization)
            Wrong2.__trunc__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong2.__trunc__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong2.__trunc__.__dict__.__setitem__('stypy_function_name', 'Wrong2.__trunc__')
            Wrong2.__trunc__.__dict__.__setitem__('stypy_param_names_list', [])
            Wrong2.__trunc__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong2.__trunc__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong2.__trunc__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong2.__trunc__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong2.__trunc__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong2.__trunc__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong2.__trunc__', [], None, None, defaults, varargs, kwargs)

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

            str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 19), 'str', 'str')
            # Assigning a type to the variable 'stypy_return_type' (line 27)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'stypy_return_type', str_6)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 26)
            stypy_return_type_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_7)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__trunc__'
            return stypy_return_type_7


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 25, 4, False)
            # Assigning a type to the variable 'self' (line 26)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Wrong2' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'Wrong2', Wrong2)
    
    # Assigning a Call to a Name (line 32):
    
    # Call to file(...): (line 32)
    # Processing the call arguments (line 32)
    str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 15), 'str', 'f.py')
    str_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'str', 'r')
    
    # Call to Sample(...): (line 32)
    # Processing the call keyword arguments (line 32)
    kwargs_12 = {}
    # Getting the type of 'Sample' (line 32)
    Sample_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 28), 'Sample', False)
    # Calling Sample(args, kwargs) (line 32)
    Sample_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 32, 28), Sample_11, *[], **kwargs_12)
    
    # Processing the call keyword arguments (line 32)
    kwargs_14 = {}
    # Getting the type of 'file' (line 32)
    file_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 10), 'file', False)
    # Calling file(args, kwargs) (line 32)
    file_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 32, 10), file_8, *[str_9, str_10, Sample_call_result_13], **kwargs_14)
    
    # Assigning a type to the variable 'ret' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'ret', file_call_result_15)
    
    # Assigning a Call to a Name (line 36):
    
    # Call to file(...): (line 36)
    # Processing the call arguments (line 36)
    str_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 15), 'str', 'f.py')
    str_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'str', 'r')
    
    # Call to Wrong1(...): (line 36)
    # Processing the call keyword arguments (line 36)
    kwargs_20 = {}
    # Getting the type of 'Wrong1' (line 36)
    Wrong1_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 28), 'Wrong1', False)
    # Calling Wrong1(args, kwargs) (line 36)
    Wrong1_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 36, 28), Wrong1_19, *[], **kwargs_20)
    
    # Processing the call keyword arguments (line 36)
    kwargs_22 = {}
    # Getting the type of 'file' (line 36)
    file_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 10), 'file', False)
    # Calling file(args, kwargs) (line 36)
    file_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 36, 10), file_16, *[str_17, str_18, Wrong1_call_result_21], **kwargs_22)
    
    # Assigning a type to the variable 'ret' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'ret', file_call_result_23)
    
    # Assigning a Call to a Name (line 38):
    
    # Call to file(...): (line 38)
    # Processing the call arguments (line 38)
    str_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 15), 'str', 'f.py')
    str_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 23), 'str', 'r')
    
    # Call to Wrong2(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_28 = {}
    # Getting the type of 'Wrong2' (line 38)
    Wrong2_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 28), 'Wrong2', False)
    # Calling Wrong2(args, kwargs) (line 38)
    Wrong2_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 38, 28), Wrong2_27, *[], **kwargs_28)
    
    # Processing the call keyword arguments (line 38)
    kwargs_30 = {}
    # Getting the type of 'file' (line 38)
    file_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 10), 'file', False)
    # Calling file(args, kwargs) (line 38)
    file_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 38, 10), file_24, *[str_25, str_26, Wrong2_call_result_29], **kwargs_30)
    
    # Assigning a type to the variable 'ret' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'ret', file_call_result_31)
    
    # Assigning a Call to a Name (line 40):
    
    # Call to file(...): (line 40)
    # Processing the call arguments (line 40)
    str_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 15), 'str', 'f.py')
    str_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 23), 'str', 'r')
    
    # Call to Empty(...): (line 40)
    # Processing the call keyword arguments (line 40)
    kwargs_36 = {}
    # Getting the type of 'Empty' (line 40)
    Empty_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 28), 'Empty', False)
    # Calling Empty(args, kwargs) (line 40)
    Empty_call_result_37 = invoke(stypy.reporting.localization.Localization(__file__, 40, 28), Empty_35, *[], **kwargs_36)
    
    # Processing the call keyword arguments (line 40)
    kwargs_38 = {}
    # Getting the type of 'file' (line 40)
    file_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 10), 'file', False)
    # Calling file(args, kwargs) (line 40)
    file_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 40, 10), file_32, *[str_33, str_34, Empty_call_result_37], **kwargs_38)
    
    # Assigning a type to the variable 'ret' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'ret', file_call_result_39)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
