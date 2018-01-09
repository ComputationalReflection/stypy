
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "unicode builtin is invoked, but classes and instances with special name methods are passed "
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'unicode'>
7:     # (Has__str__) -> <type 'unicode'>
8:     # (AnyType) -> <type 'unicode'>
9:     class Empty:
10:         pass
11: 
12: 
13:     class Sample:
14:         def __str__(self):
15:             return "str"
16: 
17: 
18:     class Wrong1:
19:         def __str__(self, x):
20:             return x
21: 
22: 
23:     class Wrong2:
24:         def __str__(self):
25:             return 4
26: 
27: 
28:     # Call the builtin with correct parameters
29:     ret = unicode(Sample())
30: 
31:     # Call the builtin with incorrect types of parameters
32:     # Type error
33:     ret = unicode(Wrong1())
34:     # Type error
35:     ret = unicode(Wrong2())
36: 
37:     ret = unicode(Empty())
38: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'unicode builtin is invoked, but classes and instances with special name methods are passed ')
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
            module_type_store = module_type_store.open_function_context('__init__', 9, 4, False)
            # Assigning a type to the variable 'self' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Empty' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'Empty', Empty)
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def stypy__str__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__str__'
            module_type_store = module_type_store.open_function_context('__str__', 14, 8, False)
            # Assigning a type to the variable 'self' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
            Sample.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.stypy__str__.__dict__.__setitem__('stypy_function_name', 'Sample.__str__')
            Sample.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
            Sample.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__str__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__str__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__str__(...)' code ##################

            str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 19), 'str', 'str')
            # Assigning a type to the variable 'stypy_return_type' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'stypy_return_type', str_2)
            
            # ################# End of '__str__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__str__' in the type store
            # Getting the type of 'stypy_return_type' (line 14)
            stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__str__'
            return stypy_return_type_3


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

    
    # Assigning a type to the variable 'Sample' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'Sample', Sample)
    # Declaration of the 'Wrong1' class

    class Wrong1:

        @norecursion
        def stypy__str__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__str__'
            module_type_store = module_type_store.open_function_context('__str__', 19, 8, False)
            # Assigning a type to the variable 'self' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong1.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
            Wrong1.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong1.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong1.stypy__str__.__dict__.__setitem__('stypy_function_name', 'Wrong1.__str__')
            Wrong1.stypy__str__.__dict__.__setitem__('stypy_param_names_list', ['x'])
            Wrong1.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong1.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong1.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong1.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong1.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong1.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong1.__str__', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__str__', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__str__(...)' code ##################

            # Getting the type of 'x' (line 20)
            x_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 19), 'x')
            # Assigning a type to the variable 'stypy_return_type' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'stypy_return_type', x_4)
            
            # ################# End of '__str__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__str__' in the type store
            # Getting the type of 'stypy_return_type' (line 19)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__str__'
            return stypy_return_type_5


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

    
    # Assigning a type to the variable 'Wrong1' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'Wrong1', Wrong1)
    # Declaration of the 'Wrong2' class

    class Wrong2:

        @norecursion
        def stypy__str__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__str__'
            module_type_store = module_type_store.open_function_context('__str__', 24, 8, False)
            # Assigning a type to the variable 'self' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong2.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
            Wrong2.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong2.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong2.stypy__str__.__dict__.__setitem__('stypy_function_name', 'Wrong2.__str__')
            Wrong2.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
            Wrong2.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong2.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong2.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong2.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong2.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong2.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong2.__str__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__str__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__str__(...)' code ##################

            int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'stypy_return_type', int_6)
            
            # ################# End of '__str__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__str__' in the type store
            # Getting the type of 'stypy_return_type' (line 24)
            stypy_return_type_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_7)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__str__'
            return stypy_return_type_7


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 23, 4, False)
            # Assigning a type to the variable 'self' (line 24)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Wrong2' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'Wrong2', Wrong2)
    
    # Assigning a Call to a Name (line 29):
    
    # Call to unicode(...): (line 29)
    # Processing the call arguments (line 29)
    
    # Call to Sample(...): (line 29)
    # Processing the call keyword arguments (line 29)
    kwargs_10 = {}
    # Getting the type of 'Sample' (line 29)
    Sample_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'Sample', False)
    # Calling Sample(args, kwargs) (line 29)
    Sample_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 29, 18), Sample_9, *[], **kwargs_10)
    
    # Processing the call keyword arguments (line 29)
    kwargs_12 = {}
    # Getting the type of 'unicode' (line 29)
    unicode_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'unicode', False)
    # Calling unicode(args, kwargs) (line 29)
    unicode_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 29, 10), unicode_8, *[Sample_call_result_11], **kwargs_12)
    
    # Assigning a type to the variable 'ret' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'ret', unicode_call_result_13)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to unicode(...): (line 33)
    # Processing the call arguments (line 33)
    
    # Call to Wrong1(...): (line 33)
    # Processing the call keyword arguments (line 33)
    kwargs_16 = {}
    # Getting the type of 'Wrong1' (line 33)
    Wrong1_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 18), 'Wrong1', False)
    # Calling Wrong1(args, kwargs) (line 33)
    Wrong1_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 33, 18), Wrong1_15, *[], **kwargs_16)
    
    # Processing the call keyword arguments (line 33)
    kwargs_18 = {}
    # Getting the type of 'unicode' (line 33)
    unicode_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 10), 'unicode', False)
    # Calling unicode(args, kwargs) (line 33)
    unicode_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 33, 10), unicode_14, *[Wrong1_call_result_17], **kwargs_18)
    
    # Assigning a type to the variable 'ret' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'ret', unicode_call_result_19)
    
    # Assigning a Call to a Name (line 35):
    
    # Call to unicode(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Call to Wrong2(...): (line 35)
    # Processing the call keyword arguments (line 35)
    kwargs_22 = {}
    # Getting the type of 'Wrong2' (line 35)
    Wrong2_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 18), 'Wrong2', False)
    # Calling Wrong2(args, kwargs) (line 35)
    Wrong2_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 35, 18), Wrong2_21, *[], **kwargs_22)
    
    # Processing the call keyword arguments (line 35)
    kwargs_24 = {}
    # Getting the type of 'unicode' (line 35)
    unicode_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 10), 'unicode', False)
    # Calling unicode(args, kwargs) (line 35)
    unicode_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 35, 10), unicode_20, *[Wrong2_call_result_23], **kwargs_24)
    
    # Assigning a type to the variable 'ret' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'ret', unicode_call_result_25)
    
    # Assigning a Call to a Name (line 37):
    
    # Call to unicode(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Call to Empty(...): (line 37)
    # Processing the call keyword arguments (line 37)
    kwargs_28 = {}
    # Getting the type of 'Empty' (line 37)
    Empty_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 18), 'Empty', False)
    # Calling Empty(args, kwargs) (line 37)
    Empty_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 37, 18), Empty_27, *[], **kwargs_28)
    
    # Processing the call keyword arguments (line 37)
    kwargs_30 = {}
    # Getting the type of 'unicode' (line 37)
    unicode_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 10), 'unicode', False)
    # Calling unicode(args, kwargs) (line 37)
    unicode_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 37, 10), unicode_26, *[Empty_call_result_29], **kwargs_30)
    
    # Assigning a type to the variable 'ret' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'ret', unicode_call_result_31)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
