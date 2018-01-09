
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "divmod builtin is invoked, but classes and instances with special name methods are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Number, Number) -> <type 'tuple'>
7:     # (Overloads__divmod__, Number) -> <type 'tuple'>
8:     # (Number, Overloads__rdivmod__) -> <type 'tuple'>
9:     class Empty:
10:         pass
11: 
12: 
13:     class Sample:
14:         def __divmod__(self, other):
15:             return 4, other
16: 
17: 
18:     class Wrong2:
19:         def __divmod__(self):
20:             return 4, 5
21: 
22: 
23:     # Call the builtin with correct parameters
24:     # No error
25:     ret = divmod(Sample(), 4)
26: 
27:     # Call the builtin with incorrect types of parameters
28:     # Type error
29:     ret = divmod(Wrong2(), 4)
30:     # Type error
31:     ret = divmod(Empty(), 4)
32: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'divmod builtin is invoked, but classes and instances with special name methods are passed')
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
        def __divmod__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__divmod__'
            module_type_store = module_type_store.open_function_context('__divmod__', 14, 8, False)
            # Assigning a type to the variable 'self' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__divmod__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__divmod__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__divmod__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__divmod__.__dict__.__setitem__('stypy_function_name', 'Sample.__divmod__')
            Sample.__divmod__.__dict__.__setitem__('stypy_param_names_list', ['other'])
            Sample.__divmod__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__divmod__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__divmod__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__divmod__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__divmod__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__divmod__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__divmod__', ['other'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__divmod__', localization, ['other'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__divmod__(...)' code ##################

            
            # Obtaining an instance of the builtin type 'tuple' (line 15)
            tuple_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 15)
            # Adding element type (line 15)
            int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 19), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 19), tuple_2, int_3)
            # Adding element type (line 15)
            # Getting the type of 'other' (line 15)
            other_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 22), 'other')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 19), tuple_2, other_4)
            
            # Assigning a type to the variable 'stypy_return_type' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'stypy_return_type', tuple_2)
            
            # ################# End of '__divmod__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__divmod__' in the type store
            # Getting the type of 'stypy_return_type' (line 14)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__divmod__'
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
    # Declaration of the 'Wrong2' class

    class Wrong2:

        @norecursion
        def __divmod__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__divmod__'
            module_type_store = module_type_store.open_function_context('__divmod__', 19, 8, False)
            # Assigning a type to the variable 'self' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Wrong2.__divmod__.__dict__.__setitem__('stypy_localization', localization)
            Wrong2.__divmod__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Wrong2.__divmod__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Wrong2.__divmod__.__dict__.__setitem__('stypy_function_name', 'Wrong2.__divmod__')
            Wrong2.__divmod__.__dict__.__setitem__('stypy_param_names_list', [])
            Wrong2.__divmod__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Wrong2.__divmod__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Wrong2.__divmod__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Wrong2.__divmod__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Wrong2.__divmod__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Wrong2.__divmod__.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Wrong2.__divmod__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__divmod__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__divmod__(...)' code ##################

            
            # Obtaining an instance of the builtin type 'tuple' (line 20)
            tuple_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 20)
            # Adding element type (line 20)
            int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 19), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 19), tuple_6, int_7)
            # Adding element type (line 20)
            int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 22), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 19), tuple_6, int_8)
            
            # Assigning a type to the variable 'stypy_return_type' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'stypy_return_type', tuple_6)
            
            # ################# End of '__divmod__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__divmod__' in the type store
            # Getting the type of 'stypy_return_type' (line 19)
            stypy_return_type_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_9)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__divmod__'
            return stypy_return_type_9


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
    
    # Assigning a Call to a Name (line 25):
    
    # Call to divmod(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Call to Sample(...): (line 25)
    # Processing the call keyword arguments (line 25)
    kwargs_12 = {}
    # Getting the type of 'Sample' (line 25)
    Sample_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'Sample', False)
    # Calling Sample(args, kwargs) (line 25)
    Sample_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 25, 17), Sample_11, *[], **kwargs_12)
    
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 27), 'int')
    # Processing the call keyword arguments (line 25)
    kwargs_15 = {}
    # Getting the type of 'divmod' (line 25)
    divmod_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'divmod', False)
    # Calling divmod(args, kwargs) (line 25)
    divmod_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 25, 10), divmod_10, *[Sample_call_result_13, int_14], **kwargs_15)
    
    # Assigning a type to the variable 'ret' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'ret', divmod_call_result_16)
    
    # Assigning a Call to a Name (line 29):
    
    # Call to divmod(...): (line 29)
    # Processing the call arguments (line 29)
    
    # Call to Wrong2(...): (line 29)
    # Processing the call keyword arguments (line 29)
    kwargs_19 = {}
    # Getting the type of 'Wrong2' (line 29)
    Wrong2_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'Wrong2', False)
    # Calling Wrong2(args, kwargs) (line 29)
    Wrong2_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 29, 17), Wrong2_18, *[], **kwargs_19)
    
    int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 27), 'int')
    # Processing the call keyword arguments (line 29)
    kwargs_22 = {}
    # Getting the type of 'divmod' (line 29)
    divmod_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'divmod', False)
    # Calling divmod(args, kwargs) (line 29)
    divmod_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 29, 10), divmod_17, *[Wrong2_call_result_20, int_21], **kwargs_22)
    
    # Assigning a type to the variable 'ret' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'ret', divmod_call_result_23)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to divmod(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Call to Empty(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_26 = {}
    # Getting the type of 'Empty' (line 31)
    Empty_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'Empty', False)
    # Calling Empty(args, kwargs) (line 31)
    Empty_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 31, 17), Empty_25, *[], **kwargs_26)
    
    int_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 26), 'int')
    # Processing the call keyword arguments (line 31)
    kwargs_29 = {}
    # Getting the type of 'divmod' (line 31)
    divmod_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 10), 'divmod', False)
    # Calling divmod(args, kwargs) (line 31)
    divmod_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 31, 10), divmod_24, *[Empty_call_result_27, int_28], **kwargs_29)
    
    # Assigning a type to the variable 'ret' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'ret', divmod_call_result_30)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
