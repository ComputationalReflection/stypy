
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "divmod builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Number, Number) -> <type 'tuple'>
7:     # (Overloads__divmod__, Number) -> <type 'tuple'>
8:     # (Number, Overloads__rdivmod__) -> <type 'tuple'>
9: 
10:     class Sample:
11:         def __divmod__(self, other):
12:             return 4, other
13: 
14: 
15:     # Type error
16:     ret = divmod(int, int)
17:     # Type error
18:     ret = divmod(Sample, 4)
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'divmod builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __divmod__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__divmod__'
            module_type_store = module_type_store.open_function_context('__divmod__', 11, 8, False)
            # Assigning a type to the variable 'self' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'self', type_of_self)
            
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

            
            # Obtaining an instance of the builtin type 'tuple' (line 12)
            tuple_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 12)
            # Adding element type (line 12)
            int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 19), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), tuple_2, int_3)
            # Adding element type (line 12)
            # Getting the type of 'other' (line 12)
            other_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 22), 'other')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), tuple_2, other_4)
            
            # Assigning a type to the variable 'stypy_return_type' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'stypy_return_type', tuple_2)
            
            # ################# End of '__divmod__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__divmod__' in the type store
            # Getting the type of 'stypy_return_type' (line 11)
            stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type')
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
            module_type_store = module_type_store.open_function_context('__init__', 10, 4, False)
            # Assigning a type to the variable 'self' (line 11)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'self', type_of_self)
            
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

    
    # Assigning a type to the variable 'Sample' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'Sample', Sample)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to divmod(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'int' (line 16)
    int_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 17), 'int', False)
    # Getting the type of 'int' (line 16)
    int_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 22), 'int', False)
    # Processing the call keyword arguments (line 16)
    kwargs_9 = {}
    # Getting the type of 'divmod' (line 16)
    divmod_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'divmod', False)
    # Calling divmod(args, kwargs) (line 16)
    divmod_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), divmod_6, *[int_7, int_8], **kwargs_9)
    
    # Assigning a type to the variable 'ret' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', divmod_call_result_10)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to divmod(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'Sample' (line 18)
    Sample_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'Sample', False)
    int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_14 = {}
    # Getting the type of 'divmod' (line 18)
    divmod_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'divmod', False)
    # Calling divmod(args, kwargs) (line 18)
    divmod_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), divmod_11, *[Sample_12, int_13], **kwargs_14)
    
    # Assigning a type to the variable 'ret' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', divmod_call_result_15)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
