
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "bytearray builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # () -> <type 'bytearray'>
7:     # (IterableDataStructureWithTypedElements(Integer, Overloads__trunc__)) -> <type 'bytearray'>
8:     # (Integer) -> <type 'bytearray'>
9:     # (Str) -> <type 'bytearray'>
10: 
11: 
12:     # Type error
13:     ret = bytearray(list)
14:     # Type error
15:     ret = bytearray(int)
16:     # Type error
17:     ret = bytearray(str)
18: 
19: 
20:     class Sample:
21:         def __trunc__(self):
22:             return 4
23: 
24: 
25:     list_trunc = [Sample, Sample]
26:     # Type error
27:     ret = bytearray(list_trunc)
28: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'bytearray builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to bytearray(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'list' (line 13)
    list_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 20), 'list', False)
    # Processing the call keyword arguments (line 13)
    kwargs_4 = {}
    # Getting the type of 'bytearray' (line 13)
    bytearray_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'bytearray', False)
    # Calling bytearray(args, kwargs) (line 13)
    bytearray_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), bytearray_2, *[list_3], **kwargs_4)
    
    # Assigning a type to the variable 'ret' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'ret', bytearray_call_result_5)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to bytearray(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'int' (line 15)
    int_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 20), 'int', False)
    # Processing the call keyword arguments (line 15)
    kwargs_8 = {}
    # Getting the type of 'bytearray' (line 15)
    bytearray_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'bytearray', False)
    # Calling bytearray(args, kwargs) (line 15)
    bytearray_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), bytearray_6, *[int_7], **kwargs_8)
    
    # Assigning a type to the variable 'ret' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'ret', bytearray_call_result_9)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to bytearray(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'str' (line 17)
    str_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 'str', False)
    # Processing the call keyword arguments (line 17)
    kwargs_12 = {}
    # Getting the type of 'bytearray' (line 17)
    bytearray_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'bytearray', False)
    # Calling bytearray(args, kwargs) (line 17)
    bytearray_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), bytearray_10, *[str_11], **kwargs_12)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', bytearray_call_result_13)
    # Declaration of the 'Sample' class

    class Sample:

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

            int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'stypy_return_type', int_14)
            
            # ################# End of '__trunc__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__trunc__' in the type store
            # Getting the type of 'stypy_return_type' (line 21)
            stypy_return_type_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_15)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__trunc__'
            return stypy_return_type_15


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

    
    # Assigning a type to the variable 'Sample' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'Sample', Sample)
    
    # Assigning a List to a Name (line 25):
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    # Getting the type of 'Sample' (line 25)
    Sample_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 18), 'Sample')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 17), list_16, Sample_17)
    # Adding element type (line 25)
    # Getting the type of 'Sample' (line 25)
    Sample_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 26), 'Sample')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 17), list_16, Sample_18)
    
    # Assigning a type to the variable 'list_trunc' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'list_trunc', list_16)
    
    # Assigning a Call to a Name (line 27):
    
    # Call to bytearray(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'list_trunc' (line 27)
    list_trunc_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 20), 'list_trunc', False)
    # Processing the call keyword arguments (line 27)
    kwargs_21 = {}
    # Getting the type of 'bytearray' (line 27)
    bytearray_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 10), 'bytearray', False)
    # Calling bytearray(args, kwargs) (line 27)
    bytearray_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 27, 10), bytearray_19, *[list_trunc_20], **kwargs_21)
    
    # Assigning a type to the variable 'ret' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'ret', bytearray_call_result_22)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
