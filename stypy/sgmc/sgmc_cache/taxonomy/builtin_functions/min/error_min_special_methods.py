
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "min builtin is invoked, but classes and instances with special name methods are passed "
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (IterableObject) -> DynamicType
7:     # (Str) -> DynamicType
8:     # (IterableObject, Has__call__) -> DynamicType
9:     # (Str, Has__call__) -> DynamicType
10:     # (AnyType, AnyType) -> DynamicType
11:     # (AnyType, AnyType, Has__call__) -> DynamicType
12:     # (AnyType, AnyType, AnyType) -> DynamicType
13:     # (AnyType, AnyType, AnyType, Has__call__) -> DynamicType
14:     # (AnyType, VarArgs) -> DynamicType
15: 
16:     class Sample:
17:         def __call__(self, x):
18:             return str(x)
19: 
20: 
21:     # Call the builtin with correct parameters
22:     ret = min([1, 2, 3], Sample())
23:     ret = min("str", Sample())
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'min builtin is invoked, but classes and instances with special name methods are passed ')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __call__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__call__'
            module_type_store = module_type_store.open_function_context('__call__', 17, 8, False)
            # Assigning a type to the variable 'self' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Sample.__call__.__dict__.__setitem__('stypy_localization', localization)
            Sample.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Sample.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
            Sample.__call__.__dict__.__setitem__('stypy_function_name', 'Sample.__call__')
            Sample.__call__.__dict__.__setitem__('stypy_param_names_list', ['x'])
            Sample.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
            Sample.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Sample.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
            Sample.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
            Sample.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Sample.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__call__', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '__call__', localization, ['x'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__call__(...)' code ##################

            
            # Call to str(...): (line 18)
            # Processing the call arguments (line 18)
            # Getting the type of 'x' (line 18)
            x_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 23), 'x', False)
            # Processing the call keyword arguments (line 18)
            kwargs_4 = {}
            # Getting the type of 'str' (line 18)
            str_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 19), 'str', False)
            # Calling str(args, kwargs) (line 18)
            str_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 18, 19), str_2, *[x_3], **kwargs_4)
            
            # Assigning a type to the variable 'stypy_return_type' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'stypy_return_type', str_call_result_5)
            
            # ################# End of '__call__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '__call__' in the type store
            # Getting the type of 'stypy_return_type' (line 17)
            stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '__call__'
            return stypy_return_type_6


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

    
    # Assigning a type to the variable 'Sample' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'Sample', Sample)
    
    # Assigning a Call to a Name (line 22):
    
    # Call to min(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 14), list_8, int_9)
    # Adding element type (line 22)
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 14), list_8, int_10)
    # Adding element type (line 22)
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 14), list_8, int_11)
    
    
    # Call to Sample(...): (line 22)
    # Processing the call keyword arguments (line 22)
    kwargs_13 = {}
    # Getting the type of 'Sample' (line 22)
    Sample_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'Sample', False)
    # Calling Sample(args, kwargs) (line 22)
    Sample_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 22, 25), Sample_12, *[], **kwargs_13)
    
    # Processing the call keyword arguments (line 22)
    kwargs_15 = {}
    # Getting the type of 'min' (line 22)
    min_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 10), 'min', False)
    # Calling min(args, kwargs) (line 22)
    min_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 22, 10), min_7, *[list_8, Sample_call_result_14], **kwargs_15)
    
    # Assigning a type to the variable 'ret' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'ret', min_call_result_16)
    
    # Assigning a Call to a Name (line 23):
    
    # Call to min(...): (line 23)
    # Processing the call arguments (line 23)
    str_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 14), 'str', 'str')
    
    # Call to Sample(...): (line 23)
    # Processing the call keyword arguments (line 23)
    kwargs_20 = {}
    # Getting the type of 'Sample' (line 23)
    Sample_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 21), 'Sample', False)
    # Calling Sample(args, kwargs) (line 23)
    Sample_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 23, 21), Sample_19, *[], **kwargs_20)
    
    # Processing the call keyword arguments (line 23)
    kwargs_22 = {}
    # Getting the type of 'min' (line 23)
    min_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'min', False)
    # Calling min(args, kwargs) (line 23)
    min_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), min_17, *[str_18, Sample_call_result_21], **kwargs_22)
    
    # Assigning a type to the variable 'ret' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'ret', min_call_result_23)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
