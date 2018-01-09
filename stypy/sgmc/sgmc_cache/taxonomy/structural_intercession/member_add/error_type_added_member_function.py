
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Collect types of the members added to a function object"
3: 
4: if __name__ == '__main__':
5:     import types
6: 
7: 
8:     def dummy_func():
9:         pass
10: 
11: 
12:     def new_class_method(self):
13:         return "new method"
14: 
15: 
16:     def new_class_method2(self):
17:         return "new method 2"
18: 
19: 
20:     dummy_func.new_class_method = types.MethodType(new_class_method, dummy_func)
21:     dummy_func.new_class_attribute = 0.0
22: 
23:     print dummy_func.new_class_attribute
24:     print dummy_func.new_class_method()
25: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Collect types of the members added to a function object')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 4))
    
    # 'import types' statement (line 5)
    import types

    import_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'types', types, module_type_store)
    

    @norecursion
    def dummy_func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dummy_func'
        module_type_store = module_type_store.open_function_context('dummy_func', 8, 4, False)
        
        # Passed parameters checking function
        dummy_func.stypy_localization = localization
        dummy_func.stypy_type_of_self = None
        dummy_func.stypy_type_store = module_type_store
        dummy_func.stypy_function_name = 'dummy_func'
        dummy_func.stypy_param_names_list = []
        dummy_func.stypy_varargs_param_name = None
        dummy_func.stypy_kwargs_param_name = None
        dummy_func.stypy_call_defaults = defaults
        dummy_func.stypy_call_varargs = varargs
        dummy_func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'dummy_func', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dummy_func', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dummy_func(...)' code ##################

        pass
        
        # ################# End of 'dummy_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dummy_func' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dummy_func'
        return stypy_return_type_2

    # Assigning a type to the variable 'dummy_func' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'dummy_func', dummy_func)

    @norecursion
    def new_class_method(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_class_method'
        module_type_store = module_type_store.open_function_context('new_class_method', 12, 4, False)
        
        # Passed parameters checking function
        new_class_method.stypy_localization = localization
        new_class_method.stypy_type_of_self = None
        new_class_method.stypy_type_store = module_type_store
        new_class_method.stypy_function_name = 'new_class_method'
        new_class_method.stypy_param_names_list = ['self']
        new_class_method.stypy_varargs_param_name = None
        new_class_method.stypy_kwargs_param_name = None
        new_class_method.stypy_call_defaults = defaults
        new_class_method.stypy_call_varargs = varargs
        new_class_method.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'new_class_method', ['self'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'new_class_method', localization, ['self'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'new_class_method(...)' code ##################

        str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'new method')
        # Assigning a type to the variable 'stypy_return_type' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str_3)
        
        # ################# End of 'new_class_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_class_method' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_class_method'
        return stypy_return_type_4

    # Assigning a type to the variable 'new_class_method' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'new_class_method', new_class_method)

    @norecursion
    def new_class_method2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_class_method2'
        module_type_store = module_type_store.open_function_context('new_class_method2', 16, 4, False)
        
        # Passed parameters checking function
        new_class_method2.stypy_localization = localization
        new_class_method2.stypy_type_of_self = None
        new_class_method2.stypy_type_store = module_type_store
        new_class_method2.stypy_function_name = 'new_class_method2'
        new_class_method2.stypy_param_names_list = ['self']
        new_class_method2.stypy_varargs_param_name = None
        new_class_method2.stypy_kwargs_param_name = None
        new_class_method2.stypy_call_defaults = defaults
        new_class_method2.stypy_call_varargs = varargs
        new_class_method2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'new_class_method2', ['self'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'new_class_method2', localization, ['self'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'new_class_method2(...)' code ##################

        str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 15), 'str', 'new method 2')
        # Assigning a type to the variable 'stypy_return_type' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'stypy_return_type', str_5)
        
        # ################# End of 'new_class_method2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_class_method2' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_class_method2'
        return stypy_return_type_6

    # Assigning a type to the variable 'new_class_method2' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'new_class_method2', new_class_method2)
    
    # Assigning a Call to a Attribute (line 20):
    
    # Call to MethodType(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'new_class_method' (line 20)
    new_class_method_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 51), 'new_class_method', False)
    # Getting the type of 'dummy_func' (line 20)
    dummy_func_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 69), 'dummy_func', False)
    # Processing the call keyword arguments (line 20)
    kwargs_11 = {}
    # Getting the type of 'types' (line 20)
    types_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 34), 'types', False)
    # Obtaining the member 'MethodType' of a type (line 20)
    MethodType_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 34), types_7, 'MethodType')
    # Calling MethodType(args, kwargs) (line 20)
    MethodType_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 20, 34), MethodType_8, *[new_class_method_9, dummy_func_10], **kwargs_11)
    
    # Getting the type of 'dummy_func' (line 20)
    dummy_func_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'dummy_func')
    # Setting the type of the member 'new_class_method' of a type (line 20)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), dummy_func_13, 'new_class_method', MethodType_call_result_12)
    
    # Assigning a Num to a Attribute (line 21):
    float_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 37), 'float')
    # Getting the type of 'dummy_func' (line 21)
    dummy_func_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'dummy_func')
    # Setting the type of the member 'new_class_attribute' of a type (line 21)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 4), dummy_func_15, 'new_class_attribute', float_14)
    # Getting the type of 'dummy_func' (line 23)
    dummy_func_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'dummy_func')
    # Obtaining the member 'new_class_attribute' of a type (line 23)
    new_class_attribute_17 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 10), dummy_func_16, 'new_class_attribute')
    
    # Call to new_class_method(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_20 = {}
    # Getting the type of 'dummy_func' (line 24)
    dummy_func_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 10), 'dummy_func', False)
    # Obtaining the member 'new_class_method' of a type (line 24)
    new_class_method_19 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 10), dummy_func_18, 'new_class_method')
    # Calling new_class_method(args, kwargs) (line 24)
    new_class_method_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 24, 10), new_class_method_19, *[], **kwargs_20)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
