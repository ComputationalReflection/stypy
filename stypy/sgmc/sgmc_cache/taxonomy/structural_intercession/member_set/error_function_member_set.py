
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Set the type of a member of a function object"
3: 
4: if __name__ == '__main__':
5:     import types
6: 
7: 
8:     def dummy_func():
9:         pass
10: 
11: 
12:     def new_method(self, param):
13:         return "new method: " + str(param)
14: 
15: 
16:     met = types.MethodType(new_method, dummy_func)
17:     setattr(dummy_func, '__str__', met)
18:     setattr(dummy_func, '__doc__', 1)
19: 
20:     # Type error
21:     print dummy_func.__str__()
22:     print dummy_func.__str__(3)
23: 
24:     # Type error
25:     print dummy_func.__doc__ + "str"
26: 
27:     print dummy_func.__doc__ / 2
28: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Set the type of a member of a function object')
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
    def new_method(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_method'
        module_type_store = module_type_store.open_function_context('new_method', 12, 4, False)
        
        # Passed parameters checking function
        new_method.stypy_localization = localization
        new_method.stypy_type_of_self = None
        new_method.stypy_type_store = module_type_store
        new_method.stypy_function_name = 'new_method'
        new_method.stypy_param_names_list = ['self', 'param']
        new_method.stypy_varargs_param_name = None
        new_method.stypy_kwargs_param_name = None
        new_method.stypy_call_defaults = defaults
        new_method.stypy_call_varargs = varargs
        new_method.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'new_method', ['self', 'param'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'new_method', localization, ['self', 'param'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'new_method(...)' code ##################

        str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'str', 'new method: ')
        
        # Call to str(...): (line 13)
        # Processing the call arguments (line 13)
        # Getting the type of 'param' (line 13)
        param_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 36), 'param', False)
        # Processing the call keyword arguments (line 13)
        kwargs_6 = {}
        # Getting the type of 'str' (line 13)
        str_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 32), 'str', False)
        # Calling str(args, kwargs) (line 13)
        str_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 13, 32), str_4, *[param_5], **kwargs_6)
        
        # Applying the binary operator '+' (line 13)
        result_add_8 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 15), '+', str_3, str_call_result_7)
        
        # Assigning a type to the variable 'stypy_return_type' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', result_add_8)
        
        # ################# End of 'new_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_method' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_method'
        return stypy_return_type_9

    # Assigning a type to the variable 'new_method' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'new_method', new_method)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to MethodType(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'new_method' (line 16)
    new_method_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 27), 'new_method', False)
    # Getting the type of 'dummy_func' (line 16)
    dummy_func_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 39), 'dummy_func', False)
    # Processing the call keyword arguments (line 16)
    kwargs_14 = {}
    # Getting the type of 'types' (line 16)
    types_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'types', False)
    # Obtaining the member 'MethodType' of a type (line 16)
    MethodType_11 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 10), types_10, 'MethodType')
    # Calling MethodType(args, kwargs) (line 16)
    MethodType_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), MethodType_11, *[new_method_12, dummy_func_13], **kwargs_14)
    
    # Assigning a type to the variable 'met' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'met', MethodType_call_result_15)
    
    # Call to setattr(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'dummy_func' (line 17)
    dummy_func_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'dummy_func', False)
    str_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 24), 'str', '__str__')
    # Getting the type of 'met' (line 17)
    met_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 35), 'met', False)
    # Processing the call keyword arguments (line 17)
    kwargs_20 = {}
    # Getting the type of 'setattr' (line 17)
    setattr_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'setattr', False)
    # Calling setattr(args, kwargs) (line 17)
    setattr_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), setattr_16, *[dummy_func_17, str_18, met_19], **kwargs_20)
    
    
    # Call to setattr(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'dummy_func' (line 18)
    dummy_func_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'dummy_func', False)
    str_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 24), 'str', '__doc__')
    int_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 35), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_26 = {}
    # Getting the type of 'setattr' (line 18)
    setattr_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'setattr', False)
    # Calling setattr(args, kwargs) (line 18)
    setattr_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), setattr_22, *[dummy_func_23, str_24, int_25], **kwargs_26)
    
    
    # Call to __str__(...): (line 21)
    # Processing the call keyword arguments (line 21)
    kwargs_30 = {}
    # Getting the type of 'dummy_func' (line 21)
    dummy_func_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'dummy_func', False)
    # Obtaining the member '__str__' of a type (line 21)
    str___29 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 10), dummy_func_28, '__str__')
    # Calling __str__(args, kwargs) (line 21)
    str___call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), str___29, *[], **kwargs_30)
    
    
    # Call to __str__(...): (line 22)
    # Processing the call arguments (line 22)
    int_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 29), 'int')
    # Processing the call keyword arguments (line 22)
    kwargs_35 = {}
    # Getting the type of 'dummy_func' (line 22)
    dummy_func_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 10), 'dummy_func', False)
    # Obtaining the member '__str__' of a type (line 22)
    str___33 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 10), dummy_func_32, '__str__')
    # Calling __str__(args, kwargs) (line 22)
    str___call_result_36 = invoke(stypy.reporting.localization.Localization(__file__, 22, 10), str___33, *[int_34], **kwargs_35)
    
    # Getting the type of 'dummy_func' (line 25)
    dummy_func_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 10), 'dummy_func')
    # Obtaining the member '__doc__' of a type (line 25)
    doc___38 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 10), dummy_func_37, '__doc__')
    str_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 31), 'str', 'str')
    # Applying the binary operator '+' (line 25)
    result_add_40 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 10), '+', doc___38, str_39)
    
    # Getting the type of 'dummy_func' (line 27)
    dummy_func_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 10), 'dummy_func')
    # Obtaining the member '__doc__' of a type (line 27)
    doc___42 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 10), dummy_func_41, '__doc__')
    int_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 31), 'int')
    # Applying the binary operator 'div' (line 27)
    result_div_44 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 10), 'div', doc___42, int_43)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
