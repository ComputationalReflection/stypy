
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Set the type of a member of a module inside a function"
3: 
4: if __name__ == '__main__':
5:     import math
6: 
7: 
8:     def new_func():
9:         return "new function"
10: 
11: 
12:     def func():
13:         setattr(math, 'cos', new_func)
14:         setattr(math, 'pi', "str")
15: 
16: 
17:     func()
18: 
19:     print math.cos()
20:     print len(math.pi)
21: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Set the type of a member of a module inside a function')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 4))
    
    # 'import math' statement (line 5)
    import math

    import_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'math', math, module_type_store)
    

    @norecursion
    def new_func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'new_func'
        module_type_store = module_type_store.open_function_context('new_func', 8, 4, False)
        
        # Passed parameters checking function
        new_func.stypy_localization = localization
        new_func.stypy_type_of_self = None
        new_func.stypy_type_store = module_type_store
        new_func.stypy_function_name = 'new_func'
        new_func.stypy_param_names_list = []
        new_func.stypy_varargs_param_name = None
        new_func.stypy_kwargs_param_name = None
        new_func.stypy_call_defaults = defaults
        new_func.stypy_call_varargs = varargs
        new_func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'new_func', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'new_func', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'new_func(...)' code ##################

        str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'str', 'new function')
        # Assigning a type to the variable 'stypy_return_type' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'stypy_return_type', str_2)
        
        # ################# End of 'new_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'new_func' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'new_func'
        return stypy_return_type_3

    # Assigning a type to the variable 'new_func' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'new_func', new_func)

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 12, 4, False)
        
        # Passed parameters checking function
        func.stypy_localization = localization
        func.stypy_type_of_self = None
        func.stypy_type_store = module_type_store
        func.stypy_function_name = 'func'
        func.stypy_param_names_list = []
        func.stypy_varargs_param_name = None
        func.stypy_kwargs_param_name = None
        func.stypy_call_defaults = defaults
        func.stypy_call_varargs = varargs
        func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func(...)' code ##################

        
        # Call to setattr(...): (line 13)
        # Processing the call arguments (line 13)
        # Getting the type of 'math' (line 13)
        math_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'math', False)
        str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'str', 'cos')
        # Getting the type of 'new_func' (line 13)
        new_func_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 29), 'new_func', False)
        # Processing the call keyword arguments (line 13)
        kwargs_8 = {}
        # Getting the type of 'setattr' (line 13)
        setattr_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'setattr', False)
        # Calling setattr(args, kwargs) (line 13)
        setattr_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), setattr_4, *[math_5, str_6, new_func_7], **kwargs_8)
        
        
        # Call to setattr(...): (line 14)
        # Processing the call arguments (line 14)
        # Getting the type of 'math' (line 14)
        math_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'math', False)
        str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 22), 'str', 'pi')
        str_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 28), 'str', 'str')
        # Processing the call keyword arguments (line 14)
        kwargs_14 = {}
        # Getting the type of 'setattr' (line 14)
        setattr_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'setattr', False)
        # Calling setattr(args, kwargs) (line 14)
        setattr_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), setattr_10, *[math_11, str_12, str_13], **kwargs_14)
        
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_16

    # Assigning a type to the variable 'func' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'func', func)
    
    # Call to func(...): (line 17)
    # Processing the call keyword arguments (line 17)
    kwargs_18 = {}
    # Getting the type of 'func' (line 17)
    func_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'func', False)
    # Calling func(args, kwargs) (line 17)
    func_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), func_17, *[], **kwargs_18)
    
    
    # Call to cos(...): (line 19)
    # Processing the call keyword arguments (line 19)
    kwargs_22 = {}
    # Getting the type of 'math' (line 19)
    math_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'math', False)
    # Obtaining the member 'cos' of a type (line 19)
    cos_21 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 10), math_20, 'cos')
    # Calling cos(args, kwargs) (line 19)
    cos_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 19, 10), cos_21, *[], **kwargs_22)
    
    
    # Call to len(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'math' (line 20)
    math_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 14), 'math', False)
    # Obtaining the member 'pi' of a type (line 20)
    pi_26 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 14), math_25, 'pi')
    # Processing the call keyword arguments (line 20)
    kwargs_27 = {}
    # Getting the type of 'len' (line 20)
    len_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'len', False)
    # Calling len(args, kwargs) (line 20)
    len_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 20, 10), len_24, *[pi_26], **kwargs_27)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
