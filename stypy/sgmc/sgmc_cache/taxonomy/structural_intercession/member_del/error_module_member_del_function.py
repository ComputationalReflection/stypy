
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Del a member of a module inside a function"
3: 
4: if __name__ == '__main__':
5:     import math
6: 
7: 
8:     def func():
9:         delattr(math, 'cos')
10: 
11: 
12:     func()
13:     # Type error
14:     print math.cos(3)
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Del a member of a module inside a function')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 4))
    
    # 'import math' statement (line 5)
    import math

    import_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'math', math, module_type_store)
    

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 8, 4, False)
        
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

        
        # Call to delattr(...): (line 9)
        # Processing the call arguments (line 9)
        # Getting the type of 'math' (line 9)
        math_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 16), 'math', False)
        str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 22), 'str', 'cos')
        # Processing the call keyword arguments (line 9)
        kwargs_5 = {}
        # Getting the type of 'delattr' (line 9)
        delattr_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'delattr', False)
        # Calling delattr(args, kwargs) (line 9)
        delattr_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 9, 8), delattr_2, *[math_3, str_4], **kwargs_5)
        
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_7

    # Assigning a type to the variable 'func' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'func', func)
    
    # Call to func(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_9 = {}
    # Getting the type of 'func' (line 12)
    func_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'func', False)
    # Calling func(args, kwargs) (line 12)
    func_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), func_8, *[], **kwargs_9)
    
    
    # Call to cos(...): (line 14)
    # Processing the call arguments (line 14)
    int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 19), 'int')
    # Processing the call keyword arguments (line 14)
    kwargs_14 = {}
    # Getting the type of 'math' (line 14)
    math_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'math', False)
    # Obtaining the member 'cos' of a type (line 14)
    cos_12 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 10), math_11, 'cos')
    # Calling cos(args, kwargs) (line 14)
    cos_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), cos_12, *[int_13], **kwargs_14)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
