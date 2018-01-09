
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "apply builtin is invoked, but a class is used instead of an instance"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Has__call__) -> DynamicType
7:     # (Has__call__, <type tuple>) -> DynamicType
8:     # (Has__call__, <type tuple>, <type dict>) -> DynamicType
9: 
10:     def func(param1, param2):
11:         return param1 + param2
12: 
13: 
14:     # Type error
15:     ret = apply(func, tuple, dict)
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'apply builtin is invoked, but a class is used instead of an instance')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 10, 4, False)
        
        # Passed parameters checking function
        func.stypy_localization = localization
        func.stypy_type_of_self = None
        func.stypy_type_store = module_type_store
        func.stypy_function_name = 'func'
        func.stypy_param_names_list = ['param1', 'param2']
        func.stypy_varargs_param_name = None
        func.stypy_kwargs_param_name = None
        func.stypy_call_defaults = defaults
        func.stypy_call_varargs = varargs
        func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func', ['param1', 'param2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func', localization, ['param1', 'param2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func(...)' code ##################

        # Getting the type of 'param1' (line 11)
        param1_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'param1')
        # Getting the type of 'param2' (line 11)
        param2_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 24), 'param2')
        # Applying the binary operator '+' (line 11)
        result_add_4 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 15), '+', param1_2, param2_3)
        
        # Assigning a type to the variable 'stypy_return_type' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type', result_add_4)
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 10)
        stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_5

    # Assigning a type to the variable 'func' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'func', func)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to apply(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'func' (line 15)
    func_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 16), 'func', False)
    # Getting the type of 'tuple' (line 15)
    tuple_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 22), 'tuple', False)
    # Getting the type of 'dict' (line 15)
    dict_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 29), 'dict', False)
    # Processing the call keyword arguments (line 15)
    kwargs_10 = {}
    # Getting the type of 'apply' (line 15)
    apply_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'apply', False)
    # Calling apply(args, kwargs) (line 15)
    apply_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), apply_6, *[func_7, tuple_8, dict_9], **kwargs_10)
    
    # Assigning a type to the variable 'ret' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'ret', apply_call_result_11)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
