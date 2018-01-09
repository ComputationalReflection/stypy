
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Checking deletion of unmodifiable objects"
3: 
4: if __name__ == '__main__':
5:     def dummy_func():
6:         pass
7: 
8: 
9:     # Type error
10:     delattr(dummy_func, 'func_name')
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Checking deletion of unmodifiable objects')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def dummy_func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dummy_func'
        module_type_store = module_type_store.open_function_context('dummy_func', 5, 4, False)
        
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
        # Getting the type of 'stypy_return_type' (line 5)
        stypy_return_type_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dummy_func'
        return stypy_return_type_2

    # Assigning a type to the variable 'dummy_func' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'dummy_func', dummy_func)
    
    # Call to delattr(...): (line 10)
    # Processing the call arguments (line 10)
    # Getting the type of 'dummy_func' (line 10)
    dummy_func_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'dummy_func', False)
    str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 24), 'str', 'func_name')
    # Processing the call keyword arguments (line 10)
    kwargs_6 = {}
    # Getting the type of 'delattr' (line 10)
    delattr_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'delattr', False)
    # Calling delattr(args, kwargs) (line 10)
    delattr_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), delattr_3, *[dummy_func_4, str_5], **kwargs_6)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
