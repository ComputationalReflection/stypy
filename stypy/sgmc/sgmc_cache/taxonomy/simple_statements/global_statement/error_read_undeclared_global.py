
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Write an undeclared global variable"
4: 
5: if __name__ == '__main__':
6:     def func():
7:         global g_var
8: 
9:         g_var = 3
10: 
11: 
12:     func()
13: 
14:     print g_var
15:     # Type error
16:     print g_var + "str"
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Write an undeclared global variable')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 6, 4, False)
        
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

        # Marking variables as global (line 7)
        module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 7, 8), 'g_var')
        
        # Assigning a Num to a Name (line 9):
        int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 16), 'int')
        # Assigning a type to the variable 'g_var' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'g_var', int_2)
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_3

    # Assigning a type to the variable 'func' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'func', func)
    
    # Call to func(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_5 = {}
    # Getting the type of 'func' (line 12)
    func_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'func', False)
    # Calling func(args, kwargs) (line 12)
    func_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), func_4, *[], **kwargs_5)
    
    # Getting the type of 'g_var' (line 14)
    g_var_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'g_var')
    # Getting the type of 'g_var' (line 16)
    g_var_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'g_var')
    str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 18), 'str', 'str')
    # Applying the binary operator '+' (line 16)
    result_add_10 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 10), '+', g_var_8, str_9)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
