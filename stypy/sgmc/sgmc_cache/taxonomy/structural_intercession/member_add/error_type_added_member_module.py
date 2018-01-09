
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Collect types of the members added to a module"
3: 
4: if __name__ == '__main__':
5:     import math
6: 
7: 
8:     def new_func():
9:         return "new method"
10: 
11: 
12:     math.new_func = new_func
13:     math.new_attribute = 0.0
14: 
15:     print math.new_attribute
16:     print math.new_func()
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Collect types of the members added to a module')
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

        str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'str', 'new method')
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
    
    # Assigning a Name to a Attribute (line 12):
    # Getting the type of 'new_func' (line 12)
    new_func_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 20), 'new_func')
    # Getting the type of 'math' (line 12)
    math_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'math')
    # Setting the type of the member 'new_func' of a type (line 12)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), math_5, 'new_func', new_func_4)
    
    # Assigning a Num to a Attribute (line 13):
    float_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'float')
    # Getting the type of 'math' (line 13)
    math_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'math')
    # Setting the type of the member 'new_attribute' of a type (line 13)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), math_7, 'new_attribute', float_6)
    # Getting the type of 'math' (line 15)
    math_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'math')
    # Obtaining the member 'new_attribute' of a type (line 15)
    new_attribute_9 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 10), math_8, 'new_attribute')
    
    # Call to new_func(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_12 = {}
    # Getting the type of 'math' (line 16)
    math_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'math', False)
    # Obtaining the member 'new_func' of a type (line 16)
    new_func_11 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 10), math_10, 'new_func')
    # Calling new_func(args, kwargs) (line 16)
    new_func_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), new_func_11, *[], **kwargs_12)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
