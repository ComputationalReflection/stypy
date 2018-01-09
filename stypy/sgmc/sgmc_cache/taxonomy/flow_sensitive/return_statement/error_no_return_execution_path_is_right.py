
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "No execution path has an execution flow free of type errors"
3: 
4: if __name__ == '__main__':
5: 
6:     def problematic_get():
7:         if True:
8:             return "hi"
9:         else:
10:             return [1, 2]
11: 
12: 
13:     # Type error
14:     x = problematic_get() / 3
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'No execution path has an execution flow free of type errors')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def problematic_get(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'problematic_get'
        module_type_store = module_type_store.open_function_context('problematic_get', 6, 4, False)
        
        # Passed parameters checking function
        problematic_get.stypy_localization = localization
        problematic_get.stypy_type_of_self = None
        problematic_get.stypy_type_store = module_type_store
        problematic_get.stypy_function_name = 'problematic_get'
        problematic_get.stypy_param_names_list = []
        problematic_get.stypy_varargs_param_name = None
        problematic_get.stypy_kwargs_param_name = None
        problematic_get.stypy_call_defaults = defaults
        problematic_get.stypy_call_varargs = varargs
        problematic_get.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'problematic_get', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'problematic_get', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'problematic_get(...)' code ##################

        
        # Getting the type of 'True' (line 7)
        True_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'True')
        # Testing the type of an if condition (line 7)
        if_condition_3 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 7, 8), True_2)
        # Assigning a type to the variable 'if_condition_3' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'if_condition_3', if_condition_3)
        # SSA begins for if statement (line 7)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 19), 'str', 'hi')
        # Assigning a type to the variable 'stypy_return_type' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'stypy_return_type', str_4)
        # SSA branch for the else part of an if statement (line 7)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining an instance of the builtin type 'list' (line 10)
        list_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 10)
        # Adding element type (line 10)
        int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 19), list_5, int_6)
        # Adding element type (line 10)
        int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 19), list_5, int_7)
        
        # Assigning a type to the variable 'stypy_return_type' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'stypy_return_type', list_5)
        # SSA join for if statement (line 7)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'problematic_get(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'problematic_get' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'problematic_get'
        return stypy_return_type_8

    # Assigning a type to the variable 'problematic_get' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'problematic_get', problematic_get)
    
    # Assigning a BinOp to a Name (line 14):
    
    # Call to problematic_get(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_10 = {}
    # Getting the type of 'problematic_get' (line 14)
    problematic_get_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'problematic_get', False)
    # Calling problematic_get(args, kwargs) (line 14)
    problematic_get_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), problematic_get_9, *[], **kwargs_10)
    
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 28), 'int')
    # Applying the binary operator 'div' (line 14)
    result_div_13 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 8), 'div', problematic_get_call_result_11, int_12)
    
    # Assigning a type to the variable 'x' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'x', result_div_13)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
