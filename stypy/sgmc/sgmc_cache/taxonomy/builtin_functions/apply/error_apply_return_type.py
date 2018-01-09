
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "apply builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (Has__call__) -> DynamicType
7:     # (Has__call__, <type tuple>) -> DynamicType
8:     # (Has__call__, <type tuple>, <type dict>) -> DynamicType
9: 
10: 
11:     def func(param1, param2):
12:         return param1 + param2
13: 
14: 
15:     # Call the builtin with correct parameters
16:     # Type warning
17:     ret = apply(func, (3, 5))
18: 
19:     ret.unexisting_method()
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'apply builtin is invoked and its return type is used to call an non existing method')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 11, 4, False)
        
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

        # Getting the type of 'param1' (line 12)
        param1_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'param1')
        # Getting the type of 'param2' (line 12)
        param2_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 24), 'param2')
        # Applying the binary operator '+' (line 12)
        result_add_4 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 15), '+', param1_2, param2_3)
        
        # Assigning a type to the variable 'stypy_return_type' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'stypy_return_type', result_add_4)
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 11)
        stypy_return_type_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_5

    # Assigning a type to the variable 'func' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'func', func)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to apply(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'func' (line 17)
    func_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), 'func', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 17)
    tuple_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 17)
    # Adding element type (line 17)
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 23), tuple_8, int_9)
    # Adding element type (line 17)
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 23), tuple_8, int_10)
    
    # Processing the call keyword arguments (line 17)
    kwargs_11 = {}
    # Getting the type of 'apply' (line 17)
    apply_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'apply', False)
    # Calling apply(args, kwargs) (line 17)
    apply_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), apply_6, *[func_7, tuple_8], **kwargs_11)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', apply_call_result_12)
    
    # Call to unexisting_method(...): (line 19)
    # Processing the call keyword arguments (line 19)
    kwargs_15 = {}
    # Getting the type of 'ret' (line 19)
    ret_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'ret', False)
    # Obtaining the member 'unexisting_method' of a type (line 19)
    unexisting_method_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 4), ret_13, 'unexisting_method')
    # Calling unexisting_method(args, kwargs) (line 19)
    unexisting_method_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), unexisting_method_14, *[], **kwargs_15)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
