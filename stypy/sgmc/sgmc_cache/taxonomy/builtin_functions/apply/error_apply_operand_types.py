
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "apply builtin is invoked, but incorrect parameter types are passed"
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
14:     # Call the builtin with correct parameters
15:     # Type warning
16:     ret = apply(func, (3, 5))
17: 
18:     # Call the builtin with incorrect types of parameters
19:     # Type error
20:     ret = apply(3, None)
21: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'apply builtin is invoked, but incorrect parameter types are passed')
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
    
    # Assigning a Call to a Name (line 16):
    
    # Call to apply(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'func' (line 16)
    func_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'func', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 16)
    tuple_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 16)
    # Adding element type (line 16)
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 23), tuple_8, int_9)
    # Adding element type (line 16)
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 23), tuple_8, int_10)
    
    # Processing the call keyword arguments (line 16)
    kwargs_11 = {}
    # Getting the type of 'apply' (line 16)
    apply_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'apply', False)
    # Calling apply(args, kwargs) (line 16)
    apply_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), apply_6, *[func_7, tuple_8], **kwargs_11)
    
    # Assigning a type to the variable 'ret' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'ret', apply_call_result_12)
    
    # Assigning a Call to a Name (line 20):
    
    # Call to apply(...): (line 20)
    # Processing the call arguments (line 20)
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 16), 'int')
    # Getting the type of 'None' (line 20)
    None_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 19), 'None', False)
    # Processing the call keyword arguments (line 20)
    kwargs_16 = {}
    # Getting the type of 'apply' (line 20)
    apply_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'apply', False)
    # Calling apply(args, kwargs) (line 20)
    apply_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 20, 10), apply_13, *[int_14, None_15], **kwargs_16)
    
    # Assigning a type to the variable 'ret' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'ret', apply_call_result_17)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
