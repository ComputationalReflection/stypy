
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Checking that the [] operation is applicable to a dict parameter"
4: 
5: if __name__ == '__main__':
6:     d = {
7:         "one": 1,
8:         "two": 2,
9:         "three": 3,
10:     }
11: 
12: 
13:     def func(param):
14:         # Type error
15:         print param[3]
16: 
17: 
18:     func(d)
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Checking that the [] operation is applicable to a dict parameter')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Dict to a Name (line 6):
    
    # Obtaining an instance of the builtin type 'dict' (line 6)
    dict_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 6)
    # Adding element type (key, value) (line 6)
    str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 8), 'str', 'one')
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 8), dict_2, (str_3, int_4))
    # Adding element type (key, value) (line 6)
    str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 8), 'str', 'two')
    int_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 15), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 8), dict_2, (str_5, int_6))
    # Adding element type (key, value) (line 6)
    str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 8), 'str', 'three')
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 17), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 8), dict_2, (str_7, int_8))
    
    # Assigning a type to the variable 'd' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'd', dict_2)

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 13, 4, False)
        
        # Passed parameters checking function
        func.stypy_localization = localization
        func.stypy_type_of_self = None
        func.stypy_type_store = module_type_store
        func.stypy_function_name = 'func'
        func.stypy_param_names_list = ['param']
        func.stypy_varargs_param_name = None
        func.stypy_kwargs_param_name = None
        func.stypy_call_defaults = defaults
        func.stypy_call_varargs = varargs
        func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func', ['param'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func', localization, ['param'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func(...)' code ##################

        
        # Obtaining the type of the subscript
        int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'int')
        # Getting the type of 'param' (line 15)
        param_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 14), 'param')
        # Obtaining the member '__getitem__' of a type (line 15)
        getitem___11 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 14), param_10, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 15)
        subscript_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 15, 14), getitem___11, int_9)
        
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_13

    # Assigning a type to the variable 'func' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'func', func)
    
    # Call to func(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'd' (line 18)
    d_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 9), 'd', False)
    # Processing the call keyword arguments (line 18)
    kwargs_16 = {}
    # Getting the type of 'func' (line 18)
    func_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'func', False)
    # Calling func(args, kwargs) (line 18)
    func_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), func_14, *[d_15], **kwargs_16)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
