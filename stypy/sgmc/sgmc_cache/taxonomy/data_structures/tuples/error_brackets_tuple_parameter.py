
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Checking that the [] operation is applicable to a tuple parameter"
4: 
5: if __name__ == '__main__':
6:     it_tuple = tuple(range(5))
7: 
8: 
9:     def func(param):
10:         # Type error
11:         print param["3"]
12: 
13: 
14:     func(it_tuple)
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Checking that the [] operation is applicable to a tuple parameter')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 6):
    
    # Call to tuple(...): (line 6)
    # Processing the call arguments (line 6)
    
    # Call to range(...): (line 6)
    # Processing the call arguments (line 6)
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 27), 'int')
    # Processing the call keyword arguments (line 6)
    kwargs_5 = {}
    # Getting the type of 'range' (line 6)
    range_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 21), 'range', False)
    # Calling range(args, kwargs) (line 6)
    range_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 6, 21), range_3, *[int_4], **kwargs_5)
    
    # Processing the call keyword arguments (line 6)
    kwargs_7 = {}
    # Getting the type of 'tuple' (line 6)
    tuple_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 6)
    tuple_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 6, 15), tuple_2, *[range_call_result_6], **kwargs_7)
    
    # Assigning a type to the variable 'it_tuple' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'it_tuple', tuple_call_result_8)

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 9, 4, False)
        
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
        str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 20), 'str', '3')
        # Getting the type of 'param' (line 11)
        param_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 14), 'param')
        # Obtaining the member '__getitem__' of a type (line 11)
        getitem___11 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 14), param_10, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 11)
        subscript_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 11, 14), getitem___11, str_9)
        
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 9)
        stypy_return_type_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_13

    # Assigning a type to the variable 'func' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'func', func)
    
    # Call to func(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'it_tuple' (line 14)
    it_tuple_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 9), 'it_tuple', False)
    # Processing the call keyword arguments (line 14)
    kwargs_16 = {}
    # Getting the type of 'func' (line 14)
    func_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'func', False)
    # Calling func(args, kwargs) (line 14)
    func_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), func_14, *[it_tuple_15], **kwargs_16)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
