
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Wrong usage of the generator type"
4: 
5: if __name__ == '__main__':
6:     def createGenerator2():
7:         yield "str"
8: 
9: 
10:     # Type error
11:     print len(createGenerator2())
12:     # Type error
13:     print createGenerator2().undefined
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Wrong usage of the generator type')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def createGenerator2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'createGenerator2'
        module_type_store = module_type_store.open_function_context('createGenerator2', 6, 4, False)
        
        # Passed parameters checking function
        createGenerator2.stypy_localization = localization
        createGenerator2.stypy_type_of_self = None
        createGenerator2.stypy_type_store = module_type_store
        createGenerator2.stypy_function_name = 'createGenerator2'
        createGenerator2.stypy_param_names_list = []
        createGenerator2.stypy_varargs_param_name = None
        createGenerator2.stypy_kwargs_param_name = None
        createGenerator2.stypy_call_defaults = defaults
        createGenerator2.stypy_call_varargs = varargs
        createGenerator2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'createGenerator2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'createGenerator2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'createGenerator2(...)' code ##################

        # Creating a generator
        str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'str', 'str')
        GeneratorType_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 8), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 8), GeneratorType_3, str_2)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type', GeneratorType_3)
        
        # ################# End of 'createGenerator2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'createGenerator2' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'createGenerator2'
        return stypy_return_type_4

    # Assigning a type to the variable 'createGenerator2' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'createGenerator2', createGenerator2)
    
    # Call to len(...): (line 11)
    # Processing the call arguments (line 11)
    
    # Call to createGenerator2(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_7 = {}
    # Getting the type of 'createGenerator2' (line 11)
    createGenerator2_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 14), 'createGenerator2', False)
    # Calling createGenerator2(args, kwargs) (line 11)
    createGenerator2_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 11, 14), createGenerator2_6, *[], **kwargs_7)
    
    # Processing the call keyword arguments (line 11)
    kwargs_9 = {}
    # Getting the type of 'len' (line 11)
    len_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'len', False)
    # Calling len(args, kwargs) (line 11)
    len_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), len_5, *[createGenerator2_call_result_8], **kwargs_9)
    
    
    # Call to createGenerator2(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_12 = {}
    # Getting the type of 'createGenerator2' (line 13)
    createGenerator2_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'createGenerator2', False)
    # Calling createGenerator2(args, kwargs) (line 13)
    createGenerator2_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), createGenerator2_11, *[], **kwargs_12)
    
    # Obtaining the member 'undefined' of a type (line 13)
    undefined_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 10), createGenerator2_call_result_13, 'undefined')


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
