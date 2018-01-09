
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Single-element non-iterable generator"
4: 
5: if __name__ == '__main__':
6: 
7:     def createGenerator2():
8:         yield "str"
9: 
10: 
11:     for i in createGenerator2():
12:         print i + "str"
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Single-element non-iterable generator')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def createGenerator2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'createGenerator2'
        module_type_store = module_type_store.open_function_context('createGenerator2', 7, 4, False)
        
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
        str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'str', 'str')
        GeneratorType_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 8), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 8), GeneratorType_3, str_2)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type', GeneratorType_3)
        
        # ################# End of 'createGenerator2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'createGenerator2' in the type store
        # Getting the type of 'stypy_return_type' (line 7)
        stypy_return_type_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'createGenerator2'
        return stypy_return_type_4

    # Assigning a type to the variable 'createGenerator2' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'createGenerator2', createGenerator2)
    
    
    # Call to createGenerator2(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_6 = {}
    # Getting the type of 'createGenerator2' (line 11)
    createGenerator2_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 13), 'createGenerator2', False)
    # Calling createGenerator2(args, kwargs) (line 11)
    createGenerator2_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 11, 13), createGenerator2_5, *[], **kwargs_6)
    
    # Testing the type of a for loop iterable (line 11)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 11, 4), createGenerator2_call_result_7)
    # Getting the type of the for loop variable (line 11)
    for_loop_var_8 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 11, 4), createGenerator2_call_result_7)
    # Assigning a type to the variable 'i' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'i', for_loop_var_8)
    # SSA begins for a for statement (line 11)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    # Getting the type of 'i' (line 12)
    i_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'i')
    str_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'str', 'str')
    # Applying the binary operator '+' (line 12)
    result_add_11 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 14), '+', i_9, str_10)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
