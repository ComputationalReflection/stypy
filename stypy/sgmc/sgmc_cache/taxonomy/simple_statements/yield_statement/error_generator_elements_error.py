
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Error creating the generator elements"
4: 
5: if __name__ == '__main__':
6: 
7:     def createGenerator():
8:         mylist = range(3)
9:         for i in mylist:
10:             # Type error
11:             yield str(i) + i
12: 
13: 
14:     for i in createGenerator():
15:         print i + "str"
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Error creating the generator elements')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def createGenerator(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'createGenerator'
        module_type_store = module_type_store.open_function_context('createGenerator', 7, 4, False)
        
        # Passed parameters checking function
        createGenerator.stypy_localization = localization
        createGenerator.stypy_type_of_self = None
        createGenerator.stypy_type_store = module_type_store
        createGenerator.stypy_function_name = 'createGenerator'
        createGenerator.stypy_param_names_list = []
        createGenerator.stypy_varargs_param_name = None
        createGenerator.stypy_kwargs_param_name = None
        createGenerator.stypy_call_defaults = defaults
        createGenerator.stypy_call_varargs = varargs
        createGenerator.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'createGenerator', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'createGenerator', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'createGenerator(...)' code ##################

        
        # Assigning a Call to a Name (line 8):
        
        # Call to range(...): (line 8)
        # Processing the call arguments (line 8)
        int_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 23), 'int')
        # Processing the call keyword arguments (line 8)
        kwargs_4 = {}
        # Getting the type of 'range' (line 8)
        range_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 17), 'range', False)
        # Calling range(args, kwargs) (line 8)
        range_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 8, 17), range_2, *[int_3], **kwargs_4)
        
        # Assigning a type to the variable 'mylist' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'mylist', range_call_result_5)
        
        # Getting the type of 'mylist' (line 9)
        mylist_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 17), 'mylist')
        # Testing the type of a for loop iterable (line 9)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 9, 8), mylist_6)
        # Getting the type of the for loop variable (line 9)
        for_loop_var_7 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 9, 8), mylist_6)
        # Assigning a type to the variable 'i' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'i', for_loop_var_7)
        # SSA begins for a for statement (line 9)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        # Creating a generator
        
        # Call to str(...): (line 11)
        # Processing the call arguments (line 11)
        # Getting the type of 'i' (line 11)
        i_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 22), 'i', False)
        # Processing the call keyword arguments (line 11)
        kwargs_10 = {}
        # Getting the type of 'str' (line 11)
        str_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 18), 'str', False)
        # Calling str(args, kwargs) (line 11)
        str_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 11, 18), str_8, *[i_9], **kwargs_10)
        
        # Getting the type of 'i' (line 11)
        i_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 27), 'i')
        # Applying the binary operator '+' (line 11)
        result_add_13 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 18), '+', str_call_result_11, i_12)
        
        GeneratorType_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 12), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 12), GeneratorType_14, result_add_13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'stypy_return_type', GeneratorType_14)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'createGenerator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'createGenerator' in the type store
        # Getting the type of 'stypy_return_type' (line 7)
        stypy_return_type_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_15)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'createGenerator'
        return stypy_return_type_15

    # Assigning a type to the variable 'createGenerator' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'createGenerator', createGenerator)
    
    
    # Call to createGenerator(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_17 = {}
    # Getting the type of 'createGenerator' (line 14)
    createGenerator_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 13), 'createGenerator', False)
    # Calling createGenerator(args, kwargs) (line 14)
    createGenerator_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 14, 13), createGenerator_16, *[], **kwargs_17)
    
    # Testing the type of a for loop iterable (line 14)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 14, 4), createGenerator_call_result_18)
    # Getting the type of the for loop variable (line 14)
    for_loop_var_19 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 14, 4), createGenerator_call_result_18)
    # Assigning a type to the variable 'i' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'i', for_loop_var_19)
    # SSA begins for a for statement (line 14)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    # Getting the type of 'i' (line 15)
    i_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 14), 'i')
    str_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 18), 'str', 'str')
    # Applying the binary operator '+' (line 15)
    result_add_22 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 14), '+', i_20, str_21)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
