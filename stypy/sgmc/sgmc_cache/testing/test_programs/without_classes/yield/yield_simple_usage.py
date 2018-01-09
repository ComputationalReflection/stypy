
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: def createGenerator():
4:     mylist = range(3)
5:     for i in mylist:
6:        yield i*i
7: 
8: 
9: x = createGenerator()
10: 
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def createGenerator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'createGenerator'
    module_type_store = module_type_store.open_function_context('createGenerator', 3, 0, False)
    
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

    
    # Assigning a Call to a Name (line 4):
    
    # Call to range(...): (line 4)
    # Processing the call arguments (line 4)
    int_6508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 19), 'int')
    # Processing the call keyword arguments (line 4)
    kwargs_6509 = {}
    # Getting the type of 'range' (line 4)
    range_6507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 13), 'range', False)
    # Calling range(args, kwargs) (line 4)
    range_call_result_6510 = invoke(stypy.reporting.localization.Localization(__file__, 4, 13), range_6507, *[int_6508], **kwargs_6509)
    
    # Assigning a type to the variable 'mylist' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'mylist', range_call_result_6510)
    
    # Getting the type of 'mylist' (line 5)
    mylist_6511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 13), 'mylist')
    # Testing the type of a for loop iterable (line 5)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 5, 4), mylist_6511)
    # Getting the type of the for loop variable (line 5)
    for_loop_var_6512 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 5, 4), mylist_6511)
    # Assigning a type to the variable 'i' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'i', for_loop_var_6512)
    # SSA begins for a for statement (line 5)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    # Creating a generator
    # Getting the type of 'i' (line 6)
    i_6513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'i')
    # Getting the type of 'i' (line 6)
    i_6514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), 'i')
    # Applying the binary operator '*' (line 6)
    result_mul_6515 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 13), '*', i_6513, i_6514)
    
    GeneratorType_6516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 7), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 7), GeneratorType_6516, result_mul_6515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 7), 'stypy_return_type', GeneratorType_6516)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'createGenerator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'createGenerator' in the type store
    # Getting the type of 'stypy_return_type' (line 3)
    stypy_return_type_6517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6517)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'createGenerator'
    return stypy_return_type_6517

# Assigning a type to the variable 'createGenerator' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'createGenerator', createGenerator)

# Assigning a Call to a Name (line 9):

# Call to createGenerator(...): (line 9)
# Processing the call keyword arguments (line 9)
kwargs_6519 = {}
# Getting the type of 'createGenerator' (line 9)
createGenerator_6518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'createGenerator', False)
# Calling createGenerator(args, kwargs) (line 9)
createGenerator_call_result_6520 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), createGenerator_6518, *[], **kwargs_6519)

# Assigning a type to the variable 'x' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'x', createGenerator_call_result_6520)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
