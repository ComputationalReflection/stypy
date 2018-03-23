
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from mm import mastermind
2: 
3: ''' copyright Sean McCarthy, license GPL v2 or later '''
4: 
5: 
6: def main():
7:     mastermind.main()
8: 
9: 
10: def run():
11:     for i in range(100):
12:         main()
13:     return True
14: 
15: 
16: run()
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from mm import mastermind' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/benchmark_suite/shedskin/')
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'mm')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'mm', sys_modules_2.module_type_store, module_type_store, ['mastermind'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_2, sys_modules_2.module_type_store, module_type_store)
    else:
        from mm import mastermind

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'mm', None, module_type_store, ['mastermind'], [mastermind])

else:
    # Assigning a type to the variable 'mm' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'mm', import_1)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/benchmark_suite/shedskin/')

str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 0), 'str', ' copyright Sean McCarthy, license GPL v2 or later ')

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 6, 0, False)
    
    # Passed parameters checking function
    main.stypy_localization = localization
    main.stypy_type_of_self = None
    main.stypy_type_store = module_type_store
    main.stypy_function_name = 'main'
    main.stypy_param_names_list = []
    main.stypy_varargs_param_name = None
    main.stypy_kwargs_param_name = None
    main.stypy_call_defaults = defaults
    main.stypy_call_varargs = varargs
    main.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'main', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'main', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'main(...)' code ##################

    
    # Call to main(...): (line 7)
    # Processing the call keyword arguments (line 7)
    kwargs_6 = {}
    # Getting the type of 'mastermind' (line 7)
    mastermind_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'mastermind', False)
    # Obtaining the member 'main' of a type (line 7)
    main_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), mastermind_4, 'main')
    # Calling main(args, kwargs) (line 7)
    main_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), main_5, *[], **kwargs_6)
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_8

# Assigning a type to the variable 'main' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'main', main)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 10, 0, False)
    
    # Passed parameters checking function
    run.stypy_localization = localization
    run.stypy_type_of_self = None
    run.stypy_type_store = module_type_store
    run.stypy_function_name = 'run'
    run.stypy_param_names_list = []
    run.stypy_varargs_param_name = None
    run.stypy_kwargs_param_name = None
    run.stypy_call_defaults = defaults
    run.stypy_call_varargs = varargs
    run.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'run', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'run', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'run(...)' code ##################

    
    
    # Call to range(...): (line 11)
    # Processing the call arguments (line 11)
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 19), 'int')
    # Processing the call keyword arguments (line 11)
    kwargs_11 = {}
    # Getting the type of 'range' (line 11)
    range_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 13), 'range', False)
    # Calling range(args, kwargs) (line 11)
    range_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 11, 13), range_9, *[int_10], **kwargs_11)
    
    # Assigning a type to the variable 'range_call_result_12' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'range_call_result_12', range_call_result_12)
    # Testing if the for loop is going to be iterated (line 11)
    # Testing the type of a for loop iterable (line 11)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 11, 4), range_call_result_12)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 11, 4), range_call_result_12):
        # Getting the type of the for loop variable (line 11)
        for_loop_var_13 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 11, 4), range_call_result_12)
        # Assigning a type to the variable 'i' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'i', for_loop_var_13)
        # SSA begins for a for statement (line 11)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to main(...): (line 12)
        # Processing the call keyword arguments (line 12)
        kwargs_15 = {}
        # Getting the type of 'main' (line 12)
        main_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'main', False)
        # Calling main(args, kwargs) (line 12)
        main_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), main_14, *[], **kwargs_15)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'True' (line 13)
    True_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type', True_17)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_18

# Assigning a type to the variable 'run' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'run', run)

# Call to run(...): (line 16)
# Processing the call keyword arguments (line 16)
kwargs_20 = {}
# Getting the type of 'run' (line 16)
run_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'run', False)
# Calling run(args, kwargs) (line 16)
run_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 16, 0), run_19, *[], **kwargs_20)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
