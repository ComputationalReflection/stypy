
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def generators():
2:     def f(x):
3:         if False:
4:             return str(x)
5:         return x
6: 
7:     r = [f(x) for x in range(10)]
8:     r2 = r[0].capitalize()  # Unreported, runtime crash
9: 
10: 
11: generators()
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def generators(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'generators'
    module_type_store = module_type_store.open_function_context('generators', 1, 0, False)
    
    # Passed parameters checking function
    generators.stypy_localization = localization
    generators.stypy_type_of_self = None
    generators.stypy_type_store = module_type_store
    generators.stypy_function_name = 'generators'
    generators.stypy_param_names_list = []
    generators.stypy_varargs_param_name = None
    generators.stypy_kwargs_param_name = None
    generators.stypy_call_defaults = defaults
    generators.stypy_call_varargs = varargs
    generators.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generators', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generators', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generators(...)' code ##################


    @norecursion
    def f(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 2, 4, False)
        
        # Passed parameters checking function
        f.stypy_localization = localization
        f.stypy_type_of_self = None
        f.stypy_type_store = module_type_store
        f.stypy_function_name = 'f'
        f.stypy_param_names_list = ['x']
        f.stypy_varargs_param_name = None
        f.stypy_kwargs_param_name = None
        f.stypy_call_defaults = defaults
        f.stypy_call_varargs = varargs
        f.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'f', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        
        # Getting the type of 'False' (line 3)
        False_7669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 11), 'False')
        # Testing the type of an if condition (line 3)
        if_condition_7670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3, 8), False_7669)
        # Assigning a type to the variable 'if_condition_7670' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'if_condition_7670', if_condition_7670)
        # SSA begins for if statement (line 3)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to str(...): (line 4)
        # Processing the call arguments (line 4)
        # Getting the type of 'x' (line 4)
        x_7672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 23), 'x', False)
        # Processing the call keyword arguments (line 4)
        kwargs_7673 = {}
        # Getting the type of 'str' (line 4)
        str_7671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 19), 'str', False)
        # Calling str(args, kwargs) (line 4)
        str_call_result_7674 = invoke(stypy.reporting.localization.Localization(__file__, 4, 19), str_7671, *[x_7672], **kwargs_7673)
        
        # Assigning a type to the variable 'stypy_return_type' (line 4)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 12), 'stypy_return_type', str_call_result_7674)
        # SSA join for if statement (line 3)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'x' (line 5)
        x_7675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 15), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'stypy_return_type', x_7675)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 2)
        stypy_return_type_7676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7676)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_7676

    # Assigning a type to the variable 'f' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'f', f)
    
    # Assigning a ListComp to a Name (line 7):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 7)
    # Processing the call arguments (line 7)
    int_7682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 29), 'int')
    # Processing the call keyword arguments (line 7)
    kwargs_7683 = {}
    # Getting the type of 'range' (line 7)
    range_7681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 23), 'range', False)
    # Calling range(args, kwargs) (line 7)
    range_call_result_7684 = invoke(stypy.reporting.localization.Localization(__file__, 7, 23), range_7681, *[int_7682], **kwargs_7683)
    
    comprehension_7685 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 9), range_call_result_7684)
    # Assigning a type to the variable 'x' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'x', comprehension_7685)
    
    # Call to f(...): (line 7)
    # Processing the call arguments (line 7)
    # Getting the type of 'x' (line 7)
    x_7678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'x', False)
    # Processing the call keyword arguments (line 7)
    kwargs_7679 = {}
    # Getting the type of 'f' (line 7)
    f_7677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'f', False)
    # Calling f(args, kwargs) (line 7)
    f_call_result_7680 = invoke(stypy.reporting.localization.Localization(__file__, 7, 9), f_7677, *[x_7678], **kwargs_7679)
    
    list_7686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 9), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 9), list_7686, f_call_result_7680)
    # Assigning a type to the variable 'r' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'r', list_7686)
    
    # Assigning a Call to a Name (line 8):
    
    # Call to capitalize(...): (line 8)
    # Processing the call keyword arguments (line 8)
    kwargs_7692 = {}
    
    # Obtaining the type of the subscript
    int_7687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'int')
    # Getting the type of 'r' (line 8)
    r_7688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 9), 'r', False)
    # Obtaining the member '__getitem__' of a type (line 8)
    getitem___7689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 9), r_7688, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 8)
    subscript_call_result_7690 = invoke(stypy.reporting.localization.Localization(__file__, 8, 9), getitem___7689, int_7687)
    
    # Obtaining the member 'capitalize' of a type (line 8)
    capitalize_7691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 9), subscript_call_result_7690, 'capitalize')
    # Calling capitalize(args, kwargs) (line 8)
    capitalize_call_result_7693 = invoke(stypy.reporting.localization.Localization(__file__, 8, 9), capitalize_7691, *[], **kwargs_7692)
    
    # Assigning a type to the variable 'r2' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'r2', capitalize_call_result_7693)
    
    # ################# End of 'generators(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generators' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_7694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7694)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generators'
    return stypy_return_type_7694

# Assigning a type to the variable 'generators' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'generators', generators)

# Call to generators(...): (line 11)
# Processing the call keyword arguments (line 11)
kwargs_7696 = {}
# Getting the type of 'generators' (line 11)
generators_7695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'generators', False)
# Calling generators(args, kwargs) (line 11)
generators_call_result_7697 = invoke(stypy.reporting.localization.Localization(__file__, 11, 0), generators_7695, *[], **kwargs_7696)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
