
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def f(x, y, z, *arguments, **kwarguments):
2:     pass
3: 
4: def f2(x, y, z, *arguments, **kwarguments):
5:     pass
6: 
7: def f3 (x=5, y=6, z=4, *args, **kwargs):
8:     pass
9: 
10: 
11: f(2, 3, 4, 5, 6, 7)
12: f2(1, 2, 8, 6, 4, r=23)
13: 
14: f3(z="1", x=4, y=True, r=11, s="12")
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def f(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f'
    module_type_store = module_type_store.open_function_context('f', 1, 0, False)
    
    # Passed parameters checking function
    f.stypy_localization = localization
    f.stypy_type_of_self = None
    f.stypy_type_store = module_type_store
    f.stypy_function_name = 'f'
    f.stypy_param_names_list = ['x', 'y', 'z']
    f.stypy_varargs_param_name = 'arguments'
    f.stypy_kwargs_param_name = 'kwarguments'
    f.stypy_call_defaults = defaults
    f.stypy_call_varargs = varargs
    f.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f', ['x', 'y', 'z'], 'arguments', 'kwarguments', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f', localization, ['x', 'y', 'z'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f(...)' code ##################

    pass
    
    # ################# End of 'f(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_631)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f'
    return stypy_return_type_631

# Assigning a type to the variable 'f' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'f', f)

@norecursion
def f2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f2'
    module_type_store = module_type_store.open_function_context('f2', 4, 0, False)
    
    # Passed parameters checking function
    f2.stypy_localization = localization
    f2.stypy_type_of_self = None
    f2.stypy_type_store = module_type_store
    f2.stypy_function_name = 'f2'
    f2.stypy_param_names_list = ['x', 'y', 'z']
    f2.stypy_varargs_param_name = 'arguments'
    f2.stypy_kwargs_param_name = 'kwarguments'
    f2.stypy_call_defaults = defaults
    f2.stypy_call_varargs = varargs
    f2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f2', ['x', 'y', 'z'], 'arguments', 'kwarguments', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f2', localization, ['x', 'y', 'z'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f2(...)' code ##################

    pass
    
    # ################# End of 'f2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f2' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_632)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f2'
    return stypy_return_type_632

# Assigning a type to the variable 'f2' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'f2', f2)

@norecursion
def f3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 10), 'int')
    int_634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'int')
    int_635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 20), 'int')
    defaults = [int_633, int_634, int_635]
    # Create a new context for function 'f3'
    module_type_store = module_type_store.open_function_context('f3', 7, 0, False)
    
    # Passed parameters checking function
    f3.stypy_localization = localization
    f3.stypy_type_of_self = None
    f3.stypy_type_store = module_type_store
    f3.stypy_function_name = 'f3'
    f3.stypy_param_names_list = ['x', 'y', 'z']
    f3.stypy_varargs_param_name = 'args'
    f3.stypy_kwargs_param_name = 'kwargs'
    f3.stypy_call_defaults = defaults
    f3.stypy_call_varargs = varargs
    f3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f3', ['x', 'y', 'z'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f3', localization, ['x', 'y', 'z'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f3(...)' code ##################

    pass
    
    # ################# End of 'f3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f3' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_636)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f3'
    return stypy_return_type_636

# Assigning a type to the variable 'f3' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'f3', f3)

# Call to f(...): (line 11)
# Processing the call arguments (line 11)
int_638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 2), 'int')
int_639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 5), 'int')
int_640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 8), 'int')
int_641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 11), 'int')
int_642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 14), 'int')
int_643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 17), 'int')
# Processing the call keyword arguments (line 11)
kwargs_644 = {}
# Getting the type of 'f' (line 11)
f_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'f', False)
# Calling f(args, kwargs) (line 11)
f_call_result_645 = invoke(stypy.reporting.localization.Localization(__file__, 11, 0), f_637, *[int_638, int_639, int_640, int_641, int_642, int_643], **kwargs_644)


# Call to f2(...): (line 12)
# Processing the call arguments (line 12)
int_647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 3), 'int')
int_648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 6), 'int')
int_649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 9), 'int')
int_650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 12), 'int')
int_651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'int')
# Processing the call keyword arguments (line 12)
int_652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 20), 'int')
keyword_653 = int_652
kwargs_654 = {'r': keyword_653}
# Getting the type of 'f2' (line 12)
f2_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'f2', False)
# Calling f2(args, kwargs) (line 12)
f2_call_result_655 = invoke(stypy.reporting.localization.Localization(__file__, 12, 0), f2_646, *[int_647, int_648, int_649, int_650, int_651], **kwargs_654)


# Call to f3(...): (line 14)
# Processing the call keyword arguments (line 14)
str_657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 5), 'str', '1')
keyword_658 = str_657
int_659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'int')
keyword_660 = int_659
# Getting the type of 'True' (line 14)
True_661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 17), 'True', False)
keyword_662 = True_661
int_663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'int')
keyword_664 = int_663
str_665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 31), 'str', '12')
keyword_666 = str_665
kwargs_667 = {'y': keyword_662, 'x': keyword_660, 'r': keyword_664, 'z': keyword_658, 's': keyword_666}
# Getting the type of 'f3' (line 14)
f3_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'f3', False)
# Calling f3(args, kwargs) (line 14)
f3_call_result_668 = invoke(stypy.reporting.localization.Localization(__file__, 14, 0), f3_656, *[], **kwargs_667)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
