
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: def functionargs(*args):
2:     return args[0]  # Should warn about None
3: 
4: 
5: r1 = functionargs("hi")
6: 
7: x1 = r1.thisdonotexist()  # Unreported
8: 
9: 
10: def functionkw(**kwargs):
11:     return kwargs[0]  # Accepts anyting as key, even if we know that kwargs has always str keys
12: 
13: 
14: def functionkw2(**kwargs):
15:     return kwargs["val"]  # Accepts anyting as key, even if we know that kwargs has always str keys
16: 
17: 
18: r2 = functionkw(val="hi")
19: 
20: x2 = r2.thisdonotexist()  # Unreported
21: 
22: r3 = functionkw2(val="hi")
23: 
24: x3 = r2.thisdonotexist()  # Unreported
25: 
26: r4 = functionkw2(not_exist="hi")

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def functionargs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'functionargs'
    module_type_store = module_type_store.open_function_context('functionargs', 1, 0, False)
    
    # Passed parameters checking function
    functionargs.stypy_localization = localization
    functionargs.stypy_type_of_self = None
    functionargs.stypy_type_store = module_type_store
    functionargs.stypy_function_name = 'functionargs'
    functionargs.stypy_param_names_list = []
    functionargs.stypy_varargs_param_name = 'args'
    functionargs.stypy_kwargs_param_name = None
    functionargs.stypy_call_defaults = defaults
    functionargs.stypy_call_varargs = varargs
    functionargs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'functionargs', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'functionargs', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'functionargs(...)' code ##################

    
    # Obtaining the type of the subscript
    int_7358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 16), 'int')
    # Getting the type of 'args' (line 2)
    args_7359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 11), 'args')
    # Obtaining the member '__getitem__' of a type (line 2)
    getitem___7360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 2, 11), args_7359, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 2)
    subscript_call_result_7361 = invoke(stypy.reporting.localization.Localization(__file__, 2, 11), getitem___7360, int_7358)
    
    # Assigning a type to the variable 'stypy_return_type' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type', subscript_call_result_7361)
    
    # ################# End of 'functionargs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'functionargs' in the type store
    # Getting the type of 'stypy_return_type' (line 1)
    stypy_return_type_7362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7362)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'functionargs'
    return stypy_return_type_7362

# Assigning a type to the variable 'functionargs' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'functionargs', functionargs)

# Assigning a Call to a Name (line 5):

# Call to functionargs(...): (line 5)
# Processing the call arguments (line 5)
str_7364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 18), 'str', 'hi')
# Processing the call keyword arguments (line 5)
kwargs_7365 = {}
# Getting the type of 'functionargs' (line 5)
functionargs_7363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 5), 'functionargs', False)
# Calling functionargs(args, kwargs) (line 5)
functionargs_call_result_7366 = invoke(stypy.reporting.localization.Localization(__file__, 5, 5), functionargs_7363, *[str_7364], **kwargs_7365)

# Assigning a type to the variable 'r1' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'r1', functionargs_call_result_7366)

# Assigning a Call to a Name (line 7):

# Call to thisdonotexist(...): (line 7)
# Processing the call keyword arguments (line 7)
kwargs_7369 = {}
# Getting the type of 'r1' (line 7)
r1_7367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'r1', False)
# Obtaining the member 'thisdonotexist' of a type (line 7)
thisdonotexist_7368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 5), r1_7367, 'thisdonotexist')
# Calling thisdonotexist(args, kwargs) (line 7)
thisdonotexist_call_result_7370 = invoke(stypy.reporting.localization.Localization(__file__, 7, 5), thisdonotexist_7368, *[], **kwargs_7369)

# Assigning a type to the variable 'x1' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'x1', thisdonotexist_call_result_7370)

@norecursion
def functionkw(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'functionkw'
    module_type_store = module_type_store.open_function_context('functionkw', 10, 0, False)
    
    # Passed parameters checking function
    functionkw.stypy_localization = localization
    functionkw.stypy_type_of_self = None
    functionkw.stypy_type_store = module_type_store
    functionkw.stypy_function_name = 'functionkw'
    functionkw.stypy_param_names_list = []
    functionkw.stypy_varargs_param_name = None
    functionkw.stypy_kwargs_param_name = 'kwargs'
    functionkw.stypy_call_defaults = defaults
    functionkw.stypy_call_varargs = varargs
    functionkw.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'functionkw', [], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'functionkw', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'functionkw(...)' code ##################

    
    # Obtaining the type of the subscript
    int_7371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'int')
    # Getting the type of 'kwargs' (line 11)
    kwargs_7372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'kwargs')
    # Obtaining the member '__getitem__' of a type (line 11)
    getitem___7373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 11), kwargs_7372, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 11)
    subscript_call_result_7374 = invoke(stypy.reporting.localization.Localization(__file__, 11, 11), getitem___7373, int_7371)
    
    # Assigning a type to the variable 'stypy_return_type' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type', subscript_call_result_7374)
    
    # ################# End of 'functionkw(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'functionkw' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_7375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7375)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'functionkw'
    return stypy_return_type_7375

# Assigning a type to the variable 'functionkw' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'functionkw', functionkw)

@norecursion
def functionkw2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'functionkw2'
    module_type_store = module_type_store.open_function_context('functionkw2', 14, 0, False)
    
    # Passed parameters checking function
    functionkw2.stypy_localization = localization
    functionkw2.stypy_type_of_self = None
    functionkw2.stypy_type_store = module_type_store
    functionkw2.stypy_function_name = 'functionkw2'
    functionkw2.stypy_param_names_list = []
    functionkw2.stypy_varargs_param_name = None
    functionkw2.stypy_kwargs_param_name = 'kwargs'
    functionkw2.stypy_call_defaults = defaults
    functionkw2.stypy_call_varargs = varargs
    functionkw2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'functionkw2', [], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'functionkw2', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'functionkw2(...)' code ##################

    
    # Obtaining the type of the subscript
    str_7376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 18), 'str', 'val')
    # Getting the type of 'kwargs' (line 15)
    kwargs_7377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'kwargs')
    # Obtaining the member '__getitem__' of a type (line 15)
    getitem___7378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 11), kwargs_7377, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 15)
    subscript_call_result_7379 = invoke(stypy.reporting.localization.Localization(__file__, 15, 11), getitem___7378, str_7376)
    
    # Assigning a type to the variable 'stypy_return_type' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type', subscript_call_result_7379)
    
    # ################# End of 'functionkw2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'functionkw2' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_7380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7380)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'functionkw2'
    return stypy_return_type_7380

# Assigning a type to the variable 'functionkw2' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'functionkw2', functionkw2)

# Assigning a Call to a Name (line 18):

# Call to functionkw(...): (line 18)
# Processing the call keyword arguments (line 18)
str_7382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 20), 'str', 'hi')
keyword_7383 = str_7382
kwargs_7384 = {'val': keyword_7383}
# Getting the type of 'functionkw' (line 18)
functionkw_7381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'functionkw', False)
# Calling functionkw(args, kwargs) (line 18)
functionkw_call_result_7385 = invoke(stypy.reporting.localization.Localization(__file__, 18, 5), functionkw_7381, *[], **kwargs_7384)

# Assigning a type to the variable 'r2' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'r2', functionkw_call_result_7385)

# Assigning a Call to a Name (line 20):

# Call to thisdonotexist(...): (line 20)
# Processing the call keyword arguments (line 20)
kwargs_7388 = {}
# Getting the type of 'r2' (line 20)
r2_7386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 5), 'r2', False)
# Obtaining the member 'thisdonotexist' of a type (line 20)
thisdonotexist_7387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 5), r2_7386, 'thisdonotexist')
# Calling thisdonotexist(args, kwargs) (line 20)
thisdonotexist_call_result_7389 = invoke(stypy.reporting.localization.Localization(__file__, 20, 5), thisdonotexist_7387, *[], **kwargs_7388)

# Assigning a type to the variable 'x2' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'x2', thisdonotexist_call_result_7389)

# Assigning a Call to a Name (line 22):

# Call to functionkw2(...): (line 22)
# Processing the call keyword arguments (line 22)
str_7391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'str', 'hi')
keyword_7392 = str_7391
kwargs_7393 = {'val': keyword_7392}
# Getting the type of 'functionkw2' (line 22)
functionkw2_7390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 5), 'functionkw2', False)
# Calling functionkw2(args, kwargs) (line 22)
functionkw2_call_result_7394 = invoke(stypy.reporting.localization.Localization(__file__, 22, 5), functionkw2_7390, *[], **kwargs_7393)

# Assigning a type to the variable 'r3' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'r3', functionkw2_call_result_7394)

# Assigning a Call to a Name (line 24):

# Call to thisdonotexist(...): (line 24)
# Processing the call keyword arguments (line 24)
kwargs_7397 = {}
# Getting the type of 'r2' (line 24)
r2_7395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 5), 'r2', False)
# Obtaining the member 'thisdonotexist' of a type (line 24)
thisdonotexist_7396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 5), r2_7395, 'thisdonotexist')
# Calling thisdonotexist(args, kwargs) (line 24)
thisdonotexist_call_result_7398 = invoke(stypy.reporting.localization.Localization(__file__, 24, 5), thisdonotexist_7396, *[], **kwargs_7397)

# Assigning a type to the variable 'x3' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'x3', thisdonotexist_call_result_7398)

# Assigning a Call to a Name (line 26):

# Call to functionkw2(...): (line 26)
# Processing the call keyword arguments (line 26)
str_7400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 27), 'str', 'hi')
keyword_7401 = str_7400
kwargs_7402 = {'not_exist': keyword_7401}
# Getting the type of 'functionkw2' (line 26)
functionkw2_7399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 5), 'functionkw2', False)
# Calling functionkw2(args, kwargs) (line 26)
functionkw2_call_result_7403 = invoke(stypy.reporting.localization.Localization(__file__, 26, 5), functionkw2_7399, *[], **kwargs_7402)

# Assigning a type to the variable 'r4' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'r4', functionkw2_call_result_7403)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
