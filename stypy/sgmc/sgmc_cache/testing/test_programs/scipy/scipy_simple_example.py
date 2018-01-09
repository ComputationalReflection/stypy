
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: from scipy.special import jv
4: from scipy.optimize import minimize
5: from numpy import linspace
6: 
7: f = lambda x: -jv(3, x)
8: sol = minimize(f, 1.0)
9: x = linspace(0, 10, 5000)
10: 
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy.special import jv' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/scipy/')
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.special')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.special', sys_modules_2.module_type_store, module_type_store, ['jv'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_2, sys_modules_2.module_type_store, module_type_store)
    else:
        from scipy.special import jv

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.special', None, module_type_store, ['jv'], [jv])

else:
    # Assigning a type to the variable 'scipy.special' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.special', import_1)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/scipy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.optimize import minimize' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/scipy/')
import_3 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.optimize')

if (type(import_3) is not StypyTypeError):

    if (import_3 != 'pyd_module'):
        __import__(import_3)
        sys_modules_4 = sys.modules[import_3]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.optimize', sys_modules_4.module_type_store, module_type_store, ['minimize'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_4, sys_modules_4.module_type_store, module_type_store)
    else:
        from scipy.optimize import minimize

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.optimize', None, module_type_store, ['minimize'], [minimize])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.optimize', import_3)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/scipy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy import linspace' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/scipy/')
import_5 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_5) is not StypyTypeError):

    if (import_5 != 'pyd_module'):
        __import__(import_5)
        sys_modules_6 = sys.modules[import_5]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', sys_modules_6.module_type_store, module_type_store, ['linspace'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_6, sys_modules_6.module_type_store, module_type_store)
    else:
        from numpy import linspace

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', None, module_type_store, ['linspace'], [linspace])

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_5)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/scipy/')


# Assigning a Lambda to a Name (line 7):

@norecursion
def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_1'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 7, 4, True)
    # Passed parameters checking function
    _stypy_temp_lambda_1.stypy_localization = localization
    _stypy_temp_lambda_1.stypy_type_of_self = None
    _stypy_temp_lambda_1.stypy_type_store = module_type_store
    _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
    _stypy_temp_lambda_1.stypy_param_names_list = ['x']
    _stypy_temp_lambda_1.stypy_varargs_param_name = None
    _stypy_temp_lambda_1.stypy_kwargs_param_name = None
    _stypy_temp_lambda_1.stypy_call_defaults = defaults
    _stypy_temp_lambda_1.stypy_call_varargs = varargs
    _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_1', ['x'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    
    # Call to jv(...): (line 7)
    # Processing the call arguments (line 7)
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 18), 'int')
    # Getting the type of 'x' (line 7)
    x_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 21), 'x', False)
    # Processing the call keyword arguments (line 7)
    kwargs_10 = {}
    # Getting the type of 'jv' (line 7)
    jv_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 15), 'jv', False)
    # Calling jv(args, kwargs) (line 7)
    jv_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 7, 15), jv_7, *[int_8, x_9], **kwargs_10)
    
    # Applying the 'usub' unary operator (line 7)
    result___neg___12 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 14), 'usub', jv_call_result_11)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type', result___neg___12)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_1' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_1'
    return stypy_return_type_13

# Assigning a type to the variable '_stypy_temp_lambda_1' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
# Getting the type of '_stypy_temp_lambda_1' (line 7)
_stypy_temp_lambda_1_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), '_stypy_temp_lambda_1')
# Assigning a type to the variable 'f' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'f', _stypy_temp_lambda_1_14)

# Assigning a Call to a Name (line 8):

# Call to minimize(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'f' (line 8)
f_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 15), 'f', False)
float_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 18), 'float')
# Processing the call keyword arguments (line 8)
kwargs_18 = {}
# Getting the type of 'minimize' (line 8)
minimize_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 6), 'minimize', False)
# Calling minimize(args, kwargs) (line 8)
minimize_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 8, 6), minimize_15, *[f_16, float_17], **kwargs_18)

# Assigning a type to the variable 'sol' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'sol', minimize_call_result_19)

# Assigning a Call to a Name (line 9):

# Call to linspace(...): (line 9)
# Processing the call arguments (line 9)
int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'int')
int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 16), 'int')
int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 20), 'int')
# Processing the call keyword arguments (line 9)
kwargs_24 = {}
# Getting the type of 'linspace' (line 9)
linspace_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'linspace', False)
# Calling linspace(args, kwargs) (line 9)
linspace_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), linspace_20, *[int_21, int_22, int_23], **kwargs_24)

# Assigning a type to the variable 'x' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'x', linspace_call_result_25)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
