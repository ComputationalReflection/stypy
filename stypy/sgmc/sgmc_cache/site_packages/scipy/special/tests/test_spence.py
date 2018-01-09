
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy import sqrt, log, pi
5: from scipy.special._testutils import FuncData
6: from scipy.special import spence
7: 
8: 
9: def test_consistency():
10:     # Make sure the implementation of spence for real arguments
11:     # agrees with the implementation of spence for imaginary arguments.
12: 
13:     x = np.logspace(-30, 300, 200)
14:     dataset = np.vstack((x + 0j, spence(x))).T
15:     FuncData(spence, dataset, 0, 1, rtol=1e-14).check()
16: 
17: 
18: def test_special_points():
19:     # Check against known values of Spence's function.
20: 
21:     phi = (1 + sqrt(5))/2
22:     dataset = [(1, 0),
23:                (2, -pi**2/12),
24:                (0.5, pi**2/12 - log(2)**2/2),
25:                (0, pi**2/6),
26:                (-1, pi**2/4 - 1j*pi*log(2)),
27:                ((-1 + sqrt(5))/2, pi**2/15 - log(phi)**2),
28:                ((3 - sqrt(5))/2, pi**2/10 - log(phi)**2),
29:                (phi, -pi**2/15 + log(phi)**2/2),
30:                # Corrected from Zagier, "The Dilogarithm Function"
31:                ((3 + sqrt(5))/2, -pi**2/10 - log(phi)**2)]
32: 
33:     dataset = np.asarray(dataset)
34:     FuncData(spence, dataset, 0, 1, rtol=1e-14).check()
35: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_560489 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_560489) is not StypyTypeError):

    if (import_560489 != 'pyd_module'):
        __import__(import_560489)
        sys_modules_560490 = sys.modules[import_560489]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_560490.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_560489)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy import sqrt, log, pi' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_560491 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_560491) is not StypyTypeError):

    if (import_560491 != 'pyd_module'):
        __import__(import_560491)
        sys_modules_560492 = sys.modules[import_560491]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', sys_modules_560492.module_type_store, module_type_store, ['sqrt', 'log', 'pi'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_560492, sys_modules_560492.module_type_store, module_type_store)
    else:
        from numpy import sqrt, log, pi

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', None, module_type_store, ['sqrt', 'log', 'pi'], [sqrt, log, pi])

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_560491)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.special._testutils import FuncData' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_560493 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special._testutils')

if (type(import_560493) is not StypyTypeError):

    if (import_560493 != 'pyd_module'):
        __import__(import_560493)
        sys_modules_560494 = sys.modules[import_560493]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special._testutils', sys_modules_560494.module_type_store, module_type_store, ['FuncData'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_560494, sys_modules_560494.module_type_store, module_type_store)
    else:
        from scipy.special._testutils import FuncData

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special._testutils', None, module_type_store, ['FuncData'], [FuncData])

else:
    # Assigning a type to the variable 'scipy.special._testutils' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special._testutils', import_560493)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.special import spence' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_560495 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special')

if (type(import_560495) is not StypyTypeError):

    if (import_560495 != 'pyd_module'):
        __import__(import_560495)
        sys_modules_560496 = sys.modules[import_560495]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special', sys_modules_560496.module_type_store, module_type_store, ['spence'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_560496, sys_modules_560496.module_type_store, module_type_store)
    else:
        from scipy.special import spence

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special', None, module_type_store, ['spence'], [spence])

else:
    # Assigning a type to the variable 'scipy.special' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special', import_560495)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')


@norecursion
def test_consistency(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_consistency'
    module_type_store = module_type_store.open_function_context('test_consistency', 9, 0, False)
    
    # Passed parameters checking function
    test_consistency.stypy_localization = localization
    test_consistency.stypy_type_of_self = None
    test_consistency.stypy_type_store = module_type_store
    test_consistency.stypy_function_name = 'test_consistency'
    test_consistency.stypy_param_names_list = []
    test_consistency.stypy_varargs_param_name = None
    test_consistency.stypy_kwargs_param_name = None
    test_consistency.stypy_call_defaults = defaults
    test_consistency.stypy_call_varargs = varargs
    test_consistency.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_consistency', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_consistency', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_consistency(...)' code ##################

    
    # Assigning a Call to a Name (line 13):
    
    # Call to logspace(...): (line 13)
    # Processing the call arguments (line 13)
    int_560499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'int')
    int_560500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'int')
    int_560501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 30), 'int')
    # Processing the call keyword arguments (line 13)
    kwargs_560502 = {}
    # Getting the type of 'np' (line 13)
    np_560497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'np', False)
    # Obtaining the member 'logspace' of a type (line 13)
    logspace_560498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), np_560497, 'logspace')
    # Calling logspace(args, kwargs) (line 13)
    logspace_call_result_560503 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), logspace_560498, *[int_560499, int_560500, int_560501], **kwargs_560502)
    
    # Assigning a type to the variable 'x' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'x', logspace_call_result_560503)
    
    # Assigning a Attribute to a Name (line 14):
    
    # Call to vstack(...): (line 14)
    # Processing the call arguments (line 14)
    
    # Obtaining an instance of the builtin type 'tuple' (line 14)
    tuple_560506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 14)
    # Adding element type (line 14)
    # Getting the type of 'x' (line 14)
    x_560507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 25), 'x', False)
    complex_560508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 29), 'complex')
    # Applying the binary operator '+' (line 14)
    result_add_560509 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 25), '+', x_560507, complex_560508)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 25), tuple_560506, result_add_560509)
    # Adding element type (line 14)
    
    # Call to spence(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'x' (line 14)
    x_560511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 40), 'x', False)
    # Processing the call keyword arguments (line 14)
    kwargs_560512 = {}
    # Getting the type of 'spence' (line 14)
    spence_560510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 33), 'spence', False)
    # Calling spence(args, kwargs) (line 14)
    spence_call_result_560513 = invoke(stypy.reporting.localization.Localization(__file__, 14, 33), spence_560510, *[x_560511], **kwargs_560512)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 25), tuple_560506, spence_call_result_560513)
    
    # Processing the call keyword arguments (line 14)
    kwargs_560514 = {}
    # Getting the type of 'np' (line 14)
    np_560504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'np', False)
    # Obtaining the member 'vstack' of a type (line 14)
    vstack_560505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 14), np_560504, 'vstack')
    # Calling vstack(args, kwargs) (line 14)
    vstack_call_result_560515 = invoke(stypy.reporting.localization.Localization(__file__, 14, 14), vstack_560505, *[tuple_560506], **kwargs_560514)
    
    # Obtaining the member 'T' of a type (line 14)
    T_560516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 14), vstack_call_result_560515, 'T')
    # Assigning a type to the variable 'dataset' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'dataset', T_560516)
    
    # Call to check(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_560527 = {}
    
    # Call to FuncData(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'spence' (line 15)
    spence_560518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 13), 'spence', False)
    # Getting the type of 'dataset' (line 15)
    dataset_560519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 21), 'dataset', False)
    int_560520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 30), 'int')
    int_560521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 33), 'int')
    # Processing the call keyword arguments (line 15)
    float_560522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 41), 'float')
    keyword_560523 = float_560522
    kwargs_560524 = {'rtol': keyword_560523}
    # Getting the type of 'FuncData' (line 15)
    FuncData_560517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'FuncData', False)
    # Calling FuncData(args, kwargs) (line 15)
    FuncData_call_result_560525 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), FuncData_560517, *[spence_560518, dataset_560519, int_560520, int_560521], **kwargs_560524)
    
    # Obtaining the member 'check' of a type (line 15)
    check_560526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), FuncData_call_result_560525, 'check')
    # Calling check(args, kwargs) (line 15)
    check_call_result_560528 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), check_560526, *[], **kwargs_560527)
    
    
    # ################# End of 'test_consistency(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_consistency' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_560529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_560529)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_consistency'
    return stypy_return_type_560529

# Assigning a type to the variable 'test_consistency' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test_consistency', test_consistency)

@norecursion
def test_special_points(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_special_points'
    module_type_store = module_type_store.open_function_context('test_special_points', 18, 0, False)
    
    # Passed parameters checking function
    test_special_points.stypy_localization = localization
    test_special_points.stypy_type_of_self = None
    test_special_points.stypy_type_store = module_type_store
    test_special_points.stypy_function_name = 'test_special_points'
    test_special_points.stypy_param_names_list = []
    test_special_points.stypy_varargs_param_name = None
    test_special_points.stypy_kwargs_param_name = None
    test_special_points.stypy_call_defaults = defaults
    test_special_points.stypy_call_varargs = varargs
    test_special_points.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_special_points', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_special_points', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_special_points(...)' code ##################

    
    # Assigning a BinOp to a Name (line 21):
    int_560530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 11), 'int')
    
    # Call to sqrt(...): (line 21)
    # Processing the call arguments (line 21)
    int_560532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 20), 'int')
    # Processing the call keyword arguments (line 21)
    kwargs_560533 = {}
    # Getting the type of 'sqrt' (line 21)
    sqrt_560531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 21)
    sqrt_call_result_560534 = invoke(stypy.reporting.localization.Localization(__file__, 21, 15), sqrt_560531, *[int_560532], **kwargs_560533)
    
    # Applying the binary operator '+' (line 21)
    result_add_560535 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 11), '+', int_560530, sqrt_call_result_560534)
    
    int_560536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'int')
    # Applying the binary operator 'div' (line 21)
    result_div_560537 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 10), 'div', result_add_560535, int_560536)
    
    # Assigning a type to the variable 'phi' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'phi', result_div_560537)
    
    # Assigning a List to a Name (line 22):
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_560538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'tuple' (line 22)
    tuple_560539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 22)
    # Adding element type (line 22)
    int_560540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), tuple_560539, int_560540)
    # Adding element type (line 22)
    int_560541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), tuple_560539, int_560541)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 14), list_560538, tuple_560539)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'tuple' (line 23)
    tuple_560542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 23)
    # Adding element type (line 23)
    int_560543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 16), tuple_560542, int_560543)
    # Adding element type (line 23)
    
    # Getting the type of 'pi' (line 23)
    pi_560544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 20), 'pi')
    int_560545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'int')
    # Applying the binary operator '**' (line 23)
    result_pow_560546 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 20), '**', pi_560544, int_560545)
    
    # Applying the 'usub' unary operator (line 23)
    result___neg___560547 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 19), 'usub', result_pow_560546)
    
    int_560548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 26), 'int')
    # Applying the binary operator 'div' (line 23)
    result_div_560549 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 19), 'div', result___neg___560547, int_560548)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 16), tuple_560542, result_div_560549)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 14), list_560538, tuple_560542)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'tuple' (line 24)
    tuple_560550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 24)
    # Adding element type (line 24)
    float_560551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 16), tuple_560550, float_560551)
    # Adding element type (line 24)
    # Getting the type of 'pi' (line 24)
    pi_560552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 21), 'pi')
    int_560553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'int')
    # Applying the binary operator '**' (line 24)
    result_pow_560554 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 21), '**', pi_560552, int_560553)
    
    int_560555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 27), 'int')
    # Applying the binary operator 'div' (line 24)
    result_div_560556 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 21), 'div', result_pow_560554, int_560555)
    
    
    # Call to log(...): (line 24)
    # Processing the call arguments (line 24)
    int_560558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 36), 'int')
    # Processing the call keyword arguments (line 24)
    kwargs_560559 = {}
    # Getting the type of 'log' (line 24)
    log_560557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 32), 'log', False)
    # Calling log(args, kwargs) (line 24)
    log_call_result_560560 = invoke(stypy.reporting.localization.Localization(__file__, 24, 32), log_560557, *[int_560558], **kwargs_560559)
    
    int_560561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 40), 'int')
    # Applying the binary operator '**' (line 24)
    result_pow_560562 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 32), '**', log_call_result_560560, int_560561)
    
    int_560563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 42), 'int')
    # Applying the binary operator 'div' (line 24)
    result_div_560564 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 32), 'div', result_pow_560562, int_560563)
    
    # Applying the binary operator '-' (line 24)
    result_sub_560565 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 21), '-', result_div_560556, result_div_560564)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 16), tuple_560550, result_sub_560565)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 14), list_560538, tuple_560550)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'tuple' (line 25)
    tuple_560566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 25)
    # Adding element type (line 25)
    int_560567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 16), tuple_560566, int_560567)
    # Adding element type (line 25)
    # Getting the type of 'pi' (line 25)
    pi_560568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'pi')
    int_560569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 23), 'int')
    # Applying the binary operator '**' (line 25)
    result_pow_560570 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 19), '**', pi_560568, int_560569)
    
    int_560571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'int')
    # Applying the binary operator 'div' (line 25)
    result_div_560572 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 19), 'div', result_pow_560570, int_560571)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 16), tuple_560566, result_div_560572)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 14), list_560538, tuple_560566)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'tuple' (line 26)
    tuple_560573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 26)
    # Adding element type (line 26)
    int_560574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 16), tuple_560573, int_560574)
    # Adding element type (line 26)
    # Getting the type of 'pi' (line 26)
    pi_560575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'pi')
    int_560576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 24), 'int')
    # Applying the binary operator '**' (line 26)
    result_pow_560577 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 20), '**', pi_560575, int_560576)
    
    int_560578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 26), 'int')
    # Applying the binary operator 'div' (line 26)
    result_div_560579 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 20), 'div', result_pow_560577, int_560578)
    
    complex_560580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 30), 'complex')
    # Getting the type of 'pi' (line 26)
    pi_560581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 33), 'pi')
    # Applying the binary operator '*' (line 26)
    result_mul_560582 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 30), '*', complex_560580, pi_560581)
    
    
    # Call to log(...): (line 26)
    # Processing the call arguments (line 26)
    int_560584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 40), 'int')
    # Processing the call keyword arguments (line 26)
    kwargs_560585 = {}
    # Getting the type of 'log' (line 26)
    log_560583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 36), 'log', False)
    # Calling log(args, kwargs) (line 26)
    log_call_result_560586 = invoke(stypy.reporting.localization.Localization(__file__, 26, 36), log_560583, *[int_560584], **kwargs_560585)
    
    # Applying the binary operator '*' (line 26)
    result_mul_560587 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 35), '*', result_mul_560582, log_call_result_560586)
    
    # Applying the binary operator '-' (line 26)
    result_sub_560588 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 20), '-', result_div_560579, result_mul_560587)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 16), tuple_560573, result_sub_560588)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 14), list_560538, tuple_560573)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'tuple' (line 27)
    tuple_560589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 27)
    # Adding element type (line 27)
    int_560590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 17), 'int')
    
    # Call to sqrt(...): (line 27)
    # Processing the call arguments (line 27)
    int_560592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 27), 'int')
    # Processing the call keyword arguments (line 27)
    kwargs_560593 = {}
    # Getting the type of 'sqrt' (line 27)
    sqrt_560591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 22), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 27)
    sqrt_call_result_560594 = invoke(stypy.reporting.localization.Localization(__file__, 27, 22), sqrt_560591, *[int_560592], **kwargs_560593)
    
    # Applying the binary operator '+' (line 27)
    result_add_560595 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 17), '+', int_560590, sqrt_call_result_560594)
    
    int_560596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 31), 'int')
    # Applying the binary operator 'div' (line 27)
    result_div_560597 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 16), 'div', result_add_560595, int_560596)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 16), tuple_560589, result_div_560597)
    # Adding element type (line 27)
    # Getting the type of 'pi' (line 27)
    pi_560598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 34), 'pi')
    int_560599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 38), 'int')
    # Applying the binary operator '**' (line 27)
    result_pow_560600 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 34), '**', pi_560598, int_560599)
    
    int_560601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 40), 'int')
    # Applying the binary operator 'div' (line 27)
    result_div_560602 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 34), 'div', result_pow_560600, int_560601)
    
    
    # Call to log(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'phi' (line 27)
    phi_560604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 49), 'phi', False)
    # Processing the call keyword arguments (line 27)
    kwargs_560605 = {}
    # Getting the type of 'log' (line 27)
    log_560603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 45), 'log', False)
    # Calling log(args, kwargs) (line 27)
    log_call_result_560606 = invoke(stypy.reporting.localization.Localization(__file__, 27, 45), log_560603, *[phi_560604], **kwargs_560605)
    
    int_560607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 55), 'int')
    # Applying the binary operator '**' (line 27)
    result_pow_560608 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 45), '**', log_call_result_560606, int_560607)
    
    # Applying the binary operator '-' (line 27)
    result_sub_560609 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 34), '-', result_div_560602, result_pow_560608)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 16), tuple_560589, result_sub_560609)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 14), list_560538, tuple_560589)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'tuple' (line 28)
    tuple_560610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 28)
    # Adding element type (line 28)
    int_560611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 17), 'int')
    
    # Call to sqrt(...): (line 28)
    # Processing the call arguments (line 28)
    int_560613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 26), 'int')
    # Processing the call keyword arguments (line 28)
    kwargs_560614 = {}
    # Getting the type of 'sqrt' (line 28)
    sqrt_560612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 21), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 28)
    sqrt_call_result_560615 = invoke(stypy.reporting.localization.Localization(__file__, 28, 21), sqrt_560612, *[int_560613], **kwargs_560614)
    
    # Applying the binary operator '-' (line 28)
    result_sub_560616 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 17), '-', int_560611, sqrt_call_result_560615)
    
    int_560617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 30), 'int')
    # Applying the binary operator 'div' (line 28)
    result_div_560618 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 16), 'div', result_sub_560616, int_560617)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 16), tuple_560610, result_div_560618)
    # Adding element type (line 28)
    # Getting the type of 'pi' (line 28)
    pi_560619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 33), 'pi')
    int_560620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 37), 'int')
    # Applying the binary operator '**' (line 28)
    result_pow_560621 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 33), '**', pi_560619, int_560620)
    
    int_560622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 39), 'int')
    # Applying the binary operator 'div' (line 28)
    result_div_560623 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 33), 'div', result_pow_560621, int_560622)
    
    
    # Call to log(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'phi' (line 28)
    phi_560625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 48), 'phi', False)
    # Processing the call keyword arguments (line 28)
    kwargs_560626 = {}
    # Getting the type of 'log' (line 28)
    log_560624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 44), 'log', False)
    # Calling log(args, kwargs) (line 28)
    log_call_result_560627 = invoke(stypy.reporting.localization.Localization(__file__, 28, 44), log_560624, *[phi_560625], **kwargs_560626)
    
    int_560628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 54), 'int')
    # Applying the binary operator '**' (line 28)
    result_pow_560629 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 44), '**', log_call_result_560627, int_560628)
    
    # Applying the binary operator '-' (line 28)
    result_sub_560630 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 33), '-', result_div_560623, result_pow_560629)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 16), tuple_560610, result_sub_560630)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 14), list_560538, tuple_560610)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'tuple' (line 29)
    tuple_560631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 29)
    # Adding element type (line 29)
    # Getting the type of 'phi' (line 29)
    phi_560632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), 'phi')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 16), tuple_560631, phi_560632)
    # Adding element type (line 29)
    
    # Getting the type of 'pi' (line 29)
    pi_560633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'pi')
    int_560634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 26), 'int')
    # Applying the binary operator '**' (line 29)
    result_pow_560635 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 22), '**', pi_560633, int_560634)
    
    # Applying the 'usub' unary operator (line 29)
    result___neg___560636 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 21), 'usub', result_pow_560635)
    
    int_560637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 28), 'int')
    # Applying the binary operator 'div' (line 29)
    result_div_560638 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 21), 'div', result___neg___560636, int_560637)
    
    
    # Call to log(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'phi' (line 29)
    phi_560640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 37), 'phi', False)
    # Processing the call keyword arguments (line 29)
    kwargs_560641 = {}
    # Getting the type of 'log' (line 29)
    log_560639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 33), 'log', False)
    # Calling log(args, kwargs) (line 29)
    log_call_result_560642 = invoke(stypy.reporting.localization.Localization(__file__, 29, 33), log_560639, *[phi_560640], **kwargs_560641)
    
    int_560643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 43), 'int')
    # Applying the binary operator '**' (line 29)
    result_pow_560644 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 33), '**', log_call_result_560642, int_560643)
    
    int_560645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 45), 'int')
    # Applying the binary operator 'div' (line 29)
    result_div_560646 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 33), 'div', result_pow_560644, int_560645)
    
    # Applying the binary operator '+' (line 29)
    result_add_560647 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 21), '+', result_div_560638, result_div_560646)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 16), tuple_560631, result_add_560647)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 14), list_560538, tuple_560631)
    # Adding element type (line 22)
    
    # Obtaining an instance of the builtin type 'tuple' (line 31)
    tuple_560648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 31)
    # Adding element type (line 31)
    int_560649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 17), 'int')
    
    # Call to sqrt(...): (line 31)
    # Processing the call arguments (line 31)
    int_560651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 26), 'int')
    # Processing the call keyword arguments (line 31)
    kwargs_560652 = {}
    # Getting the type of 'sqrt' (line 31)
    sqrt_560650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 21), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 31)
    sqrt_call_result_560653 = invoke(stypy.reporting.localization.Localization(__file__, 31, 21), sqrt_560650, *[int_560651], **kwargs_560652)
    
    # Applying the binary operator '+' (line 31)
    result_add_560654 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 17), '+', int_560649, sqrt_call_result_560653)
    
    int_560655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 30), 'int')
    # Applying the binary operator 'div' (line 31)
    result_div_560656 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 16), 'div', result_add_560654, int_560655)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 16), tuple_560648, result_div_560656)
    # Adding element type (line 31)
    
    # Getting the type of 'pi' (line 31)
    pi_560657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 34), 'pi')
    int_560658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 38), 'int')
    # Applying the binary operator '**' (line 31)
    result_pow_560659 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 34), '**', pi_560657, int_560658)
    
    # Applying the 'usub' unary operator (line 31)
    result___neg___560660 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 33), 'usub', result_pow_560659)
    
    int_560661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 40), 'int')
    # Applying the binary operator 'div' (line 31)
    result_div_560662 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 33), 'div', result___neg___560660, int_560661)
    
    
    # Call to log(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'phi' (line 31)
    phi_560664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 49), 'phi', False)
    # Processing the call keyword arguments (line 31)
    kwargs_560665 = {}
    # Getting the type of 'log' (line 31)
    log_560663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 45), 'log', False)
    # Calling log(args, kwargs) (line 31)
    log_call_result_560666 = invoke(stypy.reporting.localization.Localization(__file__, 31, 45), log_560663, *[phi_560664], **kwargs_560665)
    
    int_560667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 55), 'int')
    # Applying the binary operator '**' (line 31)
    result_pow_560668 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 45), '**', log_call_result_560666, int_560667)
    
    # Applying the binary operator '-' (line 31)
    result_sub_560669 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 33), '-', result_div_560662, result_pow_560668)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 16), tuple_560648, result_sub_560669)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 14), list_560538, tuple_560648)
    
    # Assigning a type to the variable 'dataset' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'dataset', list_560538)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to asarray(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'dataset' (line 33)
    dataset_560672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 25), 'dataset', False)
    # Processing the call keyword arguments (line 33)
    kwargs_560673 = {}
    # Getting the type of 'np' (line 33)
    np_560670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 14), 'np', False)
    # Obtaining the member 'asarray' of a type (line 33)
    asarray_560671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 14), np_560670, 'asarray')
    # Calling asarray(args, kwargs) (line 33)
    asarray_call_result_560674 = invoke(stypy.reporting.localization.Localization(__file__, 33, 14), asarray_560671, *[dataset_560672], **kwargs_560673)
    
    # Assigning a type to the variable 'dataset' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'dataset', asarray_call_result_560674)
    
    # Call to check(...): (line 34)
    # Processing the call keyword arguments (line 34)
    kwargs_560685 = {}
    
    # Call to FuncData(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'spence' (line 34)
    spence_560676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 13), 'spence', False)
    # Getting the type of 'dataset' (line 34)
    dataset_560677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'dataset', False)
    int_560678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 30), 'int')
    int_560679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 33), 'int')
    # Processing the call keyword arguments (line 34)
    float_560680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 41), 'float')
    keyword_560681 = float_560680
    kwargs_560682 = {'rtol': keyword_560681}
    # Getting the type of 'FuncData' (line 34)
    FuncData_560675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'FuncData', False)
    # Calling FuncData(args, kwargs) (line 34)
    FuncData_call_result_560683 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), FuncData_560675, *[spence_560676, dataset_560677, int_560678, int_560679], **kwargs_560682)
    
    # Obtaining the member 'check' of a type (line 34)
    check_560684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 4), FuncData_call_result_560683, 'check')
    # Calling check(args, kwargs) (line 34)
    check_call_result_560686 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), check_560684, *[], **kwargs_560685)
    
    
    # ################# End of 'test_special_points(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_special_points' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_560687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_560687)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_special_points'
    return stypy_return_type_560687

# Assigning a type to the variable 'test_special_points' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'test_special_points', test_special_points)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
