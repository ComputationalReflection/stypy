
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_, assert_allclose
5: import scipy.linalg
6: from scipy.optimize import minimize
7: 
8: 
9: def test_1():
10:     def f(x):
11:         return x**4, 4*x**3
12: 
13:     for gtol in [1e-8, 1e-12, 1e-20]:
14:         for maxcor in range(20, 35):
15:             result = minimize(fun=f, jac=True, method='L-BFGS-B', x0=20,
16:                 options={'gtol': gtol, 'maxcor': maxcor})
17: 
18:             H1 = result.hess_inv(np.array([1])).reshape(1,1)
19:             H2 = result.hess_inv.todense()
20: 
21:             assert_allclose(H1, H2)
22: 
23: 
24: def test_2():
25:     H0 = [[3, 0], [1, 2]]
26: 
27:     def f(x):
28:         return np.dot(x, np.dot(scipy.linalg.inv(H0), x))
29: 
30:     result1 = minimize(fun=f, method='L-BFGS-B', x0=[10, 20])
31:     result2 = minimize(fun=f, method='BFGS', x0=[10, 20])
32: 
33:     H1 = result1.hess_inv.todense()
34: 
35:     H2 = np.vstack((
36:         result1.hess_inv(np.array([1, 0])),
37:         result1.hess_inv(np.array([0, 1]))))
38: 
39:     assert_allclose(
40:         result1.hess_inv(np.array([1, 0]).reshape(2,1)).reshape(-1),
41:         result1.hess_inv(np.array([1, 0])))
42:     assert_allclose(H1, H2)
43:     assert_allclose(H1, result2.hess_inv, rtol=1e-2, atol=0.03)
44: 
45: 
46: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205581 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_205581) is not StypyTypeError):

    if (import_205581 != 'pyd_module'):
        __import__(import_205581)
        sys_modules_205582 = sys.modules[import_205581]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_205582.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_205581)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_, assert_allclose' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205583 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_205583) is not StypyTypeError):

    if (import_205583 != 'pyd_module'):
        __import__(import_205583)
        sys_modules_205584 = sys.modules[import_205583]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_205584.module_type_store, module_type_store, ['assert_', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_205584, sys_modules_205584.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_allclose'], [assert_, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_205583)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import scipy.linalg' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205585 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg')

if (type(import_205585) is not StypyTypeError):

    if (import_205585 != 'pyd_module'):
        __import__(import_205585)
        sys_modules_205586 = sys.modules[import_205585]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg', sys_modules_205586.module_type_store, module_type_store)
    else:
        import scipy.linalg

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg', scipy.linalg, module_type_store)

else:
    # Assigning a type to the variable 'scipy.linalg' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.linalg', import_205585)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.optimize import minimize' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_205587 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize')

if (type(import_205587) is not StypyTypeError):

    if (import_205587 != 'pyd_module'):
        __import__(import_205587)
        sys_modules_205588 = sys.modules[import_205587]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize', sys_modules_205588.module_type_store, module_type_store, ['minimize'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_205588, sys_modules_205588.module_type_store, module_type_store)
    else:
        from scipy.optimize import minimize

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize', None, module_type_store, ['minimize'], [minimize])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.optimize', import_205587)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')


@norecursion
def test_1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_1'
    module_type_store = module_type_store.open_function_context('test_1', 9, 0, False)
    
    # Passed parameters checking function
    test_1.stypy_localization = localization
    test_1.stypy_type_of_self = None
    test_1.stypy_type_store = module_type_store
    test_1.stypy_function_name = 'test_1'
    test_1.stypy_param_names_list = []
    test_1.stypy_varargs_param_name = None
    test_1.stypy_kwargs_param_name = None
    test_1.stypy_call_defaults = defaults
    test_1.stypy_call_varargs = varargs
    test_1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_1', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_1', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_1(...)' code ##################


    @norecursion
    def f(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 10, 4, False)
        
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

        
        # Obtaining an instance of the builtin type 'tuple' (line 11)
        tuple_205589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 11)
        # Adding element type (line 11)
        # Getting the type of 'x' (line 11)
        x_205590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'x')
        int_205591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'int')
        # Applying the binary operator '**' (line 11)
        result_pow_205592 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 15), '**', x_205590, int_205591)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 15), tuple_205589, result_pow_205592)
        # Adding element type (line 11)
        int_205593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 21), 'int')
        # Getting the type of 'x' (line 11)
        x_205594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 23), 'x')
        int_205595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'int')
        # Applying the binary operator '**' (line 11)
        result_pow_205596 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 23), '**', x_205594, int_205595)
        
        # Applying the binary operator '*' (line 11)
        result_mul_205597 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 21), '*', int_205593, result_pow_205596)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 15), tuple_205589, result_mul_205597)
        
        # Assigning a type to the variable 'stypy_return_type' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type', tuple_205589)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 10)
        stypy_return_type_205598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205598)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_205598

    # Assigning a type to the variable 'f' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'f', f)
    
    
    # Obtaining an instance of the builtin type 'list' (line 13)
    list_205599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 13)
    # Adding element type (line 13)
    float_205600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 17), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), list_205599, float_205600)
    # Adding element type (line 13)
    float_205601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 23), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), list_205599, float_205601)
    # Adding element type (line 13)
    float_205602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), list_205599, float_205602)
    
    # Testing the type of a for loop iterable (line 13)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 13, 4), list_205599)
    # Getting the type of the for loop variable (line 13)
    for_loop_var_205603 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 13, 4), list_205599)
    # Assigning a type to the variable 'gtol' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'gtol', for_loop_var_205603)
    # SSA begins for a for statement (line 13)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 14)
    # Processing the call arguments (line 14)
    int_205605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 28), 'int')
    int_205606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 32), 'int')
    # Processing the call keyword arguments (line 14)
    kwargs_205607 = {}
    # Getting the type of 'range' (line 14)
    range_205604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 22), 'range', False)
    # Calling range(args, kwargs) (line 14)
    range_call_result_205608 = invoke(stypy.reporting.localization.Localization(__file__, 14, 22), range_205604, *[int_205605, int_205606], **kwargs_205607)
    
    # Testing the type of a for loop iterable (line 14)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 14, 8), range_call_result_205608)
    # Getting the type of the for loop variable (line 14)
    for_loop_var_205609 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 14, 8), range_call_result_205608)
    # Assigning a type to the variable 'maxcor' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'maxcor', for_loop_var_205609)
    # SSA begins for a for statement (line 14)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 15):
    
    # Call to minimize(...): (line 15)
    # Processing the call keyword arguments (line 15)
    # Getting the type of 'f' (line 15)
    f_205611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 34), 'f', False)
    keyword_205612 = f_205611
    # Getting the type of 'True' (line 15)
    True_205613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 41), 'True', False)
    keyword_205614 = True_205613
    str_205615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 54), 'str', 'L-BFGS-B')
    keyword_205616 = str_205615
    int_205617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 69), 'int')
    keyword_205618 = int_205617
    
    # Obtaining an instance of the builtin type 'dict' (line 16)
    dict_205619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 24), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 16)
    # Adding element type (key, value) (line 16)
    str_205620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'str', 'gtol')
    # Getting the type of 'gtol' (line 16)
    gtol_205621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 33), 'gtol', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 24), dict_205619, (str_205620, gtol_205621))
    # Adding element type (key, value) (line 16)
    str_205622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 39), 'str', 'maxcor')
    # Getting the type of 'maxcor' (line 16)
    maxcor_205623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 49), 'maxcor', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 24), dict_205619, (str_205622, maxcor_205623))
    
    keyword_205624 = dict_205619
    kwargs_205625 = {'fun': keyword_205612, 'x0': keyword_205618, 'options': keyword_205624, 'jac': keyword_205614, 'method': keyword_205616}
    # Getting the type of 'minimize' (line 15)
    minimize_205610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 21), 'minimize', False)
    # Calling minimize(args, kwargs) (line 15)
    minimize_call_result_205626 = invoke(stypy.reporting.localization.Localization(__file__, 15, 21), minimize_205610, *[], **kwargs_205625)
    
    # Assigning a type to the variable 'result' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'result', minimize_call_result_205626)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to reshape(...): (line 18)
    # Processing the call arguments (line 18)
    int_205638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 56), 'int')
    int_205639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 58), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_205640 = {}
    
    # Call to hess_inv(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Call to array(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_205631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 42), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    int_205632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 42), list_205631, int_205632)
    
    # Processing the call keyword arguments (line 18)
    kwargs_205633 = {}
    # Getting the type of 'np' (line 18)
    np_205629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 33), 'np', False)
    # Obtaining the member 'array' of a type (line 18)
    array_205630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 33), np_205629, 'array')
    # Calling array(args, kwargs) (line 18)
    array_call_result_205634 = invoke(stypy.reporting.localization.Localization(__file__, 18, 33), array_205630, *[list_205631], **kwargs_205633)
    
    # Processing the call keyword arguments (line 18)
    kwargs_205635 = {}
    # Getting the type of 'result' (line 18)
    result_205627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'result', False)
    # Obtaining the member 'hess_inv' of a type (line 18)
    hess_inv_205628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 17), result_205627, 'hess_inv')
    # Calling hess_inv(args, kwargs) (line 18)
    hess_inv_call_result_205636 = invoke(stypy.reporting.localization.Localization(__file__, 18, 17), hess_inv_205628, *[array_call_result_205634], **kwargs_205635)
    
    # Obtaining the member 'reshape' of a type (line 18)
    reshape_205637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 17), hess_inv_call_result_205636, 'reshape')
    # Calling reshape(args, kwargs) (line 18)
    reshape_call_result_205641 = invoke(stypy.reporting.localization.Localization(__file__, 18, 17), reshape_205637, *[int_205638, int_205639], **kwargs_205640)
    
    # Assigning a type to the variable 'H1' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'H1', reshape_call_result_205641)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to todense(...): (line 19)
    # Processing the call keyword arguments (line 19)
    kwargs_205645 = {}
    # Getting the type of 'result' (line 19)
    result_205642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 17), 'result', False)
    # Obtaining the member 'hess_inv' of a type (line 19)
    hess_inv_205643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 17), result_205642, 'hess_inv')
    # Obtaining the member 'todense' of a type (line 19)
    todense_205644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 17), hess_inv_205643, 'todense')
    # Calling todense(args, kwargs) (line 19)
    todense_call_result_205646 = invoke(stypy.reporting.localization.Localization(__file__, 19, 17), todense_205644, *[], **kwargs_205645)
    
    # Assigning a type to the variable 'H2' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'H2', todense_call_result_205646)
    
    # Call to assert_allclose(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'H1' (line 21)
    H1_205648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 28), 'H1', False)
    # Getting the type of 'H2' (line 21)
    H2_205649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 32), 'H2', False)
    # Processing the call keyword arguments (line 21)
    kwargs_205650 = {}
    # Getting the type of 'assert_allclose' (line 21)
    assert_allclose_205647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 21)
    assert_allclose_call_result_205651 = invoke(stypy.reporting.localization.Localization(__file__, 21, 12), assert_allclose_205647, *[H1_205648, H2_205649], **kwargs_205650)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_1' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_205652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_205652)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_1'
    return stypy_return_type_205652

# Assigning a type to the variable 'test_1' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test_1', test_1)

@norecursion
def test_2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_2'
    module_type_store = module_type_store.open_function_context('test_2', 24, 0, False)
    
    # Passed parameters checking function
    test_2.stypy_localization = localization
    test_2.stypy_type_of_self = None
    test_2.stypy_type_store = module_type_store
    test_2.stypy_function_name = 'test_2'
    test_2.stypy_param_names_list = []
    test_2.stypy_varargs_param_name = None
    test_2.stypy_kwargs_param_name = None
    test_2.stypy_call_defaults = defaults
    test_2.stypy_call_varargs = varargs
    test_2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_2', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_2', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_2(...)' code ##################

    
    # Assigning a List to a Name (line 25):
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_205653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_205654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    int_205655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 11), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 10), list_205654, int_205655)
    # Adding element type (line 25)
    int_205656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 10), list_205654, int_205656)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 9), list_205653, list_205654)
    # Adding element type (line 25)
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_205657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    int_205658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 18), list_205657, int_205658)
    # Adding element type (line 25)
    int_205659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 18), list_205657, int_205659)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 9), list_205653, list_205657)
    
    # Assigning a type to the variable 'H0' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'H0', list_205653)

    @norecursion
    def f(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 27, 4, False)
        
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

        
        # Call to dot(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'x' (line 28)
        x_205662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'x', False)
        
        # Call to dot(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Call to inv(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'H0' (line 28)
        H0_205668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 49), 'H0', False)
        # Processing the call keyword arguments (line 28)
        kwargs_205669 = {}
        # Getting the type of 'scipy' (line 28)
        scipy_205665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 32), 'scipy', False)
        # Obtaining the member 'linalg' of a type (line 28)
        linalg_205666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 32), scipy_205665, 'linalg')
        # Obtaining the member 'inv' of a type (line 28)
        inv_205667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 32), linalg_205666, 'inv')
        # Calling inv(args, kwargs) (line 28)
        inv_call_result_205670 = invoke(stypy.reporting.localization.Localization(__file__, 28, 32), inv_205667, *[H0_205668], **kwargs_205669)
        
        # Getting the type of 'x' (line 28)
        x_205671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 54), 'x', False)
        # Processing the call keyword arguments (line 28)
        kwargs_205672 = {}
        # Getting the type of 'np' (line 28)
        np_205663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 25), 'np', False)
        # Obtaining the member 'dot' of a type (line 28)
        dot_205664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 25), np_205663, 'dot')
        # Calling dot(args, kwargs) (line 28)
        dot_call_result_205673 = invoke(stypy.reporting.localization.Localization(__file__, 28, 25), dot_205664, *[inv_call_result_205670, x_205671], **kwargs_205672)
        
        # Processing the call keyword arguments (line 28)
        kwargs_205674 = {}
        # Getting the type of 'np' (line 28)
        np_205660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'np', False)
        # Obtaining the member 'dot' of a type (line 28)
        dot_205661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 15), np_205660, 'dot')
        # Calling dot(args, kwargs) (line 28)
        dot_call_result_205675 = invoke(stypy.reporting.localization.Localization(__file__, 28, 15), dot_205661, *[x_205662, dot_call_result_205673], **kwargs_205674)
        
        # Assigning a type to the variable 'stypy_return_type' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type', dot_call_result_205675)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_205676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_205676)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_205676

    # Assigning a type to the variable 'f' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'f', f)
    
    # Assigning a Call to a Name (line 30):
    
    # Call to minimize(...): (line 30)
    # Processing the call keyword arguments (line 30)
    # Getting the type of 'f' (line 30)
    f_205678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 27), 'f', False)
    keyword_205679 = f_205678
    str_205680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 37), 'str', 'L-BFGS-B')
    keyword_205681 = str_205680
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_205682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 52), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    int_205683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 52), list_205682, int_205683)
    # Adding element type (line 30)
    int_205684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 57), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 52), list_205682, int_205684)
    
    keyword_205685 = list_205682
    kwargs_205686 = {'fun': keyword_205679, 'x0': keyword_205685, 'method': keyword_205681}
    # Getting the type of 'minimize' (line 30)
    minimize_205677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 14), 'minimize', False)
    # Calling minimize(args, kwargs) (line 30)
    minimize_call_result_205687 = invoke(stypy.reporting.localization.Localization(__file__, 30, 14), minimize_205677, *[], **kwargs_205686)
    
    # Assigning a type to the variable 'result1' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'result1', minimize_call_result_205687)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to minimize(...): (line 31)
    # Processing the call keyword arguments (line 31)
    # Getting the type of 'f' (line 31)
    f_205689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 27), 'f', False)
    keyword_205690 = f_205689
    str_205691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 37), 'str', 'BFGS')
    keyword_205692 = str_205691
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_205693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 48), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    # Adding element type (line 31)
    int_205694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 49), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 48), list_205693, int_205694)
    # Adding element type (line 31)
    int_205695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 53), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 48), list_205693, int_205695)
    
    keyword_205696 = list_205693
    kwargs_205697 = {'fun': keyword_205690, 'x0': keyword_205696, 'method': keyword_205692}
    # Getting the type of 'minimize' (line 31)
    minimize_205688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'minimize', False)
    # Calling minimize(args, kwargs) (line 31)
    minimize_call_result_205698 = invoke(stypy.reporting.localization.Localization(__file__, 31, 14), minimize_205688, *[], **kwargs_205697)
    
    # Assigning a type to the variable 'result2' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'result2', minimize_call_result_205698)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to todense(...): (line 33)
    # Processing the call keyword arguments (line 33)
    kwargs_205702 = {}
    # Getting the type of 'result1' (line 33)
    result1_205699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 9), 'result1', False)
    # Obtaining the member 'hess_inv' of a type (line 33)
    hess_inv_205700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 9), result1_205699, 'hess_inv')
    # Obtaining the member 'todense' of a type (line 33)
    todense_205701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 9), hess_inv_205700, 'todense')
    # Calling todense(args, kwargs) (line 33)
    todense_call_result_205703 = invoke(stypy.reporting.localization.Localization(__file__, 33, 9), todense_205701, *[], **kwargs_205702)
    
    # Assigning a type to the variable 'H1' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'H1', todense_call_result_205703)
    
    # Assigning a Call to a Name (line 35):
    
    # Call to vstack(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Obtaining an instance of the builtin type 'tuple' (line 36)
    tuple_205706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 8), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 36)
    # Adding element type (line 36)
    
    # Call to hess_inv(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Call to array(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Obtaining an instance of the builtin type 'list' (line 36)
    list_205711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 36)
    # Adding element type (line 36)
    int_205712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 34), list_205711, int_205712)
    # Adding element type (line 36)
    int_205713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 34), list_205711, int_205713)
    
    # Processing the call keyword arguments (line 36)
    kwargs_205714 = {}
    # Getting the type of 'np' (line 36)
    np_205709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 25), 'np', False)
    # Obtaining the member 'array' of a type (line 36)
    array_205710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 25), np_205709, 'array')
    # Calling array(args, kwargs) (line 36)
    array_call_result_205715 = invoke(stypy.reporting.localization.Localization(__file__, 36, 25), array_205710, *[list_205711], **kwargs_205714)
    
    # Processing the call keyword arguments (line 36)
    kwargs_205716 = {}
    # Getting the type of 'result1' (line 36)
    result1_205707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'result1', False)
    # Obtaining the member 'hess_inv' of a type (line 36)
    hess_inv_205708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), result1_205707, 'hess_inv')
    # Calling hess_inv(args, kwargs) (line 36)
    hess_inv_call_result_205717 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), hess_inv_205708, *[array_call_result_205715], **kwargs_205716)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 8), tuple_205706, hess_inv_call_result_205717)
    # Adding element type (line 36)
    
    # Call to hess_inv(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Call to array(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_205722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    int_205723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 34), list_205722, int_205723)
    # Adding element type (line 37)
    int_205724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 34), list_205722, int_205724)
    
    # Processing the call keyword arguments (line 37)
    kwargs_205725 = {}
    # Getting the type of 'np' (line 37)
    np_205720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 25), 'np', False)
    # Obtaining the member 'array' of a type (line 37)
    array_205721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 25), np_205720, 'array')
    # Calling array(args, kwargs) (line 37)
    array_call_result_205726 = invoke(stypy.reporting.localization.Localization(__file__, 37, 25), array_205721, *[list_205722], **kwargs_205725)
    
    # Processing the call keyword arguments (line 37)
    kwargs_205727 = {}
    # Getting the type of 'result1' (line 37)
    result1_205718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'result1', False)
    # Obtaining the member 'hess_inv' of a type (line 37)
    hess_inv_205719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), result1_205718, 'hess_inv')
    # Calling hess_inv(args, kwargs) (line 37)
    hess_inv_call_result_205728 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), hess_inv_205719, *[array_call_result_205726], **kwargs_205727)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 8), tuple_205706, hess_inv_call_result_205728)
    
    # Processing the call keyword arguments (line 35)
    kwargs_205729 = {}
    # Getting the type of 'np' (line 35)
    np_205704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 9), 'np', False)
    # Obtaining the member 'vstack' of a type (line 35)
    vstack_205705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 9), np_205704, 'vstack')
    # Calling vstack(args, kwargs) (line 35)
    vstack_call_result_205730 = invoke(stypy.reporting.localization.Localization(__file__, 35, 9), vstack_205705, *[tuple_205706], **kwargs_205729)
    
    # Assigning a type to the variable 'H2' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'H2', vstack_call_result_205730)
    
    # Call to assert_allclose(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Call to reshape(...): (line 40)
    # Processing the call arguments (line 40)
    int_205749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 64), 'int')
    # Processing the call keyword arguments (line 40)
    kwargs_205750 = {}
    
    # Call to hess_inv(...): (line 40)
    # Processing the call arguments (line 40)
    
    # Call to reshape(...): (line 40)
    # Processing the call arguments (line 40)
    int_205742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 50), 'int')
    int_205743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 52), 'int')
    # Processing the call keyword arguments (line 40)
    kwargs_205744 = {}
    
    # Call to array(...): (line 40)
    # Processing the call arguments (line 40)
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_205736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    # Adding element type (line 40)
    int_205737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 34), list_205736, int_205737)
    # Adding element type (line 40)
    int_205738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 34), list_205736, int_205738)
    
    # Processing the call keyword arguments (line 40)
    kwargs_205739 = {}
    # Getting the type of 'np' (line 40)
    np_205734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 25), 'np', False)
    # Obtaining the member 'array' of a type (line 40)
    array_205735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 25), np_205734, 'array')
    # Calling array(args, kwargs) (line 40)
    array_call_result_205740 = invoke(stypy.reporting.localization.Localization(__file__, 40, 25), array_205735, *[list_205736], **kwargs_205739)
    
    # Obtaining the member 'reshape' of a type (line 40)
    reshape_205741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 25), array_call_result_205740, 'reshape')
    # Calling reshape(args, kwargs) (line 40)
    reshape_call_result_205745 = invoke(stypy.reporting.localization.Localization(__file__, 40, 25), reshape_205741, *[int_205742, int_205743], **kwargs_205744)
    
    # Processing the call keyword arguments (line 40)
    kwargs_205746 = {}
    # Getting the type of 'result1' (line 40)
    result1_205732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'result1', False)
    # Obtaining the member 'hess_inv' of a type (line 40)
    hess_inv_205733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), result1_205732, 'hess_inv')
    # Calling hess_inv(args, kwargs) (line 40)
    hess_inv_call_result_205747 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), hess_inv_205733, *[reshape_call_result_205745], **kwargs_205746)
    
    # Obtaining the member 'reshape' of a type (line 40)
    reshape_205748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), hess_inv_call_result_205747, 'reshape')
    # Calling reshape(args, kwargs) (line 40)
    reshape_call_result_205751 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), reshape_205748, *[int_205749], **kwargs_205750)
    
    
    # Call to hess_inv(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Call to array(...): (line 41)
    # Processing the call arguments (line 41)
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_205756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    int_205757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 34), list_205756, int_205757)
    # Adding element type (line 41)
    int_205758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 38), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 34), list_205756, int_205758)
    
    # Processing the call keyword arguments (line 41)
    kwargs_205759 = {}
    # Getting the type of 'np' (line 41)
    np_205754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'np', False)
    # Obtaining the member 'array' of a type (line 41)
    array_205755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 25), np_205754, 'array')
    # Calling array(args, kwargs) (line 41)
    array_call_result_205760 = invoke(stypy.reporting.localization.Localization(__file__, 41, 25), array_205755, *[list_205756], **kwargs_205759)
    
    # Processing the call keyword arguments (line 41)
    kwargs_205761 = {}
    # Getting the type of 'result1' (line 41)
    result1_205752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'result1', False)
    # Obtaining the member 'hess_inv' of a type (line 41)
    hess_inv_205753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), result1_205752, 'hess_inv')
    # Calling hess_inv(args, kwargs) (line 41)
    hess_inv_call_result_205762 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), hess_inv_205753, *[array_call_result_205760], **kwargs_205761)
    
    # Processing the call keyword arguments (line 39)
    kwargs_205763 = {}
    # Getting the type of 'assert_allclose' (line 39)
    assert_allclose_205731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 39)
    assert_allclose_call_result_205764 = invoke(stypy.reporting.localization.Localization(__file__, 39, 4), assert_allclose_205731, *[reshape_call_result_205751, hess_inv_call_result_205762], **kwargs_205763)
    
    
    # Call to assert_allclose(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'H1' (line 42)
    H1_205766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'H1', False)
    # Getting the type of 'H2' (line 42)
    H2_205767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'H2', False)
    # Processing the call keyword arguments (line 42)
    kwargs_205768 = {}
    # Getting the type of 'assert_allclose' (line 42)
    assert_allclose_205765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 42)
    assert_allclose_call_result_205769 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), assert_allclose_205765, *[H1_205766, H2_205767], **kwargs_205768)
    
    
    # Call to assert_allclose(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'H1' (line 43)
    H1_205771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'H1', False)
    # Getting the type of 'result2' (line 43)
    result2_205772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'result2', False)
    # Obtaining the member 'hess_inv' of a type (line 43)
    hess_inv_205773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 24), result2_205772, 'hess_inv')
    # Processing the call keyword arguments (line 43)
    float_205774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 47), 'float')
    keyword_205775 = float_205774
    float_205776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 58), 'float')
    keyword_205777 = float_205776
    kwargs_205778 = {'rtol': keyword_205775, 'atol': keyword_205777}
    # Getting the type of 'assert_allclose' (line 43)
    assert_allclose_205770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 43)
    assert_allclose_call_result_205779 = invoke(stypy.reporting.localization.Localization(__file__, 43, 4), assert_allclose_205770, *[H1_205771, hess_inv_205773], **kwargs_205778)
    
    
    # ################# End of 'test_2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_2' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_205780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_205780)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_2'
    return stypy_return_type_205780

# Assigning a type to the variable 'test_2' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'test_2', test_2)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
