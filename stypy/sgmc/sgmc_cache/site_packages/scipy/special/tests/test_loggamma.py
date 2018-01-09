
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_allclose
5: 
6: from scipy.special._testutils import FuncData
7: from scipy.special import gamma, gammaln, loggamma
8: 
9: 
10: def test_identities1():
11:     # test the identity exp(loggamma(z)) = gamma(z)
12:     x = np.array([-99.5, -9.5, -0.5, 0.5, 9.5, 99.5])
13:     y = x.copy()
14:     x, y = np.meshgrid(x, y)
15:     z = (x + 1J*y).flatten()
16:     dataset = np.vstack((z, gamma(z))).T
17: 
18:     def f(z):
19:         return np.exp(loggamma(z))
20: 
21:     FuncData(f, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()
22: 
23: 
24: def test_identities2():
25:     # test the identity loggamma(z + 1) = log(z) + loggamma(z)
26:     x = np.array([-99.5, -9.5, -0.5, 0.5, 9.5, 99.5])
27:     y = x.copy()
28:     x, y = np.meshgrid(x, y)
29:     z = (x + 1J*y).flatten()
30:     dataset = np.vstack((z, np.log(z) + loggamma(z))).T
31: 
32:     def f(z):
33:         return loggamma(z + 1)
34: 
35:     FuncData(f, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()
36: 
37: 
38: def test_realpart():
39:     # Test that the real parts of loggamma and gammaln agree on the
40:     # real axis.
41:     x = np.r_[-np.logspace(10, -10), np.logspace(-10, 10)] + 0.5
42:     dataset = np.vstack((x, gammaln(x))).T
43: 
44:     def f(z):
45:         return loggamma(z).real
46:     
47:     FuncData(f, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()
48: 
49: 
50: def test_gh_6536():
51:     z = loggamma(complex(-3.4, +0.0))
52:     zbar = loggamma(complex(-3.4, -0.0))
53:     assert_allclose(z, zbar.conjugate(), rtol=1e-15, atol=0)
54:     
55: 
56: def test_branch_cut():
57:     # Make sure negative zero is treated correctly
58:     x = -np.logspace(300, -30, 100)
59:     z = np.asarray([complex(x0, 0.0) for x0 in x])
60:     zbar = np.asarray([complex(x0, -0.0) for x0 in x])
61:     assert_allclose(z, zbar.conjugate(), rtol=1e-15, atol=0)
62: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_541433 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_541433) is not StypyTypeError):

    if (import_541433 != 'pyd_module'):
        __import__(import_541433)
        sys_modules_541434 = sys.modules[import_541433]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_541434.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_541433)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_allclose' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_541435 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_541435) is not StypyTypeError):

    if (import_541435 != 'pyd_module'):
        __import__(import_541435)
        sys_modules_541436 = sys.modules[import_541435]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_541436.module_type_store, module_type_store, ['assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_541436, sys_modules_541436.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_allclose'], [assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_541435)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.special._testutils import FuncData' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_541437 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._testutils')

if (type(import_541437) is not StypyTypeError):

    if (import_541437 != 'pyd_module'):
        __import__(import_541437)
        sys_modules_541438 = sys.modules[import_541437]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._testutils', sys_modules_541438.module_type_store, module_type_store, ['FuncData'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_541438, sys_modules_541438.module_type_store, module_type_store)
    else:
        from scipy.special._testutils import FuncData

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._testutils', None, module_type_store, ['FuncData'], [FuncData])

else:
    # Assigning a type to the variable 'scipy.special._testutils' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._testutils', import_541437)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.special import gamma, gammaln, loggamma' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_541439 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special')

if (type(import_541439) is not StypyTypeError):

    if (import_541439 != 'pyd_module'):
        __import__(import_541439)
        sys_modules_541440 = sys.modules[import_541439]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special', sys_modules_541440.module_type_store, module_type_store, ['gamma', 'gammaln', 'loggamma'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_541440, sys_modules_541440.module_type_store, module_type_store)
    else:
        from scipy.special import gamma, gammaln, loggamma

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special', None, module_type_store, ['gamma', 'gammaln', 'loggamma'], [gamma, gammaln, loggamma])

else:
    # Assigning a type to the variable 'scipy.special' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special', import_541439)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')


@norecursion
def test_identities1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_identities1'
    module_type_store = module_type_store.open_function_context('test_identities1', 10, 0, False)
    
    # Passed parameters checking function
    test_identities1.stypy_localization = localization
    test_identities1.stypy_type_of_self = None
    test_identities1.stypy_type_store = module_type_store
    test_identities1.stypy_function_name = 'test_identities1'
    test_identities1.stypy_param_names_list = []
    test_identities1.stypy_varargs_param_name = None
    test_identities1.stypy_kwargs_param_name = None
    test_identities1.stypy_call_defaults = defaults
    test_identities1.stypy_call_varargs = varargs
    test_identities1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_identities1', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_identities1', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_identities1(...)' code ##################

    
    # Assigning a Call to a Name (line 12):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to array(...): (line 12)
    # Processing the call arguments (line 12)
    
    # Obtaining an instance of the builtin type 'list' (line 12)
    list_541443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 12)
    # Adding element type (line 12)
    float_541444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 17), list_541443, float_541444)
    # Adding element type (line 12)
    float_541445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 17), list_541443, float_541445)
    # Adding element type (line 12)
    float_541446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 17), list_541443, float_541446)
    # Adding element type (line 12)
    float_541447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 37), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 17), list_541443, float_541447)
    # Adding element type (line 12)
    float_541448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 42), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 17), list_541443, float_541448)
    # Adding element type (line 12)
    float_541449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 47), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 17), list_541443, float_541449)
    
    # Processing the call keyword arguments (line 12)
    kwargs_541450 = {}
    # Getting the type of 'np' (line 12)
    np_541441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 12)
    array_541442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), np_541441, 'array')
    # Calling array(args, kwargs) (line 12)
    array_call_result_541451 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), array_541442, *[list_541443], **kwargs_541450)
    
    # Assigning a type to the variable 'x' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'x', array_call_result_541451)
    
    # Assigning a Call to a Name (line 13):
    
    # Assigning a Call to a Name (line 13):
    
    # Call to copy(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_541454 = {}
    # Getting the type of 'x' (line 13)
    x_541452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'x', False)
    # Obtaining the member 'copy' of a type (line 13)
    copy_541453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), x_541452, 'copy')
    # Calling copy(args, kwargs) (line 13)
    copy_call_result_541455 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), copy_541453, *[], **kwargs_541454)
    
    # Assigning a type to the variable 'y' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'y', copy_call_result_541455)
    
    # Assigning a Call to a Tuple (line 14):
    
    # Assigning a Subscript to a Name (line 14):
    
    # Obtaining the type of the subscript
    int_541456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'int')
    
    # Call to meshgrid(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'x' (line 14)
    x_541459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 23), 'x', False)
    # Getting the type of 'y' (line 14)
    y_541460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 26), 'y', False)
    # Processing the call keyword arguments (line 14)
    kwargs_541461 = {}
    # Getting the type of 'np' (line 14)
    np_541457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 14)
    meshgrid_541458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 11), np_541457, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 14)
    meshgrid_call_result_541462 = invoke(stypy.reporting.localization.Localization(__file__, 14, 11), meshgrid_541458, *[x_541459, y_541460], **kwargs_541461)
    
    # Obtaining the member '__getitem__' of a type (line 14)
    getitem___541463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), meshgrid_call_result_541462, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 14)
    subscript_call_result_541464 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), getitem___541463, int_541456)
    
    # Assigning a type to the variable 'tuple_var_assignment_541429' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'tuple_var_assignment_541429', subscript_call_result_541464)
    
    # Assigning a Subscript to a Name (line 14):
    
    # Obtaining the type of the subscript
    int_541465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'int')
    
    # Call to meshgrid(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'x' (line 14)
    x_541468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 23), 'x', False)
    # Getting the type of 'y' (line 14)
    y_541469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 26), 'y', False)
    # Processing the call keyword arguments (line 14)
    kwargs_541470 = {}
    # Getting the type of 'np' (line 14)
    np_541466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 14)
    meshgrid_541467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 11), np_541466, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 14)
    meshgrid_call_result_541471 = invoke(stypy.reporting.localization.Localization(__file__, 14, 11), meshgrid_541467, *[x_541468, y_541469], **kwargs_541470)
    
    # Obtaining the member '__getitem__' of a type (line 14)
    getitem___541472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), meshgrid_call_result_541471, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 14)
    subscript_call_result_541473 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), getitem___541472, int_541465)
    
    # Assigning a type to the variable 'tuple_var_assignment_541430' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'tuple_var_assignment_541430', subscript_call_result_541473)
    
    # Assigning a Name to a Name (line 14):
    # Getting the type of 'tuple_var_assignment_541429' (line 14)
    tuple_var_assignment_541429_541474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'tuple_var_assignment_541429')
    # Assigning a type to the variable 'x' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'x', tuple_var_assignment_541429_541474)
    
    # Assigning a Name to a Name (line 14):
    # Getting the type of 'tuple_var_assignment_541430' (line 14)
    tuple_var_assignment_541430_541475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'tuple_var_assignment_541430')
    # Assigning a type to the variable 'y' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 7), 'y', tuple_var_assignment_541430_541475)
    
    # Assigning a Call to a Name (line 15):
    
    # Assigning a Call to a Name (line 15):
    
    # Call to flatten(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_541482 = {}
    # Getting the type of 'x' (line 15)
    x_541476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'x', False)
    complex_541477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 13), 'complex')
    # Getting the type of 'y' (line 15)
    y_541478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 16), 'y', False)
    # Applying the binary operator '*' (line 15)
    result_mul_541479 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 13), '*', complex_541477, y_541478)
    
    # Applying the binary operator '+' (line 15)
    result_add_541480 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 9), '+', x_541476, result_mul_541479)
    
    # Obtaining the member 'flatten' of a type (line 15)
    flatten_541481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 9), result_add_541480, 'flatten')
    # Calling flatten(args, kwargs) (line 15)
    flatten_call_result_541483 = invoke(stypy.reporting.localization.Localization(__file__, 15, 9), flatten_541481, *[], **kwargs_541482)
    
    # Assigning a type to the variable 'z' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'z', flatten_call_result_541483)
    
    # Assigning a Attribute to a Name (line 16):
    
    # Assigning a Attribute to a Name (line 16):
    
    # Call to vstack(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Obtaining an instance of the builtin type 'tuple' (line 16)
    tuple_541486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 16)
    # Adding element type (line 16)
    # Getting the type of 'z' (line 16)
    z_541487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 25), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 25), tuple_541486, z_541487)
    # Adding element type (line 16)
    
    # Call to gamma(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'z' (line 16)
    z_541489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 34), 'z', False)
    # Processing the call keyword arguments (line 16)
    kwargs_541490 = {}
    # Getting the type of 'gamma' (line 16)
    gamma_541488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 28), 'gamma', False)
    # Calling gamma(args, kwargs) (line 16)
    gamma_call_result_541491 = invoke(stypy.reporting.localization.Localization(__file__, 16, 28), gamma_541488, *[z_541489], **kwargs_541490)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 25), tuple_541486, gamma_call_result_541491)
    
    # Processing the call keyword arguments (line 16)
    kwargs_541492 = {}
    # Getting the type of 'np' (line 16)
    np_541484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'np', False)
    # Obtaining the member 'vstack' of a type (line 16)
    vstack_541485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 14), np_541484, 'vstack')
    # Calling vstack(args, kwargs) (line 16)
    vstack_call_result_541493 = invoke(stypy.reporting.localization.Localization(__file__, 16, 14), vstack_541485, *[tuple_541486], **kwargs_541492)
    
    # Obtaining the member 'T' of a type (line 16)
    T_541494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 14), vstack_call_result_541493, 'T')
    # Assigning a type to the variable 'dataset' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'dataset', T_541494)

    @norecursion
    def f(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 18, 4, False)
        
        # Passed parameters checking function
        f.stypy_localization = localization
        f.stypy_type_of_self = None
        f.stypy_type_store = module_type_store
        f.stypy_function_name = 'f'
        f.stypy_param_names_list = ['z']
        f.stypy_varargs_param_name = None
        f.stypy_kwargs_param_name = None
        f.stypy_call_defaults = defaults
        f.stypy_call_varargs = varargs
        f.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'f', ['z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        
        # Call to exp(...): (line 19)
        # Processing the call arguments (line 19)
        
        # Call to loggamma(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'z' (line 19)
        z_541498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 31), 'z', False)
        # Processing the call keyword arguments (line 19)
        kwargs_541499 = {}
        # Getting the type of 'loggamma' (line 19)
        loggamma_541497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'loggamma', False)
        # Calling loggamma(args, kwargs) (line 19)
        loggamma_call_result_541500 = invoke(stypy.reporting.localization.Localization(__file__, 19, 22), loggamma_541497, *[z_541498], **kwargs_541499)
        
        # Processing the call keyword arguments (line 19)
        kwargs_541501 = {}
        # Getting the type of 'np' (line 19)
        np_541495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'np', False)
        # Obtaining the member 'exp' of a type (line 19)
        exp_541496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 15), np_541495, 'exp')
        # Calling exp(args, kwargs) (line 19)
        exp_call_result_541502 = invoke(stypy.reporting.localization.Localization(__file__, 19, 15), exp_541496, *[loggamma_call_result_541500], **kwargs_541501)
        
        # Assigning a type to the variable 'stypy_return_type' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'stypy_return_type', exp_call_result_541502)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_541503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_541503)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_541503

    # Assigning a type to the variable 'f' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'f', f)
    
    # Call to check(...): (line 21)
    # Processing the call keyword arguments (line 21)
    kwargs_541516 = {}
    
    # Call to FuncData(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'f' (line 21)
    f_541505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 13), 'f', False)
    # Getting the type of 'dataset' (line 21)
    dataset_541506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'dataset', False)
    int_541507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'int')
    int_541508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'int')
    # Processing the call keyword arguments (line 21)
    float_541509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 36), 'float')
    keyword_541510 = float_541509
    float_541511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 48), 'float')
    keyword_541512 = float_541511
    kwargs_541513 = {'rtol': keyword_541510, 'atol': keyword_541512}
    # Getting the type of 'FuncData' (line 21)
    FuncData_541504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'FuncData', False)
    # Calling FuncData(args, kwargs) (line 21)
    FuncData_call_result_541514 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), FuncData_541504, *[f_541505, dataset_541506, int_541507, int_541508], **kwargs_541513)
    
    # Obtaining the member 'check' of a type (line 21)
    check_541515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 4), FuncData_call_result_541514, 'check')
    # Calling check(args, kwargs) (line 21)
    check_call_result_541517 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), check_541515, *[], **kwargs_541516)
    
    
    # ################# End of 'test_identities1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_identities1' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_541518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_541518)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_identities1'
    return stypy_return_type_541518

# Assigning a type to the variable 'test_identities1' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'test_identities1', test_identities1)

@norecursion
def test_identities2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_identities2'
    module_type_store = module_type_store.open_function_context('test_identities2', 24, 0, False)
    
    # Passed parameters checking function
    test_identities2.stypy_localization = localization
    test_identities2.stypy_type_of_self = None
    test_identities2.stypy_type_store = module_type_store
    test_identities2.stypy_function_name = 'test_identities2'
    test_identities2.stypy_param_names_list = []
    test_identities2.stypy_varargs_param_name = None
    test_identities2.stypy_kwargs_param_name = None
    test_identities2.stypy_call_defaults = defaults
    test_identities2.stypy_call_varargs = varargs
    test_identities2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_identities2', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_identities2', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_identities2(...)' code ##################

    
    # Assigning a Call to a Name (line 26):
    
    # Assigning a Call to a Name (line 26):
    
    # Call to array(...): (line 26)
    # Processing the call arguments (line 26)
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_541521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    float_541522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 17), list_541521, float_541522)
    # Adding element type (line 26)
    float_541523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 17), list_541521, float_541523)
    # Adding element type (line 26)
    float_541524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 17), list_541521, float_541524)
    # Adding element type (line 26)
    float_541525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 37), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 17), list_541521, float_541525)
    # Adding element type (line 26)
    float_541526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 42), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 17), list_541521, float_541526)
    # Adding element type (line 26)
    float_541527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 47), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 17), list_541521, float_541527)
    
    # Processing the call keyword arguments (line 26)
    kwargs_541528 = {}
    # Getting the type of 'np' (line 26)
    np_541519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 26)
    array_541520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), np_541519, 'array')
    # Calling array(args, kwargs) (line 26)
    array_call_result_541529 = invoke(stypy.reporting.localization.Localization(__file__, 26, 8), array_541520, *[list_541521], **kwargs_541528)
    
    # Assigning a type to the variable 'x' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'x', array_call_result_541529)
    
    # Assigning a Call to a Name (line 27):
    
    # Assigning a Call to a Name (line 27):
    
    # Call to copy(...): (line 27)
    # Processing the call keyword arguments (line 27)
    kwargs_541532 = {}
    # Getting the type of 'x' (line 27)
    x_541530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'x', False)
    # Obtaining the member 'copy' of a type (line 27)
    copy_541531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), x_541530, 'copy')
    # Calling copy(args, kwargs) (line 27)
    copy_call_result_541533 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), copy_541531, *[], **kwargs_541532)
    
    # Assigning a type to the variable 'y' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'y', copy_call_result_541533)
    
    # Assigning a Call to a Tuple (line 28):
    
    # Assigning a Subscript to a Name (line 28):
    
    # Obtaining the type of the subscript
    int_541534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'int')
    
    # Call to meshgrid(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'x' (line 28)
    x_541537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 'x', False)
    # Getting the type of 'y' (line 28)
    y_541538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'y', False)
    # Processing the call keyword arguments (line 28)
    kwargs_541539 = {}
    # Getting the type of 'np' (line 28)
    np_541535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 28)
    meshgrid_541536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 11), np_541535, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 28)
    meshgrid_call_result_541540 = invoke(stypy.reporting.localization.Localization(__file__, 28, 11), meshgrid_541536, *[x_541537, y_541538], **kwargs_541539)
    
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___541541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 4), meshgrid_call_result_541540, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_541542 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), getitem___541541, int_541534)
    
    # Assigning a type to the variable 'tuple_var_assignment_541431' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'tuple_var_assignment_541431', subscript_call_result_541542)
    
    # Assigning a Subscript to a Name (line 28):
    
    # Obtaining the type of the subscript
    int_541543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'int')
    
    # Call to meshgrid(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'x' (line 28)
    x_541546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 'x', False)
    # Getting the type of 'y' (line 28)
    y_541547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'y', False)
    # Processing the call keyword arguments (line 28)
    kwargs_541548 = {}
    # Getting the type of 'np' (line 28)
    np_541544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 28)
    meshgrid_541545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 11), np_541544, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 28)
    meshgrid_call_result_541549 = invoke(stypy.reporting.localization.Localization(__file__, 28, 11), meshgrid_541545, *[x_541546, y_541547], **kwargs_541548)
    
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___541550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 4), meshgrid_call_result_541549, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_541551 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), getitem___541550, int_541543)
    
    # Assigning a type to the variable 'tuple_var_assignment_541432' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'tuple_var_assignment_541432', subscript_call_result_541551)
    
    # Assigning a Name to a Name (line 28):
    # Getting the type of 'tuple_var_assignment_541431' (line 28)
    tuple_var_assignment_541431_541552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'tuple_var_assignment_541431')
    # Assigning a type to the variable 'x' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'x', tuple_var_assignment_541431_541552)
    
    # Assigning a Name to a Name (line 28):
    # Getting the type of 'tuple_var_assignment_541432' (line 28)
    tuple_var_assignment_541432_541553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'tuple_var_assignment_541432')
    # Assigning a type to the variable 'y' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 7), 'y', tuple_var_assignment_541432_541553)
    
    # Assigning a Call to a Name (line 29):
    
    # Assigning a Call to a Name (line 29):
    
    # Call to flatten(...): (line 29)
    # Processing the call keyword arguments (line 29)
    kwargs_541560 = {}
    # Getting the type of 'x' (line 29)
    x_541554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 9), 'x', False)
    complex_541555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 13), 'complex')
    # Getting the type of 'y' (line 29)
    y_541556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), 'y', False)
    # Applying the binary operator '*' (line 29)
    result_mul_541557 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 13), '*', complex_541555, y_541556)
    
    # Applying the binary operator '+' (line 29)
    result_add_541558 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 9), '+', x_541554, result_mul_541557)
    
    # Obtaining the member 'flatten' of a type (line 29)
    flatten_541559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 9), result_add_541558, 'flatten')
    # Calling flatten(args, kwargs) (line 29)
    flatten_call_result_541561 = invoke(stypy.reporting.localization.Localization(__file__, 29, 9), flatten_541559, *[], **kwargs_541560)
    
    # Assigning a type to the variable 'z' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'z', flatten_call_result_541561)
    
    # Assigning a Attribute to a Name (line 30):
    
    # Assigning a Attribute to a Name (line 30):
    
    # Call to vstack(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Obtaining an instance of the builtin type 'tuple' (line 30)
    tuple_541564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 30)
    # Adding element type (line 30)
    # Getting the type of 'z' (line 30)
    z_541565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), 'z', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 25), tuple_541564, z_541565)
    # Adding element type (line 30)
    
    # Call to log(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'z' (line 30)
    z_541568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 35), 'z', False)
    # Processing the call keyword arguments (line 30)
    kwargs_541569 = {}
    # Getting the type of 'np' (line 30)
    np_541566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 28), 'np', False)
    # Obtaining the member 'log' of a type (line 30)
    log_541567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 28), np_541566, 'log')
    # Calling log(args, kwargs) (line 30)
    log_call_result_541570 = invoke(stypy.reporting.localization.Localization(__file__, 30, 28), log_541567, *[z_541568], **kwargs_541569)
    
    
    # Call to loggamma(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'z' (line 30)
    z_541572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 49), 'z', False)
    # Processing the call keyword arguments (line 30)
    kwargs_541573 = {}
    # Getting the type of 'loggamma' (line 30)
    loggamma_541571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 40), 'loggamma', False)
    # Calling loggamma(args, kwargs) (line 30)
    loggamma_call_result_541574 = invoke(stypy.reporting.localization.Localization(__file__, 30, 40), loggamma_541571, *[z_541572], **kwargs_541573)
    
    # Applying the binary operator '+' (line 30)
    result_add_541575 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 28), '+', log_call_result_541570, loggamma_call_result_541574)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 25), tuple_541564, result_add_541575)
    
    # Processing the call keyword arguments (line 30)
    kwargs_541576 = {}
    # Getting the type of 'np' (line 30)
    np_541562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 14), 'np', False)
    # Obtaining the member 'vstack' of a type (line 30)
    vstack_541563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 14), np_541562, 'vstack')
    # Calling vstack(args, kwargs) (line 30)
    vstack_call_result_541577 = invoke(stypy.reporting.localization.Localization(__file__, 30, 14), vstack_541563, *[tuple_541564], **kwargs_541576)
    
    # Obtaining the member 'T' of a type (line 30)
    T_541578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 14), vstack_call_result_541577, 'T')
    # Assigning a type to the variable 'dataset' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'dataset', T_541578)

    @norecursion
    def f(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 32, 4, False)
        
        # Passed parameters checking function
        f.stypy_localization = localization
        f.stypy_type_of_self = None
        f.stypy_type_store = module_type_store
        f.stypy_function_name = 'f'
        f.stypy_param_names_list = ['z']
        f.stypy_varargs_param_name = None
        f.stypy_kwargs_param_name = None
        f.stypy_call_defaults = defaults
        f.stypy_call_varargs = varargs
        f.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'f', ['z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        
        # Call to loggamma(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'z' (line 33)
        z_541580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 24), 'z', False)
        int_541581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 28), 'int')
        # Applying the binary operator '+' (line 33)
        result_add_541582 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 24), '+', z_541580, int_541581)
        
        # Processing the call keyword arguments (line 33)
        kwargs_541583 = {}
        # Getting the type of 'loggamma' (line 33)
        loggamma_541579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'loggamma', False)
        # Calling loggamma(args, kwargs) (line 33)
        loggamma_call_result_541584 = invoke(stypy.reporting.localization.Localization(__file__, 33, 15), loggamma_541579, *[result_add_541582], **kwargs_541583)
        
        # Assigning a type to the variable 'stypy_return_type' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type', loggamma_call_result_541584)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_541585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_541585)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_541585

    # Assigning a type to the variable 'f' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'f', f)
    
    # Call to check(...): (line 35)
    # Processing the call keyword arguments (line 35)
    kwargs_541598 = {}
    
    # Call to FuncData(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'f' (line 35)
    f_541587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 13), 'f', False)
    # Getting the type of 'dataset' (line 35)
    dataset_541588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'dataset', False)
    int_541589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 25), 'int')
    int_541590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 28), 'int')
    # Processing the call keyword arguments (line 35)
    float_541591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'float')
    keyword_541592 = float_541591
    float_541593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 48), 'float')
    keyword_541594 = float_541593
    kwargs_541595 = {'rtol': keyword_541592, 'atol': keyword_541594}
    # Getting the type of 'FuncData' (line 35)
    FuncData_541586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'FuncData', False)
    # Calling FuncData(args, kwargs) (line 35)
    FuncData_call_result_541596 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), FuncData_541586, *[f_541587, dataset_541588, int_541589, int_541590], **kwargs_541595)
    
    # Obtaining the member 'check' of a type (line 35)
    check_541597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), FuncData_call_result_541596, 'check')
    # Calling check(args, kwargs) (line 35)
    check_call_result_541599 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), check_541597, *[], **kwargs_541598)
    
    
    # ################# End of 'test_identities2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_identities2' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_541600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_541600)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_identities2'
    return stypy_return_type_541600

# Assigning a type to the variable 'test_identities2' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'test_identities2', test_identities2)

@norecursion
def test_realpart(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_realpart'
    module_type_store = module_type_store.open_function_context('test_realpart', 38, 0, False)
    
    # Passed parameters checking function
    test_realpart.stypy_localization = localization
    test_realpart.stypy_type_of_self = None
    test_realpart.stypy_type_store = module_type_store
    test_realpart.stypy_function_name = 'test_realpart'
    test_realpart.stypy_param_names_list = []
    test_realpart.stypy_varargs_param_name = None
    test_realpart.stypy_kwargs_param_name = None
    test_realpart.stypy_call_defaults = defaults
    test_realpart.stypy_call_varargs = varargs
    test_realpart.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_realpart', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_realpart', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_realpart(...)' code ##################

    
    # Assigning a BinOp to a Name (line 41):
    
    # Assigning a BinOp to a Name (line 41):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 41)
    tuple_541601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 41)
    # Adding element type (line 41)
    
    
    # Call to logspace(...): (line 41)
    # Processing the call arguments (line 41)
    int_541604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 27), 'int')
    int_541605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 31), 'int')
    # Processing the call keyword arguments (line 41)
    kwargs_541606 = {}
    # Getting the type of 'np' (line 41)
    np_541602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'np', False)
    # Obtaining the member 'logspace' of a type (line 41)
    logspace_541603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), np_541602, 'logspace')
    # Calling logspace(args, kwargs) (line 41)
    logspace_call_result_541607 = invoke(stypy.reporting.localization.Localization(__file__, 41, 15), logspace_541603, *[int_541604, int_541605], **kwargs_541606)
    
    # Applying the 'usub' unary operator (line 41)
    result___neg___541608 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 14), 'usub', logspace_call_result_541607)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), tuple_541601, result___neg___541608)
    # Adding element type (line 41)
    
    # Call to logspace(...): (line 41)
    # Processing the call arguments (line 41)
    int_541611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 49), 'int')
    int_541612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 54), 'int')
    # Processing the call keyword arguments (line 41)
    kwargs_541613 = {}
    # Getting the type of 'np' (line 41)
    np_541609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 37), 'np', False)
    # Obtaining the member 'logspace' of a type (line 41)
    logspace_541610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 37), np_541609, 'logspace')
    # Calling logspace(args, kwargs) (line 41)
    logspace_call_result_541614 = invoke(stypy.reporting.localization.Localization(__file__, 41, 37), logspace_541610, *[int_541611, int_541612], **kwargs_541613)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), tuple_541601, logspace_call_result_541614)
    
    # Getting the type of 'np' (line 41)
    np_541615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'np')
    # Obtaining the member 'r_' of a type (line 41)
    r__541616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), np_541615, 'r_')
    # Obtaining the member '__getitem__' of a type (line 41)
    getitem___541617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), r__541616, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 41)
    subscript_call_result_541618 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), getitem___541617, tuple_541601)
    
    float_541619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 61), 'float')
    # Applying the binary operator '+' (line 41)
    result_add_541620 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 8), '+', subscript_call_result_541618, float_541619)
    
    # Assigning a type to the variable 'x' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'x', result_add_541620)
    
    # Assigning a Attribute to a Name (line 42):
    
    # Assigning a Attribute to a Name (line 42):
    
    # Call to vstack(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Obtaining an instance of the builtin type 'tuple' (line 42)
    tuple_541623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 42)
    # Adding element type (line 42)
    # Getting the type of 'x' (line 42)
    x_541624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 25), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 25), tuple_541623, x_541624)
    # Adding element type (line 42)
    
    # Call to gammaln(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'x' (line 42)
    x_541626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 36), 'x', False)
    # Processing the call keyword arguments (line 42)
    kwargs_541627 = {}
    # Getting the type of 'gammaln' (line 42)
    gammaln_541625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), 'gammaln', False)
    # Calling gammaln(args, kwargs) (line 42)
    gammaln_call_result_541628 = invoke(stypy.reporting.localization.Localization(__file__, 42, 28), gammaln_541625, *[x_541626], **kwargs_541627)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 25), tuple_541623, gammaln_call_result_541628)
    
    # Processing the call keyword arguments (line 42)
    kwargs_541629 = {}
    # Getting the type of 'np' (line 42)
    np_541621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 14), 'np', False)
    # Obtaining the member 'vstack' of a type (line 42)
    vstack_541622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 14), np_541621, 'vstack')
    # Calling vstack(args, kwargs) (line 42)
    vstack_call_result_541630 = invoke(stypy.reporting.localization.Localization(__file__, 42, 14), vstack_541622, *[tuple_541623], **kwargs_541629)
    
    # Obtaining the member 'T' of a type (line 42)
    T_541631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 14), vstack_call_result_541630, 'T')
    # Assigning a type to the variable 'dataset' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'dataset', T_541631)

    @norecursion
    def f(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'f'
        module_type_store = module_type_store.open_function_context('f', 44, 4, False)
        
        # Passed parameters checking function
        f.stypy_localization = localization
        f.stypy_type_of_self = None
        f.stypy_type_store = module_type_store
        f.stypy_function_name = 'f'
        f.stypy_param_names_list = ['z']
        f.stypy_varargs_param_name = None
        f.stypy_kwargs_param_name = None
        f.stypy_call_defaults = defaults
        f.stypy_call_varargs = varargs
        f.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'f', ['z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'f', localization, ['z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'f(...)' code ##################

        
        # Call to loggamma(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'z' (line 45)
        z_541633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 24), 'z', False)
        # Processing the call keyword arguments (line 45)
        kwargs_541634 = {}
        # Getting the type of 'loggamma' (line 45)
        loggamma_541632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'loggamma', False)
        # Calling loggamma(args, kwargs) (line 45)
        loggamma_call_result_541635 = invoke(stypy.reporting.localization.Localization(__file__, 45, 15), loggamma_541632, *[z_541633], **kwargs_541634)
        
        # Obtaining the member 'real' of a type (line 45)
        real_541636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 15), loggamma_call_result_541635, 'real')
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', real_541636)
        
        # ################# End of 'f(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'f' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_541637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_541637)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'f'
        return stypy_return_type_541637

    # Assigning a type to the variable 'f' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'f', f)
    
    # Call to check(...): (line 47)
    # Processing the call keyword arguments (line 47)
    kwargs_541650 = {}
    
    # Call to FuncData(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'f' (line 47)
    f_541639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 13), 'f', False)
    # Getting the type of 'dataset' (line 47)
    dataset_541640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'dataset', False)
    int_541641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 25), 'int')
    int_541642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 28), 'int')
    # Processing the call keyword arguments (line 47)
    float_541643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 36), 'float')
    keyword_541644 = float_541643
    float_541645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 48), 'float')
    keyword_541646 = float_541645
    kwargs_541647 = {'rtol': keyword_541644, 'atol': keyword_541646}
    # Getting the type of 'FuncData' (line 47)
    FuncData_541638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'FuncData', False)
    # Calling FuncData(args, kwargs) (line 47)
    FuncData_call_result_541648 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), FuncData_541638, *[f_541639, dataset_541640, int_541641, int_541642], **kwargs_541647)
    
    # Obtaining the member 'check' of a type (line 47)
    check_541649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 4), FuncData_call_result_541648, 'check')
    # Calling check(args, kwargs) (line 47)
    check_call_result_541651 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), check_541649, *[], **kwargs_541650)
    
    
    # ################# End of 'test_realpart(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_realpart' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_541652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_541652)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_realpart'
    return stypy_return_type_541652

# Assigning a type to the variable 'test_realpart' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'test_realpart', test_realpart)

@norecursion
def test_gh_6536(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_gh_6536'
    module_type_store = module_type_store.open_function_context('test_gh_6536', 50, 0, False)
    
    # Passed parameters checking function
    test_gh_6536.stypy_localization = localization
    test_gh_6536.stypy_type_of_self = None
    test_gh_6536.stypy_type_store = module_type_store
    test_gh_6536.stypy_function_name = 'test_gh_6536'
    test_gh_6536.stypy_param_names_list = []
    test_gh_6536.stypy_varargs_param_name = None
    test_gh_6536.stypy_kwargs_param_name = None
    test_gh_6536.stypy_call_defaults = defaults
    test_gh_6536.stypy_call_varargs = varargs
    test_gh_6536.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_gh_6536', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_gh_6536', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_gh_6536(...)' code ##################

    
    # Assigning a Call to a Name (line 51):
    
    # Assigning a Call to a Name (line 51):
    
    # Call to loggamma(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Call to complex(...): (line 51)
    # Processing the call arguments (line 51)
    float_541655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 25), 'float')
    
    float_541656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 32), 'float')
    # Applying the 'uadd' unary operator (line 51)
    result___pos___541657 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 31), 'uadd', float_541656)
    
    # Processing the call keyword arguments (line 51)
    kwargs_541658 = {}
    # Getting the type of 'complex' (line 51)
    complex_541654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'complex', False)
    # Calling complex(args, kwargs) (line 51)
    complex_call_result_541659 = invoke(stypy.reporting.localization.Localization(__file__, 51, 17), complex_541654, *[float_541655, result___pos___541657], **kwargs_541658)
    
    # Processing the call keyword arguments (line 51)
    kwargs_541660 = {}
    # Getting the type of 'loggamma' (line 51)
    loggamma_541653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'loggamma', False)
    # Calling loggamma(args, kwargs) (line 51)
    loggamma_call_result_541661 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), loggamma_541653, *[complex_call_result_541659], **kwargs_541660)
    
    # Assigning a type to the variable 'z' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'z', loggamma_call_result_541661)
    
    # Assigning a Call to a Name (line 52):
    
    # Assigning a Call to a Name (line 52):
    
    # Call to loggamma(...): (line 52)
    # Processing the call arguments (line 52)
    
    # Call to complex(...): (line 52)
    # Processing the call arguments (line 52)
    float_541664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 28), 'float')
    float_541665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 34), 'float')
    # Processing the call keyword arguments (line 52)
    kwargs_541666 = {}
    # Getting the type of 'complex' (line 52)
    complex_541663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'complex', False)
    # Calling complex(args, kwargs) (line 52)
    complex_call_result_541667 = invoke(stypy.reporting.localization.Localization(__file__, 52, 20), complex_541663, *[float_541664, float_541665], **kwargs_541666)
    
    # Processing the call keyword arguments (line 52)
    kwargs_541668 = {}
    # Getting the type of 'loggamma' (line 52)
    loggamma_541662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'loggamma', False)
    # Calling loggamma(args, kwargs) (line 52)
    loggamma_call_result_541669 = invoke(stypy.reporting.localization.Localization(__file__, 52, 11), loggamma_541662, *[complex_call_result_541667], **kwargs_541668)
    
    # Assigning a type to the variable 'zbar' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'zbar', loggamma_call_result_541669)
    
    # Call to assert_allclose(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'z' (line 53)
    z_541671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'z', False)
    
    # Call to conjugate(...): (line 53)
    # Processing the call keyword arguments (line 53)
    kwargs_541674 = {}
    # Getting the type of 'zbar' (line 53)
    zbar_541672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 23), 'zbar', False)
    # Obtaining the member 'conjugate' of a type (line 53)
    conjugate_541673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 23), zbar_541672, 'conjugate')
    # Calling conjugate(args, kwargs) (line 53)
    conjugate_call_result_541675 = invoke(stypy.reporting.localization.Localization(__file__, 53, 23), conjugate_541673, *[], **kwargs_541674)
    
    # Processing the call keyword arguments (line 53)
    float_541676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 46), 'float')
    keyword_541677 = float_541676
    int_541678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 58), 'int')
    keyword_541679 = int_541678
    kwargs_541680 = {'rtol': keyword_541677, 'atol': keyword_541679}
    # Getting the type of 'assert_allclose' (line 53)
    assert_allclose_541670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 53)
    assert_allclose_call_result_541681 = invoke(stypy.reporting.localization.Localization(__file__, 53, 4), assert_allclose_541670, *[z_541671, conjugate_call_result_541675], **kwargs_541680)
    
    
    # ################# End of 'test_gh_6536(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_gh_6536' in the type store
    # Getting the type of 'stypy_return_type' (line 50)
    stypy_return_type_541682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_541682)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_gh_6536'
    return stypy_return_type_541682

# Assigning a type to the variable 'test_gh_6536' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'test_gh_6536', test_gh_6536)

@norecursion
def test_branch_cut(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_branch_cut'
    module_type_store = module_type_store.open_function_context('test_branch_cut', 56, 0, False)
    
    # Passed parameters checking function
    test_branch_cut.stypy_localization = localization
    test_branch_cut.stypy_type_of_self = None
    test_branch_cut.stypy_type_store = module_type_store
    test_branch_cut.stypy_function_name = 'test_branch_cut'
    test_branch_cut.stypy_param_names_list = []
    test_branch_cut.stypy_varargs_param_name = None
    test_branch_cut.stypy_kwargs_param_name = None
    test_branch_cut.stypy_call_defaults = defaults
    test_branch_cut.stypy_call_varargs = varargs
    test_branch_cut.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_branch_cut', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_branch_cut', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_branch_cut(...)' code ##################

    
    # Assigning a UnaryOp to a Name (line 58):
    
    # Assigning a UnaryOp to a Name (line 58):
    
    
    # Call to logspace(...): (line 58)
    # Processing the call arguments (line 58)
    int_541685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 21), 'int')
    int_541686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 26), 'int')
    int_541687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 31), 'int')
    # Processing the call keyword arguments (line 58)
    kwargs_541688 = {}
    # Getting the type of 'np' (line 58)
    np_541683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 9), 'np', False)
    # Obtaining the member 'logspace' of a type (line 58)
    logspace_541684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 9), np_541683, 'logspace')
    # Calling logspace(args, kwargs) (line 58)
    logspace_call_result_541689 = invoke(stypy.reporting.localization.Localization(__file__, 58, 9), logspace_541684, *[int_541685, int_541686, int_541687], **kwargs_541688)
    
    # Applying the 'usub' unary operator (line 58)
    result___neg___541690 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 8), 'usub', logspace_call_result_541689)
    
    # Assigning a type to the variable 'x' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'x', result___neg___541690)
    
    # Assigning a Call to a Name (line 59):
    
    # Assigning a Call to a Name (line 59):
    
    # Call to asarray(...): (line 59)
    # Processing the call arguments (line 59)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'x' (line 59)
    x_541698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 47), 'x', False)
    comprehension_541699 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 20), x_541698)
    # Assigning a type to the variable 'x0' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'x0', comprehension_541699)
    
    # Call to complex(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'x0' (line 59)
    x0_541694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 28), 'x0', False)
    float_541695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 32), 'float')
    # Processing the call keyword arguments (line 59)
    kwargs_541696 = {}
    # Getting the type of 'complex' (line 59)
    complex_541693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'complex', False)
    # Calling complex(args, kwargs) (line 59)
    complex_call_result_541697 = invoke(stypy.reporting.localization.Localization(__file__, 59, 20), complex_541693, *[x0_541694, float_541695], **kwargs_541696)
    
    list_541700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 20), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 20), list_541700, complex_call_result_541697)
    # Processing the call keyword arguments (line 59)
    kwargs_541701 = {}
    # Getting the type of 'np' (line 59)
    np_541691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 59)
    asarray_541692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), np_541691, 'asarray')
    # Calling asarray(args, kwargs) (line 59)
    asarray_call_result_541702 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), asarray_541692, *[list_541700], **kwargs_541701)
    
    # Assigning a type to the variable 'z' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'z', asarray_call_result_541702)
    
    # Assigning a Call to a Name (line 60):
    
    # Assigning a Call to a Name (line 60):
    
    # Call to asarray(...): (line 60)
    # Processing the call arguments (line 60)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'x' (line 60)
    x_541710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 51), 'x', False)
    comprehension_541711 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 23), x_541710)
    # Assigning a type to the variable 'x0' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'x0', comprehension_541711)
    
    # Call to complex(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'x0' (line 60)
    x0_541706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 31), 'x0', False)
    float_541707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 35), 'float')
    # Processing the call keyword arguments (line 60)
    kwargs_541708 = {}
    # Getting the type of 'complex' (line 60)
    complex_541705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'complex', False)
    # Calling complex(args, kwargs) (line 60)
    complex_call_result_541709 = invoke(stypy.reporting.localization.Localization(__file__, 60, 23), complex_541705, *[x0_541706, float_541707], **kwargs_541708)
    
    list_541712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 23), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 23), list_541712, complex_call_result_541709)
    # Processing the call keyword arguments (line 60)
    kwargs_541713 = {}
    # Getting the type of 'np' (line 60)
    np_541703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'np', False)
    # Obtaining the member 'asarray' of a type (line 60)
    asarray_541704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 11), np_541703, 'asarray')
    # Calling asarray(args, kwargs) (line 60)
    asarray_call_result_541714 = invoke(stypy.reporting.localization.Localization(__file__, 60, 11), asarray_541704, *[list_541712], **kwargs_541713)
    
    # Assigning a type to the variable 'zbar' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'zbar', asarray_call_result_541714)
    
    # Call to assert_allclose(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'z' (line 61)
    z_541716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'z', False)
    
    # Call to conjugate(...): (line 61)
    # Processing the call keyword arguments (line 61)
    kwargs_541719 = {}
    # Getting the type of 'zbar' (line 61)
    zbar_541717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 23), 'zbar', False)
    # Obtaining the member 'conjugate' of a type (line 61)
    conjugate_541718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 23), zbar_541717, 'conjugate')
    # Calling conjugate(args, kwargs) (line 61)
    conjugate_call_result_541720 = invoke(stypy.reporting.localization.Localization(__file__, 61, 23), conjugate_541718, *[], **kwargs_541719)
    
    # Processing the call keyword arguments (line 61)
    float_541721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 46), 'float')
    keyword_541722 = float_541721
    int_541723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 58), 'int')
    keyword_541724 = int_541723
    kwargs_541725 = {'rtol': keyword_541722, 'atol': keyword_541724}
    # Getting the type of 'assert_allclose' (line 61)
    assert_allclose_541715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 61)
    assert_allclose_call_result_541726 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), assert_allclose_541715, *[z_541716, conjugate_call_result_541720], **kwargs_541725)
    
    
    # ################# End of 'test_branch_cut(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_branch_cut' in the type store
    # Getting the type of 'stypy_return_type' (line 56)
    stypy_return_type_541727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_541727)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_branch_cut'
    return stypy_return_type_541727

# Assigning a type to the variable 'test_branch_cut' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'test_branch_cut', test_branch_cut)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
