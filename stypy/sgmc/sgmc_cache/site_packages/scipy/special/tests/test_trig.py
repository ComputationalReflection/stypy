
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_equal, assert_allclose
5: 
6: from scipy.special._ufuncs import _sinpi as sinpi
7: from scipy.special._ufuncs import _cospi as cospi
8: 
9: 
10: def test_integer_real_part():
11:     x = np.arange(-100, 101)
12:     y = np.hstack((-np.linspace(310, -30, 10), np.linspace(-30, 310, 10)))
13:     x, y = np.meshgrid(x, y)
14:     z = x + 1j*y
15:     # In the following we should be *exactly* right
16:     res = sinpi(z)
17:     assert_equal(res.real, 0.0)
18:     res = cospi(z)
19:     assert_equal(res.imag, 0.0)
20: 
21: 
22: def test_half_integer_real_part():
23:     x = np.arange(-100, 101) + 0.5
24:     y = np.hstack((-np.linspace(310, -30, 10), np.linspace(-30, 310, 10)))
25:     x, y = np.meshgrid(x, y)
26:     z = x + 1j*y
27:     # In the following we should be *exactly* right
28:     res = sinpi(z)
29:     assert_equal(res.imag, 0.0)
30:     res = cospi(z)
31:     assert_equal(res.real, 0.0)
32: 
33: 
34: def test_intermediate_overlow():
35:     # Make sure we avoid overflow in situations where cosh/sinh would
36:     # overflow but the product with sin/cos would not
37:     sinpi_pts = [complex(1 + 1e-14, 227),
38:                  complex(1e-35, 250),
39:                  complex(1e-301, 445)]
40:     # Data generated with mpmath
41:     sinpi_std = [complex(-8.113438309924894e+295, -np.inf),
42:                  complex(1.9507801934611995e+306, np.inf),
43:                  complex(2.205958493464539e+306, np.inf)]
44:     for p, std in zip(sinpi_pts, sinpi_std):
45:         assert_allclose(sinpi(p), std)
46: 
47:     # Test for cosine, less interesting because cos(0) = 1.
48:     p = complex(0.5 + 1e-14, 227)
49:     std = complex(-8.113438309924894e+295, -np.inf)
50:     assert_allclose(cospi(p), std)
51: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_562969 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_562969) is not StypyTypeError):

    if (import_562969 != 'pyd_module'):
        __import__(import_562969)
        sys_modules_562970 = sys.modules[import_562969]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_562970.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_562969)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_equal, assert_allclose' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_562971 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_562971) is not StypyTypeError):

    if (import_562971 != 'pyd_module'):
        __import__(import_562971)
        sys_modules_562972 = sys.modules[import_562971]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_562972.module_type_store, module_type_store, ['assert_equal', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_562972, sys_modules_562972.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_allclose'], [assert_equal, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_562971)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.special._ufuncs import sinpi' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_562973 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._ufuncs')

if (type(import_562973) is not StypyTypeError):

    if (import_562973 != 'pyd_module'):
        __import__(import_562973)
        sys_modules_562974 = sys.modules[import_562973]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._ufuncs', sys_modules_562974.module_type_store, module_type_store, ['_sinpi'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_562974, sys_modules_562974.module_type_store, module_type_store)
    else:
        from scipy.special._ufuncs import _sinpi as sinpi

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._ufuncs', None, module_type_store, ['_sinpi'], [sinpi])

else:
    # Assigning a type to the variable 'scipy.special._ufuncs' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special._ufuncs', import_562973)

# Adding an alias
module_type_store.add_alias('sinpi', '_sinpi')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.special._ufuncs import cospi' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_562975 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._ufuncs')

if (type(import_562975) is not StypyTypeError):

    if (import_562975 != 'pyd_module'):
        __import__(import_562975)
        sys_modules_562976 = sys.modules[import_562975]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._ufuncs', sys_modules_562976.module_type_store, module_type_store, ['_cospi'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_562976, sys_modules_562976.module_type_store, module_type_store)
    else:
        from scipy.special._ufuncs import _cospi as cospi

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._ufuncs', None, module_type_store, ['_cospi'], [cospi])

else:
    # Assigning a type to the variable 'scipy.special._ufuncs' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special._ufuncs', import_562975)

# Adding an alias
module_type_store.add_alias('cospi', '_cospi')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')


@norecursion
def test_integer_real_part(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_integer_real_part'
    module_type_store = module_type_store.open_function_context('test_integer_real_part', 10, 0, False)
    
    # Passed parameters checking function
    test_integer_real_part.stypy_localization = localization
    test_integer_real_part.stypy_type_of_self = None
    test_integer_real_part.stypy_type_store = module_type_store
    test_integer_real_part.stypy_function_name = 'test_integer_real_part'
    test_integer_real_part.stypy_param_names_list = []
    test_integer_real_part.stypy_varargs_param_name = None
    test_integer_real_part.stypy_kwargs_param_name = None
    test_integer_real_part.stypy_call_defaults = defaults
    test_integer_real_part.stypy_call_varargs = varargs
    test_integer_real_part.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_integer_real_part', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_integer_real_part', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_integer_real_part(...)' code ##################

    
    # Assigning a Call to a Name (line 11):
    
    # Assigning a Call to a Name (line 11):
    
    # Call to arange(...): (line 11)
    # Processing the call arguments (line 11)
    int_562979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'int')
    int_562980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 24), 'int')
    # Processing the call keyword arguments (line 11)
    kwargs_562981 = {}
    # Getting the type of 'np' (line 11)
    np_562977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 11)
    arange_562978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), np_562977, 'arange')
    # Calling arange(args, kwargs) (line 11)
    arange_call_result_562982 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), arange_562978, *[int_562979, int_562980], **kwargs_562981)
    
    # Assigning a type to the variable 'x' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'x', arange_call_result_562982)
    
    # Assigning a Call to a Name (line 12):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to hstack(...): (line 12)
    # Processing the call arguments (line 12)
    
    # Obtaining an instance of the builtin type 'tuple' (line 12)
    tuple_562985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 12)
    # Adding element type (line 12)
    
    
    # Call to linspace(...): (line 12)
    # Processing the call arguments (line 12)
    int_562988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 32), 'int')
    int_562989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 37), 'int')
    int_562990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 42), 'int')
    # Processing the call keyword arguments (line 12)
    kwargs_562991 = {}
    # Getting the type of 'np' (line 12)
    np_562986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 20), 'np', False)
    # Obtaining the member 'linspace' of a type (line 12)
    linspace_562987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 20), np_562986, 'linspace')
    # Calling linspace(args, kwargs) (line 12)
    linspace_call_result_562992 = invoke(stypy.reporting.localization.Localization(__file__, 12, 20), linspace_562987, *[int_562988, int_562989, int_562990], **kwargs_562991)
    
    # Applying the 'usub' unary operator (line 12)
    result___neg___562993 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 19), 'usub', linspace_call_result_562992)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), tuple_562985, result___neg___562993)
    # Adding element type (line 12)
    
    # Call to linspace(...): (line 12)
    # Processing the call arguments (line 12)
    int_562996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 59), 'int')
    int_562997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 64), 'int')
    int_562998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 69), 'int')
    # Processing the call keyword arguments (line 12)
    kwargs_562999 = {}
    # Getting the type of 'np' (line 12)
    np_562994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 47), 'np', False)
    # Obtaining the member 'linspace' of a type (line 12)
    linspace_562995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 47), np_562994, 'linspace')
    # Calling linspace(args, kwargs) (line 12)
    linspace_call_result_563000 = invoke(stypy.reporting.localization.Localization(__file__, 12, 47), linspace_562995, *[int_562996, int_562997, int_562998], **kwargs_562999)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), tuple_562985, linspace_call_result_563000)
    
    # Processing the call keyword arguments (line 12)
    kwargs_563001 = {}
    # Getting the type of 'np' (line 12)
    np_562983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'np', False)
    # Obtaining the member 'hstack' of a type (line 12)
    hstack_562984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), np_562983, 'hstack')
    # Calling hstack(args, kwargs) (line 12)
    hstack_call_result_563002 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), hstack_562984, *[tuple_562985], **kwargs_563001)
    
    # Assigning a type to the variable 'y' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'y', hstack_call_result_563002)
    
    # Assigning a Call to a Tuple (line 13):
    
    # Assigning a Subscript to a Name (line 13):
    
    # Obtaining the type of the subscript
    int_563003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 4), 'int')
    
    # Call to meshgrid(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'x' (line 13)
    x_563006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 23), 'x', False)
    # Getting the type of 'y' (line 13)
    y_563007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 26), 'y', False)
    # Processing the call keyword arguments (line 13)
    kwargs_563008 = {}
    # Getting the type of 'np' (line 13)
    np_563004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 13)
    meshgrid_563005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 11), np_563004, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 13)
    meshgrid_call_result_563009 = invoke(stypy.reporting.localization.Localization(__file__, 13, 11), meshgrid_563005, *[x_563006, y_563007], **kwargs_563008)
    
    # Obtaining the member '__getitem__' of a type (line 13)
    getitem___563010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), meshgrid_call_result_563009, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 13)
    subscript_call_result_563011 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), getitem___563010, int_563003)
    
    # Assigning a type to the variable 'tuple_var_assignment_562965' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'tuple_var_assignment_562965', subscript_call_result_563011)
    
    # Assigning a Subscript to a Name (line 13):
    
    # Obtaining the type of the subscript
    int_563012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 4), 'int')
    
    # Call to meshgrid(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'x' (line 13)
    x_563015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 23), 'x', False)
    # Getting the type of 'y' (line 13)
    y_563016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 26), 'y', False)
    # Processing the call keyword arguments (line 13)
    kwargs_563017 = {}
    # Getting the type of 'np' (line 13)
    np_563013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 13)
    meshgrid_563014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 11), np_563013, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 13)
    meshgrid_call_result_563018 = invoke(stypy.reporting.localization.Localization(__file__, 13, 11), meshgrid_563014, *[x_563015, y_563016], **kwargs_563017)
    
    # Obtaining the member '__getitem__' of a type (line 13)
    getitem___563019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), meshgrid_call_result_563018, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 13)
    subscript_call_result_563020 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), getitem___563019, int_563012)
    
    # Assigning a type to the variable 'tuple_var_assignment_562966' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'tuple_var_assignment_562966', subscript_call_result_563020)
    
    # Assigning a Name to a Name (line 13):
    # Getting the type of 'tuple_var_assignment_562965' (line 13)
    tuple_var_assignment_562965_563021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'tuple_var_assignment_562965')
    # Assigning a type to the variable 'x' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'x', tuple_var_assignment_562965_563021)
    
    # Assigning a Name to a Name (line 13):
    # Getting the type of 'tuple_var_assignment_562966' (line 13)
    tuple_var_assignment_562966_563022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'tuple_var_assignment_562966')
    # Assigning a type to the variable 'y' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 7), 'y', tuple_var_assignment_562966_563022)
    
    # Assigning a BinOp to a Name (line 14):
    
    # Assigning a BinOp to a Name (line 14):
    # Getting the type of 'x' (line 14)
    x_563023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'x')
    complex_563024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'complex')
    # Getting the type of 'y' (line 14)
    y_563025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'y')
    # Applying the binary operator '*' (line 14)
    result_mul_563026 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 12), '*', complex_563024, y_563025)
    
    # Applying the binary operator '+' (line 14)
    result_add_563027 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 8), '+', x_563023, result_mul_563026)
    
    # Assigning a type to the variable 'z' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'z', result_add_563027)
    
    # Assigning a Call to a Name (line 16):
    
    # Assigning a Call to a Name (line 16):
    
    # Call to sinpi(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'z' (line 16)
    z_563029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'z', False)
    # Processing the call keyword arguments (line 16)
    kwargs_563030 = {}
    # Getting the type of 'sinpi' (line 16)
    sinpi_563028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 10), 'sinpi', False)
    # Calling sinpi(args, kwargs) (line 16)
    sinpi_call_result_563031 = invoke(stypy.reporting.localization.Localization(__file__, 16, 10), sinpi_563028, *[z_563029], **kwargs_563030)
    
    # Assigning a type to the variable 'res' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'res', sinpi_call_result_563031)
    
    # Call to assert_equal(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'res' (line 17)
    res_563033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 17), 'res', False)
    # Obtaining the member 'real' of a type (line 17)
    real_563034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 17), res_563033, 'real')
    float_563035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 27), 'float')
    # Processing the call keyword arguments (line 17)
    kwargs_563036 = {}
    # Getting the type of 'assert_equal' (line 17)
    assert_equal_563032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 17)
    assert_equal_call_result_563037 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), assert_equal_563032, *[real_563034, float_563035], **kwargs_563036)
    
    
    # Assigning a Call to a Name (line 18):
    
    # Assigning a Call to a Name (line 18):
    
    # Call to cospi(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'z' (line 18)
    z_563039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'z', False)
    # Processing the call keyword arguments (line 18)
    kwargs_563040 = {}
    # Getting the type of 'cospi' (line 18)
    cospi_563038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'cospi', False)
    # Calling cospi(args, kwargs) (line 18)
    cospi_call_result_563041 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), cospi_563038, *[z_563039], **kwargs_563040)
    
    # Assigning a type to the variable 'res' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'res', cospi_call_result_563041)
    
    # Call to assert_equal(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'res' (line 19)
    res_563043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 17), 'res', False)
    # Obtaining the member 'imag' of a type (line 19)
    imag_563044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 17), res_563043, 'imag')
    float_563045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 27), 'float')
    # Processing the call keyword arguments (line 19)
    kwargs_563046 = {}
    # Getting the type of 'assert_equal' (line 19)
    assert_equal_563042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 19)
    assert_equal_call_result_563047 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), assert_equal_563042, *[imag_563044, float_563045], **kwargs_563046)
    
    
    # ################# End of 'test_integer_real_part(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_integer_real_part' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_563048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_563048)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_integer_real_part'
    return stypy_return_type_563048

# Assigning a type to the variable 'test_integer_real_part' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'test_integer_real_part', test_integer_real_part)

@norecursion
def test_half_integer_real_part(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_half_integer_real_part'
    module_type_store = module_type_store.open_function_context('test_half_integer_real_part', 22, 0, False)
    
    # Passed parameters checking function
    test_half_integer_real_part.stypy_localization = localization
    test_half_integer_real_part.stypy_type_of_self = None
    test_half_integer_real_part.stypy_type_store = module_type_store
    test_half_integer_real_part.stypy_function_name = 'test_half_integer_real_part'
    test_half_integer_real_part.stypy_param_names_list = []
    test_half_integer_real_part.stypy_varargs_param_name = None
    test_half_integer_real_part.stypy_kwargs_param_name = None
    test_half_integer_real_part.stypy_call_defaults = defaults
    test_half_integer_real_part.stypy_call_varargs = varargs
    test_half_integer_real_part.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_half_integer_real_part', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_half_integer_real_part', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_half_integer_real_part(...)' code ##################

    
    # Assigning a BinOp to a Name (line 23):
    
    # Assigning a BinOp to a Name (line 23):
    
    # Call to arange(...): (line 23)
    # Processing the call arguments (line 23)
    int_563051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 18), 'int')
    int_563052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'int')
    # Processing the call keyword arguments (line 23)
    kwargs_563053 = {}
    # Getting the type of 'np' (line 23)
    np_563049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 23)
    arange_563050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), np_563049, 'arange')
    # Calling arange(args, kwargs) (line 23)
    arange_call_result_563054 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), arange_563050, *[int_563051, int_563052], **kwargs_563053)
    
    float_563055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 31), 'float')
    # Applying the binary operator '+' (line 23)
    result_add_563056 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 8), '+', arange_call_result_563054, float_563055)
    
    # Assigning a type to the variable 'x' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'x', result_add_563056)
    
    # Assigning a Call to a Name (line 24):
    
    # Assigning a Call to a Name (line 24):
    
    # Call to hstack(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Obtaining an instance of the builtin type 'tuple' (line 24)
    tuple_563059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 24)
    # Adding element type (line 24)
    
    
    # Call to linspace(...): (line 24)
    # Processing the call arguments (line 24)
    int_563062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 32), 'int')
    int_563063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 37), 'int')
    int_563064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 42), 'int')
    # Processing the call keyword arguments (line 24)
    kwargs_563065 = {}
    # Getting the type of 'np' (line 24)
    np_563060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'np', False)
    # Obtaining the member 'linspace' of a type (line 24)
    linspace_563061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 20), np_563060, 'linspace')
    # Calling linspace(args, kwargs) (line 24)
    linspace_call_result_563066 = invoke(stypy.reporting.localization.Localization(__file__, 24, 20), linspace_563061, *[int_563062, int_563063, int_563064], **kwargs_563065)
    
    # Applying the 'usub' unary operator (line 24)
    result___neg___563067 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 19), 'usub', linspace_call_result_563066)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), tuple_563059, result___neg___563067)
    # Adding element type (line 24)
    
    # Call to linspace(...): (line 24)
    # Processing the call arguments (line 24)
    int_563070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 59), 'int')
    int_563071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 64), 'int')
    int_563072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 69), 'int')
    # Processing the call keyword arguments (line 24)
    kwargs_563073 = {}
    # Getting the type of 'np' (line 24)
    np_563068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 47), 'np', False)
    # Obtaining the member 'linspace' of a type (line 24)
    linspace_563069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 47), np_563068, 'linspace')
    # Calling linspace(args, kwargs) (line 24)
    linspace_call_result_563074 = invoke(stypy.reporting.localization.Localization(__file__, 24, 47), linspace_563069, *[int_563070, int_563071, int_563072], **kwargs_563073)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), tuple_563059, linspace_call_result_563074)
    
    # Processing the call keyword arguments (line 24)
    kwargs_563075 = {}
    # Getting the type of 'np' (line 24)
    np_563057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'np', False)
    # Obtaining the member 'hstack' of a type (line 24)
    hstack_563058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), np_563057, 'hstack')
    # Calling hstack(args, kwargs) (line 24)
    hstack_call_result_563076 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), hstack_563058, *[tuple_563059], **kwargs_563075)
    
    # Assigning a type to the variable 'y' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'y', hstack_call_result_563076)
    
    # Assigning a Call to a Tuple (line 25):
    
    # Assigning a Subscript to a Name (line 25):
    
    # Obtaining the type of the subscript
    int_563077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'int')
    
    # Call to meshgrid(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'x' (line 25)
    x_563080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 23), 'x', False)
    # Getting the type of 'y' (line 25)
    y_563081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 26), 'y', False)
    # Processing the call keyword arguments (line 25)
    kwargs_563082 = {}
    # Getting the type of 'np' (line 25)
    np_563078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 25)
    meshgrid_563079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 11), np_563078, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 25)
    meshgrid_call_result_563083 = invoke(stypy.reporting.localization.Localization(__file__, 25, 11), meshgrid_563079, *[x_563080, y_563081], **kwargs_563082)
    
    # Obtaining the member '__getitem__' of a type (line 25)
    getitem___563084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), meshgrid_call_result_563083, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 25)
    subscript_call_result_563085 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), getitem___563084, int_563077)
    
    # Assigning a type to the variable 'tuple_var_assignment_562967' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'tuple_var_assignment_562967', subscript_call_result_563085)
    
    # Assigning a Subscript to a Name (line 25):
    
    # Obtaining the type of the subscript
    int_563086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'int')
    
    # Call to meshgrid(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'x' (line 25)
    x_563089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 23), 'x', False)
    # Getting the type of 'y' (line 25)
    y_563090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 26), 'y', False)
    # Processing the call keyword arguments (line 25)
    kwargs_563091 = {}
    # Getting the type of 'np' (line 25)
    np_563087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'np', False)
    # Obtaining the member 'meshgrid' of a type (line 25)
    meshgrid_563088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 11), np_563087, 'meshgrid')
    # Calling meshgrid(args, kwargs) (line 25)
    meshgrid_call_result_563092 = invoke(stypy.reporting.localization.Localization(__file__, 25, 11), meshgrid_563088, *[x_563089, y_563090], **kwargs_563091)
    
    # Obtaining the member '__getitem__' of a type (line 25)
    getitem___563093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), meshgrid_call_result_563092, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 25)
    subscript_call_result_563094 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), getitem___563093, int_563086)
    
    # Assigning a type to the variable 'tuple_var_assignment_562968' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'tuple_var_assignment_562968', subscript_call_result_563094)
    
    # Assigning a Name to a Name (line 25):
    # Getting the type of 'tuple_var_assignment_562967' (line 25)
    tuple_var_assignment_562967_563095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'tuple_var_assignment_562967')
    # Assigning a type to the variable 'x' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'x', tuple_var_assignment_562967_563095)
    
    # Assigning a Name to a Name (line 25):
    # Getting the type of 'tuple_var_assignment_562968' (line 25)
    tuple_var_assignment_562968_563096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'tuple_var_assignment_562968')
    # Assigning a type to the variable 'y' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 7), 'y', tuple_var_assignment_562968_563096)
    
    # Assigning a BinOp to a Name (line 26):
    
    # Assigning a BinOp to a Name (line 26):
    # Getting the type of 'x' (line 26)
    x_563097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'x')
    complex_563098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 12), 'complex')
    # Getting the type of 'y' (line 26)
    y_563099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'y')
    # Applying the binary operator '*' (line 26)
    result_mul_563100 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 12), '*', complex_563098, y_563099)
    
    # Applying the binary operator '+' (line 26)
    result_add_563101 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 8), '+', x_563097, result_mul_563100)
    
    # Assigning a type to the variable 'z' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'z', result_add_563101)
    
    # Assigning a Call to a Name (line 28):
    
    # Assigning a Call to a Name (line 28):
    
    # Call to sinpi(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'z' (line 28)
    z_563103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'z', False)
    # Processing the call keyword arguments (line 28)
    kwargs_563104 = {}
    # Getting the type of 'sinpi' (line 28)
    sinpi_563102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 10), 'sinpi', False)
    # Calling sinpi(args, kwargs) (line 28)
    sinpi_call_result_563105 = invoke(stypy.reporting.localization.Localization(__file__, 28, 10), sinpi_563102, *[z_563103], **kwargs_563104)
    
    # Assigning a type to the variable 'res' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'res', sinpi_call_result_563105)
    
    # Call to assert_equal(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'res' (line 29)
    res_563107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'res', False)
    # Obtaining the member 'imag' of a type (line 29)
    imag_563108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 17), res_563107, 'imag')
    float_563109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 27), 'float')
    # Processing the call keyword arguments (line 29)
    kwargs_563110 = {}
    # Getting the type of 'assert_equal' (line 29)
    assert_equal_563106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 29)
    assert_equal_call_result_563111 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), assert_equal_563106, *[imag_563108, float_563109], **kwargs_563110)
    
    
    # Assigning a Call to a Name (line 30):
    
    # Assigning a Call to a Name (line 30):
    
    # Call to cospi(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'z' (line 30)
    z_563113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'z', False)
    # Processing the call keyword arguments (line 30)
    kwargs_563114 = {}
    # Getting the type of 'cospi' (line 30)
    cospi_563112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 10), 'cospi', False)
    # Calling cospi(args, kwargs) (line 30)
    cospi_call_result_563115 = invoke(stypy.reporting.localization.Localization(__file__, 30, 10), cospi_563112, *[z_563113], **kwargs_563114)
    
    # Assigning a type to the variable 'res' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'res', cospi_call_result_563115)
    
    # Call to assert_equal(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'res' (line 31)
    res_563117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'res', False)
    # Obtaining the member 'real' of a type (line 31)
    real_563118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 17), res_563117, 'real')
    float_563119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 27), 'float')
    # Processing the call keyword arguments (line 31)
    kwargs_563120 = {}
    # Getting the type of 'assert_equal' (line 31)
    assert_equal_563116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 31)
    assert_equal_call_result_563121 = invoke(stypy.reporting.localization.Localization(__file__, 31, 4), assert_equal_563116, *[real_563118, float_563119], **kwargs_563120)
    
    
    # ################# End of 'test_half_integer_real_part(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_half_integer_real_part' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_563122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_563122)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_half_integer_real_part'
    return stypy_return_type_563122

# Assigning a type to the variable 'test_half_integer_real_part' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'test_half_integer_real_part', test_half_integer_real_part)

@norecursion
def test_intermediate_overlow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_intermediate_overlow'
    module_type_store = module_type_store.open_function_context('test_intermediate_overlow', 34, 0, False)
    
    # Passed parameters checking function
    test_intermediate_overlow.stypy_localization = localization
    test_intermediate_overlow.stypy_type_of_self = None
    test_intermediate_overlow.stypy_type_store = module_type_store
    test_intermediate_overlow.stypy_function_name = 'test_intermediate_overlow'
    test_intermediate_overlow.stypy_param_names_list = []
    test_intermediate_overlow.stypy_varargs_param_name = None
    test_intermediate_overlow.stypy_kwargs_param_name = None
    test_intermediate_overlow.stypy_call_defaults = defaults
    test_intermediate_overlow.stypy_call_varargs = varargs
    test_intermediate_overlow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_intermediate_overlow', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_intermediate_overlow', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_intermediate_overlow(...)' code ##################

    
    # Assigning a List to a Name (line 37):
    
    # Assigning a List to a Name (line 37):
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_563123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    
    # Call to complex(...): (line 37)
    # Processing the call arguments (line 37)
    int_563125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 25), 'int')
    float_563126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 29), 'float')
    # Applying the binary operator '+' (line 37)
    result_add_563127 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 25), '+', int_563125, float_563126)
    
    int_563128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 36), 'int')
    # Processing the call keyword arguments (line 37)
    kwargs_563129 = {}
    # Getting the type of 'complex' (line 37)
    complex_563124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'complex', False)
    # Calling complex(args, kwargs) (line 37)
    complex_call_result_563130 = invoke(stypy.reporting.localization.Localization(__file__, 37, 17), complex_563124, *[result_add_563127, int_563128], **kwargs_563129)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), list_563123, complex_call_result_563130)
    # Adding element type (line 37)
    
    # Call to complex(...): (line 38)
    # Processing the call arguments (line 38)
    float_563132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 25), 'float')
    int_563133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 32), 'int')
    # Processing the call keyword arguments (line 38)
    kwargs_563134 = {}
    # Getting the type of 'complex' (line 38)
    complex_563131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'complex', False)
    # Calling complex(args, kwargs) (line 38)
    complex_call_result_563135 = invoke(stypy.reporting.localization.Localization(__file__, 38, 17), complex_563131, *[float_563132, int_563133], **kwargs_563134)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), list_563123, complex_call_result_563135)
    # Adding element type (line 37)
    
    # Call to complex(...): (line 39)
    # Processing the call arguments (line 39)
    float_563137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 25), 'float')
    int_563138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 33), 'int')
    # Processing the call keyword arguments (line 39)
    kwargs_563139 = {}
    # Getting the type of 'complex' (line 39)
    complex_563136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 'complex', False)
    # Calling complex(args, kwargs) (line 39)
    complex_call_result_563140 = invoke(stypy.reporting.localization.Localization(__file__, 39, 17), complex_563136, *[float_563137, int_563138], **kwargs_563139)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), list_563123, complex_call_result_563140)
    
    # Assigning a type to the variable 'sinpi_pts' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'sinpi_pts', list_563123)
    
    # Assigning a List to a Name (line 41):
    
    # Assigning a List to a Name (line 41):
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_563141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    
    # Call to complex(...): (line 41)
    # Processing the call arguments (line 41)
    float_563143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 25), 'float')
    
    # Getting the type of 'np' (line 41)
    np_563144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 51), 'np', False)
    # Obtaining the member 'inf' of a type (line 41)
    inf_563145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 51), np_563144, 'inf')
    # Applying the 'usub' unary operator (line 41)
    result___neg___563146 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 50), 'usub', inf_563145)
    
    # Processing the call keyword arguments (line 41)
    kwargs_563147 = {}
    # Getting the type of 'complex' (line 41)
    complex_563142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'complex', False)
    # Calling complex(args, kwargs) (line 41)
    complex_call_result_563148 = invoke(stypy.reporting.localization.Localization(__file__, 41, 17), complex_563142, *[float_563143, result___neg___563146], **kwargs_563147)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 16), list_563141, complex_call_result_563148)
    # Adding element type (line 41)
    
    # Call to complex(...): (line 42)
    # Processing the call arguments (line 42)
    float_563150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'float')
    # Getting the type of 'np' (line 42)
    np_563151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 50), 'np', False)
    # Obtaining the member 'inf' of a type (line 42)
    inf_563152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 50), np_563151, 'inf')
    # Processing the call keyword arguments (line 42)
    kwargs_563153 = {}
    # Getting the type of 'complex' (line 42)
    complex_563149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'complex', False)
    # Calling complex(args, kwargs) (line 42)
    complex_call_result_563154 = invoke(stypy.reporting.localization.Localization(__file__, 42, 17), complex_563149, *[float_563150, inf_563152], **kwargs_563153)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 16), list_563141, complex_call_result_563154)
    # Adding element type (line 41)
    
    # Call to complex(...): (line 43)
    # Processing the call arguments (line 43)
    float_563156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 25), 'float')
    # Getting the type of 'np' (line 43)
    np_563157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 49), 'np', False)
    # Obtaining the member 'inf' of a type (line 43)
    inf_563158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 49), np_563157, 'inf')
    # Processing the call keyword arguments (line 43)
    kwargs_563159 = {}
    # Getting the type of 'complex' (line 43)
    complex_563155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'complex', False)
    # Calling complex(args, kwargs) (line 43)
    complex_call_result_563160 = invoke(stypy.reporting.localization.Localization(__file__, 43, 17), complex_563155, *[float_563156, inf_563158], **kwargs_563159)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 16), list_563141, complex_call_result_563160)
    
    # Assigning a type to the variable 'sinpi_std' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'sinpi_std', list_563141)
    
    
    # Call to zip(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'sinpi_pts' (line 44)
    sinpi_pts_563162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'sinpi_pts', False)
    # Getting the type of 'sinpi_std' (line 44)
    sinpi_std_563163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 33), 'sinpi_std', False)
    # Processing the call keyword arguments (line 44)
    kwargs_563164 = {}
    # Getting the type of 'zip' (line 44)
    zip_563161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 18), 'zip', False)
    # Calling zip(args, kwargs) (line 44)
    zip_call_result_563165 = invoke(stypy.reporting.localization.Localization(__file__, 44, 18), zip_563161, *[sinpi_pts_563162, sinpi_std_563163], **kwargs_563164)
    
    # Testing the type of a for loop iterable (line 44)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 44, 4), zip_call_result_563165)
    # Getting the type of the for loop variable (line 44)
    for_loop_var_563166 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 44, 4), zip_call_result_563165)
    # Assigning a type to the variable 'p' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'p', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 4), for_loop_var_563166))
    # Assigning a type to the variable 'std' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'std', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 4), for_loop_var_563166))
    # SSA begins for a for statement (line 44)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_allclose(...): (line 45)
    # Processing the call arguments (line 45)
    
    # Call to sinpi(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'p' (line 45)
    p_563169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 30), 'p', False)
    # Processing the call keyword arguments (line 45)
    kwargs_563170 = {}
    # Getting the type of 'sinpi' (line 45)
    sinpi_563168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 24), 'sinpi', False)
    # Calling sinpi(args, kwargs) (line 45)
    sinpi_call_result_563171 = invoke(stypy.reporting.localization.Localization(__file__, 45, 24), sinpi_563168, *[p_563169], **kwargs_563170)
    
    # Getting the type of 'std' (line 45)
    std_563172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 34), 'std', False)
    # Processing the call keyword arguments (line 45)
    kwargs_563173 = {}
    # Getting the type of 'assert_allclose' (line 45)
    assert_allclose_563167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 45)
    assert_allclose_call_result_563174 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), assert_allclose_563167, *[sinpi_call_result_563171, std_563172], **kwargs_563173)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 48):
    
    # Assigning a Call to a Name (line 48):
    
    # Call to complex(...): (line 48)
    # Processing the call arguments (line 48)
    float_563176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 16), 'float')
    float_563177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 22), 'float')
    # Applying the binary operator '+' (line 48)
    result_add_563178 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 16), '+', float_563176, float_563177)
    
    int_563179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 29), 'int')
    # Processing the call keyword arguments (line 48)
    kwargs_563180 = {}
    # Getting the type of 'complex' (line 48)
    complex_563175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'complex', False)
    # Calling complex(args, kwargs) (line 48)
    complex_call_result_563181 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), complex_563175, *[result_add_563178, int_563179], **kwargs_563180)
    
    # Assigning a type to the variable 'p' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'p', complex_call_result_563181)
    
    # Assigning a Call to a Name (line 49):
    
    # Assigning a Call to a Name (line 49):
    
    # Call to complex(...): (line 49)
    # Processing the call arguments (line 49)
    float_563183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 18), 'float')
    
    # Getting the type of 'np' (line 49)
    np_563184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 44), 'np', False)
    # Obtaining the member 'inf' of a type (line 49)
    inf_563185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 44), np_563184, 'inf')
    # Applying the 'usub' unary operator (line 49)
    result___neg___563186 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 43), 'usub', inf_563185)
    
    # Processing the call keyword arguments (line 49)
    kwargs_563187 = {}
    # Getting the type of 'complex' (line 49)
    complex_563182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 10), 'complex', False)
    # Calling complex(args, kwargs) (line 49)
    complex_call_result_563188 = invoke(stypy.reporting.localization.Localization(__file__, 49, 10), complex_563182, *[float_563183, result___neg___563186], **kwargs_563187)
    
    # Assigning a type to the variable 'std' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'std', complex_call_result_563188)
    
    # Call to assert_allclose(...): (line 50)
    # Processing the call arguments (line 50)
    
    # Call to cospi(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'p' (line 50)
    p_563191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'p', False)
    # Processing the call keyword arguments (line 50)
    kwargs_563192 = {}
    # Getting the type of 'cospi' (line 50)
    cospi_563190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'cospi', False)
    # Calling cospi(args, kwargs) (line 50)
    cospi_call_result_563193 = invoke(stypy.reporting.localization.Localization(__file__, 50, 20), cospi_563190, *[p_563191], **kwargs_563192)
    
    # Getting the type of 'std' (line 50)
    std_563194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 30), 'std', False)
    # Processing the call keyword arguments (line 50)
    kwargs_563195 = {}
    # Getting the type of 'assert_allclose' (line 50)
    assert_allclose_563189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 50)
    assert_allclose_call_result_563196 = invoke(stypy.reporting.localization.Localization(__file__, 50, 4), assert_allclose_563189, *[cospi_call_result_563193, std_563194], **kwargs_563195)
    
    
    # ################# End of 'test_intermediate_overlow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_intermediate_overlow' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_563197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_563197)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_intermediate_overlow'
    return stypy_return_type_563197

# Assigning a type to the variable 'test_intermediate_overlow' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'test_intermediate_overlow', test_intermediate_overlow)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
