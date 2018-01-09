
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_, assert_equal
5: 
6: import scipy.special as sc
7: 
8: 
9: def test_wrightomega_nan():
10:     pts = [complex(np.nan, 0),
11:            complex(0, np.nan),
12:            complex(np.nan, np.nan),
13:            complex(np.nan, 1),
14:            complex(1, np.nan)]
15:     for p in pts:
16:         res = sc.wrightomega(p)
17:         assert_(np.isnan(res.real))
18:         assert_(np.isnan(res.imag))
19: 
20: 
21: def test_wrightomega_inf_branch():
22:     pts = [complex(-np.inf, np.pi/4),
23:            complex(-np.inf, -np.pi/4),
24:            complex(-np.inf, 3*np.pi/4),
25:            complex(-np.inf, -3*np.pi/4)]
26:     expected_results = [complex(0.0, 0.0),
27:                         complex(0.0, -0.0),
28:                         complex(-0.0, 0.0),
29:                         complex(-0.0, -0.0)]
30:     for p, expected in zip(pts, expected_results):
31:         res = sc.wrightomega(p)
32:         # We can't use assert_equal(res, expected) because in older versions of
33:         # numpy, assert_equal doesn't check the sign of the real and imaginary
34:         # parts when comparing complex zeros. It does check the sign when the
35:         # arguments are *real* scalars.
36:         assert_equal(res.real, expected.real)
37:         assert_equal(res.imag, expected.imag)
38: 
39: 
40: def test_wrightomega_inf():
41:     pts = [complex(np.inf, 10),
42:            complex(-np.inf, 10),
43:            complex(10, np.inf),
44:            complex(10, -np.inf)]
45:     for p in pts:
46:         assert_equal(sc.wrightomega(p), p)
47: 
48: 
49: def test_wrightomega_singular():
50:     pts = [complex(-1.0, np.pi),
51:            complex(-1.0, -np.pi)]
52:     for p in pts:
53:         res = sc.wrightomega(p)
54:         assert_equal(res, -1.0)
55:         assert_(np.signbit(res.imag) == False)
56: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_563198 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_563198) is not StypyTypeError):

    if (import_563198 != 'pyd_module'):
        __import__(import_563198)
        sys_modules_563199 = sys.modules[import_563198]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_563199.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_563198)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_, assert_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_563200 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_563200) is not StypyTypeError):

    if (import_563200 != 'pyd_module'):
        __import__(import_563200)
        sys_modules_563201 = sys.modules[import_563200]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_563201.module_type_store, module_type_store, ['assert_', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_563201, sys_modules_563201.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_equal'], [assert_, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_563200)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import scipy.special' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_563202 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special')

if (type(import_563202) is not StypyTypeError):

    if (import_563202 != 'pyd_module'):
        __import__(import_563202)
        sys_modules_563203 = sys.modules[import_563202]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'sc', sys_modules_563203.module_type_store, module_type_store)
    else:
        import scipy.special as sc

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'sc', scipy.special, module_type_store)

else:
    # Assigning a type to the variable 'scipy.special' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.special', import_563202)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')


@norecursion
def test_wrightomega_nan(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_wrightomega_nan'
    module_type_store = module_type_store.open_function_context('test_wrightomega_nan', 9, 0, False)
    
    # Passed parameters checking function
    test_wrightomega_nan.stypy_localization = localization
    test_wrightomega_nan.stypy_type_of_self = None
    test_wrightomega_nan.stypy_type_store = module_type_store
    test_wrightomega_nan.stypy_function_name = 'test_wrightomega_nan'
    test_wrightomega_nan.stypy_param_names_list = []
    test_wrightomega_nan.stypy_varargs_param_name = None
    test_wrightomega_nan.stypy_kwargs_param_name = None
    test_wrightomega_nan.stypy_call_defaults = defaults
    test_wrightomega_nan.stypy_call_varargs = varargs
    test_wrightomega_nan.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_wrightomega_nan', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_wrightomega_nan', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_wrightomega_nan(...)' code ##################

    
    # Assigning a List to a Name (line 10):
    
    # Obtaining an instance of the builtin type 'list' (line 10)
    list_563204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 10)
    # Adding element type (line 10)
    
    # Call to complex(...): (line 10)
    # Processing the call arguments (line 10)
    # Getting the type of 'np' (line 10)
    np_563206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 19), 'np', False)
    # Obtaining the member 'nan' of a type (line 10)
    nan_563207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 19), np_563206, 'nan')
    int_563208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 27), 'int')
    # Processing the call keyword arguments (line 10)
    kwargs_563209 = {}
    # Getting the type of 'complex' (line 10)
    complex_563205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'complex', False)
    # Calling complex(args, kwargs) (line 10)
    complex_call_result_563210 = invoke(stypy.reporting.localization.Localization(__file__, 10, 11), complex_563205, *[nan_563207, int_563208], **kwargs_563209)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_563204, complex_call_result_563210)
    # Adding element type (line 10)
    
    # Call to complex(...): (line 11)
    # Processing the call arguments (line 11)
    int_563212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 19), 'int')
    # Getting the type of 'np' (line 11)
    np_563213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 22), 'np', False)
    # Obtaining the member 'nan' of a type (line 11)
    nan_563214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 22), np_563213, 'nan')
    # Processing the call keyword arguments (line 11)
    kwargs_563215 = {}
    # Getting the type of 'complex' (line 11)
    complex_563211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'complex', False)
    # Calling complex(args, kwargs) (line 11)
    complex_call_result_563216 = invoke(stypy.reporting.localization.Localization(__file__, 11, 11), complex_563211, *[int_563212, nan_563214], **kwargs_563215)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_563204, complex_call_result_563216)
    # Adding element type (line 10)
    
    # Call to complex(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'np' (line 12)
    np_563218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'np', False)
    # Obtaining the member 'nan' of a type (line 12)
    nan_563219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 19), np_563218, 'nan')
    # Getting the type of 'np' (line 12)
    np_563220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 27), 'np', False)
    # Obtaining the member 'nan' of a type (line 12)
    nan_563221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 27), np_563220, 'nan')
    # Processing the call keyword arguments (line 12)
    kwargs_563222 = {}
    # Getting the type of 'complex' (line 12)
    complex_563217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 11), 'complex', False)
    # Calling complex(args, kwargs) (line 12)
    complex_call_result_563223 = invoke(stypy.reporting.localization.Localization(__file__, 12, 11), complex_563217, *[nan_563219, nan_563221], **kwargs_563222)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_563204, complex_call_result_563223)
    # Adding element type (line 10)
    
    # Call to complex(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'np' (line 13)
    np_563225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 19), 'np', False)
    # Obtaining the member 'nan' of a type (line 13)
    nan_563226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 19), np_563225, 'nan')
    int_563227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 27), 'int')
    # Processing the call keyword arguments (line 13)
    kwargs_563228 = {}
    # Getting the type of 'complex' (line 13)
    complex_563224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'complex', False)
    # Calling complex(args, kwargs) (line 13)
    complex_call_result_563229 = invoke(stypy.reporting.localization.Localization(__file__, 13, 11), complex_563224, *[nan_563226, int_563227], **kwargs_563228)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_563204, complex_call_result_563229)
    # Adding element type (line 10)
    
    # Call to complex(...): (line 14)
    # Processing the call arguments (line 14)
    int_563231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 19), 'int')
    # Getting the type of 'np' (line 14)
    np_563232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 22), 'np', False)
    # Obtaining the member 'nan' of a type (line 14)
    nan_563233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 22), np_563232, 'nan')
    # Processing the call keyword arguments (line 14)
    kwargs_563234 = {}
    # Getting the type of 'complex' (line 14)
    complex_563230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'complex', False)
    # Calling complex(args, kwargs) (line 14)
    complex_call_result_563235 = invoke(stypy.reporting.localization.Localization(__file__, 14, 11), complex_563230, *[int_563231, nan_563233], **kwargs_563234)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_563204, complex_call_result_563235)
    
    # Assigning a type to the variable 'pts' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'pts', list_563204)
    
    # Getting the type of 'pts' (line 15)
    pts_563236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 13), 'pts')
    # Testing the type of a for loop iterable (line 15)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 15, 4), pts_563236)
    # Getting the type of the for loop variable (line 15)
    for_loop_var_563237 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 15, 4), pts_563236)
    # Assigning a type to the variable 'p' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'p', for_loop_var_563237)
    # SSA begins for a for statement (line 15)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 16):
    
    # Call to wrightomega(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'p' (line 16)
    p_563240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 29), 'p', False)
    # Processing the call keyword arguments (line 16)
    kwargs_563241 = {}
    # Getting the type of 'sc' (line 16)
    sc_563238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'sc', False)
    # Obtaining the member 'wrightomega' of a type (line 16)
    wrightomega_563239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 14), sc_563238, 'wrightomega')
    # Calling wrightomega(args, kwargs) (line 16)
    wrightomega_call_result_563242 = invoke(stypy.reporting.localization.Localization(__file__, 16, 14), wrightomega_563239, *[p_563240], **kwargs_563241)
    
    # Assigning a type to the variable 'res' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'res', wrightomega_call_result_563242)
    
    # Call to assert_(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Call to isnan(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'res' (line 17)
    res_563246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 25), 'res', False)
    # Obtaining the member 'real' of a type (line 17)
    real_563247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 25), res_563246, 'real')
    # Processing the call keyword arguments (line 17)
    kwargs_563248 = {}
    # Getting the type of 'np' (line 17)
    np_563244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), 'np', False)
    # Obtaining the member 'isnan' of a type (line 17)
    isnan_563245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 16), np_563244, 'isnan')
    # Calling isnan(args, kwargs) (line 17)
    isnan_call_result_563249 = invoke(stypy.reporting.localization.Localization(__file__, 17, 16), isnan_563245, *[real_563247], **kwargs_563248)
    
    # Processing the call keyword arguments (line 17)
    kwargs_563250 = {}
    # Getting the type of 'assert_' (line 17)
    assert__563243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 17)
    assert__call_result_563251 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), assert__563243, *[isnan_call_result_563249], **kwargs_563250)
    
    
    # Call to assert_(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Call to isnan(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'res' (line 18)
    res_563255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'res', False)
    # Obtaining the member 'imag' of a type (line 18)
    imag_563256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 25), res_563255, 'imag')
    # Processing the call keyword arguments (line 18)
    kwargs_563257 = {}
    # Getting the type of 'np' (line 18)
    np_563253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'np', False)
    # Obtaining the member 'isnan' of a type (line 18)
    isnan_563254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 16), np_563253, 'isnan')
    # Calling isnan(args, kwargs) (line 18)
    isnan_call_result_563258 = invoke(stypy.reporting.localization.Localization(__file__, 18, 16), isnan_563254, *[imag_563256], **kwargs_563257)
    
    # Processing the call keyword arguments (line 18)
    kwargs_563259 = {}
    # Getting the type of 'assert_' (line 18)
    assert__563252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 18)
    assert__call_result_563260 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), assert__563252, *[isnan_call_result_563258], **kwargs_563259)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_wrightomega_nan(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_wrightomega_nan' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_563261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_563261)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_wrightomega_nan'
    return stypy_return_type_563261

# Assigning a type to the variable 'test_wrightomega_nan' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'test_wrightomega_nan', test_wrightomega_nan)

@norecursion
def test_wrightomega_inf_branch(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_wrightomega_inf_branch'
    module_type_store = module_type_store.open_function_context('test_wrightomega_inf_branch', 21, 0, False)
    
    # Passed parameters checking function
    test_wrightomega_inf_branch.stypy_localization = localization
    test_wrightomega_inf_branch.stypy_type_of_self = None
    test_wrightomega_inf_branch.stypy_type_store = module_type_store
    test_wrightomega_inf_branch.stypy_function_name = 'test_wrightomega_inf_branch'
    test_wrightomega_inf_branch.stypy_param_names_list = []
    test_wrightomega_inf_branch.stypy_varargs_param_name = None
    test_wrightomega_inf_branch.stypy_kwargs_param_name = None
    test_wrightomega_inf_branch.stypy_call_defaults = defaults
    test_wrightomega_inf_branch.stypy_call_varargs = varargs
    test_wrightomega_inf_branch.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_wrightomega_inf_branch', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_wrightomega_inf_branch', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_wrightomega_inf_branch(...)' code ##################

    
    # Assigning a List to a Name (line 22):
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_563262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    
    # Call to complex(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Getting the type of 'np' (line 22)
    np_563264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), 'np', False)
    # Obtaining the member 'inf' of a type (line 22)
    inf_563265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 20), np_563264, 'inf')
    # Applying the 'usub' unary operator (line 22)
    result___neg___563266 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 19), 'usub', inf_563265)
    
    # Getting the type of 'np' (line 22)
    np_563267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 28), 'np', False)
    # Obtaining the member 'pi' of a type (line 22)
    pi_563268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 28), np_563267, 'pi')
    int_563269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 34), 'int')
    # Applying the binary operator 'div' (line 22)
    result_div_563270 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 28), 'div', pi_563268, int_563269)
    
    # Processing the call keyword arguments (line 22)
    kwargs_563271 = {}
    # Getting the type of 'complex' (line 22)
    complex_563263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'complex', False)
    # Calling complex(args, kwargs) (line 22)
    complex_call_result_563272 = invoke(stypy.reporting.localization.Localization(__file__, 22, 11), complex_563263, *[result___neg___563266, result_div_563270], **kwargs_563271)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_563262, complex_call_result_563272)
    # Adding element type (line 22)
    
    # Call to complex(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Getting the type of 'np' (line 23)
    np_563274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 20), 'np', False)
    # Obtaining the member 'inf' of a type (line 23)
    inf_563275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 20), np_563274, 'inf')
    # Applying the 'usub' unary operator (line 23)
    result___neg___563276 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 19), 'usub', inf_563275)
    
    
    # Getting the type of 'np' (line 23)
    np_563277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 29), 'np', False)
    # Obtaining the member 'pi' of a type (line 23)
    pi_563278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 29), np_563277, 'pi')
    # Applying the 'usub' unary operator (line 23)
    result___neg___563279 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 28), 'usub', pi_563278)
    
    int_563280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 35), 'int')
    # Applying the binary operator 'div' (line 23)
    result_div_563281 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 28), 'div', result___neg___563279, int_563280)
    
    # Processing the call keyword arguments (line 23)
    kwargs_563282 = {}
    # Getting the type of 'complex' (line 23)
    complex_563273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'complex', False)
    # Calling complex(args, kwargs) (line 23)
    complex_call_result_563283 = invoke(stypy.reporting.localization.Localization(__file__, 23, 11), complex_563273, *[result___neg___563276, result_div_563281], **kwargs_563282)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_563262, complex_call_result_563283)
    # Adding element type (line 22)
    
    # Call to complex(...): (line 24)
    # Processing the call arguments (line 24)
    
    # Getting the type of 'np' (line 24)
    np_563285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'np', False)
    # Obtaining the member 'inf' of a type (line 24)
    inf_563286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 20), np_563285, 'inf')
    # Applying the 'usub' unary operator (line 24)
    result___neg___563287 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 19), 'usub', inf_563286)
    
    int_563288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 28), 'int')
    # Getting the type of 'np' (line 24)
    np_563289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 30), 'np', False)
    # Obtaining the member 'pi' of a type (line 24)
    pi_563290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 30), np_563289, 'pi')
    # Applying the binary operator '*' (line 24)
    result_mul_563291 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 28), '*', int_563288, pi_563290)
    
    int_563292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 36), 'int')
    # Applying the binary operator 'div' (line 24)
    result_div_563293 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 35), 'div', result_mul_563291, int_563292)
    
    # Processing the call keyword arguments (line 24)
    kwargs_563294 = {}
    # Getting the type of 'complex' (line 24)
    complex_563284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'complex', False)
    # Calling complex(args, kwargs) (line 24)
    complex_call_result_563295 = invoke(stypy.reporting.localization.Localization(__file__, 24, 11), complex_563284, *[result___neg___563287, result_div_563293], **kwargs_563294)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_563262, complex_call_result_563295)
    # Adding element type (line 22)
    
    # Call to complex(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Getting the type of 'np' (line 25)
    np_563297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'np', False)
    # Obtaining the member 'inf' of a type (line 25)
    inf_563298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 20), np_563297, 'inf')
    # Applying the 'usub' unary operator (line 25)
    result___neg___563299 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 19), 'usub', inf_563298)
    
    int_563300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 28), 'int')
    # Getting the type of 'np' (line 25)
    np_563301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 31), 'np', False)
    # Obtaining the member 'pi' of a type (line 25)
    pi_563302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 31), np_563301, 'pi')
    # Applying the binary operator '*' (line 25)
    result_mul_563303 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 28), '*', int_563300, pi_563302)
    
    int_563304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 37), 'int')
    # Applying the binary operator 'div' (line 25)
    result_div_563305 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 36), 'div', result_mul_563303, int_563304)
    
    # Processing the call keyword arguments (line 25)
    kwargs_563306 = {}
    # Getting the type of 'complex' (line 25)
    complex_563296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'complex', False)
    # Calling complex(args, kwargs) (line 25)
    complex_call_result_563307 = invoke(stypy.reporting.localization.Localization(__file__, 25, 11), complex_563296, *[result___neg___563299, result_div_563305], **kwargs_563306)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 10), list_563262, complex_call_result_563307)
    
    # Assigning a type to the variable 'pts' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'pts', list_563262)
    
    # Assigning a List to a Name (line 26):
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_563308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    
    # Call to complex(...): (line 26)
    # Processing the call arguments (line 26)
    float_563310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 32), 'float')
    float_563311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 37), 'float')
    # Processing the call keyword arguments (line 26)
    kwargs_563312 = {}
    # Getting the type of 'complex' (line 26)
    complex_563309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 24), 'complex', False)
    # Calling complex(args, kwargs) (line 26)
    complex_call_result_563313 = invoke(stypy.reporting.localization.Localization(__file__, 26, 24), complex_563309, *[float_563310, float_563311], **kwargs_563312)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 23), list_563308, complex_call_result_563313)
    # Adding element type (line 26)
    
    # Call to complex(...): (line 27)
    # Processing the call arguments (line 27)
    float_563315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 32), 'float')
    float_563316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 37), 'float')
    # Processing the call keyword arguments (line 27)
    kwargs_563317 = {}
    # Getting the type of 'complex' (line 27)
    complex_563314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 24), 'complex', False)
    # Calling complex(args, kwargs) (line 27)
    complex_call_result_563318 = invoke(stypy.reporting.localization.Localization(__file__, 27, 24), complex_563314, *[float_563315, float_563316], **kwargs_563317)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 23), list_563308, complex_call_result_563318)
    # Adding element type (line 26)
    
    # Call to complex(...): (line 28)
    # Processing the call arguments (line 28)
    float_563320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 32), 'float')
    float_563321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 38), 'float')
    # Processing the call keyword arguments (line 28)
    kwargs_563322 = {}
    # Getting the type of 'complex' (line 28)
    complex_563319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'complex', False)
    # Calling complex(args, kwargs) (line 28)
    complex_call_result_563323 = invoke(stypy.reporting.localization.Localization(__file__, 28, 24), complex_563319, *[float_563320, float_563321], **kwargs_563322)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 23), list_563308, complex_call_result_563323)
    # Adding element type (line 26)
    
    # Call to complex(...): (line 29)
    # Processing the call arguments (line 29)
    float_563325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 32), 'float')
    float_563326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 38), 'float')
    # Processing the call keyword arguments (line 29)
    kwargs_563327 = {}
    # Getting the type of 'complex' (line 29)
    complex_563324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 24), 'complex', False)
    # Calling complex(args, kwargs) (line 29)
    complex_call_result_563328 = invoke(stypy.reporting.localization.Localization(__file__, 29, 24), complex_563324, *[float_563325, float_563326], **kwargs_563327)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 23), list_563308, complex_call_result_563328)
    
    # Assigning a type to the variable 'expected_results' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'expected_results', list_563308)
    
    
    # Call to zip(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'pts' (line 30)
    pts_563330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 27), 'pts', False)
    # Getting the type of 'expected_results' (line 30)
    expected_results_563331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 32), 'expected_results', False)
    # Processing the call keyword arguments (line 30)
    kwargs_563332 = {}
    # Getting the type of 'zip' (line 30)
    zip_563329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'zip', False)
    # Calling zip(args, kwargs) (line 30)
    zip_call_result_563333 = invoke(stypy.reporting.localization.Localization(__file__, 30, 23), zip_563329, *[pts_563330, expected_results_563331], **kwargs_563332)
    
    # Testing the type of a for loop iterable (line 30)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 30, 4), zip_call_result_563333)
    # Getting the type of the for loop variable (line 30)
    for_loop_var_563334 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 30, 4), zip_call_result_563333)
    # Assigning a type to the variable 'p' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'p', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 4), for_loop_var_563334))
    # Assigning a type to the variable 'expected' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'expected', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 4), for_loop_var_563334))
    # SSA begins for a for statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 31):
    
    # Call to wrightomega(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'p' (line 31)
    p_563337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 29), 'p', False)
    # Processing the call keyword arguments (line 31)
    kwargs_563338 = {}
    # Getting the type of 'sc' (line 31)
    sc_563335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'sc', False)
    # Obtaining the member 'wrightomega' of a type (line 31)
    wrightomega_563336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 14), sc_563335, 'wrightomega')
    # Calling wrightomega(args, kwargs) (line 31)
    wrightomega_call_result_563339 = invoke(stypy.reporting.localization.Localization(__file__, 31, 14), wrightomega_563336, *[p_563337], **kwargs_563338)
    
    # Assigning a type to the variable 'res' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'res', wrightomega_call_result_563339)
    
    # Call to assert_equal(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'res' (line 36)
    res_563341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 21), 'res', False)
    # Obtaining the member 'real' of a type (line 36)
    real_563342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 21), res_563341, 'real')
    # Getting the type of 'expected' (line 36)
    expected_563343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'expected', False)
    # Obtaining the member 'real' of a type (line 36)
    real_563344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 31), expected_563343, 'real')
    # Processing the call keyword arguments (line 36)
    kwargs_563345 = {}
    # Getting the type of 'assert_equal' (line 36)
    assert_equal_563340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 36)
    assert_equal_call_result_563346 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), assert_equal_563340, *[real_563342, real_563344], **kwargs_563345)
    
    
    # Call to assert_equal(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'res' (line 37)
    res_563348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 21), 'res', False)
    # Obtaining the member 'imag' of a type (line 37)
    imag_563349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 21), res_563348, 'imag')
    # Getting the type of 'expected' (line 37)
    expected_563350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), 'expected', False)
    # Obtaining the member 'imag' of a type (line 37)
    imag_563351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 31), expected_563350, 'imag')
    # Processing the call keyword arguments (line 37)
    kwargs_563352 = {}
    # Getting the type of 'assert_equal' (line 37)
    assert_equal_563347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 37)
    assert_equal_call_result_563353 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), assert_equal_563347, *[imag_563349, imag_563351], **kwargs_563352)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_wrightomega_inf_branch(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_wrightomega_inf_branch' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_563354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_563354)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_wrightomega_inf_branch'
    return stypy_return_type_563354

# Assigning a type to the variable 'test_wrightomega_inf_branch' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'test_wrightomega_inf_branch', test_wrightomega_inf_branch)

@norecursion
def test_wrightomega_inf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_wrightomega_inf'
    module_type_store = module_type_store.open_function_context('test_wrightomega_inf', 40, 0, False)
    
    # Passed parameters checking function
    test_wrightomega_inf.stypy_localization = localization
    test_wrightomega_inf.stypy_type_of_self = None
    test_wrightomega_inf.stypy_type_store = module_type_store
    test_wrightomega_inf.stypy_function_name = 'test_wrightomega_inf'
    test_wrightomega_inf.stypy_param_names_list = []
    test_wrightomega_inf.stypy_varargs_param_name = None
    test_wrightomega_inf.stypy_kwargs_param_name = None
    test_wrightomega_inf.stypy_call_defaults = defaults
    test_wrightomega_inf.stypy_call_varargs = varargs
    test_wrightomega_inf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_wrightomega_inf', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_wrightomega_inf', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_wrightomega_inf(...)' code ##################

    
    # Assigning a List to a Name (line 41):
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_563355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    
    # Call to complex(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'np' (line 41)
    np_563357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 19), 'np', False)
    # Obtaining the member 'inf' of a type (line 41)
    inf_563358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 19), np_563357, 'inf')
    int_563359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 27), 'int')
    # Processing the call keyword arguments (line 41)
    kwargs_563360 = {}
    # Getting the type of 'complex' (line 41)
    complex_563356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'complex', False)
    # Calling complex(args, kwargs) (line 41)
    complex_call_result_563361 = invoke(stypy.reporting.localization.Localization(__file__, 41, 11), complex_563356, *[inf_563358, int_563359], **kwargs_563360)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 10), list_563355, complex_call_result_563361)
    # Adding element type (line 41)
    
    # Call to complex(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Getting the type of 'np' (line 42)
    np_563363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'np', False)
    # Obtaining the member 'inf' of a type (line 42)
    inf_563364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 20), np_563363, 'inf')
    # Applying the 'usub' unary operator (line 42)
    result___neg___563365 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 19), 'usub', inf_563364)
    
    int_563366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 28), 'int')
    # Processing the call keyword arguments (line 42)
    kwargs_563367 = {}
    # Getting the type of 'complex' (line 42)
    complex_563362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'complex', False)
    # Calling complex(args, kwargs) (line 42)
    complex_call_result_563368 = invoke(stypy.reporting.localization.Localization(__file__, 42, 11), complex_563362, *[result___neg___563365, int_563366], **kwargs_563367)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 10), list_563355, complex_call_result_563368)
    # Adding element type (line 41)
    
    # Call to complex(...): (line 43)
    # Processing the call arguments (line 43)
    int_563370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'int')
    # Getting the type of 'np' (line 43)
    np_563371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 23), 'np', False)
    # Obtaining the member 'inf' of a type (line 43)
    inf_563372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 23), np_563371, 'inf')
    # Processing the call keyword arguments (line 43)
    kwargs_563373 = {}
    # Getting the type of 'complex' (line 43)
    complex_563369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'complex', False)
    # Calling complex(args, kwargs) (line 43)
    complex_call_result_563374 = invoke(stypy.reporting.localization.Localization(__file__, 43, 11), complex_563369, *[int_563370, inf_563372], **kwargs_563373)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 10), list_563355, complex_call_result_563374)
    # Adding element type (line 41)
    
    # Call to complex(...): (line 44)
    # Processing the call arguments (line 44)
    int_563376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 19), 'int')
    
    # Getting the type of 'np' (line 44)
    np_563377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'np', False)
    # Obtaining the member 'inf' of a type (line 44)
    inf_563378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 24), np_563377, 'inf')
    # Applying the 'usub' unary operator (line 44)
    result___neg___563379 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 23), 'usub', inf_563378)
    
    # Processing the call keyword arguments (line 44)
    kwargs_563380 = {}
    # Getting the type of 'complex' (line 44)
    complex_563375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'complex', False)
    # Calling complex(args, kwargs) (line 44)
    complex_call_result_563381 = invoke(stypy.reporting.localization.Localization(__file__, 44, 11), complex_563375, *[int_563376, result___neg___563379], **kwargs_563380)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 10), list_563355, complex_call_result_563381)
    
    # Assigning a type to the variable 'pts' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'pts', list_563355)
    
    # Getting the type of 'pts' (line 45)
    pts_563382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 13), 'pts')
    # Testing the type of a for loop iterable (line 45)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 45, 4), pts_563382)
    # Getting the type of the for loop variable (line 45)
    for_loop_var_563383 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 45, 4), pts_563382)
    # Assigning a type to the variable 'p' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'p', for_loop_var_563383)
    # SSA begins for a for statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_equal(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Call to wrightomega(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'p' (line 46)
    p_563387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 36), 'p', False)
    # Processing the call keyword arguments (line 46)
    kwargs_563388 = {}
    # Getting the type of 'sc' (line 46)
    sc_563385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 21), 'sc', False)
    # Obtaining the member 'wrightomega' of a type (line 46)
    wrightomega_563386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 21), sc_563385, 'wrightomega')
    # Calling wrightomega(args, kwargs) (line 46)
    wrightomega_call_result_563389 = invoke(stypy.reporting.localization.Localization(__file__, 46, 21), wrightomega_563386, *[p_563387], **kwargs_563388)
    
    # Getting the type of 'p' (line 46)
    p_563390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 40), 'p', False)
    # Processing the call keyword arguments (line 46)
    kwargs_563391 = {}
    # Getting the type of 'assert_equal' (line 46)
    assert_equal_563384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 46)
    assert_equal_call_result_563392 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), assert_equal_563384, *[wrightomega_call_result_563389, p_563390], **kwargs_563391)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_wrightomega_inf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_wrightomega_inf' in the type store
    # Getting the type of 'stypy_return_type' (line 40)
    stypy_return_type_563393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_563393)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_wrightomega_inf'
    return stypy_return_type_563393

# Assigning a type to the variable 'test_wrightomega_inf' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'test_wrightomega_inf', test_wrightomega_inf)

@norecursion
def test_wrightomega_singular(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_wrightomega_singular'
    module_type_store = module_type_store.open_function_context('test_wrightomega_singular', 49, 0, False)
    
    # Passed parameters checking function
    test_wrightomega_singular.stypy_localization = localization
    test_wrightomega_singular.stypy_type_of_self = None
    test_wrightomega_singular.stypy_type_store = module_type_store
    test_wrightomega_singular.stypy_function_name = 'test_wrightomega_singular'
    test_wrightomega_singular.stypy_param_names_list = []
    test_wrightomega_singular.stypy_varargs_param_name = None
    test_wrightomega_singular.stypy_kwargs_param_name = None
    test_wrightomega_singular.stypy_call_defaults = defaults
    test_wrightomega_singular.stypy_call_varargs = varargs
    test_wrightomega_singular.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_wrightomega_singular', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_wrightomega_singular', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_wrightomega_singular(...)' code ##################

    
    # Assigning a List to a Name (line 50):
    
    # Obtaining an instance of the builtin type 'list' (line 50)
    list_563394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 50)
    # Adding element type (line 50)
    
    # Call to complex(...): (line 50)
    # Processing the call arguments (line 50)
    float_563396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 19), 'float')
    # Getting the type of 'np' (line 50)
    np_563397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 25), 'np', False)
    # Obtaining the member 'pi' of a type (line 50)
    pi_563398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 25), np_563397, 'pi')
    # Processing the call keyword arguments (line 50)
    kwargs_563399 = {}
    # Getting the type of 'complex' (line 50)
    complex_563395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'complex', False)
    # Calling complex(args, kwargs) (line 50)
    complex_call_result_563400 = invoke(stypy.reporting.localization.Localization(__file__, 50, 11), complex_563395, *[float_563396, pi_563398], **kwargs_563399)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_563394, complex_call_result_563400)
    # Adding element type (line 50)
    
    # Call to complex(...): (line 51)
    # Processing the call arguments (line 51)
    float_563402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 19), 'float')
    
    # Getting the type of 'np' (line 51)
    np_563403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), 'np', False)
    # Obtaining the member 'pi' of a type (line 51)
    pi_563404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 26), np_563403, 'pi')
    # Applying the 'usub' unary operator (line 51)
    result___neg___563405 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 25), 'usub', pi_563404)
    
    # Processing the call keyword arguments (line 51)
    kwargs_563406 = {}
    # Getting the type of 'complex' (line 51)
    complex_563401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'complex', False)
    # Calling complex(args, kwargs) (line 51)
    complex_call_result_563407 = invoke(stypy.reporting.localization.Localization(__file__, 51, 11), complex_563401, *[float_563402, result___neg___563405], **kwargs_563406)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_563394, complex_call_result_563407)
    
    # Assigning a type to the variable 'pts' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'pts', list_563394)
    
    # Getting the type of 'pts' (line 52)
    pts_563408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 13), 'pts')
    # Testing the type of a for loop iterable (line 52)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 52, 4), pts_563408)
    # Getting the type of the for loop variable (line 52)
    for_loop_var_563409 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 52, 4), pts_563408)
    # Assigning a type to the variable 'p' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'p', for_loop_var_563409)
    # SSA begins for a for statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 53):
    
    # Call to wrightomega(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'p' (line 53)
    p_563412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 29), 'p', False)
    # Processing the call keyword arguments (line 53)
    kwargs_563413 = {}
    # Getting the type of 'sc' (line 53)
    sc_563410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 14), 'sc', False)
    # Obtaining the member 'wrightomega' of a type (line 53)
    wrightomega_563411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 14), sc_563410, 'wrightomega')
    # Calling wrightomega(args, kwargs) (line 53)
    wrightomega_call_result_563414 = invoke(stypy.reporting.localization.Localization(__file__, 53, 14), wrightomega_563411, *[p_563412], **kwargs_563413)
    
    # Assigning a type to the variable 'res' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'res', wrightomega_call_result_563414)
    
    # Call to assert_equal(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'res' (line 54)
    res_563416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'res', False)
    float_563417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 26), 'float')
    # Processing the call keyword arguments (line 54)
    kwargs_563418 = {}
    # Getting the type of 'assert_equal' (line 54)
    assert_equal_563415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 54)
    assert_equal_call_result_563419 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), assert_equal_563415, *[res_563416, float_563417], **kwargs_563418)
    
    
    # Call to assert_(...): (line 55)
    # Processing the call arguments (line 55)
    
    
    # Call to signbit(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'res' (line 55)
    res_563423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 27), 'res', False)
    # Obtaining the member 'imag' of a type (line 55)
    imag_563424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 27), res_563423, 'imag')
    # Processing the call keyword arguments (line 55)
    kwargs_563425 = {}
    # Getting the type of 'np' (line 55)
    np_563421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'np', False)
    # Obtaining the member 'signbit' of a type (line 55)
    signbit_563422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), np_563421, 'signbit')
    # Calling signbit(args, kwargs) (line 55)
    signbit_call_result_563426 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), signbit_563422, *[imag_563424], **kwargs_563425)
    
    # Getting the type of 'False' (line 55)
    False_563427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 40), 'False', False)
    # Applying the binary operator '==' (line 55)
    result_eq_563428 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 16), '==', signbit_call_result_563426, False_563427)
    
    # Processing the call keyword arguments (line 55)
    kwargs_563429 = {}
    # Getting the type of 'assert_' (line 55)
    assert__563420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 55)
    assert__call_result_563430 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), assert__563420, *[result_eq_563428], **kwargs_563429)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_wrightomega_singular(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_wrightomega_singular' in the type store
    # Getting the type of 'stypy_return_type' (line 49)
    stypy_return_type_563431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_563431)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_wrightomega_singular'
    return stypy_return_type_563431

# Assigning a type to the variable 'test_wrightomega_singular' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'test_wrightomega_singular', test_wrightomega_singular)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
