
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import assert_allclose, assert_equal
5: 
6: from scipy.stats._tukeylambda_stats import (tukeylambda_variance,
7:                                             tukeylambda_kurtosis)
8: 
9: 
10: def test_tukeylambda_stats_known_exact():
11:     '''Compare results with some known exact formulas.'''
12:     # Some exact values of the Tukey Lambda variance and kurtosis:
13:     # lambda   var      kurtosis
14:     #   0     pi**2/3     6/5     (logistic distribution)
15:     #  0.5    4 - pi    (5/3 - pi/2)/(pi/4 - 1)**2 - 3
16:     #   1      1/3       -6/5     (uniform distribution on (-1,1))
17:     #   2      1/12      -6/5     (uniform distribution on (-1/2, 1/2))
18: 
19:     # lambda = 0
20:     var = tukeylambda_variance(0)
21:     assert_allclose(var, np.pi**2 / 3, atol=1e-12)
22:     kurt = tukeylambda_kurtosis(0)
23:     assert_allclose(kurt, 1.2, atol=1e-10)
24: 
25:     # lambda = 0.5
26:     var = tukeylambda_variance(0.5)
27:     assert_allclose(var, 4 - np.pi, atol=1e-12)
28:     kurt = tukeylambda_kurtosis(0.5)
29:     desired = (5./3 - np.pi/2) / (np.pi/4 - 1)**2 - 3
30:     assert_allclose(kurt, desired, atol=1e-10)
31: 
32:     # lambda = 1
33:     var = tukeylambda_variance(1)
34:     assert_allclose(var, 1.0 / 3, atol=1e-12)
35:     kurt = tukeylambda_kurtosis(1)
36:     assert_allclose(kurt, -1.2, atol=1e-10)
37: 
38:     # lambda = 2
39:     var = tukeylambda_variance(2)
40:     assert_allclose(var, 1.0 / 12, atol=1e-12)
41:     kurt = tukeylambda_kurtosis(2)
42:     assert_allclose(kurt, -1.2, atol=1e-10)
43: 
44: 
45: def test_tukeylambda_stats_mpmath():
46:     '''Compare results with some values that were computed using mpmath.'''
47:     a10 = dict(atol=1e-10, rtol=0)
48:     a12 = dict(atol=1e-12, rtol=0)
49:     data = [
50:         # lambda        variance              kurtosis
51:         [-0.1, 4.78050217874253547, 3.78559520346454510],
52:         [-0.0649, 4.16428023599895777, 2.52019675947435718],
53:         [-0.05, 3.93672267890775277, 2.13129793057777277],
54:         [-0.001, 3.30128380390964882, 1.21452460083542988],
55:         [0.001, 3.27850775649572176, 1.18560634779287585],
56:         [0.03125, 2.95927803254615800, 0.804487555161819980],
57:         [0.05, 2.78281053405464501, 0.611604043886644327],
58:         [0.0649, 2.65282386754100551, 0.476834119532774540],
59:         [1.2, 0.242153920578588346, -1.23428047169049726],
60:         [10.0, 0.00095237579757703597, 2.37810697355144933],
61:         [20.0, 0.00012195121951131043, 7.37654321002709531],
62:     ]
63: 
64:     for lam, var_expected, kurt_expected in data:
65:         var = tukeylambda_variance(lam)
66:         assert_allclose(var, var_expected, **a12)
67:         kurt = tukeylambda_kurtosis(lam)
68:         assert_allclose(kurt, kurt_expected, **a10)
69: 
70:     # Test with vector arguments (most of the other tests are for single
71:     # values).
72:     lam, var_expected, kurt_expected = zip(*data)
73:     var = tukeylambda_variance(lam)
74:     assert_allclose(var, var_expected, **a12)
75:     kurt = tukeylambda_kurtosis(lam)
76:     assert_allclose(kurt, kurt_expected, **a10)
77: 
78: 
79: def test_tukeylambda_stats_invalid():
80:     '''Test values of lambda outside the domains of the functions.'''
81:     lam = [-1.0, -0.5]
82:     var = tukeylambda_variance(lam)
83:     assert_equal(var, np.array([np.nan, np.inf]))
84: 
85:     lam = [-1.0, -0.25]
86:     kurt = tukeylambda_kurtosis(lam)
87:     assert_equal(kurt, np.array([np.nan, np.inf]))
88: 
89: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_705556 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_705556) is not StypyTypeError):

    if (import_705556 != 'pyd_module'):
        __import__(import_705556)
        sys_modules_705557 = sys.modules[import_705556]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_705557.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_705556)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_allclose, assert_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_705558 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_705558) is not StypyTypeError):

    if (import_705558 != 'pyd_module'):
        __import__(import_705558)
        sys_modules_705559 = sys.modules[import_705558]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_705559.module_type_store, module_type_store, ['assert_allclose', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_705559, sys_modules_705559.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_allclose', 'assert_equal'], [assert_allclose, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_705558)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.stats._tukeylambda_stats import tukeylambda_variance, tukeylambda_kurtosis' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_705560 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.stats._tukeylambda_stats')

if (type(import_705560) is not StypyTypeError):

    if (import_705560 != 'pyd_module'):
        __import__(import_705560)
        sys_modules_705561 = sys.modules[import_705560]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.stats._tukeylambda_stats', sys_modules_705561.module_type_store, module_type_store, ['tukeylambda_variance', 'tukeylambda_kurtosis'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_705561, sys_modules_705561.module_type_store, module_type_store)
    else:
        from scipy.stats._tukeylambda_stats import tukeylambda_variance, tukeylambda_kurtosis

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.stats._tukeylambda_stats', None, module_type_store, ['tukeylambda_variance', 'tukeylambda_kurtosis'], [tukeylambda_variance, tukeylambda_kurtosis])

else:
    # Assigning a type to the variable 'scipy.stats._tukeylambda_stats' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.stats._tukeylambda_stats', import_705560)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')


@norecursion
def test_tukeylambda_stats_known_exact(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_tukeylambda_stats_known_exact'
    module_type_store = module_type_store.open_function_context('test_tukeylambda_stats_known_exact', 10, 0, False)
    
    # Passed parameters checking function
    test_tukeylambda_stats_known_exact.stypy_localization = localization
    test_tukeylambda_stats_known_exact.stypy_type_of_self = None
    test_tukeylambda_stats_known_exact.stypy_type_store = module_type_store
    test_tukeylambda_stats_known_exact.stypy_function_name = 'test_tukeylambda_stats_known_exact'
    test_tukeylambda_stats_known_exact.stypy_param_names_list = []
    test_tukeylambda_stats_known_exact.stypy_varargs_param_name = None
    test_tukeylambda_stats_known_exact.stypy_kwargs_param_name = None
    test_tukeylambda_stats_known_exact.stypy_call_defaults = defaults
    test_tukeylambda_stats_known_exact.stypy_call_varargs = varargs
    test_tukeylambda_stats_known_exact.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_tukeylambda_stats_known_exact', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_tukeylambda_stats_known_exact', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_tukeylambda_stats_known_exact(...)' code ##################

    str_705562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 4), 'str', 'Compare results with some known exact formulas.')
    
    # Assigning a Call to a Name (line 20):
    
    # Assigning a Call to a Name (line 20):
    
    # Call to tukeylambda_variance(...): (line 20)
    # Processing the call arguments (line 20)
    int_705564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 31), 'int')
    # Processing the call keyword arguments (line 20)
    kwargs_705565 = {}
    # Getting the type of 'tukeylambda_variance' (line 20)
    tukeylambda_variance_705563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'tukeylambda_variance', False)
    # Calling tukeylambda_variance(args, kwargs) (line 20)
    tukeylambda_variance_call_result_705566 = invoke(stypy.reporting.localization.Localization(__file__, 20, 10), tukeylambda_variance_705563, *[int_705564], **kwargs_705565)
    
    # Assigning a type to the variable 'var' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'var', tukeylambda_variance_call_result_705566)
    
    # Call to assert_allclose(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'var' (line 21)
    var_705568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'var', False)
    # Getting the type of 'np' (line 21)
    np_705569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 25), 'np', False)
    # Obtaining the member 'pi' of a type (line 21)
    pi_705570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 25), np_705569, 'pi')
    int_705571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 32), 'int')
    # Applying the binary operator '**' (line 21)
    result_pow_705572 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 25), '**', pi_705570, int_705571)
    
    int_705573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 36), 'int')
    # Applying the binary operator 'div' (line 21)
    result_div_705574 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 25), 'div', result_pow_705572, int_705573)
    
    # Processing the call keyword arguments (line 21)
    float_705575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 44), 'float')
    keyword_705576 = float_705575
    kwargs_705577 = {'atol': keyword_705576}
    # Getting the type of 'assert_allclose' (line 21)
    assert_allclose_705567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 21)
    assert_allclose_call_result_705578 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), assert_allclose_705567, *[var_705568, result_div_705574], **kwargs_705577)
    
    
    # Assigning a Call to a Name (line 22):
    
    # Assigning a Call to a Name (line 22):
    
    # Call to tukeylambda_kurtosis(...): (line 22)
    # Processing the call arguments (line 22)
    int_705580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 32), 'int')
    # Processing the call keyword arguments (line 22)
    kwargs_705581 = {}
    # Getting the type of 'tukeylambda_kurtosis' (line 22)
    tukeylambda_kurtosis_705579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'tukeylambda_kurtosis', False)
    # Calling tukeylambda_kurtosis(args, kwargs) (line 22)
    tukeylambda_kurtosis_call_result_705582 = invoke(stypy.reporting.localization.Localization(__file__, 22, 11), tukeylambda_kurtosis_705579, *[int_705580], **kwargs_705581)
    
    # Assigning a type to the variable 'kurt' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'kurt', tukeylambda_kurtosis_call_result_705582)
    
    # Call to assert_allclose(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'kurt' (line 23)
    kurt_705584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 20), 'kurt', False)
    float_705585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 26), 'float')
    # Processing the call keyword arguments (line 23)
    float_705586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 36), 'float')
    keyword_705587 = float_705586
    kwargs_705588 = {'atol': keyword_705587}
    # Getting the type of 'assert_allclose' (line 23)
    assert_allclose_705583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 23)
    assert_allclose_call_result_705589 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), assert_allclose_705583, *[kurt_705584, float_705585], **kwargs_705588)
    
    
    # Assigning a Call to a Name (line 26):
    
    # Assigning a Call to a Name (line 26):
    
    # Call to tukeylambda_variance(...): (line 26)
    # Processing the call arguments (line 26)
    float_705591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 31), 'float')
    # Processing the call keyword arguments (line 26)
    kwargs_705592 = {}
    # Getting the type of 'tukeylambda_variance' (line 26)
    tukeylambda_variance_705590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 10), 'tukeylambda_variance', False)
    # Calling tukeylambda_variance(args, kwargs) (line 26)
    tukeylambda_variance_call_result_705593 = invoke(stypy.reporting.localization.Localization(__file__, 26, 10), tukeylambda_variance_705590, *[float_705591], **kwargs_705592)
    
    # Assigning a type to the variable 'var' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'var', tukeylambda_variance_call_result_705593)
    
    # Call to assert_allclose(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'var' (line 27)
    var_705595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 20), 'var', False)
    int_705596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'int')
    # Getting the type of 'np' (line 27)
    np_705597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 29), 'np', False)
    # Obtaining the member 'pi' of a type (line 27)
    pi_705598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 29), np_705597, 'pi')
    # Applying the binary operator '-' (line 27)
    result_sub_705599 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 25), '-', int_705596, pi_705598)
    
    # Processing the call keyword arguments (line 27)
    float_705600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 41), 'float')
    keyword_705601 = float_705600
    kwargs_705602 = {'atol': keyword_705601}
    # Getting the type of 'assert_allclose' (line 27)
    assert_allclose_705594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 27)
    assert_allclose_call_result_705603 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), assert_allclose_705594, *[var_705595, result_sub_705599], **kwargs_705602)
    
    
    # Assigning a Call to a Name (line 28):
    
    # Assigning a Call to a Name (line 28):
    
    # Call to tukeylambda_kurtosis(...): (line 28)
    # Processing the call arguments (line 28)
    float_705605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 32), 'float')
    # Processing the call keyword arguments (line 28)
    kwargs_705606 = {}
    # Getting the type of 'tukeylambda_kurtosis' (line 28)
    tukeylambda_kurtosis_705604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'tukeylambda_kurtosis', False)
    # Calling tukeylambda_kurtosis(args, kwargs) (line 28)
    tukeylambda_kurtosis_call_result_705607 = invoke(stypy.reporting.localization.Localization(__file__, 28, 11), tukeylambda_kurtosis_705604, *[float_705605], **kwargs_705606)
    
    # Assigning a type to the variable 'kurt' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'kurt', tukeylambda_kurtosis_call_result_705607)
    
    # Assigning a BinOp to a Name (line 29):
    
    # Assigning a BinOp to a Name (line 29):
    float_705608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 15), 'float')
    int_705609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 18), 'int')
    # Applying the binary operator 'div' (line 29)
    result_div_705610 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 15), 'div', float_705608, int_705609)
    
    # Getting the type of 'np' (line 29)
    np_705611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'np')
    # Obtaining the member 'pi' of a type (line 29)
    pi_705612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 22), np_705611, 'pi')
    int_705613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 28), 'int')
    # Applying the binary operator 'div' (line 29)
    result_div_705614 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 22), 'div', pi_705612, int_705613)
    
    # Applying the binary operator '-' (line 29)
    result_sub_705615 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 15), '-', result_div_705610, result_div_705614)
    
    # Getting the type of 'np' (line 29)
    np_705616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 34), 'np')
    # Obtaining the member 'pi' of a type (line 29)
    pi_705617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 34), np_705616, 'pi')
    int_705618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 40), 'int')
    # Applying the binary operator 'div' (line 29)
    result_div_705619 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 34), 'div', pi_705617, int_705618)
    
    int_705620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 44), 'int')
    # Applying the binary operator '-' (line 29)
    result_sub_705621 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 34), '-', result_div_705619, int_705620)
    
    int_705622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 48), 'int')
    # Applying the binary operator '**' (line 29)
    result_pow_705623 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 33), '**', result_sub_705621, int_705622)
    
    # Applying the binary operator 'div' (line 29)
    result_div_705624 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 14), 'div', result_sub_705615, result_pow_705623)
    
    int_705625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 52), 'int')
    # Applying the binary operator '-' (line 29)
    result_sub_705626 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 14), '-', result_div_705624, int_705625)
    
    # Assigning a type to the variable 'desired' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'desired', result_sub_705626)
    
    # Call to assert_allclose(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'kurt' (line 30)
    kurt_705628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 20), 'kurt', False)
    # Getting the type of 'desired' (line 30)
    desired_705629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 26), 'desired', False)
    # Processing the call keyword arguments (line 30)
    float_705630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 40), 'float')
    keyword_705631 = float_705630
    kwargs_705632 = {'atol': keyword_705631}
    # Getting the type of 'assert_allclose' (line 30)
    assert_allclose_705627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 30)
    assert_allclose_call_result_705633 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), assert_allclose_705627, *[kurt_705628, desired_705629], **kwargs_705632)
    
    
    # Assigning a Call to a Name (line 33):
    
    # Assigning a Call to a Name (line 33):
    
    # Call to tukeylambda_variance(...): (line 33)
    # Processing the call arguments (line 33)
    int_705635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 31), 'int')
    # Processing the call keyword arguments (line 33)
    kwargs_705636 = {}
    # Getting the type of 'tukeylambda_variance' (line 33)
    tukeylambda_variance_705634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 10), 'tukeylambda_variance', False)
    # Calling tukeylambda_variance(args, kwargs) (line 33)
    tukeylambda_variance_call_result_705637 = invoke(stypy.reporting.localization.Localization(__file__, 33, 10), tukeylambda_variance_705634, *[int_705635], **kwargs_705636)
    
    # Assigning a type to the variable 'var' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'var', tukeylambda_variance_call_result_705637)
    
    # Call to assert_allclose(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'var' (line 34)
    var_705639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'var', False)
    float_705640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 25), 'float')
    int_705641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 31), 'int')
    # Applying the binary operator 'div' (line 34)
    result_div_705642 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 25), 'div', float_705640, int_705641)
    
    # Processing the call keyword arguments (line 34)
    float_705643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 39), 'float')
    keyword_705644 = float_705643
    kwargs_705645 = {'atol': keyword_705644}
    # Getting the type of 'assert_allclose' (line 34)
    assert_allclose_705638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 34)
    assert_allclose_call_result_705646 = invoke(stypy.reporting.localization.Localization(__file__, 34, 4), assert_allclose_705638, *[var_705639, result_div_705642], **kwargs_705645)
    
    
    # Assigning a Call to a Name (line 35):
    
    # Assigning a Call to a Name (line 35):
    
    # Call to tukeylambda_kurtosis(...): (line 35)
    # Processing the call arguments (line 35)
    int_705648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 32), 'int')
    # Processing the call keyword arguments (line 35)
    kwargs_705649 = {}
    # Getting the type of 'tukeylambda_kurtosis' (line 35)
    tukeylambda_kurtosis_705647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'tukeylambda_kurtosis', False)
    # Calling tukeylambda_kurtosis(args, kwargs) (line 35)
    tukeylambda_kurtosis_call_result_705650 = invoke(stypy.reporting.localization.Localization(__file__, 35, 11), tukeylambda_kurtosis_705647, *[int_705648], **kwargs_705649)
    
    # Assigning a type to the variable 'kurt' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'kurt', tukeylambda_kurtosis_call_result_705650)
    
    # Call to assert_allclose(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'kurt' (line 36)
    kurt_705652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'kurt', False)
    float_705653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 26), 'float')
    # Processing the call keyword arguments (line 36)
    float_705654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 37), 'float')
    keyword_705655 = float_705654
    kwargs_705656 = {'atol': keyword_705655}
    # Getting the type of 'assert_allclose' (line 36)
    assert_allclose_705651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 36)
    assert_allclose_call_result_705657 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), assert_allclose_705651, *[kurt_705652, float_705653], **kwargs_705656)
    
    
    # Assigning a Call to a Name (line 39):
    
    # Assigning a Call to a Name (line 39):
    
    # Call to tukeylambda_variance(...): (line 39)
    # Processing the call arguments (line 39)
    int_705659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 31), 'int')
    # Processing the call keyword arguments (line 39)
    kwargs_705660 = {}
    # Getting the type of 'tukeylambda_variance' (line 39)
    tukeylambda_variance_705658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 10), 'tukeylambda_variance', False)
    # Calling tukeylambda_variance(args, kwargs) (line 39)
    tukeylambda_variance_call_result_705661 = invoke(stypy.reporting.localization.Localization(__file__, 39, 10), tukeylambda_variance_705658, *[int_705659], **kwargs_705660)
    
    # Assigning a type to the variable 'var' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'var', tukeylambda_variance_call_result_705661)
    
    # Call to assert_allclose(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'var' (line 40)
    var_705663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'var', False)
    float_705664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 25), 'float')
    int_705665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 31), 'int')
    # Applying the binary operator 'div' (line 40)
    result_div_705666 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 25), 'div', float_705664, int_705665)
    
    # Processing the call keyword arguments (line 40)
    float_705667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 40), 'float')
    keyword_705668 = float_705667
    kwargs_705669 = {'atol': keyword_705668}
    # Getting the type of 'assert_allclose' (line 40)
    assert_allclose_705662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 40)
    assert_allclose_call_result_705670 = invoke(stypy.reporting.localization.Localization(__file__, 40, 4), assert_allclose_705662, *[var_705663, result_div_705666], **kwargs_705669)
    
    
    # Assigning a Call to a Name (line 41):
    
    # Assigning a Call to a Name (line 41):
    
    # Call to tukeylambda_kurtosis(...): (line 41)
    # Processing the call arguments (line 41)
    int_705672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 32), 'int')
    # Processing the call keyword arguments (line 41)
    kwargs_705673 = {}
    # Getting the type of 'tukeylambda_kurtosis' (line 41)
    tukeylambda_kurtosis_705671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'tukeylambda_kurtosis', False)
    # Calling tukeylambda_kurtosis(args, kwargs) (line 41)
    tukeylambda_kurtosis_call_result_705674 = invoke(stypy.reporting.localization.Localization(__file__, 41, 11), tukeylambda_kurtosis_705671, *[int_705672], **kwargs_705673)
    
    # Assigning a type to the variable 'kurt' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'kurt', tukeylambda_kurtosis_call_result_705674)
    
    # Call to assert_allclose(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'kurt' (line 42)
    kurt_705676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'kurt', False)
    float_705677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 26), 'float')
    # Processing the call keyword arguments (line 42)
    float_705678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 37), 'float')
    keyword_705679 = float_705678
    kwargs_705680 = {'atol': keyword_705679}
    # Getting the type of 'assert_allclose' (line 42)
    assert_allclose_705675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 42)
    assert_allclose_call_result_705681 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), assert_allclose_705675, *[kurt_705676, float_705677], **kwargs_705680)
    
    
    # ################# End of 'test_tukeylambda_stats_known_exact(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_tukeylambda_stats_known_exact' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_705682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_705682)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_tukeylambda_stats_known_exact'
    return stypy_return_type_705682

# Assigning a type to the variable 'test_tukeylambda_stats_known_exact' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'test_tukeylambda_stats_known_exact', test_tukeylambda_stats_known_exact)

@norecursion
def test_tukeylambda_stats_mpmath(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_tukeylambda_stats_mpmath'
    module_type_store = module_type_store.open_function_context('test_tukeylambda_stats_mpmath', 45, 0, False)
    
    # Passed parameters checking function
    test_tukeylambda_stats_mpmath.stypy_localization = localization
    test_tukeylambda_stats_mpmath.stypy_type_of_self = None
    test_tukeylambda_stats_mpmath.stypy_type_store = module_type_store
    test_tukeylambda_stats_mpmath.stypy_function_name = 'test_tukeylambda_stats_mpmath'
    test_tukeylambda_stats_mpmath.stypy_param_names_list = []
    test_tukeylambda_stats_mpmath.stypy_varargs_param_name = None
    test_tukeylambda_stats_mpmath.stypy_kwargs_param_name = None
    test_tukeylambda_stats_mpmath.stypy_call_defaults = defaults
    test_tukeylambda_stats_mpmath.stypy_call_varargs = varargs
    test_tukeylambda_stats_mpmath.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_tukeylambda_stats_mpmath', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_tukeylambda_stats_mpmath', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_tukeylambda_stats_mpmath(...)' code ##################

    str_705683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'str', 'Compare results with some values that were computed using mpmath.')
    
    # Assigning a Call to a Name (line 47):
    
    # Assigning a Call to a Name (line 47):
    
    # Call to dict(...): (line 47)
    # Processing the call keyword arguments (line 47)
    float_705685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 20), 'float')
    keyword_705686 = float_705685
    int_705687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 32), 'int')
    keyword_705688 = int_705687
    kwargs_705689 = {'rtol': keyword_705688, 'atol': keyword_705686}
    # Getting the type of 'dict' (line 47)
    dict_705684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 10), 'dict', False)
    # Calling dict(args, kwargs) (line 47)
    dict_call_result_705690 = invoke(stypy.reporting.localization.Localization(__file__, 47, 10), dict_705684, *[], **kwargs_705689)
    
    # Assigning a type to the variable 'a10' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'a10', dict_call_result_705690)
    
    # Assigning a Call to a Name (line 48):
    
    # Assigning a Call to a Name (line 48):
    
    # Call to dict(...): (line 48)
    # Processing the call keyword arguments (line 48)
    float_705692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 20), 'float')
    keyword_705693 = float_705692
    int_705694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 32), 'int')
    keyword_705695 = int_705694
    kwargs_705696 = {'rtol': keyword_705695, 'atol': keyword_705693}
    # Getting the type of 'dict' (line 48)
    dict_705691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 10), 'dict', False)
    # Calling dict(args, kwargs) (line 48)
    dict_call_result_705697 = invoke(stypy.reporting.localization.Localization(__file__, 48, 10), dict_705691, *[], **kwargs_705696)
    
    # Assigning a type to the variable 'a12' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'a12', dict_call_result_705697)
    
    # Assigning a List to a Name (line 49):
    
    # Assigning a List to a Name (line 49):
    
    # Obtaining an instance of the builtin type 'list' (line 49)
    list_705698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 49)
    # Adding element type (line 49)
    
    # Obtaining an instance of the builtin type 'list' (line 51)
    list_705699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 51)
    # Adding element type (line 51)
    float_705700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 8), list_705699, float_705700)
    # Adding element type (line 51)
    float_705701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 15), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 8), list_705699, float_705701)
    # Adding element type (line 51)
    float_705702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 36), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 8), list_705699, float_705702)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 11), list_705698, list_705699)
    # Adding element type (line 49)
    
    # Obtaining an instance of the builtin type 'list' (line 52)
    list_705703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 52)
    # Adding element type (line 52)
    float_705704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 8), list_705703, float_705704)
    # Adding element type (line 52)
    float_705705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 8), list_705703, float_705705)
    # Adding element type (line 52)
    float_705706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 8), list_705703, float_705706)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 11), list_705698, list_705703)
    # Adding element type (line 49)
    
    # Obtaining an instance of the builtin type 'list' (line 53)
    list_705707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 53)
    # Adding element type (line 53)
    float_705708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 8), list_705707, float_705708)
    # Adding element type (line 53)
    float_705709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 16), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 8), list_705707, float_705709)
    # Adding element type (line 53)
    float_705710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 37), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 8), list_705707, float_705710)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 11), list_705698, list_705707)
    # Adding element type (line 49)
    
    # Obtaining an instance of the builtin type 'list' (line 54)
    list_705711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 54)
    # Adding element type (line 54)
    float_705712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 8), list_705711, float_705712)
    # Adding element type (line 54)
    float_705713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 17), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 8), list_705711, float_705713)
    # Adding element type (line 54)
    float_705714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 8), list_705711, float_705714)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 11), list_705698, list_705711)
    # Adding element type (line 49)
    
    # Obtaining an instance of the builtin type 'list' (line 55)
    list_705715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 55)
    # Adding element type (line 55)
    float_705716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 8), list_705715, float_705716)
    # Adding element type (line 55)
    float_705717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 16), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 8), list_705715, float_705717)
    # Adding element type (line 55)
    float_705718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 37), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 8), list_705715, float_705718)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 11), list_705698, list_705715)
    # Adding element type (line 49)
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_705719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    float_705720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 8), list_705719, float_705720)
    # Adding element type (line 56)
    float_705721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 18), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 8), list_705719, float_705721)
    # Adding element type (line 56)
    float_705722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 8), list_705719, float_705722)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 11), list_705698, list_705719)
    # Adding element type (line 49)
    
    # Obtaining an instance of the builtin type 'list' (line 57)
    list_705723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 57)
    # Adding element type (line 57)
    float_705724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 8), list_705723, float_705724)
    # Adding element type (line 57)
    float_705725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 15), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 8), list_705723, float_705725)
    # Adding element type (line 57)
    float_705726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 36), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 8), list_705723, float_705726)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 11), list_705698, list_705723)
    # Adding element type (line 49)
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_705727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    float_705728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 8), list_705727, float_705728)
    # Adding element type (line 58)
    float_705729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 17), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 8), list_705727, float_705729)
    # Adding element type (line 58)
    float_705730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 8), list_705727, float_705730)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 11), list_705698, list_705727)
    # Adding element type (line 49)
    
    # Obtaining an instance of the builtin type 'list' (line 59)
    list_705731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 59)
    # Adding element type (line 59)
    float_705732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 8), list_705731, float_705732)
    # Adding element type (line 59)
    float_705733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 14), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 8), list_705731, float_705733)
    # Adding element type (line 59)
    float_705734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 36), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 8), list_705731, float_705734)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 11), list_705698, list_705731)
    # Adding element type (line 49)
    
    # Obtaining an instance of the builtin type 'list' (line 60)
    list_705735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 60)
    # Adding element type (line 60)
    float_705736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 8), list_705735, float_705736)
    # Adding element type (line 60)
    float_705737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 15), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 8), list_705735, float_705737)
    # Adding element type (line 60)
    float_705738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 8), list_705735, float_705738)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 11), list_705698, list_705735)
    # Adding element type (line 49)
    
    # Obtaining an instance of the builtin type 'list' (line 61)
    list_705739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 61)
    # Adding element type (line 61)
    float_705740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 9), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 8), list_705739, float_705740)
    # Adding element type (line 61)
    float_705741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 15), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 8), list_705739, float_705741)
    # Adding element type (line 61)
    float_705742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 8), list_705739, float_705742)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 11), list_705698, list_705739)
    
    # Assigning a type to the variable 'data' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'data', list_705698)
    
    # Getting the type of 'data' (line 64)
    data_705743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 44), 'data')
    # Testing the type of a for loop iterable (line 64)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 64, 4), data_705743)
    # Getting the type of the for loop variable (line 64)
    for_loop_var_705744 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 64, 4), data_705743)
    # Assigning a type to the variable 'lam' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'lam', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 4), for_loop_var_705744))
    # Assigning a type to the variable 'var_expected' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'var_expected', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 4), for_loop_var_705744))
    # Assigning a type to the variable 'kurt_expected' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'kurt_expected', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 4), for_loop_var_705744))
    # SSA begins for a for statement (line 64)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 65):
    
    # Assigning a Call to a Name (line 65):
    
    # Call to tukeylambda_variance(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'lam' (line 65)
    lam_705746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 35), 'lam', False)
    # Processing the call keyword arguments (line 65)
    kwargs_705747 = {}
    # Getting the type of 'tukeylambda_variance' (line 65)
    tukeylambda_variance_705745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 14), 'tukeylambda_variance', False)
    # Calling tukeylambda_variance(args, kwargs) (line 65)
    tukeylambda_variance_call_result_705748 = invoke(stypy.reporting.localization.Localization(__file__, 65, 14), tukeylambda_variance_705745, *[lam_705746], **kwargs_705747)
    
    # Assigning a type to the variable 'var' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'var', tukeylambda_variance_call_result_705748)
    
    # Call to assert_allclose(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'var' (line 66)
    var_705750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'var', False)
    # Getting the type of 'var_expected' (line 66)
    var_expected_705751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 29), 'var_expected', False)
    # Processing the call keyword arguments (line 66)
    # Getting the type of 'a12' (line 66)
    a12_705752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 45), 'a12', False)
    kwargs_705753 = {'a12_705752': a12_705752}
    # Getting the type of 'assert_allclose' (line 66)
    assert_allclose_705749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 66)
    assert_allclose_call_result_705754 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), assert_allclose_705749, *[var_705750, var_expected_705751], **kwargs_705753)
    
    
    # Assigning a Call to a Name (line 67):
    
    # Assigning a Call to a Name (line 67):
    
    # Call to tukeylambda_kurtosis(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'lam' (line 67)
    lam_705756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 36), 'lam', False)
    # Processing the call keyword arguments (line 67)
    kwargs_705757 = {}
    # Getting the type of 'tukeylambda_kurtosis' (line 67)
    tukeylambda_kurtosis_705755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'tukeylambda_kurtosis', False)
    # Calling tukeylambda_kurtosis(args, kwargs) (line 67)
    tukeylambda_kurtosis_call_result_705758 = invoke(stypy.reporting.localization.Localization(__file__, 67, 15), tukeylambda_kurtosis_705755, *[lam_705756], **kwargs_705757)
    
    # Assigning a type to the variable 'kurt' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'kurt', tukeylambda_kurtosis_call_result_705758)
    
    # Call to assert_allclose(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'kurt' (line 68)
    kurt_705760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 24), 'kurt', False)
    # Getting the type of 'kurt_expected' (line 68)
    kurt_expected_705761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 30), 'kurt_expected', False)
    # Processing the call keyword arguments (line 68)
    # Getting the type of 'a10' (line 68)
    a10_705762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 47), 'a10', False)
    kwargs_705763 = {'a10_705762': a10_705762}
    # Getting the type of 'assert_allclose' (line 68)
    assert_allclose_705759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 68)
    assert_allclose_call_result_705764 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), assert_allclose_705759, *[kurt_705760, kurt_expected_705761], **kwargs_705763)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 72):
    
    # Assigning a Subscript to a Name (line 72):
    
    # Obtaining the type of the subscript
    int_705765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'int')
    
    # Call to zip(...): (line 72)
    # Getting the type of 'data' (line 72)
    data_705767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 44), 'data', False)
    # Processing the call keyword arguments (line 72)
    kwargs_705768 = {}
    # Getting the type of 'zip' (line 72)
    zip_705766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 39), 'zip', False)
    # Calling zip(args, kwargs) (line 72)
    zip_call_result_705769 = invoke(stypy.reporting.localization.Localization(__file__, 72, 39), zip_705766, *[data_705767], **kwargs_705768)
    
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___705770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), zip_call_result_705769, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_705771 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), getitem___705770, int_705765)
    
    # Assigning a type to the variable 'tuple_var_assignment_705553' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_705553', subscript_call_result_705771)
    
    # Assigning a Subscript to a Name (line 72):
    
    # Obtaining the type of the subscript
    int_705772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'int')
    
    # Call to zip(...): (line 72)
    # Getting the type of 'data' (line 72)
    data_705774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 44), 'data', False)
    # Processing the call keyword arguments (line 72)
    kwargs_705775 = {}
    # Getting the type of 'zip' (line 72)
    zip_705773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 39), 'zip', False)
    # Calling zip(args, kwargs) (line 72)
    zip_call_result_705776 = invoke(stypy.reporting.localization.Localization(__file__, 72, 39), zip_705773, *[data_705774], **kwargs_705775)
    
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___705777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), zip_call_result_705776, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_705778 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), getitem___705777, int_705772)
    
    # Assigning a type to the variable 'tuple_var_assignment_705554' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_705554', subscript_call_result_705778)
    
    # Assigning a Subscript to a Name (line 72):
    
    # Obtaining the type of the subscript
    int_705779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'int')
    
    # Call to zip(...): (line 72)
    # Getting the type of 'data' (line 72)
    data_705781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 44), 'data', False)
    # Processing the call keyword arguments (line 72)
    kwargs_705782 = {}
    # Getting the type of 'zip' (line 72)
    zip_705780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 39), 'zip', False)
    # Calling zip(args, kwargs) (line 72)
    zip_call_result_705783 = invoke(stypy.reporting.localization.Localization(__file__, 72, 39), zip_705780, *[data_705781], **kwargs_705782)
    
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___705784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), zip_call_result_705783, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_705785 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), getitem___705784, int_705779)
    
    # Assigning a type to the variable 'tuple_var_assignment_705555' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_705555', subscript_call_result_705785)
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'tuple_var_assignment_705553' (line 72)
    tuple_var_assignment_705553_705786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_705553')
    # Assigning a type to the variable 'lam' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'lam', tuple_var_assignment_705553_705786)
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'tuple_var_assignment_705554' (line 72)
    tuple_var_assignment_705554_705787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_705554')
    # Assigning a type to the variable 'var_expected' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 9), 'var_expected', tuple_var_assignment_705554_705787)
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'tuple_var_assignment_705555' (line 72)
    tuple_var_assignment_705555_705788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'tuple_var_assignment_705555')
    # Assigning a type to the variable 'kurt_expected' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'kurt_expected', tuple_var_assignment_705555_705788)
    
    # Assigning a Call to a Name (line 73):
    
    # Assigning a Call to a Name (line 73):
    
    # Call to tukeylambda_variance(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'lam' (line 73)
    lam_705790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 31), 'lam', False)
    # Processing the call keyword arguments (line 73)
    kwargs_705791 = {}
    # Getting the type of 'tukeylambda_variance' (line 73)
    tukeylambda_variance_705789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 10), 'tukeylambda_variance', False)
    # Calling tukeylambda_variance(args, kwargs) (line 73)
    tukeylambda_variance_call_result_705792 = invoke(stypy.reporting.localization.Localization(__file__, 73, 10), tukeylambda_variance_705789, *[lam_705790], **kwargs_705791)
    
    # Assigning a type to the variable 'var' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'var', tukeylambda_variance_call_result_705792)
    
    # Call to assert_allclose(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'var' (line 74)
    var_705794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'var', False)
    # Getting the type of 'var_expected' (line 74)
    var_expected_705795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'var_expected', False)
    # Processing the call keyword arguments (line 74)
    # Getting the type of 'a12' (line 74)
    a12_705796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 41), 'a12', False)
    kwargs_705797 = {'a12_705796': a12_705796}
    # Getting the type of 'assert_allclose' (line 74)
    assert_allclose_705793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 74)
    assert_allclose_call_result_705798 = invoke(stypy.reporting.localization.Localization(__file__, 74, 4), assert_allclose_705793, *[var_705794, var_expected_705795], **kwargs_705797)
    
    
    # Assigning a Call to a Name (line 75):
    
    # Assigning a Call to a Name (line 75):
    
    # Call to tukeylambda_kurtosis(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'lam' (line 75)
    lam_705800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'lam', False)
    # Processing the call keyword arguments (line 75)
    kwargs_705801 = {}
    # Getting the type of 'tukeylambda_kurtosis' (line 75)
    tukeylambda_kurtosis_705799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'tukeylambda_kurtosis', False)
    # Calling tukeylambda_kurtosis(args, kwargs) (line 75)
    tukeylambda_kurtosis_call_result_705802 = invoke(stypy.reporting.localization.Localization(__file__, 75, 11), tukeylambda_kurtosis_705799, *[lam_705800], **kwargs_705801)
    
    # Assigning a type to the variable 'kurt' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'kurt', tukeylambda_kurtosis_call_result_705802)
    
    # Call to assert_allclose(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'kurt' (line 76)
    kurt_705804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'kurt', False)
    # Getting the type of 'kurt_expected' (line 76)
    kurt_expected_705805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 26), 'kurt_expected', False)
    # Processing the call keyword arguments (line 76)
    # Getting the type of 'a10' (line 76)
    a10_705806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 43), 'a10', False)
    kwargs_705807 = {'a10_705806': a10_705806}
    # Getting the type of 'assert_allclose' (line 76)
    assert_allclose_705803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 76)
    assert_allclose_call_result_705808 = invoke(stypy.reporting.localization.Localization(__file__, 76, 4), assert_allclose_705803, *[kurt_705804, kurt_expected_705805], **kwargs_705807)
    
    
    # ################# End of 'test_tukeylambda_stats_mpmath(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_tukeylambda_stats_mpmath' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_705809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_705809)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_tukeylambda_stats_mpmath'
    return stypy_return_type_705809

# Assigning a type to the variable 'test_tukeylambda_stats_mpmath' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'test_tukeylambda_stats_mpmath', test_tukeylambda_stats_mpmath)

@norecursion
def test_tukeylambda_stats_invalid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_tukeylambda_stats_invalid'
    module_type_store = module_type_store.open_function_context('test_tukeylambda_stats_invalid', 79, 0, False)
    
    # Passed parameters checking function
    test_tukeylambda_stats_invalid.stypy_localization = localization
    test_tukeylambda_stats_invalid.stypy_type_of_self = None
    test_tukeylambda_stats_invalid.stypy_type_store = module_type_store
    test_tukeylambda_stats_invalid.stypy_function_name = 'test_tukeylambda_stats_invalid'
    test_tukeylambda_stats_invalid.stypy_param_names_list = []
    test_tukeylambda_stats_invalid.stypy_varargs_param_name = None
    test_tukeylambda_stats_invalid.stypy_kwargs_param_name = None
    test_tukeylambda_stats_invalid.stypy_call_defaults = defaults
    test_tukeylambda_stats_invalid.stypy_call_varargs = varargs
    test_tukeylambda_stats_invalid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_tukeylambda_stats_invalid', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_tukeylambda_stats_invalid', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_tukeylambda_stats_invalid(...)' code ##################

    str_705810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 4), 'str', 'Test values of lambda outside the domains of the functions.')
    
    # Assigning a List to a Name (line 81):
    
    # Assigning a List to a Name (line 81):
    
    # Obtaining an instance of the builtin type 'list' (line 81)
    list_705811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 81)
    # Adding element type (line 81)
    float_705812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 11), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 10), list_705811, float_705812)
    # Adding element type (line 81)
    float_705813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 17), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 10), list_705811, float_705813)
    
    # Assigning a type to the variable 'lam' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'lam', list_705811)
    
    # Assigning a Call to a Name (line 82):
    
    # Assigning a Call to a Name (line 82):
    
    # Call to tukeylambda_variance(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'lam' (line 82)
    lam_705815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 31), 'lam', False)
    # Processing the call keyword arguments (line 82)
    kwargs_705816 = {}
    # Getting the type of 'tukeylambda_variance' (line 82)
    tukeylambda_variance_705814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 10), 'tukeylambda_variance', False)
    # Calling tukeylambda_variance(args, kwargs) (line 82)
    tukeylambda_variance_call_result_705817 = invoke(stypy.reporting.localization.Localization(__file__, 82, 10), tukeylambda_variance_705814, *[lam_705815], **kwargs_705816)
    
    # Assigning a type to the variable 'var' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'var', tukeylambda_variance_call_result_705817)
    
    # Call to assert_equal(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'var' (line 83)
    var_705819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 17), 'var', False)
    
    # Call to array(...): (line 83)
    # Processing the call arguments (line 83)
    
    # Obtaining an instance of the builtin type 'list' (line 83)
    list_705822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 83)
    # Adding element type (line 83)
    # Getting the type of 'np' (line 83)
    np_705823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 32), 'np', False)
    # Obtaining the member 'nan' of a type (line 83)
    nan_705824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 32), np_705823, 'nan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 31), list_705822, nan_705824)
    # Adding element type (line 83)
    # Getting the type of 'np' (line 83)
    np_705825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 40), 'np', False)
    # Obtaining the member 'inf' of a type (line 83)
    inf_705826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 40), np_705825, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 31), list_705822, inf_705826)
    
    # Processing the call keyword arguments (line 83)
    kwargs_705827 = {}
    # Getting the type of 'np' (line 83)
    np_705820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'np', False)
    # Obtaining the member 'array' of a type (line 83)
    array_705821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 22), np_705820, 'array')
    # Calling array(args, kwargs) (line 83)
    array_call_result_705828 = invoke(stypy.reporting.localization.Localization(__file__, 83, 22), array_705821, *[list_705822], **kwargs_705827)
    
    # Processing the call keyword arguments (line 83)
    kwargs_705829 = {}
    # Getting the type of 'assert_equal' (line 83)
    assert_equal_705818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 83)
    assert_equal_call_result_705830 = invoke(stypy.reporting.localization.Localization(__file__, 83, 4), assert_equal_705818, *[var_705819, array_call_result_705828], **kwargs_705829)
    
    
    # Assigning a List to a Name (line 85):
    
    # Assigning a List to a Name (line 85):
    
    # Obtaining an instance of the builtin type 'list' (line 85)
    list_705831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 85)
    # Adding element type (line 85)
    float_705832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 11), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 10), list_705831, float_705832)
    # Adding element type (line 85)
    float_705833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 17), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 10), list_705831, float_705833)
    
    # Assigning a type to the variable 'lam' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'lam', list_705831)
    
    # Assigning a Call to a Name (line 86):
    
    # Assigning a Call to a Name (line 86):
    
    # Call to tukeylambda_kurtosis(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'lam' (line 86)
    lam_705835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 32), 'lam', False)
    # Processing the call keyword arguments (line 86)
    kwargs_705836 = {}
    # Getting the type of 'tukeylambda_kurtosis' (line 86)
    tukeylambda_kurtosis_705834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'tukeylambda_kurtosis', False)
    # Calling tukeylambda_kurtosis(args, kwargs) (line 86)
    tukeylambda_kurtosis_call_result_705837 = invoke(stypy.reporting.localization.Localization(__file__, 86, 11), tukeylambda_kurtosis_705834, *[lam_705835], **kwargs_705836)
    
    # Assigning a type to the variable 'kurt' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'kurt', tukeylambda_kurtosis_call_result_705837)
    
    # Call to assert_equal(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'kurt' (line 87)
    kurt_705839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 17), 'kurt', False)
    
    # Call to array(...): (line 87)
    # Processing the call arguments (line 87)
    
    # Obtaining an instance of the builtin type 'list' (line 87)
    list_705842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 87)
    # Adding element type (line 87)
    # Getting the type of 'np' (line 87)
    np_705843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 33), 'np', False)
    # Obtaining the member 'nan' of a type (line 87)
    nan_705844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 33), np_705843, 'nan')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 32), list_705842, nan_705844)
    # Adding element type (line 87)
    # Getting the type of 'np' (line 87)
    np_705845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 41), 'np', False)
    # Obtaining the member 'inf' of a type (line 87)
    inf_705846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 41), np_705845, 'inf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 32), list_705842, inf_705846)
    
    # Processing the call keyword arguments (line 87)
    kwargs_705847 = {}
    # Getting the type of 'np' (line 87)
    np_705840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'np', False)
    # Obtaining the member 'array' of a type (line 87)
    array_705841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 23), np_705840, 'array')
    # Calling array(args, kwargs) (line 87)
    array_call_result_705848 = invoke(stypy.reporting.localization.Localization(__file__, 87, 23), array_705841, *[list_705842], **kwargs_705847)
    
    # Processing the call keyword arguments (line 87)
    kwargs_705849 = {}
    # Getting the type of 'assert_equal' (line 87)
    assert_equal_705838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 87)
    assert_equal_call_result_705850 = invoke(stypy.reporting.localization.Localization(__file__, 87, 4), assert_equal_705838, *[kurt_705839, array_call_result_705848], **kwargs_705849)
    
    
    # ################# End of 'test_tukeylambda_stats_invalid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_tukeylambda_stats_invalid' in the type store
    # Getting the type of 'stypy_return_type' (line 79)
    stypy_return_type_705851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_705851)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_tukeylambda_stats_invalid'
    return stypy_return_type_705851

# Assigning a type to the variable 'test_tukeylambda_stats_invalid' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'test_tukeylambda_stats_invalid', test_tukeylambda_stats_invalid)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
