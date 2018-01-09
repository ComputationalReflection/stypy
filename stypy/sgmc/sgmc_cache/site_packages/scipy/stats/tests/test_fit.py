
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import os
4: 
5: import numpy as np
6: from numpy.testing import assert_allclose
7: from scipy._lib._numpy_compat import suppress_warnings
8: import pytest
9: from scipy import stats
10: 
11: from .test_continuous_basic import distcont
12: 
13: # this is not a proper statistical test for convergence, but only
14: # verifies that the estimate and true values don't differ by too much
15: 
16: fit_sizes = [1000, 5000]  # sample sizes to try
17: 
18: thresh_percent = 0.25  # percent of true parameters for fail cut-off
19: thresh_min = 0.75  # minimum difference estimate - true to fail test
20: 
21: failing_fits = [
22:         'burr',
23:         'chi2',
24:         'gausshyper',
25:         'genexpon',
26:         'gengamma',
27:         'kappa4',
28:         'ksone',
29:         'mielke',
30:         'ncf',
31:         'ncx2',
32:         'pearson3',
33:         'powerlognorm',
34:         'truncexpon',
35:         'tukeylambda',
36:         'vonmises',
37:         'wrapcauchy',
38:         'levy_stable',
39:         'trapz'
40: ]
41: 
42: # Don't run the fit test on these:
43: skip_fit = [
44:     'erlang',  # Subclass of gamma, generates a warning.
45: ]
46: 
47: 
48: def cases_test_cont_fit():
49:     # this tests the closeness of the estimated parameters to the true
50:     # parameters with fit method of continuous distributions
51:     # Note: is slow, some distributions don't converge with sample size <= 10000
52:     for distname, arg in distcont:
53:         if distname not in skip_fit:
54:             yield distname, arg
55: 
56: 
57: @pytest.mark.slow
58: @pytest.mark.parametrize('distname,arg', cases_test_cont_fit())
59: def test_cont_fit(distname, arg):
60:     if distname in failing_fits:
61:         # Skip failing fits unless overridden
62:         xfail = True
63:         try:
64:             xfail = not int(os.environ['SCIPY_XFAIL'])
65:         except:
66:             pass
67:         if xfail:
68:             msg = "Fitting %s doesn't work reliably yet" % distname
69:             msg += " [Set environment variable SCIPY_XFAIL=1 to run this test nevertheless.]"
70:             pytest.xfail(msg)
71: 
72:     distfn = getattr(stats, distname)
73: 
74:     truearg = np.hstack([arg, [0.0, 1.0]])
75:     diffthreshold = np.max(np.vstack([truearg*thresh_percent,
76:                                       np.ones(distfn.numargs+2)*thresh_min]),
77:                            0)
78: 
79:     for fit_size in fit_sizes:
80:         # Note that if a fit succeeds, the other fit_sizes are skipped
81:         np.random.seed(1234)
82: 
83:         with np.errstate(all='ignore'), suppress_warnings() as sup:
84:             sup.filter(category=DeprecationWarning, message=".*frechet_")
85:             rvs = distfn.rvs(size=fit_size, *arg)
86:             est = distfn.fit(rvs)  # start with default values
87: 
88:         diff = est - truearg
89: 
90:         # threshold for location
91:         diffthreshold[-2] = np.max([np.abs(rvs.mean())*thresh_percent,thresh_min])
92: 
93:         if np.any(np.isnan(est)):
94:             raise AssertionError('nan returned in fit')
95:         else:
96:             if np.all(np.abs(diff) <= diffthreshold):
97:                 break
98:     else:
99:         txt = 'parameter: %s\n' % str(truearg)
100:         txt += 'estimated: %s\n' % str(est)
101:         txt += 'diff     : %s\n' % str(diff)
102:         raise AssertionError('fit not very good in %s\n' % distfn.name + txt)
103: 
104: 
105: def _check_loc_scale_mle_fit(name, data, desired, atol=None):
106:     d = getattr(stats, name)
107:     actual = d.fit(data)[-2:]
108:     assert_allclose(actual, desired, atol=atol,
109:                     err_msg='poor mle fit of (loc, scale) in %s' % name)
110: 
111: 
112: def test_non_default_loc_scale_mle_fit():
113:     data = np.array([1.01, 1.78, 1.78, 1.78, 1.88, 1.88, 1.88, 2.00])
114:     _check_loc_scale_mle_fit('uniform', data, [1.01, 0.99], 1e-3)
115:     _check_loc_scale_mle_fit('expon', data, [1.01, 0.73875], 1e-3)
116: 
117: 
118: def test_expon_fit():
119:     '''gh-6167'''
120:     data = [0, 0, 0, 0, 2, 2, 2, 2]
121:     phat = stats.expon.fit(data, floc=0)
122:     assert_allclose(phat, [0, 1.0], atol=1e-3)
123: 
124: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_651957 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_651957) is not StypyTypeError):

    if (import_651957 != 'pyd_module'):
        __import__(import_651957)
        sys_modules_651958 = sys.modules[import_651957]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_651958.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_651957)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.testing import assert_allclose' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_651959 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_651959) is not StypyTypeError):

    if (import_651959 != 'pyd_module'):
        __import__(import_651959)
        sys_modules_651960 = sys.modules[import_651959]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', sys_modules_651960.module_type_store, module_type_store, ['assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_651960, sys_modules_651960.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', None, module_type_store, ['assert_allclose'], [assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_651959)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_651961 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._numpy_compat')

if (type(import_651961) is not StypyTypeError):

    if (import_651961 != 'pyd_module'):
        __import__(import_651961)
        sys_modules_651962 = sys.modules[import_651961]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._numpy_compat', sys_modules_651962.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_651962, sys_modules_651962.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._numpy_compat', import_651961)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import pytest' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_651963 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_651963) is not StypyTypeError):

    if (import_651963 != 'pyd_module'):
        __import__(import_651963)
        sys_modules_651964 = sys.modules[import_651963]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_651964.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_651963)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy import stats' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_651965 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy')

if (type(import_651965) is not StypyTypeError):

    if (import_651965 != 'pyd_module'):
        __import__(import_651965)
        sys_modules_651966 = sys.modules[import_651965]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy', sys_modules_651966.module_type_store, module_type_store, ['stats'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_651966, sys_modules_651966.module_type_store, module_type_store)
    else:
        from scipy import stats

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy', None, module_type_store, ['stats'], [stats])

else:
    # Assigning a type to the variable 'scipy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy', import_651965)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.stats.tests.test_continuous_basic import distcont' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_651967 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.stats.tests.test_continuous_basic')

if (type(import_651967) is not StypyTypeError):

    if (import_651967 != 'pyd_module'):
        __import__(import_651967)
        sys_modules_651968 = sys.modules[import_651967]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.stats.tests.test_continuous_basic', sys_modules_651968.module_type_store, module_type_store, ['distcont'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_651968, sys_modules_651968.module_type_store, module_type_store)
    else:
        from scipy.stats.tests.test_continuous_basic import distcont

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.stats.tests.test_continuous_basic', None, module_type_store, ['distcont'], [distcont])

else:
    # Assigning a type to the variable 'scipy.stats.tests.test_continuous_basic' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.stats.tests.test_continuous_basic', import_651967)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')


# Assigning a List to a Name (line 16):

# Obtaining an instance of the builtin type 'list' (line 16)
list_651969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
int_651970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 12), list_651969, int_651970)
# Adding element type (line 16)
int_651971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 12), list_651969, int_651971)

# Assigning a type to the variable 'fit_sizes' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'fit_sizes', list_651969)

# Assigning a Num to a Name (line 18):
float_651972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 17), 'float')
# Assigning a type to the variable 'thresh_percent' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'thresh_percent', float_651972)

# Assigning a Num to a Name (line 19):
float_651973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 13), 'float')
# Assigning a type to the variable 'thresh_min' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'thresh_min', float_651973)

# Assigning a List to a Name (line 21):

# Obtaining an instance of the builtin type 'list' (line 21)
list_651974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_651975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 8), 'str', 'burr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651975)
# Adding element type (line 21)
str_651976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 8), 'str', 'chi2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651976)
# Adding element type (line 21)
str_651977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 8), 'str', 'gausshyper')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651977)
# Adding element type (line 21)
str_651978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 8), 'str', 'genexpon')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651978)
# Adding element type (line 21)
str_651979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 8), 'str', 'gengamma')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651979)
# Adding element type (line 21)
str_651980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 8), 'str', 'kappa4')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651980)
# Adding element type (line 21)
str_651981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'str', 'ksone')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651981)
# Adding element type (line 21)
str_651982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 8), 'str', 'mielke')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651982)
# Adding element type (line 21)
str_651983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 8), 'str', 'ncf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651983)
# Adding element type (line 21)
str_651984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 8), 'str', 'ncx2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651984)
# Adding element type (line 21)
str_651985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 8), 'str', 'pearson3')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651985)
# Adding element type (line 21)
str_651986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 8), 'str', 'powerlognorm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651986)
# Adding element type (line 21)
str_651987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 8), 'str', 'truncexpon')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651987)
# Adding element type (line 21)
str_651988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 8), 'str', 'tukeylambda')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651988)
# Adding element type (line 21)
str_651989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 8), 'str', 'vonmises')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651989)
# Adding element type (line 21)
str_651990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 8), 'str', 'wrapcauchy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651990)
# Adding element type (line 21)
str_651991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 8), 'str', 'levy_stable')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651991)
# Adding element type (line 21)
str_651992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 8), 'str', 'trapz')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_651974, str_651992)

# Assigning a type to the variable 'failing_fits' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'failing_fits', list_651974)

# Assigning a List to a Name (line 43):

# Obtaining an instance of the builtin type 'list' (line 43)
list_651993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 43)
# Adding element type (line 43)
str_651994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 4), 'str', 'erlang')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 11), list_651993, str_651994)

# Assigning a type to the variable 'skip_fit' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'skip_fit', list_651993)

@norecursion
def cases_test_cont_fit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'cases_test_cont_fit'
    module_type_store = module_type_store.open_function_context('cases_test_cont_fit', 48, 0, False)
    
    # Passed parameters checking function
    cases_test_cont_fit.stypy_localization = localization
    cases_test_cont_fit.stypy_type_of_self = None
    cases_test_cont_fit.stypy_type_store = module_type_store
    cases_test_cont_fit.stypy_function_name = 'cases_test_cont_fit'
    cases_test_cont_fit.stypy_param_names_list = []
    cases_test_cont_fit.stypy_varargs_param_name = None
    cases_test_cont_fit.stypy_kwargs_param_name = None
    cases_test_cont_fit.stypy_call_defaults = defaults
    cases_test_cont_fit.stypy_call_varargs = varargs
    cases_test_cont_fit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cases_test_cont_fit', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cases_test_cont_fit', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cases_test_cont_fit(...)' code ##################

    
    # Getting the type of 'distcont' (line 52)
    distcont_651995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 25), 'distcont')
    # Testing the type of a for loop iterable (line 52)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 52, 4), distcont_651995)
    # Getting the type of the for loop variable (line 52)
    for_loop_var_651996 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 52, 4), distcont_651995)
    # Assigning a type to the variable 'distname' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'distname', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 4), for_loop_var_651996))
    # Assigning a type to the variable 'arg' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'arg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 4), for_loop_var_651996))
    # SSA begins for a for statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'distname' (line 53)
    distname_651997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'distname')
    # Getting the type of 'skip_fit' (line 53)
    skip_fit_651998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 27), 'skip_fit')
    # Applying the binary operator 'notin' (line 53)
    result_contains_651999 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 11), 'notin', distname_651997, skip_fit_651998)
    
    # Testing the type of an if condition (line 53)
    if_condition_652000 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 8), result_contains_651999)
    # Assigning a type to the variable 'if_condition_652000' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'if_condition_652000', if_condition_652000)
    # SSA begins for if statement (line 53)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Creating a generator
    
    # Obtaining an instance of the builtin type 'tuple' (line 54)
    tuple_652001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 18), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 54)
    # Adding element type (line 54)
    # Getting the type of 'distname' (line 54)
    distname_652002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 18), 'distname')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 18), tuple_652001, distname_652002)
    # Adding element type (line 54)
    # Getting the type of 'arg' (line 54)
    arg_652003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'arg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 18), tuple_652001, arg_652003)
    
    GeneratorType_652004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 12), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 12), GeneratorType_652004, tuple_652001)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'stypy_return_type', GeneratorType_652004)
    # SSA join for if statement (line 53)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'cases_test_cont_fit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cases_test_cont_fit' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_652005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_652005)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cases_test_cont_fit'
    return stypy_return_type_652005

# Assigning a type to the variable 'cases_test_cont_fit' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'cases_test_cont_fit', cases_test_cont_fit)

@norecursion
def test_cont_fit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_cont_fit'
    module_type_store = module_type_store.open_function_context('test_cont_fit', 57, 0, False)
    
    # Passed parameters checking function
    test_cont_fit.stypy_localization = localization
    test_cont_fit.stypy_type_of_self = None
    test_cont_fit.stypy_type_store = module_type_store
    test_cont_fit.stypy_function_name = 'test_cont_fit'
    test_cont_fit.stypy_param_names_list = ['distname', 'arg']
    test_cont_fit.stypy_varargs_param_name = None
    test_cont_fit.stypy_kwargs_param_name = None
    test_cont_fit.stypy_call_defaults = defaults
    test_cont_fit.stypy_call_varargs = varargs
    test_cont_fit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_cont_fit', ['distname', 'arg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_cont_fit', localization, ['distname', 'arg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_cont_fit(...)' code ##################

    
    
    # Getting the type of 'distname' (line 60)
    distname_652006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 7), 'distname')
    # Getting the type of 'failing_fits' (line 60)
    failing_fits_652007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 19), 'failing_fits')
    # Applying the binary operator 'in' (line 60)
    result_contains_652008 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 7), 'in', distname_652006, failing_fits_652007)
    
    # Testing the type of an if condition (line 60)
    if_condition_652009 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 4), result_contains_652008)
    # Assigning a type to the variable 'if_condition_652009' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'if_condition_652009', if_condition_652009)
    # SSA begins for if statement (line 60)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 62):
    # Getting the type of 'True' (line 62)
    True_652010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'True')
    # Assigning a type to the variable 'xfail' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'xfail', True_652010)
    
    
    # SSA begins for try-except statement (line 63)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a UnaryOp to a Name (line 64):
    
    
    # Call to int(...): (line 64)
    # Processing the call arguments (line 64)
    
    # Obtaining the type of the subscript
    str_652012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 39), 'str', 'SCIPY_XFAIL')
    # Getting the type of 'os' (line 64)
    os_652013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 28), 'os', False)
    # Obtaining the member 'environ' of a type (line 64)
    environ_652014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 28), os_652013, 'environ')
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___652015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 28), environ_652014, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_652016 = invoke(stypy.reporting.localization.Localization(__file__, 64, 28), getitem___652015, str_652012)
    
    # Processing the call keyword arguments (line 64)
    kwargs_652017 = {}
    # Getting the type of 'int' (line 64)
    int_652011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'int', False)
    # Calling int(args, kwargs) (line 64)
    int_call_result_652018 = invoke(stypy.reporting.localization.Localization(__file__, 64, 24), int_652011, *[subscript_call_result_652016], **kwargs_652017)
    
    # Applying the 'not' unary operator (line 64)
    result_not__652019 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 20), 'not', int_call_result_652018)
    
    # Assigning a type to the variable 'xfail' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'xfail', result_not__652019)
    # SSA branch for the except part of a try statement (line 63)
    # SSA branch for the except '<any exception>' branch of a try statement (line 63)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 63)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'xfail' (line 67)
    xfail_652020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'xfail')
    # Testing the type of an if condition (line 67)
    if_condition_652021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), xfail_652020)
    # Assigning a type to the variable 'if_condition_652021' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'if_condition_652021', if_condition_652021)
    # SSA begins for if statement (line 67)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 68):
    str_652022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 18), 'str', "Fitting %s doesn't work reliably yet")
    # Getting the type of 'distname' (line 68)
    distname_652023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 59), 'distname')
    # Applying the binary operator '%' (line 68)
    result_mod_652024 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 18), '%', str_652022, distname_652023)
    
    # Assigning a type to the variable 'msg' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'msg', result_mod_652024)
    
    # Getting the type of 'msg' (line 69)
    msg_652025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'msg')
    str_652026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 19), 'str', ' [Set environment variable SCIPY_XFAIL=1 to run this test nevertheless.]')
    # Applying the binary operator '+=' (line 69)
    result_iadd_652027 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 12), '+=', msg_652025, str_652026)
    # Assigning a type to the variable 'msg' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'msg', result_iadd_652027)
    
    
    # Call to xfail(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'msg' (line 70)
    msg_652030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'msg', False)
    # Processing the call keyword arguments (line 70)
    kwargs_652031 = {}
    # Getting the type of 'pytest' (line 70)
    pytest_652028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'pytest', False)
    # Obtaining the member 'xfail' of a type (line 70)
    xfail_652029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), pytest_652028, 'xfail')
    # Calling xfail(args, kwargs) (line 70)
    xfail_call_result_652032 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), xfail_652029, *[msg_652030], **kwargs_652031)
    
    # SSA join for if statement (line 67)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 60)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 72):
    
    # Call to getattr(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'stats' (line 72)
    stats_652034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 21), 'stats', False)
    # Getting the type of 'distname' (line 72)
    distname_652035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'distname', False)
    # Processing the call keyword arguments (line 72)
    kwargs_652036 = {}
    # Getting the type of 'getattr' (line 72)
    getattr_652033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'getattr', False)
    # Calling getattr(args, kwargs) (line 72)
    getattr_call_result_652037 = invoke(stypy.reporting.localization.Localization(__file__, 72, 13), getattr_652033, *[stats_652034, distname_652035], **kwargs_652036)
    
    # Assigning a type to the variable 'distfn' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'distfn', getattr_call_result_652037)
    
    # Assigning a Call to a Name (line 74):
    
    # Call to hstack(...): (line 74)
    # Processing the call arguments (line 74)
    
    # Obtaining an instance of the builtin type 'list' (line 74)
    list_652040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 74)
    # Adding element type (line 74)
    # Getting the type of 'arg' (line 74)
    arg_652041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'arg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 24), list_652040, arg_652041)
    # Adding element type (line 74)
    
    # Obtaining an instance of the builtin type 'list' (line 74)
    list_652042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 74)
    # Adding element type (line 74)
    float_652043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 30), list_652042, float_652043)
    # Adding element type (line 74)
    float_652044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 36), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 30), list_652042, float_652044)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 24), list_652040, list_652042)
    
    # Processing the call keyword arguments (line 74)
    kwargs_652045 = {}
    # Getting the type of 'np' (line 74)
    np_652038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 14), 'np', False)
    # Obtaining the member 'hstack' of a type (line 74)
    hstack_652039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 14), np_652038, 'hstack')
    # Calling hstack(args, kwargs) (line 74)
    hstack_call_result_652046 = invoke(stypy.reporting.localization.Localization(__file__, 74, 14), hstack_652039, *[list_652040], **kwargs_652045)
    
    # Assigning a type to the variable 'truearg' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'truearg', hstack_call_result_652046)
    
    # Assigning a Call to a Name (line 75):
    
    # Call to max(...): (line 75)
    # Processing the call arguments (line 75)
    
    # Call to vstack(...): (line 75)
    # Processing the call arguments (line 75)
    
    # Obtaining an instance of the builtin type 'list' (line 75)
    list_652051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 75)
    # Adding element type (line 75)
    # Getting the type of 'truearg' (line 75)
    truearg_652052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 38), 'truearg', False)
    # Getting the type of 'thresh_percent' (line 75)
    thresh_percent_652053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 46), 'thresh_percent', False)
    # Applying the binary operator '*' (line 75)
    result_mul_652054 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 38), '*', truearg_652052, thresh_percent_652053)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 37), list_652051, result_mul_652054)
    # Adding element type (line 75)
    
    # Call to ones(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'distfn' (line 76)
    distfn_652057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 46), 'distfn', False)
    # Obtaining the member 'numargs' of a type (line 76)
    numargs_652058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 46), distfn_652057, 'numargs')
    int_652059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 61), 'int')
    # Applying the binary operator '+' (line 76)
    result_add_652060 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 46), '+', numargs_652058, int_652059)
    
    # Processing the call keyword arguments (line 76)
    kwargs_652061 = {}
    # Getting the type of 'np' (line 76)
    np_652055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 38), 'np', False)
    # Obtaining the member 'ones' of a type (line 76)
    ones_652056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 38), np_652055, 'ones')
    # Calling ones(args, kwargs) (line 76)
    ones_call_result_652062 = invoke(stypy.reporting.localization.Localization(__file__, 76, 38), ones_652056, *[result_add_652060], **kwargs_652061)
    
    # Getting the type of 'thresh_min' (line 76)
    thresh_min_652063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 64), 'thresh_min', False)
    # Applying the binary operator '*' (line 76)
    result_mul_652064 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 38), '*', ones_call_result_652062, thresh_min_652063)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 37), list_652051, result_mul_652064)
    
    # Processing the call keyword arguments (line 75)
    kwargs_652065 = {}
    # Getting the type of 'np' (line 75)
    np_652049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 27), 'np', False)
    # Obtaining the member 'vstack' of a type (line 75)
    vstack_652050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 27), np_652049, 'vstack')
    # Calling vstack(args, kwargs) (line 75)
    vstack_call_result_652066 = invoke(stypy.reporting.localization.Localization(__file__, 75, 27), vstack_652050, *[list_652051], **kwargs_652065)
    
    int_652067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 27), 'int')
    # Processing the call keyword arguments (line 75)
    kwargs_652068 = {}
    # Getting the type of 'np' (line 75)
    np_652047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'np', False)
    # Obtaining the member 'max' of a type (line 75)
    max_652048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 20), np_652047, 'max')
    # Calling max(args, kwargs) (line 75)
    max_call_result_652069 = invoke(stypy.reporting.localization.Localization(__file__, 75, 20), max_652048, *[vstack_call_result_652066, int_652067], **kwargs_652068)
    
    # Assigning a type to the variable 'diffthreshold' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'diffthreshold', max_call_result_652069)
    
    # Getting the type of 'fit_sizes' (line 79)
    fit_sizes_652070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'fit_sizes')
    # Testing the type of a for loop iterable (line 79)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 79, 4), fit_sizes_652070)
    # Getting the type of the for loop variable (line 79)
    for_loop_var_652071 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 79, 4), fit_sizes_652070)
    # Assigning a type to the variable 'fit_size' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'fit_size', for_loop_var_652071)
    # SSA begins for a for statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to seed(...): (line 81)
    # Processing the call arguments (line 81)
    int_652075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 23), 'int')
    # Processing the call keyword arguments (line 81)
    kwargs_652076 = {}
    # Getting the type of 'np' (line 81)
    np_652072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'np', False)
    # Obtaining the member 'random' of a type (line 81)
    random_652073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), np_652072, 'random')
    # Obtaining the member 'seed' of a type (line 81)
    seed_652074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), random_652073, 'seed')
    # Calling seed(args, kwargs) (line 81)
    seed_call_result_652077 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), seed_652074, *[int_652075], **kwargs_652076)
    
    
    # Call to errstate(...): (line 83)
    # Processing the call keyword arguments (line 83)
    str_652080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 29), 'str', 'ignore')
    keyword_652081 = str_652080
    kwargs_652082 = {'all': keyword_652081}
    # Getting the type of 'np' (line 83)
    np_652078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'np', False)
    # Obtaining the member 'errstate' of a type (line 83)
    errstate_652079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 13), np_652078, 'errstate')
    # Calling errstate(args, kwargs) (line 83)
    errstate_call_result_652083 = invoke(stypy.reporting.localization.Localization(__file__, 83, 13), errstate_652079, *[], **kwargs_652082)
    
    with_652084 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 83, 13), errstate_call_result_652083, 'with parameter', '__enter__', '__exit__')

    if with_652084:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 83)
        enter___652085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 13), errstate_call_result_652083, '__enter__')
        with_enter_652086 = invoke(stypy.reporting.localization.Localization(__file__, 83, 13), enter___652085)
        
        # Call to suppress_warnings(...): (line 83)
        # Processing the call keyword arguments (line 83)
        kwargs_652088 = {}
        # Getting the type of 'suppress_warnings' (line 83)
        suppress_warnings_652087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 40), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 83)
        suppress_warnings_call_result_652089 = invoke(stypy.reporting.localization.Localization(__file__, 83, 40), suppress_warnings_652087, *[], **kwargs_652088)
        
        with_652090 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 83, 40), suppress_warnings_call_result_652089, 'with parameter', '__enter__', '__exit__')

        if with_652090:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 83)
            enter___652091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 40), suppress_warnings_call_result_652089, '__enter__')
            with_enter_652092 = invoke(stypy.reporting.localization.Localization(__file__, 83, 40), enter___652091)
            # Assigning a type to the variable 'sup' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 40), 'sup', with_enter_652092)
            
            # Call to filter(...): (line 84)
            # Processing the call keyword arguments (line 84)
            # Getting the type of 'DeprecationWarning' (line 84)
            DeprecationWarning_652095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 32), 'DeprecationWarning', False)
            keyword_652096 = DeprecationWarning_652095
            str_652097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 60), 'str', '.*frechet_')
            keyword_652098 = str_652097
            kwargs_652099 = {'category': keyword_652096, 'message': keyword_652098}
            # Getting the type of 'sup' (line 84)
            sup_652093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 84)
            filter_652094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), sup_652093, 'filter')
            # Calling filter(args, kwargs) (line 84)
            filter_call_result_652100 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), filter_652094, *[], **kwargs_652099)
            
            
            # Assigning a Call to a Name (line 85):
            
            # Call to rvs(...): (line 85)
            # Getting the type of 'arg' (line 85)
            arg_652103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 45), 'arg', False)
            # Processing the call keyword arguments (line 85)
            # Getting the type of 'fit_size' (line 85)
            fit_size_652104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 34), 'fit_size', False)
            keyword_652105 = fit_size_652104
            kwargs_652106 = {'size': keyword_652105}
            # Getting the type of 'distfn' (line 85)
            distfn_652101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 18), 'distfn', False)
            # Obtaining the member 'rvs' of a type (line 85)
            rvs_652102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 18), distfn_652101, 'rvs')
            # Calling rvs(args, kwargs) (line 85)
            rvs_call_result_652107 = invoke(stypy.reporting.localization.Localization(__file__, 85, 18), rvs_652102, *[arg_652103], **kwargs_652106)
            
            # Assigning a type to the variable 'rvs' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'rvs', rvs_call_result_652107)
            
            # Assigning a Call to a Name (line 86):
            
            # Call to fit(...): (line 86)
            # Processing the call arguments (line 86)
            # Getting the type of 'rvs' (line 86)
            rvs_652110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 29), 'rvs', False)
            # Processing the call keyword arguments (line 86)
            kwargs_652111 = {}
            # Getting the type of 'distfn' (line 86)
            distfn_652108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 18), 'distfn', False)
            # Obtaining the member 'fit' of a type (line 86)
            fit_652109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 18), distfn_652108, 'fit')
            # Calling fit(args, kwargs) (line 86)
            fit_call_result_652112 = invoke(stypy.reporting.localization.Localization(__file__, 86, 18), fit_652109, *[rvs_652110], **kwargs_652111)
            
            # Assigning a type to the variable 'est' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'est', fit_call_result_652112)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 83)
            exit___652113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 40), suppress_warnings_call_result_652089, '__exit__')
            with_exit_652114 = invoke(stypy.reporting.localization.Localization(__file__, 83, 40), exit___652113, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 83)
        exit___652115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 13), errstate_call_result_652083, '__exit__')
        with_exit_652116 = invoke(stypy.reporting.localization.Localization(__file__, 83, 13), exit___652115, None, None, None)

    
    # Assigning a BinOp to a Name (line 88):
    # Getting the type of 'est' (line 88)
    est_652117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'est')
    # Getting the type of 'truearg' (line 88)
    truearg_652118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 21), 'truearg')
    # Applying the binary operator '-' (line 88)
    result_sub_652119 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 15), '-', est_652117, truearg_652118)
    
    # Assigning a type to the variable 'diff' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'diff', result_sub_652119)
    
    # Assigning a Call to a Subscript (line 91):
    
    # Call to max(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Obtaining an instance of the builtin type 'list' (line 91)
    list_652122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 91)
    # Adding element type (line 91)
    
    # Call to abs(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Call to mean(...): (line 91)
    # Processing the call keyword arguments (line 91)
    kwargs_652127 = {}
    # Getting the type of 'rvs' (line 91)
    rvs_652125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'rvs', False)
    # Obtaining the member 'mean' of a type (line 91)
    mean_652126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 43), rvs_652125, 'mean')
    # Calling mean(args, kwargs) (line 91)
    mean_call_result_652128 = invoke(stypy.reporting.localization.Localization(__file__, 91, 43), mean_652126, *[], **kwargs_652127)
    
    # Processing the call keyword arguments (line 91)
    kwargs_652129 = {}
    # Getting the type of 'np' (line 91)
    np_652123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 36), 'np', False)
    # Obtaining the member 'abs' of a type (line 91)
    abs_652124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 36), np_652123, 'abs')
    # Calling abs(args, kwargs) (line 91)
    abs_call_result_652130 = invoke(stypy.reporting.localization.Localization(__file__, 91, 36), abs_652124, *[mean_call_result_652128], **kwargs_652129)
    
    # Getting the type of 'thresh_percent' (line 91)
    thresh_percent_652131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 55), 'thresh_percent', False)
    # Applying the binary operator '*' (line 91)
    result_mul_652132 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 36), '*', abs_call_result_652130, thresh_percent_652131)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 35), list_652122, result_mul_652132)
    # Adding element type (line 91)
    # Getting the type of 'thresh_min' (line 91)
    thresh_min_652133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 70), 'thresh_min', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 35), list_652122, thresh_min_652133)
    
    # Processing the call keyword arguments (line 91)
    kwargs_652134 = {}
    # Getting the type of 'np' (line 91)
    np_652120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 28), 'np', False)
    # Obtaining the member 'max' of a type (line 91)
    max_652121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 28), np_652120, 'max')
    # Calling max(args, kwargs) (line 91)
    max_call_result_652135 = invoke(stypy.reporting.localization.Localization(__file__, 91, 28), max_652121, *[list_652122], **kwargs_652134)
    
    # Getting the type of 'diffthreshold' (line 91)
    diffthreshold_652136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'diffthreshold')
    int_652137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 22), 'int')
    # Storing an element on a container (line 91)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 8), diffthreshold_652136, (int_652137, max_call_result_652135))
    
    
    # Call to any(...): (line 93)
    # Processing the call arguments (line 93)
    
    # Call to isnan(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'est' (line 93)
    est_652142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 27), 'est', False)
    # Processing the call keyword arguments (line 93)
    kwargs_652143 = {}
    # Getting the type of 'np' (line 93)
    np_652140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 18), 'np', False)
    # Obtaining the member 'isnan' of a type (line 93)
    isnan_652141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 18), np_652140, 'isnan')
    # Calling isnan(args, kwargs) (line 93)
    isnan_call_result_652144 = invoke(stypy.reporting.localization.Localization(__file__, 93, 18), isnan_652141, *[est_652142], **kwargs_652143)
    
    # Processing the call keyword arguments (line 93)
    kwargs_652145 = {}
    # Getting the type of 'np' (line 93)
    np_652138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'np', False)
    # Obtaining the member 'any' of a type (line 93)
    any_652139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 11), np_652138, 'any')
    # Calling any(args, kwargs) (line 93)
    any_call_result_652146 = invoke(stypy.reporting.localization.Localization(__file__, 93, 11), any_652139, *[isnan_call_result_652144], **kwargs_652145)
    
    # Testing the type of an if condition (line 93)
    if_condition_652147 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 8), any_call_result_652146)
    # Assigning a type to the variable 'if_condition_652147' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'if_condition_652147', if_condition_652147)
    # SSA begins for if statement (line 93)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to AssertionError(...): (line 94)
    # Processing the call arguments (line 94)
    str_652149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 33), 'str', 'nan returned in fit')
    # Processing the call keyword arguments (line 94)
    kwargs_652150 = {}
    # Getting the type of 'AssertionError' (line 94)
    AssertionError_652148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 18), 'AssertionError', False)
    # Calling AssertionError(args, kwargs) (line 94)
    AssertionError_call_result_652151 = invoke(stypy.reporting.localization.Localization(__file__, 94, 18), AssertionError_652148, *[str_652149], **kwargs_652150)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 94, 12), AssertionError_call_result_652151, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 93)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to all(...): (line 96)
    # Processing the call arguments (line 96)
    
    
    # Call to abs(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'diff' (line 96)
    diff_652156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 29), 'diff', False)
    # Processing the call keyword arguments (line 96)
    kwargs_652157 = {}
    # Getting the type of 'np' (line 96)
    np_652154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 22), 'np', False)
    # Obtaining the member 'abs' of a type (line 96)
    abs_652155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 22), np_652154, 'abs')
    # Calling abs(args, kwargs) (line 96)
    abs_call_result_652158 = invoke(stypy.reporting.localization.Localization(__file__, 96, 22), abs_652155, *[diff_652156], **kwargs_652157)
    
    # Getting the type of 'diffthreshold' (line 96)
    diffthreshold_652159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 38), 'diffthreshold', False)
    # Applying the binary operator '<=' (line 96)
    result_le_652160 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 22), '<=', abs_call_result_652158, diffthreshold_652159)
    
    # Processing the call keyword arguments (line 96)
    kwargs_652161 = {}
    # Getting the type of 'np' (line 96)
    np_652152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'np', False)
    # Obtaining the member 'all' of a type (line 96)
    all_652153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 15), np_652152, 'all')
    # Calling all(args, kwargs) (line 96)
    all_call_result_652162 = invoke(stypy.reporting.localization.Localization(__file__, 96, 15), all_652153, *[result_le_652160], **kwargs_652161)
    
    # Testing the type of an if condition (line 96)
    if_condition_652163 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 12), all_call_result_652162)
    # Assigning a type to the variable 'if_condition_652163' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'if_condition_652163', if_condition_652163)
    # SSA begins for if statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 96)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 93)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of a for statement (line 79)
    module_type_store.open_ssa_branch('for loop else')
    
    # Assigning a BinOp to a Name (line 99):
    str_652164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 14), 'str', 'parameter: %s\n')
    
    # Call to str(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'truearg' (line 99)
    truearg_652166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 38), 'truearg', False)
    # Processing the call keyword arguments (line 99)
    kwargs_652167 = {}
    # Getting the type of 'str' (line 99)
    str_652165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'str', False)
    # Calling str(args, kwargs) (line 99)
    str_call_result_652168 = invoke(stypy.reporting.localization.Localization(__file__, 99, 34), str_652165, *[truearg_652166], **kwargs_652167)
    
    # Applying the binary operator '%' (line 99)
    result_mod_652169 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 14), '%', str_652164, str_call_result_652168)
    
    # Assigning a type to the variable 'txt' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'txt', result_mod_652169)
    
    # Getting the type of 'txt' (line 100)
    txt_652170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'txt')
    str_652171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 15), 'str', 'estimated: %s\n')
    
    # Call to str(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'est' (line 100)
    est_652173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 39), 'est', False)
    # Processing the call keyword arguments (line 100)
    kwargs_652174 = {}
    # Getting the type of 'str' (line 100)
    str_652172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 35), 'str', False)
    # Calling str(args, kwargs) (line 100)
    str_call_result_652175 = invoke(stypy.reporting.localization.Localization(__file__, 100, 35), str_652172, *[est_652173], **kwargs_652174)
    
    # Applying the binary operator '%' (line 100)
    result_mod_652176 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), '%', str_652171, str_call_result_652175)
    
    # Applying the binary operator '+=' (line 100)
    result_iadd_652177 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 8), '+=', txt_652170, result_mod_652176)
    # Assigning a type to the variable 'txt' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'txt', result_iadd_652177)
    
    
    # Getting the type of 'txt' (line 101)
    txt_652178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'txt')
    str_652179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 15), 'str', 'diff     : %s\n')
    
    # Call to str(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'diff' (line 101)
    diff_652181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 39), 'diff', False)
    # Processing the call keyword arguments (line 101)
    kwargs_652182 = {}
    # Getting the type of 'str' (line 101)
    str_652180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 35), 'str', False)
    # Calling str(args, kwargs) (line 101)
    str_call_result_652183 = invoke(stypy.reporting.localization.Localization(__file__, 101, 35), str_652180, *[diff_652181], **kwargs_652182)
    
    # Applying the binary operator '%' (line 101)
    result_mod_652184 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 15), '%', str_652179, str_call_result_652183)
    
    # Applying the binary operator '+=' (line 101)
    result_iadd_652185 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 8), '+=', txt_652178, result_mod_652184)
    # Assigning a type to the variable 'txt' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'txt', result_iadd_652185)
    
    
    # Call to AssertionError(...): (line 102)
    # Processing the call arguments (line 102)
    str_652187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 29), 'str', 'fit not very good in %s\n')
    # Getting the type of 'distfn' (line 102)
    distfn_652188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 59), 'distfn', False)
    # Obtaining the member 'name' of a type (line 102)
    name_652189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 59), distfn_652188, 'name')
    # Applying the binary operator '%' (line 102)
    result_mod_652190 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 29), '%', str_652187, name_652189)
    
    # Getting the type of 'txt' (line 102)
    txt_652191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 73), 'txt', False)
    # Applying the binary operator '+' (line 102)
    result_add_652192 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 29), '+', result_mod_652190, txt_652191)
    
    # Processing the call keyword arguments (line 102)
    kwargs_652193 = {}
    # Getting the type of 'AssertionError' (line 102)
    AssertionError_652186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 'AssertionError', False)
    # Calling AssertionError(args, kwargs) (line 102)
    AssertionError_call_result_652194 = invoke(stypy.reporting.localization.Localization(__file__, 102, 14), AssertionError_652186, *[result_add_652192], **kwargs_652193)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 102, 8), AssertionError_call_result_652194, 'raise parameter', BaseException)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_cont_fit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_cont_fit' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_652195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_652195)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_cont_fit'
    return stypy_return_type_652195

# Assigning a type to the variable 'test_cont_fit' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'test_cont_fit', test_cont_fit)

@norecursion
def _check_loc_scale_mle_fit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 105)
    None_652196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 55), 'None')
    defaults = [None_652196]
    # Create a new context for function '_check_loc_scale_mle_fit'
    module_type_store = module_type_store.open_function_context('_check_loc_scale_mle_fit', 105, 0, False)
    
    # Passed parameters checking function
    _check_loc_scale_mle_fit.stypy_localization = localization
    _check_loc_scale_mle_fit.stypy_type_of_self = None
    _check_loc_scale_mle_fit.stypy_type_store = module_type_store
    _check_loc_scale_mle_fit.stypy_function_name = '_check_loc_scale_mle_fit'
    _check_loc_scale_mle_fit.stypy_param_names_list = ['name', 'data', 'desired', 'atol']
    _check_loc_scale_mle_fit.stypy_varargs_param_name = None
    _check_loc_scale_mle_fit.stypy_kwargs_param_name = None
    _check_loc_scale_mle_fit.stypy_call_defaults = defaults
    _check_loc_scale_mle_fit.stypy_call_varargs = varargs
    _check_loc_scale_mle_fit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_loc_scale_mle_fit', ['name', 'data', 'desired', 'atol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_loc_scale_mle_fit', localization, ['name', 'data', 'desired', 'atol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_loc_scale_mle_fit(...)' code ##################

    
    # Assigning a Call to a Name (line 106):
    
    # Call to getattr(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'stats' (line 106)
    stats_652198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'stats', False)
    # Getting the type of 'name' (line 106)
    name_652199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 23), 'name', False)
    # Processing the call keyword arguments (line 106)
    kwargs_652200 = {}
    # Getting the type of 'getattr' (line 106)
    getattr_652197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'getattr', False)
    # Calling getattr(args, kwargs) (line 106)
    getattr_call_result_652201 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), getattr_652197, *[stats_652198, name_652199], **kwargs_652200)
    
    # Assigning a type to the variable 'd' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'd', getattr_call_result_652201)
    
    # Assigning a Subscript to a Name (line 107):
    
    # Obtaining the type of the subscript
    int_652202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 25), 'int')
    slice_652203 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 107, 13), int_652202, None, None)
    
    # Call to fit(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'data' (line 107)
    data_652206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'data', False)
    # Processing the call keyword arguments (line 107)
    kwargs_652207 = {}
    # Getting the type of 'd' (line 107)
    d_652204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 13), 'd', False)
    # Obtaining the member 'fit' of a type (line 107)
    fit_652205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 13), d_652204, 'fit')
    # Calling fit(args, kwargs) (line 107)
    fit_call_result_652208 = invoke(stypy.reporting.localization.Localization(__file__, 107, 13), fit_652205, *[data_652206], **kwargs_652207)
    
    # Obtaining the member '__getitem__' of a type (line 107)
    getitem___652209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 13), fit_call_result_652208, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 107)
    subscript_call_result_652210 = invoke(stypy.reporting.localization.Localization(__file__, 107, 13), getitem___652209, slice_652203)
    
    # Assigning a type to the variable 'actual' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'actual', subscript_call_result_652210)
    
    # Call to assert_allclose(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'actual' (line 108)
    actual_652212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'actual', False)
    # Getting the type of 'desired' (line 108)
    desired_652213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 28), 'desired', False)
    # Processing the call keyword arguments (line 108)
    # Getting the type of 'atol' (line 108)
    atol_652214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 42), 'atol', False)
    keyword_652215 = atol_652214
    str_652216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 28), 'str', 'poor mle fit of (loc, scale) in %s')
    # Getting the type of 'name' (line 109)
    name_652217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 67), 'name', False)
    # Applying the binary operator '%' (line 109)
    result_mod_652218 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 28), '%', str_652216, name_652217)
    
    keyword_652219 = result_mod_652218
    kwargs_652220 = {'err_msg': keyword_652219, 'atol': keyword_652215}
    # Getting the type of 'assert_allclose' (line 108)
    assert_allclose_652211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 108)
    assert_allclose_call_result_652221 = invoke(stypy.reporting.localization.Localization(__file__, 108, 4), assert_allclose_652211, *[actual_652212, desired_652213], **kwargs_652220)
    
    
    # ################# End of '_check_loc_scale_mle_fit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_loc_scale_mle_fit' in the type store
    # Getting the type of 'stypy_return_type' (line 105)
    stypy_return_type_652222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_652222)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_loc_scale_mle_fit'
    return stypy_return_type_652222

# Assigning a type to the variable '_check_loc_scale_mle_fit' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), '_check_loc_scale_mle_fit', _check_loc_scale_mle_fit)

@norecursion
def test_non_default_loc_scale_mle_fit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_non_default_loc_scale_mle_fit'
    module_type_store = module_type_store.open_function_context('test_non_default_loc_scale_mle_fit', 112, 0, False)
    
    # Passed parameters checking function
    test_non_default_loc_scale_mle_fit.stypy_localization = localization
    test_non_default_loc_scale_mle_fit.stypy_type_of_self = None
    test_non_default_loc_scale_mle_fit.stypy_type_store = module_type_store
    test_non_default_loc_scale_mle_fit.stypy_function_name = 'test_non_default_loc_scale_mle_fit'
    test_non_default_loc_scale_mle_fit.stypy_param_names_list = []
    test_non_default_loc_scale_mle_fit.stypy_varargs_param_name = None
    test_non_default_loc_scale_mle_fit.stypy_kwargs_param_name = None
    test_non_default_loc_scale_mle_fit.stypy_call_defaults = defaults
    test_non_default_loc_scale_mle_fit.stypy_call_varargs = varargs
    test_non_default_loc_scale_mle_fit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_non_default_loc_scale_mle_fit', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_non_default_loc_scale_mle_fit', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_non_default_loc_scale_mle_fit(...)' code ##################

    
    # Assigning a Call to a Name (line 113):
    
    # Call to array(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Obtaining an instance of the builtin type 'list' (line 113)
    list_652225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 113)
    # Adding element type (line 113)
    float_652226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 20), list_652225, float_652226)
    # Adding element type (line 113)
    float_652227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 27), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 20), list_652225, float_652227)
    # Adding element type (line 113)
    float_652228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 33), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 20), list_652225, float_652228)
    # Adding element type (line 113)
    float_652229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 39), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 20), list_652225, float_652229)
    # Adding element type (line 113)
    float_652230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 45), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 20), list_652225, float_652230)
    # Adding element type (line 113)
    float_652231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 51), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 20), list_652225, float_652231)
    # Adding element type (line 113)
    float_652232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 57), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 20), list_652225, float_652232)
    # Adding element type (line 113)
    float_652233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 63), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 20), list_652225, float_652233)
    
    # Processing the call keyword arguments (line 113)
    kwargs_652234 = {}
    # Getting the type of 'np' (line 113)
    np_652223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'np', False)
    # Obtaining the member 'array' of a type (line 113)
    array_652224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 11), np_652223, 'array')
    # Calling array(args, kwargs) (line 113)
    array_call_result_652235 = invoke(stypy.reporting.localization.Localization(__file__, 113, 11), array_652224, *[list_652225], **kwargs_652234)
    
    # Assigning a type to the variable 'data' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'data', array_call_result_652235)
    
    # Call to _check_loc_scale_mle_fit(...): (line 114)
    # Processing the call arguments (line 114)
    str_652237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 29), 'str', 'uniform')
    # Getting the type of 'data' (line 114)
    data_652238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 40), 'data', False)
    
    # Obtaining an instance of the builtin type 'list' (line 114)
    list_652239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 46), 'list')
    # Adding type elements to the builtin type 'list' instance (line 114)
    # Adding element type (line 114)
    float_652240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 47), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 46), list_652239, float_652240)
    # Adding element type (line 114)
    float_652241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 53), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 46), list_652239, float_652241)
    
    float_652242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 60), 'float')
    # Processing the call keyword arguments (line 114)
    kwargs_652243 = {}
    # Getting the type of '_check_loc_scale_mle_fit' (line 114)
    _check_loc_scale_mle_fit_652236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), '_check_loc_scale_mle_fit', False)
    # Calling _check_loc_scale_mle_fit(args, kwargs) (line 114)
    _check_loc_scale_mle_fit_call_result_652244 = invoke(stypy.reporting.localization.Localization(__file__, 114, 4), _check_loc_scale_mle_fit_652236, *[str_652237, data_652238, list_652239, float_652242], **kwargs_652243)
    
    
    # Call to _check_loc_scale_mle_fit(...): (line 115)
    # Processing the call arguments (line 115)
    str_652246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 29), 'str', 'expon')
    # Getting the type of 'data' (line 115)
    data_652247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 38), 'data', False)
    
    # Obtaining an instance of the builtin type 'list' (line 115)
    list_652248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 44), 'list')
    # Adding type elements to the builtin type 'list' instance (line 115)
    # Adding element type (line 115)
    float_652249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 45), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 44), list_652248, float_652249)
    # Adding element type (line 115)
    float_652250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 51), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 44), list_652248, float_652250)
    
    float_652251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 61), 'float')
    # Processing the call keyword arguments (line 115)
    kwargs_652252 = {}
    # Getting the type of '_check_loc_scale_mle_fit' (line 115)
    _check_loc_scale_mle_fit_652245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), '_check_loc_scale_mle_fit', False)
    # Calling _check_loc_scale_mle_fit(args, kwargs) (line 115)
    _check_loc_scale_mle_fit_call_result_652253 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), _check_loc_scale_mle_fit_652245, *[str_652246, data_652247, list_652248, float_652251], **kwargs_652252)
    
    
    # ################# End of 'test_non_default_loc_scale_mle_fit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_non_default_loc_scale_mle_fit' in the type store
    # Getting the type of 'stypy_return_type' (line 112)
    stypy_return_type_652254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_652254)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_non_default_loc_scale_mle_fit'
    return stypy_return_type_652254

# Assigning a type to the variable 'test_non_default_loc_scale_mle_fit' (line 112)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), 'test_non_default_loc_scale_mle_fit', test_non_default_loc_scale_mle_fit)

@norecursion
def test_expon_fit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_expon_fit'
    module_type_store = module_type_store.open_function_context('test_expon_fit', 118, 0, False)
    
    # Passed parameters checking function
    test_expon_fit.stypy_localization = localization
    test_expon_fit.stypy_type_of_self = None
    test_expon_fit.stypy_type_store = module_type_store
    test_expon_fit.stypy_function_name = 'test_expon_fit'
    test_expon_fit.stypy_param_names_list = []
    test_expon_fit.stypy_varargs_param_name = None
    test_expon_fit.stypy_kwargs_param_name = None
    test_expon_fit.stypy_call_defaults = defaults
    test_expon_fit.stypy_call_varargs = varargs
    test_expon_fit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_expon_fit', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_expon_fit', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_expon_fit(...)' code ##################

    str_652255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 4), 'str', 'gh-6167')
    
    # Assigning a List to a Name (line 120):
    
    # Obtaining an instance of the builtin type 'list' (line 120)
    list_652256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 120)
    # Adding element type (line 120)
    int_652257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 11), list_652256, int_652257)
    # Adding element type (line 120)
    int_652258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 11), list_652256, int_652258)
    # Adding element type (line 120)
    int_652259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 11), list_652256, int_652259)
    # Adding element type (line 120)
    int_652260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 11), list_652256, int_652260)
    # Adding element type (line 120)
    int_652261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 11), list_652256, int_652261)
    # Adding element type (line 120)
    int_652262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 11), list_652256, int_652262)
    # Adding element type (line 120)
    int_652263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 11), list_652256, int_652263)
    # Adding element type (line 120)
    int_652264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 11), list_652256, int_652264)
    
    # Assigning a type to the variable 'data' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'data', list_652256)
    
    # Assigning a Call to a Name (line 121):
    
    # Call to fit(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'data' (line 121)
    data_652268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 27), 'data', False)
    # Processing the call keyword arguments (line 121)
    int_652269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 38), 'int')
    keyword_652270 = int_652269
    kwargs_652271 = {'floc': keyword_652270}
    # Getting the type of 'stats' (line 121)
    stats_652265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'stats', False)
    # Obtaining the member 'expon' of a type (line 121)
    expon_652266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 11), stats_652265, 'expon')
    # Obtaining the member 'fit' of a type (line 121)
    fit_652267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 11), expon_652266, 'fit')
    # Calling fit(args, kwargs) (line 121)
    fit_call_result_652272 = invoke(stypy.reporting.localization.Localization(__file__, 121, 11), fit_652267, *[data_652268], **kwargs_652271)
    
    # Assigning a type to the variable 'phat' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'phat', fit_call_result_652272)
    
    # Call to assert_allclose(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'phat' (line 122)
    phat_652274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'phat', False)
    
    # Obtaining an instance of the builtin type 'list' (line 122)
    list_652275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 122)
    # Adding element type (line 122)
    int_652276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 26), list_652275, int_652276)
    # Adding element type (line 122)
    float_652277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 30), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 26), list_652275, float_652277)
    
    # Processing the call keyword arguments (line 122)
    float_652278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 41), 'float')
    keyword_652279 = float_652278
    kwargs_652280 = {'atol': keyword_652279}
    # Getting the type of 'assert_allclose' (line 122)
    assert_allclose_652273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 122)
    assert_allclose_call_result_652281 = invoke(stypy.reporting.localization.Localization(__file__, 122, 4), assert_allclose_652273, *[phat_652274, list_652275], **kwargs_652280)
    
    
    # ################# End of 'test_expon_fit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_expon_fit' in the type store
    # Getting the type of 'stypy_return_type' (line 118)
    stypy_return_type_652282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_652282)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_expon_fit'
    return stypy_return_type_652282

# Assigning a type to the variable 'test_expon_fit' (line 118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'test_expon_fit', test_expon_fit)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
