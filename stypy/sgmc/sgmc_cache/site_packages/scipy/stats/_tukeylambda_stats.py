
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy import poly1d
5: from scipy.special import beta
6: 
7: 
8: # The following code was used to generate the Pade coefficients for the
9: # Tukey Lambda variance function.  Version 0.17 of mpmath was used.
10: #---------------------------------------------------------------------------
11: # import mpmath as mp
12: #
13: # mp.mp.dps = 60
14: #
15: # one   = mp.mpf(1)
16: # two   = mp.mpf(2)
17: #
18: # def mpvar(lam):
19: #     if lam == 0:
20: #         v = mp.pi**2 / three
21: #     else:
22: #         v = (two / lam**2) * (one / (one + two*lam) -
23: #                               mp.beta(lam + one, lam + one))
24: #     return v
25: #
26: # t = mp.taylor(mpvar, 0, 8)
27: # p, q = mp.pade(t, 4, 4)
28: # print("p =", [mp.fp.mpf(c) for c in p])
29: # print("q =", [mp.fp.mpf(c) for c in q])
30: #---------------------------------------------------------------------------
31: 
32: # Pade coefficients for the Tukey Lambda variance function.
33: _tukeylambda_var_pc = [3.289868133696453, 0.7306125098871127,
34:                        -0.5370742306855439, 0.17292046290190008,
35:                        -0.02371146284628187]
36: _tukeylambda_var_qc = [1.0, 3.683605511659861, 4.184152498888124,
37:                        1.7660926747377275, 0.2643989311168465]
38: 
39: # numpy.poly1d instances for the numerator and denominator of the
40: # Pade approximation to the Tukey Lambda variance.
41: _tukeylambda_var_p = poly1d(_tukeylambda_var_pc[::-1])
42: _tukeylambda_var_q = poly1d(_tukeylambda_var_qc[::-1])
43: 
44: 
45: def tukeylambda_variance(lam):
46:     '''Variance of the Tukey Lambda distribution.
47: 
48:     Parameters
49:     ----------
50:     lam : array_like
51:         The lambda values at which to compute the variance.
52: 
53:     Returns
54:     -------
55:     v : ndarray
56:         The variance.  For lam < -0.5, the variance is not defined, so
57:         np.nan is returned.  For lam = 0.5, np.inf is returned.
58: 
59:     Notes
60:     -----
61:     In an interval around lambda=0, this function uses the [4,4] Pade
62:     approximation to compute the variance.  Otherwise it uses the standard
63:     formula (http://en.wikipedia.org/wiki/Tukey_lambda_distribution).  The
64:     Pade approximation is used because the standard formula has a removable
65:     discontinuity at lambda = 0, and does not produce accurate numerical
66:     results near lambda = 0.
67:     '''
68:     lam = np.asarray(lam)
69:     shp = lam.shape
70:     lam = np.atleast_1d(lam).astype(np.float64)
71: 
72:     # For absolute values of lam less than threshold, use the Pade
73:     # approximation.
74:     threshold = 0.075
75: 
76:     # Play games with masks to implement the conditional evaluation of
77:     # the distribution.
78:     # lambda < -0.5:  var = nan
79:     low_mask = lam < -0.5
80:     # lambda == -0.5: var = inf
81:     neghalf_mask = lam == -0.5
82:     # abs(lambda) < threshold:  use Pade approximation
83:     small_mask = np.abs(lam) < threshold
84:     # else the "regular" case:  use the explicit formula.
85:     reg_mask = ~(low_mask | neghalf_mask | small_mask)
86: 
87:     # Get the 'lam' values for the cases where they are needed.
88:     small = lam[small_mask]
89:     reg = lam[reg_mask]
90: 
91:     # Compute the function for each case.
92:     v = np.empty_like(lam)
93:     v[low_mask] = np.nan
94:     v[neghalf_mask] = np.inf
95:     if small.size > 0:
96:         # Use the Pade approximation near lambda = 0.
97:         v[small_mask] = _tukeylambda_var_p(small) / _tukeylambda_var_q(small)
98:     if reg.size > 0:
99:         v[reg_mask] = (2.0 / reg**2) * (1.0 / (1.0 + 2 * reg) -
100:                                       beta(reg + 1, reg + 1))
101:     v.shape = shp
102:     return v
103: 
104: 
105: # The following code was used to generate the Pade coefficients for the
106: # Tukey Lambda kurtosis function.  Version 0.17 of mpmath was used.
107: #---------------------------------------------------------------------------
108: # import mpmath as mp
109: #
110: # mp.mp.dps = 60
111: #
112: # one   = mp.mpf(1)
113: # two   = mp.mpf(2)
114: # three = mp.mpf(3)
115: # four  = mp.mpf(4)
116: #
117: # def mpkurt(lam):
118: #     if lam == 0:
119: #         k = mp.mpf(6)/5
120: #     else:
121: #         numer = (one/(four*lam+one) - four*mp.beta(three*lam+one, lam+one) +
122: #                  three*mp.beta(two*lam+one, two*lam+one))
123: #         denom = two*(one/(two*lam+one) - mp.beta(lam+one,lam+one))**2
124: #         k = numer / denom - three
125: #     return k
126: #
127: # # There is a bug in mpmath 0.17: when we use the 'method' keyword of the
128: # # taylor function and we request a degree 9 Taylor polynomial, we actually
129: # # get degree 8.
130: # t = mp.taylor(mpkurt, 0, 9, method='quad', radius=0.01)
131: # t = [mp.chop(c, tol=1e-15) for c in t]
132: # p, q = mp.pade(t, 4, 4)
133: # print("p =", [mp.fp.mpf(c) for c in p])
134: # print("q =", [mp.fp.mpf(c) for c in q])
135: #---------------------------------------------------------------------------
136: 
137: # Pade coefficients for the Tukey Lambda kurtosis function.
138: _tukeylambda_kurt_pc = [1.2, -5.853465139719495, -22.653447381131077,
139:                         0.20601184383406815, 4.59796302262789]
140: _tukeylambda_kurt_qc = [1.0, 7.171149192233599, 12.96663094361842,
141:                         0.43075235247853005, -2.789746758009912]
142: 
143: # numpy.poly1d instances for the numerator and denominator of the
144: # Pade approximation to the Tukey Lambda kurtosis.
145: _tukeylambda_kurt_p = poly1d(_tukeylambda_kurt_pc[::-1])
146: _tukeylambda_kurt_q = poly1d(_tukeylambda_kurt_qc[::-1])
147: 
148: 
149: def tukeylambda_kurtosis(lam):
150:     '''Kurtosis of the Tukey Lambda distribution.
151: 
152:     Parameters
153:     ----------
154:     lam : array_like
155:         The lambda values at which to compute the variance.
156: 
157:     Returns
158:     -------
159:     v : ndarray
160:         The variance.  For lam < -0.25, the variance is not defined, so
161:         np.nan is returned.  For lam = 0.25, np.inf is returned.
162: 
163:     '''
164:     lam = np.asarray(lam)
165:     shp = lam.shape
166:     lam = np.atleast_1d(lam).astype(np.float64)
167: 
168:     # For absolute values of lam less than threshold, use the Pade
169:     # approximation.
170:     threshold = 0.055
171: 
172:     # Use masks to implement the conditional evaluation of the kurtosis.
173:     # lambda < -0.25:  kurtosis = nan
174:     low_mask = lam < -0.25
175:     # lambda == -0.25: kurtosis = inf
176:     negqrtr_mask = lam == -0.25
177:     # lambda near 0:  use Pade approximation
178:     small_mask = np.abs(lam) < threshold
179:     # else the "regular" case:  use the explicit formula.
180:     reg_mask = ~(low_mask | negqrtr_mask | small_mask)
181: 
182:     # Get the 'lam' values for the cases where they are needed.
183:     small = lam[small_mask]
184:     reg = lam[reg_mask]
185: 
186:     # Compute the function for each case.
187:     k = np.empty_like(lam)
188:     k[low_mask] = np.nan
189:     k[negqrtr_mask] = np.inf
190:     if small.size > 0:
191:         k[small_mask] = _tukeylambda_kurt_p(small) / _tukeylambda_kurt_q(small)
192:     if reg.size > 0:
193:         numer = (1.0 / (4 * reg + 1) - 4 * beta(3 * reg + 1, reg + 1) +
194:                  3 * beta(2 * reg + 1, 2 * reg + 1))
195:         denom = 2 * (1.0/(2 * reg + 1) - beta(reg + 1, reg + 1))**2
196:         k[reg_mask] = numer / denom - 3
197: 
198:     # The return value will be a numpy array; resetting the shape ensures that
199:     # if `lam` was a scalar, the return value is a 0-d array.
200:     k.shape = shp
201:     return k
202: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_626464 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_626464) is not StypyTypeError):

    if (import_626464 != 'pyd_module'):
        __import__(import_626464)
        sys_modules_626465 = sys.modules[import_626464]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_626465.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_626464)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy import poly1d' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_626466 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_626466) is not StypyTypeError):

    if (import_626466 != 'pyd_module'):
        __import__(import_626466)
        sys_modules_626467 = sys.modules[import_626466]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', sys_modules_626467.module_type_store, module_type_store, ['poly1d'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_626467, sys_modules_626467.module_type_store, module_type_store)
    else:
        from numpy import poly1d

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', None, module_type_store, ['poly1d'], [poly1d])

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_626466)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.special import beta' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_626468 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special')

if (type(import_626468) is not StypyTypeError):

    if (import_626468 != 'pyd_module'):
        __import__(import_626468)
        sys_modules_626469 = sys.modules[import_626468]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special', sys_modules_626469.module_type_store, module_type_store, ['beta'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_626469, sys_modules_626469.module_type_store, module_type_store)
    else:
        from scipy.special import beta

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special', None, module_type_store, ['beta'], [beta])

else:
    # Assigning a type to the variable 'scipy.special' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.special', import_626468)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')


# Assigning a List to a Name (line 33):

# Obtaining an instance of the builtin type 'list' (line 33)
list_626470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 33)
# Adding element type (line 33)
float_626471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 22), list_626470, float_626471)
# Adding element type (line 33)
float_626472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 42), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 22), list_626470, float_626472)
# Adding element type (line 33)
float_626473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 22), list_626470, float_626473)
# Adding element type (line 33)
float_626474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 44), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 22), list_626470, float_626474)
# Adding element type (line 33)
float_626475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 22), list_626470, float_626475)

# Assigning a type to the variable '_tukeylambda_var_pc' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), '_tukeylambda_var_pc', list_626470)

# Assigning a List to a Name (line 36):

# Obtaining an instance of the builtin type 'list' (line 36)
list_626476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 36)
# Adding element type (line 36)
float_626477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 22), list_626476, float_626477)
# Adding element type (line 36)
float_626478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 28), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 22), list_626476, float_626478)
# Adding element type (line 36)
float_626479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 47), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 22), list_626476, float_626479)
# Adding element type (line 36)
float_626480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 23), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 22), list_626476, float_626480)
# Adding element type (line 36)
float_626481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 43), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 22), list_626476, float_626481)

# Assigning a type to the variable '_tukeylambda_var_qc' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), '_tukeylambda_var_qc', list_626476)

# Assigning a Call to a Name (line 41):

# Call to poly1d(...): (line 41)
# Processing the call arguments (line 41)

# Obtaining the type of the subscript
int_626483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 50), 'int')
slice_626484 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 41, 28), None, None, int_626483)
# Getting the type of '_tukeylambda_var_pc' (line 41)
_tukeylambda_var_pc_626485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 28), '_tukeylambda_var_pc', False)
# Obtaining the member '__getitem__' of a type (line 41)
getitem___626486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 28), _tukeylambda_var_pc_626485, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 41)
subscript_call_result_626487 = invoke(stypy.reporting.localization.Localization(__file__, 41, 28), getitem___626486, slice_626484)

# Processing the call keyword arguments (line 41)
kwargs_626488 = {}
# Getting the type of 'poly1d' (line 41)
poly1d_626482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 21), 'poly1d', False)
# Calling poly1d(args, kwargs) (line 41)
poly1d_call_result_626489 = invoke(stypy.reporting.localization.Localization(__file__, 41, 21), poly1d_626482, *[subscript_call_result_626487], **kwargs_626488)

# Assigning a type to the variable '_tukeylambda_var_p' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), '_tukeylambda_var_p', poly1d_call_result_626489)

# Assigning a Call to a Name (line 42):

# Call to poly1d(...): (line 42)
# Processing the call arguments (line 42)

# Obtaining the type of the subscript
int_626491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 50), 'int')
slice_626492 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 42, 28), None, None, int_626491)
# Getting the type of '_tukeylambda_var_qc' (line 42)
_tukeylambda_var_qc_626493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), '_tukeylambda_var_qc', False)
# Obtaining the member '__getitem__' of a type (line 42)
getitem___626494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 28), _tukeylambda_var_qc_626493, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 42)
subscript_call_result_626495 = invoke(stypy.reporting.localization.Localization(__file__, 42, 28), getitem___626494, slice_626492)

# Processing the call keyword arguments (line 42)
kwargs_626496 = {}
# Getting the type of 'poly1d' (line 42)
poly1d_626490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 21), 'poly1d', False)
# Calling poly1d(args, kwargs) (line 42)
poly1d_call_result_626497 = invoke(stypy.reporting.localization.Localization(__file__, 42, 21), poly1d_626490, *[subscript_call_result_626495], **kwargs_626496)

# Assigning a type to the variable '_tukeylambda_var_q' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), '_tukeylambda_var_q', poly1d_call_result_626497)

@norecursion
def tukeylambda_variance(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'tukeylambda_variance'
    module_type_store = module_type_store.open_function_context('tukeylambda_variance', 45, 0, False)
    
    # Passed parameters checking function
    tukeylambda_variance.stypy_localization = localization
    tukeylambda_variance.stypy_type_of_self = None
    tukeylambda_variance.stypy_type_store = module_type_store
    tukeylambda_variance.stypy_function_name = 'tukeylambda_variance'
    tukeylambda_variance.stypy_param_names_list = ['lam']
    tukeylambda_variance.stypy_varargs_param_name = None
    tukeylambda_variance.stypy_kwargs_param_name = None
    tukeylambda_variance.stypy_call_defaults = defaults
    tukeylambda_variance.stypy_call_varargs = varargs
    tukeylambda_variance.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tukeylambda_variance', ['lam'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tukeylambda_variance', localization, ['lam'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tukeylambda_variance(...)' code ##################

    str_626498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, (-1)), 'str', 'Variance of the Tukey Lambda distribution.\n\n    Parameters\n    ----------\n    lam : array_like\n        The lambda values at which to compute the variance.\n\n    Returns\n    -------\n    v : ndarray\n        The variance.  For lam < -0.5, the variance is not defined, so\n        np.nan is returned.  For lam = 0.5, np.inf is returned.\n\n    Notes\n    -----\n    In an interval around lambda=0, this function uses the [4,4] Pade\n    approximation to compute the variance.  Otherwise it uses the standard\n    formula (http://en.wikipedia.org/wiki/Tukey_lambda_distribution).  The\n    Pade approximation is used because the standard formula has a removable\n    discontinuity at lambda = 0, and does not produce accurate numerical\n    results near lambda = 0.\n    ')
    
    # Assigning a Call to a Name (line 68):
    
    # Call to asarray(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'lam' (line 68)
    lam_626501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 'lam', False)
    # Processing the call keyword arguments (line 68)
    kwargs_626502 = {}
    # Getting the type of 'np' (line 68)
    np_626499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 10), 'np', False)
    # Obtaining the member 'asarray' of a type (line 68)
    asarray_626500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 10), np_626499, 'asarray')
    # Calling asarray(args, kwargs) (line 68)
    asarray_call_result_626503 = invoke(stypy.reporting.localization.Localization(__file__, 68, 10), asarray_626500, *[lam_626501], **kwargs_626502)
    
    # Assigning a type to the variable 'lam' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'lam', asarray_call_result_626503)
    
    # Assigning a Attribute to a Name (line 69):
    # Getting the type of 'lam' (line 69)
    lam_626504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 10), 'lam')
    # Obtaining the member 'shape' of a type (line 69)
    shape_626505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 10), lam_626504, 'shape')
    # Assigning a type to the variable 'shp' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'shp', shape_626505)
    
    # Assigning a Call to a Name (line 70):
    
    # Call to astype(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'np' (line 70)
    np_626512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 36), 'np', False)
    # Obtaining the member 'float64' of a type (line 70)
    float64_626513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 36), np_626512, 'float64')
    # Processing the call keyword arguments (line 70)
    kwargs_626514 = {}
    
    # Call to atleast_1d(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'lam' (line 70)
    lam_626508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 24), 'lam', False)
    # Processing the call keyword arguments (line 70)
    kwargs_626509 = {}
    # Getting the type of 'np' (line 70)
    np_626506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 10), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 70)
    atleast_1d_626507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 10), np_626506, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 70)
    atleast_1d_call_result_626510 = invoke(stypy.reporting.localization.Localization(__file__, 70, 10), atleast_1d_626507, *[lam_626508], **kwargs_626509)
    
    # Obtaining the member 'astype' of a type (line 70)
    astype_626511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 10), atleast_1d_call_result_626510, 'astype')
    # Calling astype(args, kwargs) (line 70)
    astype_call_result_626515 = invoke(stypy.reporting.localization.Localization(__file__, 70, 10), astype_626511, *[float64_626513], **kwargs_626514)
    
    # Assigning a type to the variable 'lam' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'lam', astype_call_result_626515)
    
    # Assigning a Num to a Name (line 74):
    float_626516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 16), 'float')
    # Assigning a type to the variable 'threshold' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'threshold', float_626516)
    
    # Assigning a Compare to a Name (line 79):
    
    # Getting the type of 'lam' (line 79)
    lam_626517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'lam')
    float_626518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 21), 'float')
    # Applying the binary operator '<' (line 79)
    result_lt_626519 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 15), '<', lam_626517, float_626518)
    
    # Assigning a type to the variable 'low_mask' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'low_mask', result_lt_626519)
    
    # Assigning a Compare to a Name (line 81):
    
    # Getting the type of 'lam' (line 81)
    lam_626520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 19), 'lam')
    float_626521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 26), 'float')
    # Applying the binary operator '==' (line 81)
    result_eq_626522 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 19), '==', lam_626520, float_626521)
    
    # Assigning a type to the variable 'neghalf_mask' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'neghalf_mask', result_eq_626522)
    
    # Assigning a Compare to a Name (line 83):
    
    
    # Call to abs(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'lam' (line 83)
    lam_626525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'lam', False)
    # Processing the call keyword arguments (line 83)
    kwargs_626526 = {}
    # Getting the type of 'np' (line 83)
    np_626523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 17), 'np', False)
    # Obtaining the member 'abs' of a type (line 83)
    abs_626524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 17), np_626523, 'abs')
    # Calling abs(args, kwargs) (line 83)
    abs_call_result_626527 = invoke(stypy.reporting.localization.Localization(__file__, 83, 17), abs_626524, *[lam_626525], **kwargs_626526)
    
    # Getting the type of 'threshold' (line 83)
    threshold_626528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 31), 'threshold')
    # Applying the binary operator '<' (line 83)
    result_lt_626529 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 17), '<', abs_call_result_626527, threshold_626528)
    
    # Assigning a type to the variable 'small_mask' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'small_mask', result_lt_626529)
    
    # Assigning a UnaryOp to a Name (line 85):
    
    # Getting the type of 'low_mask' (line 85)
    low_mask_626530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'low_mask')
    # Getting the type of 'neghalf_mask' (line 85)
    neghalf_mask_626531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 28), 'neghalf_mask')
    # Applying the binary operator '|' (line 85)
    result_or__626532 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 17), '|', low_mask_626530, neghalf_mask_626531)
    
    # Getting the type of 'small_mask' (line 85)
    small_mask_626533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 43), 'small_mask')
    # Applying the binary operator '|' (line 85)
    result_or__626534 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 41), '|', result_or__626532, small_mask_626533)
    
    # Applying the '~' unary operator (line 85)
    result_inv_626535 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 15), '~', result_or__626534)
    
    # Assigning a type to the variable 'reg_mask' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'reg_mask', result_inv_626535)
    
    # Assigning a Subscript to a Name (line 88):
    
    # Obtaining the type of the subscript
    # Getting the type of 'small_mask' (line 88)
    small_mask_626536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'small_mask')
    # Getting the type of 'lam' (line 88)
    lam_626537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'lam')
    # Obtaining the member '__getitem__' of a type (line 88)
    getitem___626538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), lam_626537, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 88)
    subscript_call_result_626539 = invoke(stypy.reporting.localization.Localization(__file__, 88, 12), getitem___626538, small_mask_626536)
    
    # Assigning a type to the variable 'small' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'small', subscript_call_result_626539)
    
    # Assigning a Subscript to a Name (line 89):
    
    # Obtaining the type of the subscript
    # Getting the type of 'reg_mask' (line 89)
    reg_mask_626540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 14), 'reg_mask')
    # Getting the type of 'lam' (line 89)
    lam_626541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 10), 'lam')
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___626542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 10), lam_626541, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_626543 = invoke(stypy.reporting.localization.Localization(__file__, 89, 10), getitem___626542, reg_mask_626540)
    
    # Assigning a type to the variable 'reg' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'reg', subscript_call_result_626543)
    
    # Assigning a Call to a Name (line 92):
    
    # Call to empty_like(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'lam' (line 92)
    lam_626546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 22), 'lam', False)
    # Processing the call keyword arguments (line 92)
    kwargs_626547 = {}
    # Getting the type of 'np' (line 92)
    np_626544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'np', False)
    # Obtaining the member 'empty_like' of a type (line 92)
    empty_like_626545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), np_626544, 'empty_like')
    # Calling empty_like(args, kwargs) (line 92)
    empty_like_call_result_626548 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), empty_like_626545, *[lam_626546], **kwargs_626547)
    
    # Assigning a type to the variable 'v' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'v', empty_like_call_result_626548)
    
    # Assigning a Attribute to a Subscript (line 93):
    # Getting the type of 'np' (line 93)
    np_626549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 18), 'np')
    # Obtaining the member 'nan' of a type (line 93)
    nan_626550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 18), np_626549, 'nan')
    # Getting the type of 'v' (line 93)
    v_626551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'v')
    # Getting the type of 'low_mask' (line 93)
    low_mask_626552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 6), 'low_mask')
    # Storing an element on a container (line 93)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 4), v_626551, (low_mask_626552, nan_626550))
    
    # Assigning a Attribute to a Subscript (line 94):
    # Getting the type of 'np' (line 94)
    np_626553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 22), 'np')
    # Obtaining the member 'inf' of a type (line 94)
    inf_626554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 22), np_626553, 'inf')
    # Getting the type of 'v' (line 94)
    v_626555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'v')
    # Getting the type of 'neghalf_mask' (line 94)
    neghalf_mask_626556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 6), 'neghalf_mask')
    # Storing an element on a container (line 94)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 4), v_626555, (neghalf_mask_626556, inf_626554))
    
    
    # Getting the type of 'small' (line 95)
    small_626557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 7), 'small')
    # Obtaining the member 'size' of a type (line 95)
    size_626558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 7), small_626557, 'size')
    int_626559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 20), 'int')
    # Applying the binary operator '>' (line 95)
    result_gt_626560 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 7), '>', size_626558, int_626559)
    
    # Testing the type of an if condition (line 95)
    if_condition_626561 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 4), result_gt_626560)
    # Assigning a type to the variable 'if_condition_626561' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'if_condition_626561', if_condition_626561)
    # SSA begins for if statement (line 95)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 97):
    
    # Call to _tukeylambda_var_p(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'small' (line 97)
    small_626563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 43), 'small', False)
    # Processing the call keyword arguments (line 97)
    kwargs_626564 = {}
    # Getting the type of '_tukeylambda_var_p' (line 97)
    _tukeylambda_var_p_626562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), '_tukeylambda_var_p', False)
    # Calling _tukeylambda_var_p(args, kwargs) (line 97)
    _tukeylambda_var_p_call_result_626565 = invoke(stypy.reporting.localization.Localization(__file__, 97, 24), _tukeylambda_var_p_626562, *[small_626563], **kwargs_626564)
    
    
    # Call to _tukeylambda_var_q(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'small' (line 97)
    small_626567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 71), 'small', False)
    # Processing the call keyword arguments (line 97)
    kwargs_626568 = {}
    # Getting the type of '_tukeylambda_var_q' (line 97)
    _tukeylambda_var_q_626566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 52), '_tukeylambda_var_q', False)
    # Calling _tukeylambda_var_q(args, kwargs) (line 97)
    _tukeylambda_var_q_call_result_626569 = invoke(stypy.reporting.localization.Localization(__file__, 97, 52), _tukeylambda_var_q_626566, *[small_626567], **kwargs_626568)
    
    # Applying the binary operator 'div' (line 97)
    result_div_626570 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 24), 'div', _tukeylambda_var_p_call_result_626565, _tukeylambda_var_q_call_result_626569)
    
    # Getting the type of 'v' (line 97)
    v_626571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'v')
    # Getting the type of 'small_mask' (line 97)
    small_mask_626572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 10), 'small_mask')
    # Storing an element on a container (line 97)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 8), v_626571, (small_mask_626572, result_div_626570))
    # SSA join for if statement (line 95)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'reg' (line 98)
    reg_626573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 7), 'reg')
    # Obtaining the member 'size' of a type (line 98)
    size_626574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 7), reg_626573, 'size')
    int_626575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 18), 'int')
    # Applying the binary operator '>' (line 98)
    result_gt_626576 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 7), '>', size_626574, int_626575)
    
    # Testing the type of an if condition (line 98)
    if_condition_626577 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 4), result_gt_626576)
    # Assigning a type to the variable 'if_condition_626577' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'if_condition_626577', if_condition_626577)
    # SSA begins for if statement (line 98)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 99):
    float_626578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 23), 'float')
    # Getting the type of 'reg' (line 99)
    reg_626579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 29), 'reg')
    int_626580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 34), 'int')
    # Applying the binary operator '**' (line 99)
    result_pow_626581 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 29), '**', reg_626579, int_626580)
    
    # Applying the binary operator 'div' (line 99)
    result_div_626582 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 23), 'div', float_626578, result_pow_626581)
    
    float_626583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 40), 'float')
    float_626584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 47), 'float')
    int_626585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 53), 'int')
    # Getting the type of 'reg' (line 99)
    reg_626586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 57), 'reg')
    # Applying the binary operator '*' (line 99)
    result_mul_626587 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 53), '*', int_626585, reg_626586)
    
    # Applying the binary operator '+' (line 99)
    result_add_626588 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 47), '+', float_626584, result_mul_626587)
    
    # Applying the binary operator 'div' (line 99)
    result_div_626589 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 40), 'div', float_626583, result_add_626588)
    
    
    # Call to beta(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'reg' (line 100)
    reg_626591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 43), 'reg', False)
    int_626592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 49), 'int')
    # Applying the binary operator '+' (line 100)
    result_add_626593 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 43), '+', reg_626591, int_626592)
    
    # Getting the type of 'reg' (line 100)
    reg_626594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 52), 'reg', False)
    int_626595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 58), 'int')
    # Applying the binary operator '+' (line 100)
    result_add_626596 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 52), '+', reg_626594, int_626595)
    
    # Processing the call keyword arguments (line 100)
    kwargs_626597 = {}
    # Getting the type of 'beta' (line 100)
    beta_626590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 38), 'beta', False)
    # Calling beta(args, kwargs) (line 100)
    beta_call_result_626598 = invoke(stypy.reporting.localization.Localization(__file__, 100, 38), beta_626590, *[result_add_626593, result_add_626596], **kwargs_626597)
    
    # Applying the binary operator '-' (line 99)
    result_sub_626599 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 40), '-', result_div_626589, beta_call_result_626598)
    
    # Applying the binary operator '*' (line 99)
    result_mul_626600 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 22), '*', result_div_626582, result_sub_626599)
    
    # Getting the type of 'v' (line 99)
    v_626601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'v')
    # Getting the type of 'reg_mask' (line 99)
    reg_mask_626602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 10), 'reg_mask')
    # Storing an element on a container (line 99)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 8), v_626601, (reg_mask_626602, result_mul_626600))
    # SSA join for if statement (line 98)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Attribute (line 101):
    # Getting the type of 'shp' (line 101)
    shp_626603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 14), 'shp')
    # Getting the type of 'v' (line 101)
    v_626604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'v')
    # Setting the type of the member 'shape' of a type (line 101)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 4), v_626604, 'shape', shp_626603)
    # Getting the type of 'v' (line 102)
    v_626605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'v')
    # Assigning a type to the variable 'stypy_return_type' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'stypy_return_type', v_626605)
    
    # ################# End of 'tukeylambda_variance(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tukeylambda_variance' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_626606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_626606)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tukeylambda_variance'
    return stypy_return_type_626606

# Assigning a type to the variable 'tukeylambda_variance' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'tukeylambda_variance', tukeylambda_variance)

# Assigning a List to a Name (line 138):

# Obtaining an instance of the builtin type 'list' (line 138)
list_626607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 138)
# Adding element type (line 138)
float_626608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 23), list_626607, float_626608)
# Adding element type (line 138)
float_626609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 29), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 23), list_626607, float_626609)
# Adding element type (line 138)
float_626610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 49), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 23), list_626607, float_626610)
# Adding element type (line 138)
float_626611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 23), list_626607, float_626611)
# Adding element type (line 138)
float_626612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 45), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 23), list_626607, float_626612)

# Assigning a type to the variable '_tukeylambda_kurt_pc' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), '_tukeylambda_kurt_pc', list_626607)

# Assigning a List to a Name (line 140):

# Obtaining an instance of the builtin type 'list' (line 140)
list_626613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 140)
# Adding element type (line 140)
float_626614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 23), list_626613, float_626614)
# Adding element type (line 140)
float_626615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 29), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 23), list_626613, float_626615)
# Adding element type (line 140)
float_626616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 48), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 23), list_626613, float_626616)
# Adding element type (line 140)
float_626617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 23), list_626613, float_626617)
# Adding element type (line 140)
float_626618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 45), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 23), list_626613, float_626618)

# Assigning a type to the variable '_tukeylambda_kurt_qc' (line 140)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), '_tukeylambda_kurt_qc', list_626613)

# Assigning a Call to a Name (line 145):

# Call to poly1d(...): (line 145)
# Processing the call arguments (line 145)

# Obtaining the type of the subscript
int_626620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 52), 'int')
slice_626621 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 145, 29), None, None, int_626620)
# Getting the type of '_tukeylambda_kurt_pc' (line 145)
_tukeylambda_kurt_pc_626622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 29), '_tukeylambda_kurt_pc', False)
# Obtaining the member '__getitem__' of a type (line 145)
getitem___626623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 29), _tukeylambda_kurt_pc_626622, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 145)
subscript_call_result_626624 = invoke(stypy.reporting.localization.Localization(__file__, 145, 29), getitem___626623, slice_626621)

# Processing the call keyword arguments (line 145)
kwargs_626625 = {}
# Getting the type of 'poly1d' (line 145)
poly1d_626619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 22), 'poly1d', False)
# Calling poly1d(args, kwargs) (line 145)
poly1d_call_result_626626 = invoke(stypy.reporting.localization.Localization(__file__, 145, 22), poly1d_626619, *[subscript_call_result_626624], **kwargs_626625)

# Assigning a type to the variable '_tukeylambda_kurt_p' (line 145)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), '_tukeylambda_kurt_p', poly1d_call_result_626626)

# Assigning a Call to a Name (line 146):

# Call to poly1d(...): (line 146)
# Processing the call arguments (line 146)

# Obtaining the type of the subscript
int_626628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 52), 'int')
slice_626629 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 146, 29), None, None, int_626628)
# Getting the type of '_tukeylambda_kurt_qc' (line 146)
_tukeylambda_kurt_qc_626630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), '_tukeylambda_kurt_qc', False)
# Obtaining the member '__getitem__' of a type (line 146)
getitem___626631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 29), _tukeylambda_kurt_qc_626630, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 146)
subscript_call_result_626632 = invoke(stypy.reporting.localization.Localization(__file__, 146, 29), getitem___626631, slice_626629)

# Processing the call keyword arguments (line 146)
kwargs_626633 = {}
# Getting the type of 'poly1d' (line 146)
poly1d_626627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 22), 'poly1d', False)
# Calling poly1d(args, kwargs) (line 146)
poly1d_call_result_626634 = invoke(stypy.reporting.localization.Localization(__file__, 146, 22), poly1d_626627, *[subscript_call_result_626632], **kwargs_626633)

# Assigning a type to the variable '_tukeylambda_kurt_q' (line 146)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), '_tukeylambda_kurt_q', poly1d_call_result_626634)

@norecursion
def tukeylambda_kurtosis(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'tukeylambda_kurtosis'
    module_type_store = module_type_store.open_function_context('tukeylambda_kurtosis', 149, 0, False)
    
    # Passed parameters checking function
    tukeylambda_kurtosis.stypy_localization = localization
    tukeylambda_kurtosis.stypy_type_of_self = None
    tukeylambda_kurtosis.stypy_type_store = module_type_store
    tukeylambda_kurtosis.stypy_function_name = 'tukeylambda_kurtosis'
    tukeylambda_kurtosis.stypy_param_names_list = ['lam']
    tukeylambda_kurtosis.stypy_varargs_param_name = None
    tukeylambda_kurtosis.stypy_kwargs_param_name = None
    tukeylambda_kurtosis.stypy_call_defaults = defaults
    tukeylambda_kurtosis.stypy_call_varargs = varargs
    tukeylambda_kurtosis.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tukeylambda_kurtosis', ['lam'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tukeylambda_kurtosis', localization, ['lam'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tukeylambda_kurtosis(...)' code ##################

    str_626635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, (-1)), 'str', 'Kurtosis of the Tukey Lambda distribution.\n\n    Parameters\n    ----------\n    lam : array_like\n        The lambda values at which to compute the variance.\n\n    Returns\n    -------\n    v : ndarray\n        The variance.  For lam < -0.25, the variance is not defined, so\n        np.nan is returned.  For lam = 0.25, np.inf is returned.\n\n    ')
    
    # Assigning a Call to a Name (line 164):
    
    # Call to asarray(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'lam' (line 164)
    lam_626638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 21), 'lam', False)
    # Processing the call keyword arguments (line 164)
    kwargs_626639 = {}
    # Getting the type of 'np' (line 164)
    np_626636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 10), 'np', False)
    # Obtaining the member 'asarray' of a type (line 164)
    asarray_626637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 10), np_626636, 'asarray')
    # Calling asarray(args, kwargs) (line 164)
    asarray_call_result_626640 = invoke(stypy.reporting.localization.Localization(__file__, 164, 10), asarray_626637, *[lam_626638], **kwargs_626639)
    
    # Assigning a type to the variable 'lam' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'lam', asarray_call_result_626640)
    
    # Assigning a Attribute to a Name (line 165):
    # Getting the type of 'lam' (line 165)
    lam_626641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 10), 'lam')
    # Obtaining the member 'shape' of a type (line 165)
    shape_626642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 10), lam_626641, 'shape')
    # Assigning a type to the variable 'shp' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'shp', shape_626642)
    
    # Assigning a Call to a Name (line 166):
    
    # Call to astype(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'np' (line 166)
    np_626649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 36), 'np', False)
    # Obtaining the member 'float64' of a type (line 166)
    float64_626650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 36), np_626649, 'float64')
    # Processing the call keyword arguments (line 166)
    kwargs_626651 = {}
    
    # Call to atleast_1d(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'lam' (line 166)
    lam_626645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'lam', False)
    # Processing the call keyword arguments (line 166)
    kwargs_626646 = {}
    # Getting the type of 'np' (line 166)
    np_626643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 10), 'np', False)
    # Obtaining the member 'atleast_1d' of a type (line 166)
    atleast_1d_626644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 10), np_626643, 'atleast_1d')
    # Calling atleast_1d(args, kwargs) (line 166)
    atleast_1d_call_result_626647 = invoke(stypy.reporting.localization.Localization(__file__, 166, 10), atleast_1d_626644, *[lam_626645], **kwargs_626646)
    
    # Obtaining the member 'astype' of a type (line 166)
    astype_626648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 10), atleast_1d_call_result_626647, 'astype')
    # Calling astype(args, kwargs) (line 166)
    astype_call_result_626652 = invoke(stypy.reporting.localization.Localization(__file__, 166, 10), astype_626648, *[float64_626650], **kwargs_626651)
    
    # Assigning a type to the variable 'lam' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'lam', astype_call_result_626652)
    
    # Assigning a Num to a Name (line 170):
    float_626653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 16), 'float')
    # Assigning a type to the variable 'threshold' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'threshold', float_626653)
    
    # Assigning a Compare to a Name (line 174):
    
    # Getting the type of 'lam' (line 174)
    lam_626654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 15), 'lam')
    float_626655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 21), 'float')
    # Applying the binary operator '<' (line 174)
    result_lt_626656 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 15), '<', lam_626654, float_626655)
    
    # Assigning a type to the variable 'low_mask' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'low_mask', result_lt_626656)
    
    # Assigning a Compare to a Name (line 176):
    
    # Getting the type of 'lam' (line 176)
    lam_626657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'lam')
    float_626658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 26), 'float')
    # Applying the binary operator '==' (line 176)
    result_eq_626659 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 19), '==', lam_626657, float_626658)
    
    # Assigning a type to the variable 'negqrtr_mask' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'negqrtr_mask', result_eq_626659)
    
    # Assigning a Compare to a Name (line 178):
    
    
    # Call to abs(...): (line 178)
    # Processing the call arguments (line 178)
    # Getting the type of 'lam' (line 178)
    lam_626662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), 'lam', False)
    # Processing the call keyword arguments (line 178)
    kwargs_626663 = {}
    # Getting the type of 'np' (line 178)
    np_626660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 17), 'np', False)
    # Obtaining the member 'abs' of a type (line 178)
    abs_626661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 17), np_626660, 'abs')
    # Calling abs(args, kwargs) (line 178)
    abs_call_result_626664 = invoke(stypy.reporting.localization.Localization(__file__, 178, 17), abs_626661, *[lam_626662], **kwargs_626663)
    
    # Getting the type of 'threshold' (line 178)
    threshold_626665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 31), 'threshold')
    # Applying the binary operator '<' (line 178)
    result_lt_626666 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 17), '<', abs_call_result_626664, threshold_626665)
    
    # Assigning a type to the variable 'small_mask' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'small_mask', result_lt_626666)
    
    # Assigning a UnaryOp to a Name (line 180):
    
    # Getting the type of 'low_mask' (line 180)
    low_mask_626667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'low_mask')
    # Getting the type of 'negqrtr_mask' (line 180)
    negqrtr_mask_626668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 28), 'negqrtr_mask')
    # Applying the binary operator '|' (line 180)
    result_or__626669 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 17), '|', low_mask_626667, negqrtr_mask_626668)
    
    # Getting the type of 'small_mask' (line 180)
    small_mask_626670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 43), 'small_mask')
    # Applying the binary operator '|' (line 180)
    result_or__626671 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 41), '|', result_or__626669, small_mask_626670)
    
    # Applying the '~' unary operator (line 180)
    result_inv_626672 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 15), '~', result_or__626671)
    
    # Assigning a type to the variable 'reg_mask' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'reg_mask', result_inv_626672)
    
    # Assigning a Subscript to a Name (line 183):
    
    # Obtaining the type of the subscript
    # Getting the type of 'small_mask' (line 183)
    small_mask_626673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'small_mask')
    # Getting the type of 'lam' (line 183)
    lam_626674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'lam')
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___626675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), lam_626674, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_626676 = invoke(stypy.reporting.localization.Localization(__file__, 183, 12), getitem___626675, small_mask_626673)
    
    # Assigning a type to the variable 'small' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'small', subscript_call_result_626676)
    
    # Assigning a Subscript to a Name (line 184):
    
    # Obtaining the type of the subscript
    # Getting the type of 'reg_mask' (line 184)
    reg_mask_626677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 14), 'reg_mask')
    # Getting the type of 'lam' (line 184)
    lam_626678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 10), 'lam')
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___626679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 10), lam_626678, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_626680 = invoke(stypy.reporting.localization.Localization(__file__, 184, 10), getitem___626679, reg_mask_626677)
    
    # Assigning a type to the variable 'reg' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'reg', subscript_call_result_626680)
    
    # Assigning a Call to a Name (line 187):
    
    # Call to empty_like(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'lam' (line 187)
    lam_626683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 22), 'lam', False)
    # Processing the call keyword arguments (line 187)
    kwargs_626684 = {}
    # Getting the type of 'np' (line 187)
    np_626681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'np', False)
    # Obtaining the member 'empty_like' of a type (line 187)
    empty_like_626682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), np_626681, 'empty_like')
    # Calling empty_like(args, kwargs) (line 187)
    empty_like_call_result_626685 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), empty_like_626682, *[lam_626683], **kwargs_626684)
    
    # Assigning a type to the variable 'k' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'k', empty_like_call_result_626685)
    
    # Assigning a Attribute to a Subscript (line 188):
    # Getting the type of 'np' (line 188)
    np_626686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 18), 'np')
    # Obtaining the member 'nan' of a type (line 188)
    nan_626687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 18), np_626686, 'nan')
    # Getting the type of 'k' (line 188)
    k_626688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'k')
    # Getting the type of 'low_mask' (line 188)
    low_mask_626689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 6), 'low_mask')
    # Storing an element on a container (line 188)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 4), k_626688, (low_mask_626689, nan_626687))
    
    # Assigning a Attribute to a Subscript (line 189):
    # Getting the type of 'np' (line 189)
    np_626690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 22), 'np')
    # Obtaining the member 'inf' of a type (line 189)
    inf_626691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 22), np_626690, 'inf')
    # Getting the type of 'k' (line 189)
    k_626692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'k')
    # Getting the type of 'negqrtr_mask' (line 189)
    negqrtr_mask_626693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 6), 'negqrtr_mask')
    # Storing an element on a container (line 189)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 4), k_626692, (negqrtr_mask_626693, inf_626691))
    
    
    # Getting the type of 'small' (line 190)
    small_626694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 7), 'small')
    # Obtaining the member 'size' of a type (line 190)
    size_626695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 7), small_626694, 'size')
    int_626696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 20), 'int')
    # Applying the binary operator '>' (line 190)
    result_gt_626697 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 7), '>', size_626695, int_626696)
    
    # Testing the type of an if condition (line 190)
    if_condition_626698 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 4), result_gt_626697)
    # Assigning a type to the variable 'if_condition_626698' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'if_condition_626698', if_condition_626698)
    # SSA begins for if statement (line 190)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 191):
    
    # Call to _tukeylambda_kurt_p(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'small' (line 191)
    small_626700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 44), 'small', False)
    # Processing the call keyword arguments (line 191)
    kwargs_626701 = {}
    # Getting the type of '_tukeylambda_kurt_p' (line 191)
    _tukeylambda_kurt_p_626699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 24), '_tukeylambda_kurt_p', False)
    # Calling _tukeylambda_kurt_p(args, kwargs) (line 191)
    _tukeylambda_kurt_p_call_result_626702 = invoke(stypy.reporting.localization.Localization(__file__, 191, 24), _tukeylambda_kurt_p_626699, *[small_626700], **kwargs_626701)
    
    
    # Call to _tukeylambda_kurt_q(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'small' (line 191)
    small_626704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 73), 'small', False)
    # Processing the call keyword arguments (line 191)
    kwargs_626705 = {}
    # Getting the type of '_tukeylambda_kurt_q' (line 191)
    _tukeylambda_kurt_q_626703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 53), '_tukeylambda_kurt_q', False)
    # Calling _tukeylambda_kurt_q(args, kwargs) (line 191)
    _tukeylambda_kurt_q_call_result_626706 = invoke(stypy.reporting.localization.Localization(__file__, 191, 53), _tukeylambda_kurt_q_626703, *[small_626704], **kwargs_626705)
    
    # Applying the binary operator 'div' (line 191)
    result_div_626707 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 24), 'div', _tukeylambda_kurt_p_call_result_626702, _tukeylambda_kurt_q_call_result_626706)
    
    # Getting the type of 'k' (line 191)
    k_626708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'k')
    # Getting the type of 'small_mask' (line 191)
    small_mask_626709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 10), 'small_mask')
    # Storing an element on a container (line 191)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 8), k_626708, (small_mask_626709, result_div_626707))
    # SSA join for if statement (line 190)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'reg' (line 192)
    reg_626710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 7), 'reg')
    # Obtaining the member 'size' of a type (line 192)
    size_626711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 7), reg_626710, 'size')
    int_626712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 18), 'int')
    # Applying the binary operator '>' (line 192)
    result_gt_626713 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 7), '>', size_626711, int_626712)
    
    # Testing the type of an if condition (line 192)
    if_condition_626714 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 4), result_gt_626713)
    # Assigning a type to the variable 'if_condition_626714' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'if_condition_626714', if_condition_626714)
    # SSA begins for if statement (line 192)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 193):
    float_626715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 17), 'float')
    int_626716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 24), 'int')
    # Getting the type of 'reg' (line 193)
    reg_626717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 28), 'reg')
    # Applying the binary operator '*' (line 193)
    result_mul_626718 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 24), '*', int_626716, reg_626717)
    
    int_626719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 34), 'int')
    # Applying the binary operator '+' (line 193)
    result_add_626720 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 24), '+', result_mul_626718, int_626719)
    
    # Applying the binary operator 'div' (line 193)
    result_div_626721 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 17), 'div', float_626715, result_add_626720)
    
    int_626722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 39), 'int')
    
    # Call to beta(...): (line 193)
    # Processing the call arguments (line 193)
    int_626724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 48), 'int')
    # Getting the type of 'reg' (line 193)
    reg_626725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 52), 'reg', False)
    # Applying the binary operator '*' (line 193)
    result_mul_626726 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 48), '*', int_626724, reg_626725)
    
    int_626727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 58), 'int')
    # Applying the binary operator '+' (line 193)
    result_add_626728 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 48), '+', result_mul_626726, int_626727)
    
    # Getting the type of 'reg' (line 193)
    reg_626729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 61), 'reg', False)
    int_626730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 67), 'int')
    # Applying the binary operator '+' (line 193)
    result_add_626731 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 61), '+', reg_626729, int_626730)
    
    # Processing the call keyword arguments (line 193)
    kwargs_626732 = {}
    # Getting the type of 'beta' (line 193)
    beta_626723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 43), 'beta', False)
    # Calling beta(args, kwargs) (line 193)
    beta_call_result_626733 = invoke(stypy.reporting.localization.Localization(__file__, 193, 43), beta_626723, *[result_add_626728, result_add_626731], **kwargs_626732)
    
    # Applying the binary operator '*' (line 193)
    result_mul_626734 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 39), '*', int_626722, beta_call_result_626733)
    
    # Applying the binary operator '-' (line 193)
    result_sub_626735 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 17), '-', result_div_626721, result_mul_626734)
    
    int_626736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 17), 'int')
    
    # Call to beta(...): (line 194)
    # Processing the call arguments (line 194)
    int_626738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 26), 'int')
    # Getting the type of 'reg' (line 194)
    reg_626739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 30), 'reg', False)
    # Applying the binary operator '*' (line 194)
    result_mul_626740 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 26), '*', int_626738, reg_626739)
    
    int_626741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 36), 'int')
    # Applying the binary operator '+' (line 194)
    result_add_626742 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 26), '+', result_mul_626740, int_626741)
    
    int_626743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 39), 'int')
    # Getting the type of 'reg' (line 194)
    reg_626744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 43), 'reg', False)
    # Applying the binary operator '*' (line 194)
    result_mul_626745 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 39), '*', int_626743, reg_626744)
    
    int_626746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 49), 'int')
    # Applying the binary operator '+' (line 194)
    result_add_626747 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 39), '+', result_mul_626745, int_626746)
    
    # Processing the call keyword arguments (line 194)
    kwargs_626748 = {}
    # Getting the type of 'beta' (line 194)
    beta_626737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 21), 'beta', False)
    # Calling beta(args, kwargs) (line 194)
    beta_call_result_626749 = invoke(stypy.reporting.localization.Localization(__file__, 194, 21), beta_626737, *[result_add_626742, result_add_626747], **kwargs_626748)
    
    # Applying the binary operator '*' (line 194)
    result_mul_626750 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 17), '*', int_626736, beta_call_result_626749)
    
    # Applying the binary operator '+' (line 193)
    result_add_626751 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 70), '+', result_sub_626735, result_mul_626750)
    
    # Assigning a type to the variable 'numer' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'numer', result_add_626751)
    
    # Assigning a BinOp to a Name (line 195):
    int_626752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 16), 'int')
    float_626753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 21), 'float')
    int_626754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 26), 'int')
    # Getting the type of 'reg' (line 195)
    reg_626755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 30), 'reg')
    # Applying the binary operator '*' (line 195)
    result_mul_626756 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 26), '*', int_626754, reg_626755)
    
    int_626757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 36), 'int')
    # Applying the binary operator '+' (line 195)
    result_add_626758 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 26), '+', result_mul_626756, int_626757)
    
    # Applying the binary operator 'div' (line 195)
    result_div_626759 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 21), 'div', float_626753, result_add_626758)
    
    
    # Call to beta(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'reg' (line 195)
    reg_626761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 46), 'reg', False)
    int_626762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 52), 'int')
    # Applying the binary operator '+' (line 195)
    result_add_626763 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 46), '+', reg_626761, int_626762)
    
    # Getting the type of 'reg' (line 195)
    reg_626764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 55), 'reg', False)
    int_626765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 61), 'int')
    # Applying the binary operator '+' (line 195)
    result_add_626766 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 55), '+', reg_626764, int_626765)
    
    # Processing the call keyword arguments (line 195)
    kwargs_626767 = {}
    # Getting the type of 'beta' (line 195)
    beta_626760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 41), 'beta', False)
    # Calling beta(args, kwargs) (line 195)
    beta_call_result_626768 = invoke(stypy.reporting.localization.Localization(__file__, 195, 41), beta_626760, *[result_add_626763, result_add_626766], **kwargs_626767)
    
    # Applying the binary operator '-' (line 195)
    result_sub_626769 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 21), '-', result_div_626759, beta_call_result_626768)
    
    int_626770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 66), 'int')
    # Applying the binary operator '**' (line 195)
    result_pow_626771 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 20), '**', result_sub_626769, int_626770)
    
    # Applying the binary operator '*' (line 195)
    result_mul_626772 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 16), '*', int_626752, result_pow_626771)
    
    # Assigning a type to the variable 'denom' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'denom', result_mul_626772)
    
    # Assigning a BinOp to a Subscript (line 196):
    # Getting the type of 'numer' (line 196)
    numer_626773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 22), 'numer')
    # Getting the type of 'denom' (line 196)
    denom_626774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 30), 'denom')
    # Applying the binary operator 'div' (line 196)
    result_div_626775 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 22), 'div', numer_626773, denom_626774)
    
    int_626776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 38), 'int')
    # Applying the binary operator '-' (line 196)
    result_sub_626777 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 22), '-', result_div_626775, int_626776)
    
    # Getting the type of 'k' (line 196)
    k_626778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'k')
    # Getting the type of 'reg_mask' (line 196)
    reg_mask_626779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 10), 'reg_mask')
    # Storing an element on a container (line 196)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 8), k_626778, (reg_mask_626779, result_sub_626777))
    # SSA join for if statement (line 192)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Attribute (line 200):
    # Getting the type of 'shp' (line 200)
    shp_626780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 14), 'shp')
    # Getting the type of 'k' (line 200)
    k_626781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'k')
    # Setting the type of the member 'shape' of a type (line 200)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 4), k_626781, 'shape', shp_626780)
    # Getting the type of 'k' (line 201)
    k_626782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'k')
    # Assigning a type to the variable 'stypy_return_type' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type', k_626782)
    
    # ################# End of 'tukeylambda_kurtosis(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tukeylambda_kurtosis' in the type store
    # Getting the type of 'stypy_return_type' (line 149)
    stypy_return_type_626783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_626783)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tukeylambda_kurtosis'
    return stypy_return_type_626783

# Assigning a type to the variable 'tukeylambda_kurtosis' (line 149)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 0), 'tukeylambda_kurtosis', tukeylambda_kurtosis)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
