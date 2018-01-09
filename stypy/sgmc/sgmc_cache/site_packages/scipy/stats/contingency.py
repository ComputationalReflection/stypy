
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Some functions for working with contingency tables (i.e. cross tabulations).
2: '''
3: 
4: 
5: from __future__ import division, print_function, absolute_import
6: 
7: from functools import reduce
8: import numpy as np
9: from .stats import power_divergence
10: 
11: 
12: __all__ = ['margins', 'expected_freq', 'chi2_contingency']
13: 
14: 
15: def margins(a):
16:     '''Return a list of the marginal sums of the array `a`.
17: 
18:     Parameters
19:     ----------
20:     a : ndarray
21:         The array for which to compute the marginal sums.
22: 
23:     Returns
24:     -------
25:     margsums : list of ndarrays
26:         A list of length `a.ndim`.  `margsums[k]` is the result
27:         of summing `a` over all axes except `k`; it has the same
28:         number of dimensions as `a`, but the length of each axis
29:         except axis `k` will be 1.
30: 
31:     Examples
32:     --------
33:     >>> a = np.arange(12).reshape(2, 6)
34:     >>> a
35:     array([[ 0,  1,  2,  3,  4,  5],
36:            [ 6,  7,  8,  9, 10, 11]])
37:     >>> m0, m1 = margins(a)
38:     >>> m0
39:     array([[15],
40:            [51]])
41:     >>> m1
42:     array([[ 6,  8, 10, 12, 14, 16]])
43: 
44:     >>> b = np.arange(24).reshape(2,3,4)
45:     >>> m0, m1, m2 = margins(b)
46:     >>> m0
47:     array([[[ 66]],
48:            [[210]]])
49:     >>> m1
50:     array([[[ 60],
51:             [ 92],
52:             [124]]])
53:     >>> m2
54:     array([[[60, 66, 72, 78]]])
55:     '''
56:     margsums = []
57:     ranged = list(range(a.ndim))
58:     for k in ranged:
59:         marg = np.apply_over_axes(np.sum, a, [j for j in ranged if j != k])
60:         margsums.append(marg)
61:     return margsums
62: 
63: 
64: def expected_freq(observed):
65:     '''
66:     Compute the expected frequencies from a contingency table.
67: 
68:     Given an n-dimensional contingency table of observed frequencies,
69:     compute the expected frequencies for the table based on the marginal
70:     sums under the assumption that the groups associated with each
71:     dimension are independent.
72: 
73:     Parameters
74:     ----------
75:     observed : array_like
76:         The table of observed frequencies.  (While this function can handle
77:         a 1-D array, that case is trivial.  Generally `observed` is at
78:         least 2-D.)
79: 
80:     Returns
81:     -------
82:     expected : ndarray of float64
83:         The expected frequencies, based on the marginal sums of the table.
84:         Same shape as `observed`.
85: 
86:     Examples
87:     --------
88:     >>> observed = np.array([[10, 10, 20],[20, 20, 20]])
89:     >>> from scipy.stats import expected_freq
90:     >>> expected_freq(observed)
91:     array([[ 12.,  12.,  16.],
92:            [ 18.,  18.,  24.]])
93: 
94:     '''
95:     # Typically `observed` is an integer array. If `observed` has a large
96:     # number of dimensions or holds large values, some of the following
97:     # computations may overflow, so we first switch to floating point.
98:     observed = np.asarray(observed, dtype=np.float64)
99: 
100:     # Create a list of the marginal sums.
101:     margsums = margins(observed)
102: 
103:     # Create the array of expected frequencies.  The shapes of the
104:     # marginal sums returned by apply_over_axes() are just what we
105:     # need for broadcasting in the following product.
106:     d = observed.ndim
107:     expected = reduce(np.multiply, margsums) / observed.sum() ** (d - 1)
108:     return expected
109: 
110: 
111: def chi2_contingency(observed, correction=True, lambda_=None):
112:     '''Chi-square test of independence of variables in a contingency table.
113: 
114:     This function computes the chi-square statistic and p-value for the
115:     hypothesis test of independence of the observed frequencies in the
116:     contingency table [1]_ `observed`.  The expected frequencies are computed
117:     based on the marginal sums under the assumption of independence; see
118:     `scipy.stats.contingency.expected_freq`.  The number of degrees of
119:     freedom is (expressed using numpy functions and attributes)::
120: 
121:         dof = observed.size - sum(observed.shape) + observed.ndim - 1
122: 
123: 
124:     Parameters
125:     ----------
126:     observed : array_like
127:         The contingency table. The table contains the observed frequencies
128:         (i.e. number of occurrences) in each category.  In the two-dimensional
129:         case, the table is often described as an "R x C table".
130:     correction : bool, optional
131:         If True, *and* the degrees of freedom is 1, apply Yates' correction
132:         for continuity.  The effect of the correction is to adjust each
133:         observed value by 0.5 towards the corresponding expected value.
134:     lambda_ : float or str, optional.
135:         By default, the statistic computed in this test is Pearson's
136:         chi-squared statistic [2]_.  `lambda_` allows a statistic from the
137:         Cressie-Read power divergence family [3]_ to be used instead.  See
138:         `power_divergence` for details.
139: 
140:     Returns
141:     -------
142:     chi2 : float
143:         The test statistic.
144:     p : float
145:         The p-value of the test
146:     dof : int
147:         Degrees of freedom
148:     expected : ndarray, same shape as `observed`
149:         The expected frequencies, based on the marginal sums of the table.
150: 
151:     See Also
152:     --------
153:     contingency.expected_freq
154:     fisher_exact
155:     chisquare
156:     power_divergence
157: 
158:     Notes
159:     -----
160:     An often quoted guideline for the validity of this calculation is that
161:     the test should be used only if the observed and expected frequency in
162:     each cell is at least 5.
163: 
164:     This is a test for the independence of different categories of a
165:     population. The test is only meaningful when the dimension of
166:     `observed` is two or more.  Applying the test to a one-dimensional
167:     table will always result in `expected` equal to `observed` and a
168:     chi-square statistic equal to 0.
169: 
170:     This function does not handle masked arrays, because the calculation
171:     does not make sense with missing values.
172: 
173:     Like stats.chisquare, this function computes a chi-square statistic;
174:     the convenience this function provides is to figure out the expected
175:     frequencies and degrees of freedom from the given contingency table.
176:     If these were already known, and if the Yates' correction was not
177:     required, one could use stats.chisquare.  That is, if one calls::
178: 
179:         chi2, p, dof, ex = chi2_contingency(obs, correction=False)
180: 
181:     then the following is true::
182: 
183:         (chi2, p) == stats.chisquare(obs.ravel(), f_exp=ex.ravel(),
184:                                      ddof=obs.size - 1 - dof)
185: 
186:     The `lambda_` argument was added in version 0.13.0 of scipy.
187: 
188:     References
189:     ----------
190:     .. [1] "Contingency table", http://en.wikipedia.org/wiki/Contingency_table
191:     .. [2] "Pearson's chi-squared test",
192:            http://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
193:     .. [3] Cressie, N. and Read, T. R. C., "Multinomial Goodness-of-Fit
194:            Tests", J. Royal Stat. Soc. Series B, Vol. 46, No. 3 (1984),
195:            pp. 440-464.
196: 
197:     Examples
198:     --------
199:     A two-way example (2 x 3):
200: 
201:     >>> from scipy.stats import chi2_contingency
202:     >>> obs = np.array([[10, 10, 20], [20, 20, 20]])
203:     >>> chi2_contingency(obs)
204:     (2.7777777777777777,
205:      0.24935220877729619,
206:      2,
207:      array([[ 12.,  12.,  16.],
208:             [ 18.,  18.,  24.]]))
209: 
210:     Perform the test using the log-likelihood ratio (i.e. the "G-test")
211:     instead of Pearson's chi-squared statistic.
212: 
213:     >>> g, p, dof, expctd = chi2_contingency(obs, lambda_="log-likelihood")
214:     >>> g, p
215:     (2.7688587616781319, 0.25046668010954165)
216: 
217:     A four-way example (2 x 2 x 2 x 2):
218: 
219:     >>> obs = np.array(
220:     ...     [[[[12, 17],
221:     ...        [11, 16]],
222:     ...       [[11, 12],
223:     ...        [15, 16]]],
224:     ...      [[[23, 15],
225:     ...        [30, 22]],
226:     ...       [[14, 17],
227:     ...        [15, 16]]]])
228:     >>> chi2_contingency(obs)
229:     (8.7584514426741897,
230:      0.64417725029295503,
231:      11,
232:      array([[[[ 14.15462386,  14.15462386],
233:               [ 16.49423111,  16.49423111]],
234:              [[ 11.2461395 ,  11.2461395 ],
235:               [ 13.10500554,  13.10500554]]],
236:             [[[ 19.5591166 ,  19.5591166 ],
237:               [ 22.79202844,  22.79202844]],
238:              [[ 15.54012004,  15.54012004],
239:               [ 18.10873492,  18.10873492]]]]))
240:     '''
241:     observed = np.asarray(observed)
242:     if np.any(observed < 0):
243:         raise ValueError("All values in `observed` must be nonnegative.")
244:     if observed.size == 0:
245:         raise ValueError("No data; `observed` has size 0.")
246: 
247:     expected = expected_freq(observed)
248:     if np.any(expected == 0):
249:         # Include one of the positions where expected is zero in
250:         # the exception message.
251:         zeropos = list(zip(*np.where(expected == 0)))[0]
252:         raise ValueError("The internally computed table of expected "
253:                          "frequencies has a zero element at %s." % (zeropos,))
254: 
255:     # The degrees of freedom
256:     dof = expected.size - sum(expected.shape) + expected.ndim - 1
257: 
258:     if dof == 0:
259:         # Degenerate case; this occurs when `observed` is 1D (or, more
260:         # generally, when it has only one nontrivial dimension).  In this
261:         # case, we also have observed == expected, so chi2 is 0.
262:         chi2 = 0.0
263:         p = 1.0
264:     else:
265:         if dof == 1 and correction:
266:             # Adjust `observed` according to Yates' correction for continuity.
267:             observed = observed + 0.5 * np.sign(expected - observed)
268: 
269:         chi2, p = power_divergence(observed, expected,
270:                                    ddof=observed.size - 1 - dof, axis=None,
271:                                    lambda_=lambda_)
272: 
273:     return chi2, p, dof, expected
274: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_564831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', 'Some functions for working with contingency tables (i.e. cross tabulations).\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from functools import reduce' statement (line 7)
try:
    from functools import reduce

except:
    reduce = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'functools', None, module_type_store, ['reduce'], [reduce])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_564832 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_564832) is not StypyTypeError):

    if (import_564832 != 'pyd_module'):
        __import__(import_564832)
        sys_modules_564833 = sys.modules[import_564832]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_564833.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_564832)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.stats.stats import power_divergence' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_564834 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.stats.stats')

if (type(import_564834) is not StypyTypeError):

    if (import_564834 != 'pyd_module'):
        __import__(import_564834)
        sys_modules_564835 = sys.modules[import_564834]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.stats.stats', sys_modules_564835.module_type_store, module_type_store, ['power_divergence'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_564835, sys_modules_564835.module_type_store, module_type_store)
    else:
        from scipy.stats.stats import power_divergence

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.stats.stats', None, module_type_store, ['power_divergence'], [power_divergence])

else:
    # Assigning a type to the variable 'scipy.stats.stats' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.stats.stats', import_564834)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')


# Assigning a List to a Name (line 12):

# Assigning a List to a Name (line 12):
__all__ = ['margins', 'expected_freq', 'chi2_contingency']
module_type_store.set_exportable_members(['margins', 'expected_freq', 'chi2_contingency'])

# Obtaining an instance of the builtin type 'list' (line 12)
list_564836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_564837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'margins')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_564836, str_564837)
# Adding element type (line 12)
str_564838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 22), 'str', 'expected_freq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_564836, str_564838)
# Adding element type (line 12)
str_564839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 39), 'str', 'chi2_contingency')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_564836, str_564839)

# Assigning a type to the variable '__all__' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '__all__', list_564836)

@norecursion
def margins(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'margins'
    module_type_store = module_type_store.open_function_context('margins', 15, 0, False)
    
    # Passed parameters checking function
    margins.stypy_localization = localization
    margins.stypy_type_of_self = None
    margins.stypy_type_store = module_type_store
    margins.stypy_function_name = 'margins'
    margins.stypy_param_names_list = ['a']
    margins.stypy_varargs_param_name = None
    margins.stypy_kwargs_param_name = None
    margins.stypy_call_defaults = defaults
    margins.stypy_call_varargs = varargs
    margins.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'margins', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'margins', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'margins(...)' code ##################

    str_564840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, (-1)), 'str', 'Return a list of the marginal sums of the array `a`.\n\n    Parameters\n    ----------\n    a : ndarray\n        The array for which to compute the marginal sums.\n\n    Returns\n    -------\n    margsums : list of ndarrays\n        A list of length `a.ndim`.  `margsums[k]` is the result\n        of summing `a` over all axes except `k`; it has the same\n        number of dimensions as `a`, but the length of each axis\n        except axis `k` will be 1.\n\n    Examples\n    --------\n    >>> a = np.arange(12).reshape(2, 6)\n    >>> a\n    array([[ 0,  1,  2,  3,  4,  5],\n           [ 6,  7,  8,  9, 10, 11]])\n    >>> m0, m1 = margins(a)\n    >>> m0\n    array([[15],\n           [51]])\n    >>> m1\n    array([[ 6,  8, 10, 12, 14, 16]])\n\n    >>> b = np.arange(24).reshape(2,3,4)\n    >>> m0, m1, m2 = margins(b)\n    >>> m0\n    array([[[ 66]],\n           [[210]]])\n    >>> m1\n    array([[[ 60],\n            [ 92],\n            [124]]])\n    >>> m2\n    array([[[60, 66, 72, 78]]])\n    ')
    
    # Assigning a List to a Name (line 56):
    
    # Assigning a List to a Name (line 56):
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_564841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    
    # Assigning a type to the variable 'margsums' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'margsums', list_564841)
    
    # Assigning a Call to a Name (line 57):
    
    # Assigning a Call to a Name (line 57):
    
    # Call to list(...): (line 57)
    # Processing the call arguments (line 57)
    
    # Call to range(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'a' (line 57)
    a_564844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 24), 'a', False)
    # Obtaining the member 'ndim' of a type (line 57)
    ndim_564845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 24), a_564844, 'ndim')
    # Processing the call keyword arguments (line 57)
    kwargs_564846 = {}
    # Getting the type of 'range' (line 57)
    range_564843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'range', False)
    # Calling range(args, kwargs) (line 57)
    range_call_result_564847 = invoke(stypy.reporting.localization.Localization(__file__, 57, 18), range_564843, *[ndim_564845], **kwargs_564846)
    
    # Processing the call keyword arguments (line 57)
    kwargs_564848 = {}
    # Getting the type of 'list' (line 57)
    list_564842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 13), 'list', False)
    # Calling list(args, kwargs) (line 57)
    list_call_result_564849 = invoke(stypy.reporting.localization.Localization(__file__, 57, 13), list_564842, *[range_call_result_564847], **kwargs_564848)
    
    # Assigning a type to the variable 'ranged' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'ranged', list_call_result_564849)
    
    # Getting the type of 'ranged' (line 58)
    ranged_564850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 13), 'ranged')
    # Testing the type of a for loop iterable (line 58)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 58, 4), ranged_564850)
    # Getting the type of the for loop variable (line 58)
    for_loop_var_564851 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 58, 4), ranged_564850)
    # Assigning a type to the variable 'k' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'k', for_loop_var_564851)
    # SSA begins for a for statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 59):
    
    # Assigning a Call to a Name (line 59):
    
    # Call to apply_over_axes(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'np' (line 59)
    np_564854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 34), 'np', False)
    # Obtaining the member 'sum' of a type (line 59)
    sum_564855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 34), np_564854, 'sum')
    # Getting the type of 'a' (line 59)
    a_564856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 42), 'a', False)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'ranged' (line 59)
    ranged_564861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 57), 'ranged', False)
    comprehension_564862 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 46), ranged_564861)
    # Assigning a type to the variable 'j' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 46), 'j', comprehension_564862)
    
    # Getting the type of 'j' (line 59)
    j_564858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 67), 'j', False)
    # Getting the type of 'k' (line 59)
    k_564859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 72), 'k', False)
    # Applying the binary operator '!=' (line 59)
    result_ne_564860 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 67), '!=', j_564858, k_564859)
    
    # Getting the type of 'j' (line 59)
    j_564857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 46), 'j', False)
    list_564863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 46), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 46), list_564863, j_564857)
    # Processing the call keyword arguments (line 59)
    kwargs_564864 = {}
    # Getting the type of 'np' (line 59)
    np_564852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'np', False)
    # Obtaining the member 'apply_over_axes' of a type (line 59)
    apply_over_axes_564853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 15), np_564852, 'apply_over_axes')
    # Calling apply_over_axes(args, kwargs) (line 59)
    apply_over_axes_call_result_564865 = invoke(stypy.reporting.localization.Localization(__file__, 59, 15), apply_over_axes_564853, *[sum_564855, a_564856, list_564863], **kwargs_564864)
    
    # Assigning a type to the variable 'marg' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'marg', apply_over_axes_call_result_564865)
    
    # Call to append(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'marg' (line 60)
    marg_564868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 24), 'marg', False)
    # Processing the call keyword arguments (line 60)
    kwargs_564869 = {}
    # Getting the type of 'margsums' (line 60)
    margsums_564866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'margsums', False)
    # Obtaining the member 'append' of a type (line 60)
    append_564867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), margsums_564866, 'append')
    # Calling append(args, kwargs) (line 60)
    append_call_result_564870 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), append_564867, *[marg_564868], **kwargs_564869)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'margsums' (line 61)
    margsums_564871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'margsums')
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type', margsums_564871)
    
    # ################# End of 'margins(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'margins' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_564872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_564872)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'margins'
    return stypy_return_type_564872

# Assigning a type to the variable 'margins' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'margins', margins)

@norecursion
def expected_freq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'expected_freq'
    module_type_store = module_type_store.open_function_context('expected_freq', 64, 0, False)
    
    # Passed parameters checking function
    expected_freq.stypy_localization = localization
    expected_freq.stypy_type_of_self = None
    expected_freq.stypy_type_store = module_type_store
    expected_freq.stypy_function_name = 'expected_freq'
    expected_freq.stypy_param_names_list = ['observed']
    expected_freq.stypy_varargs_param_name = None
    expected_freq.stypy_kwargs_param_name = None
    expected_freq.stypy_call_defaults = defaults
    expected_freq.stypy_call_varargs = varargs
    expected_freq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'expected_freq', ['observed'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'expected_freq', localization, ['observed'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'expected_freq(...)' code ##################

    str_564873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, (-1)), 'str', '\n    Compute the expected frequencies from a contingency table.\n\n    Given an n-dimensional contingency table of observed frequencies,\n    compute the expected frequencies for the table based on the marginal\n    sums under the assumption that the groups associated with each\n    dimension are independent.\n\n    Parameters\n    ----------\n    observed : array_like\n        The table of observed frequencies.  (While this function can handle\n        a 1-D array, that case is trivial.  Generally `observed` is at\n        least 2-D.)\n\n    Returns\n    -------\n    expected : ndarray of float64\n        The expected frequencies, based on the marginal sums of the table.\n        Same shape as `observed`.\n\n    Examples\n    --------\n    >>> observed = np.array([[10, 10, 20],[20, 20, 20]])\n    >>> from scipy.stats import expected_freq\n    >>> expected_freq(observed)\n    array([[ 12.,  12.,  16.],\n           [ 18.,  18.,  24.]])\n\n    ')
    
    # Assigning a Call to a Name (line 98):
    
    # Assigning a Call to a Name (line 98):
    
    # Call to asarray(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'observed' (line 98)
    observed_564876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 'observed', False)
    # Processing the call keyword arguments (line 98)
    # Getting the type of 'np' (line 98)
    np_564877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 42), 'np', False)
    # Obtaining the member 'float64' of a type (line 98)
    float64_564878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 42), np_564877, 'float64')
    keyword_564879 = float64_564878
    kwargs_564880 = {'dtype': keyword_564879}
    # Getting the type of 'np' (line 98)
    np_564874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'np', False)
    # Obtaining the member 'asarray' of a type (line 98)
    asarray_564875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 15), np_564874, 'asarray')
    # Calling asarray(args, kwargs) (line 98)
    asarray_call_result_564881 = invoke(stypy.reporting.localization.Localization(__file__, 98, 15), asarray_564875, *[observed_564876], **kwargs_564880)
    
    # Assigning a type to the variable 'observed' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'observed', asarray_call_result_564881)
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to margins(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'observed' (line 101)
    observed_564883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'observed', False)
    # Processing the call keyword arguments (line 101)
    kwargs_564884 = {}
    # Getting the type of 'margins' (line 101)
    margins_564882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'margins', False)
    # Calling margins(args, kwargs) (line 101)
    margins_call_result_564885 = invoke(stypy.reporting.localization.Localization(__file__, 101, 15), margins_564882, *[observed_564883], **kwargs_564884)
    
    # Assigning a type to the variable 'margsums' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'margsums', margins_call_result_564885)
    
    # Assigning a Attribute to a Name (line 106):
    
    # Assigning a Attribute to a Name (line 106):
    # Getting the type of 'observed' (line 106)
    observed_564886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'observed')
    # Obtaining the member 'ndim' of a type (line 106)
    ndim_564887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), observed_564886, 'ndim')
    # Assigning a type to the variable 'd' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'd', ndim_564887)
    
    # Assigning a BinOp to a Name (line 107):
    
    # Assigning a BinOp to a Name (line 107):
    
    # Call to reduce(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'np' (line 107)
    np_564889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 22), 'np', False)
    # Obtaining the member 'multiply' of a type (line 107)
    multiply_564890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 22), np_564889, 'multiply')
    # Getting the type of 'margsums' (line 107)
    margsums_564891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 35), 'margsums', False)
    # Processing the call keyword arguments (line 107)
    kwargs_564892 = {}
    # Getting the type of 'reduce' (line 107)
    reduce_564888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), 'reduce', False)
    # Calling reduce(args, kwargs) (line 107)
    reduce_call_result_564893 = invoke(stypy.reporting.localization.Localization(__file__, 107, 15), reduce_564888, *[multiply_564890, margsums_564891], **kwargs_564892)
    
    
    # Call to sum(...): (line 107)
    # Processing the call keyword arguments (line 107)
    kwargs_564896 = {}
    # Getting the type of 'observed' (line 107)
    observed_564894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 47), 'observed', False)
    # Obtaining the member 'sum' of a type (line 107)
    sum_564895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 47), observed_564894, 'sum')
    # Calling sum(args, kwargs) (line 107)
    sum_call_result_564897 = invoke(stypy.reporting.localization.Localization(__file__, 107, 47), sum_564895, *[], **kwargs_564896)
    
    # Getting the type of 'd' (line 107)
    d_564898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 66), 'd')
    int_564899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 70), 'int')
    # Applying the binary operator '-' (line 107)
    result_sub_564900 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 66), '-', d_564898, int_564899)
    
    # Applying the binary operator '**' (line 107)
    result_pow_564901 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 47), '**', sum_call_result_564897, result_sub_564900)
    
    # Applying the binary operator 'div' (line 107)
    result_div_564902 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 15), 'div', reduce_call_result_564893, result_pow_564901)
    
    # Assigning a type to the variable 'expected' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'expected', result_div_564902)
    # Getting the type of 'expected' (line 108)
    expected_564903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'expected')
    # Assigning a type to the variable 'stypy_return_type' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type', expected_564903)
    
    # ################# End of 'expected_freq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'expected_freq' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_564904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_564904)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'expected_freq'
    return stypy_return_type_564904

# Assigning a type to the variable 'expected_freq' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'expected_freq', expected_freq)

@norecursion
def chi2_contingency(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 111)
    True_564905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 42), 'True')
    # Getting the type of 'None' (line 111)
    None_564906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 56), 'None')
    defaults = [True_564905, None_564906]
    # Create a new context for function 'chi2_contingency'
    module_type_store = module_type_store.open_function_context('chi2_contingency', 111, 0, False)
    
    # Passed parameters checking function
    chi2_contingency.stypy_localization = localization
    chi2_contingency.stypy_type_of_self = None
    chi2_contingency.stypy_type_store = module_type_store
    chi2_contingency.stypy_function_name = 'chi2_contingency'
    chi2_contingency.stypy_param_names_list = ['observed', 'correction', 'lambda_']
    chi2_contingency.stypy_varargs_param_name = None
    chi2_contingency.stypy_kwargs_param_name = None
    chi2_contingency.stypy_call_defaults = defaults
    chi2_contingency.stypy_call_varargs = varargs
    chi2_contingency.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chi2_contingency', ['observed', 'correction', 'lambda_'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chi2_contingency', localization, ['observed', 'correction', 'lambda_'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chi2_contingency(...)' code ##################

    str_564907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, (-1)), 'str', 'Chi-square test of independence of variables in a contingency table.\n\n    This function computes the chi-square statistic and p-value for the\n    hypothesis test of independence of the observed frequencies in the\n    contingency table [1]_ `observed`.  The expected frequencies are computed\n    based on the marginal sums under the assumption of independence; see\n    `scipy.stats.contingency.expected_freq`.  The number of degrees of\n    freedom is (expressed using numpy functions and attributes)::\n\n        dof = observed.size - sum(observed.shape) + observed.ndim - 1\n\n\n    Parameters\n    ----------\n    observed : array_like\n        The contingency table. The table contains the observed frequencies\n        (i.e. number of occurrences) in each category.  In the two-dimensional\n        case, the table is often described as an "R x C table".\n    correction : bool, optional\n        If True, *and* the degrees of freedom is 1, apply Yates\' correction\n        for continuity.  The effect of the correction is to adjust each\n        observed value by 0.5 towards the corresponding expected value.\n    lambda_ : float or str, optional.\n        By default, the statistic computed in this test is Pearson\'s\n        chi-squared statistic [2]_.  `lambda_` allows a statistic from the\n        Cressie-Read power divergence family [3]_ to be used instead.  See\n        `power_divergence` for details.\n\n    Returns\n    -------\n    chi2 : float\n        The test statistic.\n    p : float\n        The p-value of the test\n    dof : int\n        Degrees of freedom\n    expected : ndarray, same shape as `observed`\n        The expected frequencies, based on the marginal sums of the table.\n\n    See Also\n    --------\n    contingency.expected_freq\n    fisher_exact\n    chisquare\n    power_divergence\n\n    Notes\n    -----\n    An often quoted guideline for the validity of this calculation is that\n    the test should be used only if the observed and expected frequency in\n    each cell is at least 5.\n\n    This is a test for the independence of different categories of a\n    population. The test is only meaningful when the dimension of\n    `observed` is two or more.  Applying the test to a one-dimensional\n    table will always result in `expected` equal to `observed` and a\n    chi-square statistic equal to 0.\n\n    This function does not handle masked arrays, because the calculation\n    does not make sense with missing values.\n\n    Like stats.chisquare, this function computes a chi-square statistic;\n    the convenience this function provides is to figure out the expected\n    frequencies and degrees of freedom from the given contingency table.\n    If these were already known, and if the Yates\' correction was not\n    required, one could use stats.chisquare.  That is, if one calls::\n\n        chi2, p, dof, ex = chi2_contingency(obs, correction=False)\n\n    then the following is true::\n\n        (chi2, p) == stats.chisquare(obs.ravel(), f_exp=ex.ravel(),\n                                     ddof=obs.size - 1 - dof)\n\n    The `lambda_` argument was added in version 0.13.0 of scipy.\n\n    References\n    ----------\n    .. [1] "Contingency table", http://en.wikipedia.org/wiki/Contingency_table\n    .. [2] "Pearson\'s chi-squared test",\n           http://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test\n    .. [3] Cressie, N. and Read, T. R. C., "Multinomial Goodness-of-Fit\n           Tests", J. Royal Stat. Soc. Series B, Vol. 46, No. 3 (1984),\n           pp. 440-464.\n\n    Examples\n    --------\n    A two-way example (2 x 3):\n\n    >>> from scipy.stats import chi2_contingency\n    >>> obs = np.array([[10, 10, 20], [20, 20, 20]])\n    >>> chi2_contingency(obs)\n    (2.7777777777777777,\n     0.24935220877729619,\n     2,\n     array([[ 12.,  12.,  16.],\n            [ 18.,  18.,  24.]]))\n\n    Perform the test using the log-likelihood ratio (i.e. the "G-test")\n    instead of Pearson\'s chi-squared statistic.\n\n    >>> g, p, dof, expctd = chi2_contingency(obs, lambda_="log-likelihood")\n    >>> g, p\n    (2.7688587616781319, 0.25046668010954165)\n\n    A four-way example (2 x 2 x 2 x 2):\n\n    >>> obs = np.array(\n    ...     [[[[12, 17],\n    ...        [11, 16]],\n    ...       [[11, 12],\n    ...        [15, 16]]],\n    ...      [[[23, 15],\n    ...        [30, 22]],\n    ...       [[14, 17],\n    ...        [15, 16]]]])\n    >>> chi2_contingency(obs)\n    (8.7584514426741897,\n     0.64417725029295503,\n     11,\n     array([[[[ 14.15462386,  14.15462386],\n              [ 16.49423111,  16.49423111]],\n             [[ 11.2461395 ,  11.2461395 ],\n              [ 13.10500554,  13.10500554]]],\n            [[[ 19.5591166 ,  19.5591166 ],\n              [ 22.79202844,  22.79202844]],\n             [[ 15.54012004,  15.54012004],\n              [ 18.10873492,  18.10873492]]]]))\n    ')
    
    # Assigning a Call to a Name (line 241):
    
    # Assigning a Call to a Name (line 241):
    
    # Call to asarray(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'observed' (line 241)
    observed_564910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 26), 'observed', False)
    # Processing the call keyword arguments (line 241)
    kwargs_564911 = {}
    # Getting the type of 'np' (line 241)
    np_564908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 15), 'np', False)
    # Obtaining the member 'asarray' of a type (line 241)
    asarray_564909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 15), np_564908, 'asarray')
    # Calling asarray(args, kwargs) (line 241)
    asarray_call_result_564912 = invoke(stypy.reporting.localization.Localization(__file__, 241, 15), asarray_564909, *[observed_564910], **kwargs_564911)
    
    # Assigning a type to the variable 'observed' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 4), 'observed', asarray_call_result_564912)
    
    
    # Call to any(...): (line 242)
    # Processing the call arguments (line 242)
    
    # Getting the type of 'observed' (line 242)
    observed_564915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 14), 'observed', False)
    int_564916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 25), 'int')
    # Applying the binary operator '<' (line 242)
    result_lt_564917 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 14), '<', observed_564915, int_564916)
    
    # Processing the call keyword arguments (line 242)
    kwargs_564918 = {}
    # Getting the type of 'np' (line 242)
    np_564913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 242)
    any_564914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 7), np_564913, 'any')
    # Calling any(args, kwargs) (line 242)
    any_call_result_564919 = invoke(stypy.reporting.localization.Localization(__file__, 242, 7), any_564914, *[result_lt_564917], **kwargs_564918)
    
    # Testing the type of an if condition (line 242)
    if_condition_564920 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 4), any_call_result_564919)
    # Assigning a type to the variable 'if_condition_564920' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'if_condition_564920', if_condition_564920)
    # SSA begins for if statement (line 242)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 243)
    # Processing the call arguments (line 243)
    str_564922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 25), 'str', 'All values in `observed` must be nonnegative.')
    # Processing the call keyword arguments (line 243)
    kwargs_564923 = {}
    # Getting the type of 'ValueError' (line 243)
    ValueError_564921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 243)
    ValueError_call_result_564924 = invoke(stypy.reporting.localization.Localization(__file__, 243, 14), ValueError_564921, *[str_564922], **kwargs_564923)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 243, 8), ValueError_call_result_564924, 'raise parameter', BaseException)
    # SSA join for if statement (line 242)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'observed' (line 244)
    observed_564925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 7), 'observed')
    # Obtaining the member 'size' of a type (line 244)
    size_564926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 7), observed_564925, 'size')
    int_564927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 24), 'int')
    # Applying the binary operator '==' (line 244)
    result_eq_564928 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 7), '==', size_564926, int_564927)
    
    # Testing the type of an if condition (line 244)
    if_condition_564929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 244, 4), result_eq_564928)
    # Assigning a type to the variable 'if_condition_564929' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'if_condition_564929', if_condition_564929)
    # SSA begins for if statement (line 244)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 245)
    # Processing the call arguments (line 245)
    str_564931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 25), 'str', 'No data; `observed` has size 0.')
    # Processing the call keyword arguments (line 245)
    kwargs_564932 = {}
    # Getting the type of 'ValueError' (line 245)
    ValueError_564930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 245)
    ValueError_call_result_564933 = invoke(stypy.reporting.localization.Localization(__file__, 245, 14), ValueError_564930, *[str_564931], **kwargs_564932)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 245, 8), ValueError_call_result_564933, 'raise parameter', BaseException)
    # SSA join for if statement (line 244)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 247):
    
    # Assigning a Call to a Name (line 247):
    
    # Call to expected_freq(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'observed' (line 247)
    observed_564935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 29), 'observed', False)
    # Processing the call keyword arguments (line 247)
    kwargs_564936 = {}
    # Getting the type of 'expected_freq' (line 247)
    expected_freq_564934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 15), 'expected_freq', False)
    # Calling expected_freq(args, kwargs) (line 247)
    expected_freq_call_result_564937 = invoke(stypy.reporting.localization.Localization(__file__, 247, 15), expected_freq_564934, *[observed_564935], **kwargs_564936)
    
    # Assigning a type to the variable 'expected' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'expected', expected_freq_call_result_564937)
    
    
    # Call to any(...): (line 248)
    # Processing the call arguments (line 248)
    
    # Getting the type of 'expected' (line 248)
    expected_564940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 14), 'expected', False)
    int_564941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 26), 'int')
    # Applying the binary operator '==' (line 248)
    result_eq_564942 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 14), '==', expected_564940, int_564941)
    
    # Processing the call keyword arguments (line 248)
    kwargs_564943 = {}
    # Getting the type of 'np' (line 248)
    np_564938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 7), 'np', False)
    # Obtaining the member 'any' of a type (line 248)
    any_564939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 7), np_564938, 'any')
    # Calling any(args, kwargs) (line 248)
    any_call_result_564944 = invoke(stypy.reporting.localization.Localization(__file__, 248, 7), any_564939, *[result_eq_564942], **kwargs_564943)
    
    # Testing the type of an if condition (line 248)
    if_condition_564945 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 248, 4), any_call_result_564944)
    # Assigning a type to the variable 'if_condition_564945' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'if_condition_564945', if_condition_564945)
    # SSA begins for if statement (line 248)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 251):
    
    # Assigning a Subscript to a Name (line 251):
    
    # Obtaining the type of the subscript
    int_564946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 54), 'int')
    
    # Call to list(...): (line 251)
    # Processing the call arguments (line 251)
    
    # Call to zip(...): (line 251)
    
    # Call to where(...): (line 251)
    # Processing the call arguments (line 251)
    
    # Getting the type of 'expected' (line 251)
    expected_564951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 37), 'expected', False)
    int_564952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 49), 'int')
    # Applying the binary operator '==' (line 251)
    result_eq_564953 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 37), '==', expected_564951, int_564952)
    
    # Processing the call keyword arguments (line 251)
    kwargs_564954 = {}
    # Getting the type of 'np' (line 251)
    np_564949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 28), 'np', False)
    # Obtaining the member 'where' of a type (line 251)
    where_564950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 28), np_564949, 'where')
    # Calling where(args, kwargs) (line 251)
    where_call_result_564955 = invoke(stypy.reporting.localization.Localization(__file__, 251, 28), where_564950, *[result_eq_564953], **kwargs_564954)
    
    # Processing the call keyword arguments (line 251)
    kwargs_564956 = {}
    # Getting the type of 'zip' (line 251)
    zip_564948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 23), 'zip', False)
    # Calling zip(args, kwargs) (line 251)
    zip_call_result_564957 = invoke(stypy.reporting.localization.Localization(__file__, 251, 23), zip_564948, *[where_call_result_564955], **kwargs_564956)
    
    # Processing the call keyword arguments (line 251)
    kwargs_564958 = {}
    # Getting the type of 'list' (line 251)
    list_564947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 18), 'list', False)
    # Calling list(args, kwargs) (line 251)
    list_call_result_564959 = invoke(stypy.reporting.localization.Localization(__file__, 251, 18), list_564947, *[zip_call_result_564957], **kwargs_564958)
    
    # Obtaining the member '__getitem__' of a type (line 251)
    getitem___564960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 18), list_call_result_564959, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 251)
    subscript_call_result_564961 = invoke(stypy.reporting.localization.Localization(__file__, 251, 18), getitem___564960, int_564946)
    
    # Assigning a type to the variable 'zeropos' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'zeropos', subscript_call_result_564961)
    
    # Call to ValueError(...): (line 252)
    # Processing the call arguments (line 252)
    str_564963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 25), 'str', 'The internally computed table of expected frequencies has a zero element at %s.')
    
    # Obtaining an instance of the builtin type 'tuple' (line 253)
    tuple_564964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 68), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 253)
    # Adding element type (line 253)
    # Getting the type of 'zeropos' (line 253)
    zeropos_564965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 68), 'zeropos', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 68), tuple_564964, zeropos_564965)
    
    # Applying the binary operator '%' (line 252)
    result_mod_564966 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 25), '%', str_564963, tuple_564964)
    
    # Processing the call keyword arguments (line 252)
    kwargs_564967 = {}
    # Getting the type of 'ValueError' (line 252)
    ValueError_564962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 252)
    ValueError_call_result_564968 = invoke(stypy.reporting.localization.Localization(__file__, 252, 14), ValueError_564962, *[result_mod_564966], **kwargs_564967)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 252, 8), ValueError_call_result_564968, 'raise parameter', BaseException)
    # SSA join for if statement (line 248)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 256):
    
    # Assigning a BinOp to a Name (line 256):
    # Getting the type of 'expected' (line 256)
    expected_564969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 10), 'expected')
    # Obtaining the member 'size' of a type (line 256)
    size_564970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 10), expected_564969, 'size')
    
    # Call to sum(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'expected' (line 256)
    expected_564972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 30), 'expected', False)
    # Obtaining the member 'shape' of a type (line 256)
    shape_564973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 30), expected_564972, 'shape')
    # Processing the call keyword arguments (line 256)
    kwargs_564974 = {}
    # Getting the type of 'sum' (line 256)
    sum_564971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 26), 'sum', False)
    # Calling sum(args, kwargs) (line 256)
    sum_call_result_564975 = invoke(stypy.reporting.localization.Localization(__file__, 256, 26), sum_564971, *[shape_564973], **kwargs_564974)
    
    # Applying the binary operator '-' (line 256)
    result_sub_564976 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 10), '-', size_564970, sum_call_result_564975)
    
    # Getting the type of 'expected' (line 256)
    expected_564977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 48), 'expected')
    # Obtaining the member 'ndim' of a type (line 256)
    ndim_564978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 48), expected_564977, 'ndim')
    # Applying the binary operator '+' (line 256)
    result_add_564979 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 46), '+', result_sub_564976, ndim_564978)
    
    int_564980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 64), 'int')
    # Applying the binary operator '-' (line 256)
    result_sub_564981 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 62), '-', result_add_564979, int_564980)
    
    # Assigning a type to the variable 'dof' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'dof', result_sub_564981)
    
    
    # Getting the type of 'dof' (line 258)
    dof_564982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 7), 'dof')
    int_564983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 14), 'int')
    # Applying the binary operator '==' (line 258)
    result_eq_564984 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 7), '==', dof_564982, int_564983)
    
    # Testing the type of an if condition (line 258)
    if_condition_564985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 4), result_eq_564984)
    # Assigning a type to the variable 'if_condition_564985' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'if_condition_564985', if_condition_564985)
    # SSA begins for if statement (line 258)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 262):
    
    # Assigning a Num to a Name (line 262):
    float_564986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 15), 'float')
    # Assigning a type to the variable 'chi2' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'chi2', float_564986)
    
    # Assigning a Num to a Name (line 263):
    
    # Assigning a Num to a Name (line 263):
    float_564987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 12), 'float')
    # Assigning a type to the variable 'p' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'p', float_564987)
    # SSA branch for the else part of an if statement (line 258)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dof' (line 265)
    dof_564988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 11), 'dof')
    int_564989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 18), 'int')
    # Applying the binary operator '==' (line 265)
    result_eq_564990 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 11), '==', dof_564988, int_564989)
    
    # Getting the type of 'correction' (line 265)
    correction_564991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 24), 'correction')
    # Applying the binary operator 'and' (line 265)
    result_and_keyword_564992 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 11), 'and', result_eq_564990, correction_564991)
    
    # Testing the type of an if condition (line 265)
    if_condition_564993 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 8), result_and_keyword_564992)
    # Assigning a type to the variable 'if_condition_564993' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'if_condition_564993', if_condition_564993)
    # SSA begins for if statement (line 265)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 267):
    
    # Assigning a BinOp to a Name (line 267):
    # Getting the type of 'observed' (line 267)
    observed_564994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 23), 'observed')
    float_564995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 34), 'float')
    
    # Call to sign(...): (line 267)
    # Processing the call arguments (line 267)
    # Getting the type of 'expected' (line 267)
    expected_564998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 48), 'expected', False)
    # Getting the type of 'observed' (line 267)
    observed_564999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 59), 'observed', False)
    # Applying the binary operator '-' (line 267)
    result_sub_565000 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 48), '-', expected_564998, observed_564999)
    
    # Processing the call keyword arguments (line 267)
    kwargs_565001 = {}
    # Getting the type of 'np' (line 267)
    np_564996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 40), 'np', False)
    # Obtaining the member 'sign' of a type (line 267)
    sign_564997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 40), np_564996, 'sign')
    # Calling sign(args, kwargs) (line 267)
    sign_call_result_565002 = invoke(stypy.reporting.localization.Localization(__file__, 267, 40), sign_564997, *[result_sub_565000], **kwargs_565001)
    
    # Applying the binary operator '*' (line 267)
    result_mul_565003 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 34), '*', float_564995, sign_call_result_565002)
    
    # Applying the binary operator '+' (line 267)
    result_add_565004 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 23), '+', observed_564994, result_mul_565003)
    
    # Assigning a type to the variable 'observed' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'observed', result_add_565004)
    # SSA join for if statement (line 265)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 269):
    
    # Assigning a Subscript to a Name (line 269):
    
    # Obtaining the type of the subscript
    int_565005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 8), 'int')
    
    # Call to power_divergence(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'observed' (line 269)
    observed_565007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 35), 'observed', False)
    # Getting the type of 'expected' (line 269)
    expected_565008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 45), 'expected', False)
    # Processing the call keyword arguments (line 269)
    # Getting the type of 'observed' (line 270)
    observed_565009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 40), 'observed', False)
    # Obtaining the member 'size' of a type (line 270)
    size_565010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 40), observed_565009, 'size')
    int_565011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 56), 'int')
    # Applying the binary operator '-' (line 270)
    result_sub_565012 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 40), '-', size_565010, int_565011)
    
    # Getting the type of 'dof' (line 270)
    dof_565013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 60), 'dof', False)
    # Applying the binary operator '-' (line 270)
    result_sub_565014 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 58), '-', result_sub_565012, dof_565013)
    
    keyword_565015 = result_sub_565014
    # Getting the type of 'None' (line 270)
    None_565016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 70), 'None', False)
    keyword_565017 = None_565016
    # Getting the type of 'lambda_' (line 271)
    lambda__565018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 43), 'lambda_', False)
    keyword_565019 = lambda__565018
    kwargs_565020 = {'axis': keyword_565017, 'lambda_': keyword_565019, 'ddof': keyword_565015}
    # Getting the type of 'power_divergence' (line 269)
    power_divergence_565006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 18), 'power_divergence', False)
    # Calling power_divergence(args, kwargs) (line 269)
    power_divergence_call_result_565021 = invoke(stypy.reporting.localization.Localization(__file__, 269, 18), power_divergence_565006, *[observed_565007, expected_565008], **kwargs_565020)
    
    # Obtaining the member '__getitem__' of a type (line 269)
    getitem___565022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 8), power_divergence_call_result_565021, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 269)
    subscript_call_result_565023 = invoke(stypy.reporting.localization.Localization(__file__, 269, 8), getitem___565022, int_565005)
    
    # Assigning a type to the variable 'tuple_var_assignment_564829' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'tuple_var_assignment_564829', subscript_call_result_565023)
    
    # Assigning a Subscript to a Name (line 269):
    
    # Obtaining the type of the subscript
    int_565024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 8), 'int')
    
    # Call to power_divergence(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'observed' (line 269)
    observed_565026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 35), 'observed', False)
    # Getting the type of 'expected' (line 269)
    expected_565027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 45), 'expected', False)
    # Processing the call keyword arguments (line 269)
    # Getting the type of 'observed' (line 270)
    observed_565028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 40), 'observed', False)
    # Obtaining the member 'size' of a type (line 270)
    size_565029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 40), observed_565028, 'size')
    int_565030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 56), 'int')
    # Applying the binary operator '-' (line 270)
    result_sub_565031 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 40), '-', size_565029, int_565030)
    
    # Getting the type of 'dof' (line 270)
    dof_565032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 60), 'dof', False)
    # Applying the binary operator '-' (line 270)
    result_sub_565033 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 58), '-', result_sub_565031, dof_565032)
    
    keyword_565034 = result_sub_565033
    # Getting the type of 'None' (line 270)
    None_565035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 70), 'None', False)
    keyword_565036 = None_565035
    # Getting the type of 'lambda_' (line 271)
    lambda__565037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 43), 'lambda_', False)
    keyword_565038 = lambda__565037
    kwargs_565039 = {'axis': keyword_565036, 'lambda_': keyword_565038, 'ddof': keyword_565034}
    # Getting the type of 'power_divergence' (line 269)
    power_divergence_565025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 18), 'power_divergence', False)
    # Calling power_divergence(args, kwargs) (line 269)
    power_divergence_call_result_565040 = invoke(stypy.reporting.localization.Localization(__file__, 269, 18), power_divergence_565025, *[observed_565026, expected_565027], **kwargs_565039)
    
    # Obtaining the member '__getitem__' of a type (line 269)
    getitem___565041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 8), power_divergence_call_result_565040, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 269)
    subscript_call_result_565042 = invoke(stypy.reporting.localization.Localization(__file__, 269, 8), getitem___565041, int_565024)
    
    # Assigning a type to the variable 'tuple_var_assignment_564830' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'tuple_var_assignment_564830', subscript_call_result_565042)
    
    # Assigning a Name to a Name (line 269):
    # Getting the type of 'tuple_var_assignment_564829' (line 269)
    tuple_var_assignment_564829_565043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'tuple_var_assignment_564829')
    # Assigning a type to the variable 'chi2' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'chi2', tuple_var_assignment_564829_565043)
    
    # Assigning a Name to a Name (line 269):
    # Getting the type of 'tuple_var_assignment_564830' (line 269)
    tuple_var_assignment_564830_565044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'tuple_var_assignment_564830')
    # Assigning a type to the variable 'p' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 14), 'p', tuple_var_assignment_564830_565044)
    # SSA join for if statement (line 258)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 273)
    tuple_565045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 273)
    # Adding element type (line 273)
    # Getting the type of 'chi2' (line 273)
    chi2_565046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 11), 'chi2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 11), tuple_565045, chi2_565046)
    # Adding element type (line 273)
    # Getting the type of 'p' (line 273)
    p_565047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 17), 'p')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 11), tuple_565045, p_565047)
    # Adding element type (line 273)
    # Getting the type of 'dof' (line 273)
    dof_565048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 20), 'dof')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 11), tuple_565045, dof_565048)
    # Adding element type (line 273)
    # Getting the type of 'expected' (line 273)
    expected_565049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 25), 'expected')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 11), tuple_565045, expected_565049)
    
    # Assigning a type to the variable 'stypy_return_type' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'stypy_return_type', tuple_565045)
    
    # ################# End of 'chi2_contingency(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chi2_contingency' in the type store
    # Getting the type of 'stypy_return_type' (line 111)
    stypy_return_type_565050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_565050)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chi2_contingency'
    return stypy_return_type_565050

# Assigning a type to the variable 'chi2_contingency' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'chi2_contingency', chi2_contingency)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
