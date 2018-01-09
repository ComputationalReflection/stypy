
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from collections import namedtuple
2: 
3: import numpy as np
4: 
5: from . import distributions
6: 
7: 
8: __all__ = ['_find_repeats', 'linregress', 'theilslopes']
9: 
10: LinregressResult = namedtuple('LinregressResult', ('slope', 'intercept',
11:                                                    'rvalue', 'pvalue',
12:                                                    'stderr'))
13: 
14: def linregress(x, y=None):
15:     '''
16:     Calculate a linear least-squares regression for two sets of measurements.
17: 
18:     Parameters
19:     ----------
20:     x, y : array_like
21:         Two sets of measurements.  Both arrays should have the same length.
22:         If only x is given (and y=None), then it must be a two-dimensional
23:         array where one dimension has length 2.  The two sets of measurements
24:         are then found by splitting the array along the length-2 dimension.
25: 
26:     Returns
27:     -------
28:     slope : float
29:         slope of the regression line
30:     intercept : float
31:         intercept of the regression line
32:     rvalue : float
33:         correlation coefficient
34:     pvalue : float
35:         two-sided p-value for a hypothesis test whose null hypothesis is
36:         that the slope is zero, using Wald Test with t-distribution of
37:         the test statistic.
38:     stderr : float
39:         Standard error of the estimated gradient.
40: 
41:     See also
42:     --------
43:     :func:`scipy.optimize.curve_fit` : Use non-linear
44:      least squares to fit a function to data.
45:     :func:`scipy.optimize.leastsq` : Minimize the sum of
46:      squares of a set of equations.
47: 
48:     Examples
49:     --------
50:     >>> import matplotlib.pyplot as plt
51:     >>> from scipy import stats
52:     >>> np.random.seed(12345678)
53:     >>> x = np.random.random(10)
54:     >>> y = np.random.random(10)
55:     >>> slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
56: 
57:     To get coefficient of determination (r_squared)
58: 
59:     >>> print("r-squared:", r_value**2)
60:     r-squared: 0.080402268539
61: 
62:     Plot the data along with the fitted line
63: 
64:     >>> plt.plot(x, y, 'o', label='original data')
65:     >>> plt.plot(x, intercept + slope*x, 'r', label='fitted line')
66:     >>> plt.legend()
67:     >>> plt.show()
68: 
69:     '''
70:     TINY = 1.0e-20
71:     if y is None:  # x is a (2, N) or (N, 2) shaped array_like
72:         x = np.asarray(x)
73:         if x.shape[0] == 2:
74:             x, y = x
75:         elif x.shape[1] == 2:
76:             x, y = x.T
77:         else:
78:             msg = ("If only `x` is given as input, it has to be of shape "
79:                    "(2, N) or (N, 2), provided shape was %s" % str(x.shape))
80:             raise ValueError(msg)
81:     else:
82:         x = np.asarray(x)
83:         y = np.asarray(y)
84: 
85:     if x.size == 0 or y.size == 0:
86:         raise ValueError("Inputs must not be empty.")
87: 
88:     n = len(x)
89:     xmean = np.mean(x, None)
90:     ymean = np.mean(y, None)
91: 
92:     # average sum of squares:
93:     ssxm, ssxym, ssyxm, ssym = np.cov(x, y, bias=1).flat
94:     r_num = ssxym
95:     r_den = np.sqrt(ssxm * ssym)
96:     if r_den == 0.0:
97:         r = 0.0
98:     else:
99:         r = r_num / r_den
100:         # test for numerical error propagation
101:         if r > 1.0:
102:             r = 1.0
103:         elif r < -1.0:
104:             r = -1.0
105: 
106:     df = n - 2
107:     slope = r_num / ssxm
108:     intercept = ymean - slope*xmean
109:     if n == 2:
110:         # handle case when only two points are passed in
111:         if y[0] == y[1]:
112:             prob = 1.0
113:         else:
114:             prob = 0.0
115:         sterrest = 0.0
116:     else:
117:         t = r * np.sqrt(df / ((1.0 - r + TINY)*(1.0 + r + TINY)))
118:         prob = 2 * distributions.t.sf(np.abs(t), df)
119:         sterrest = np.sqrt((1 - r**2) * ssym / ssxm / df)
120: 
121:     return LinregressResult(slope, intercept, r, prob, sterrest)
122: 
123: 
124: def theilslopes(y, x=None, alpha=0.95):
125:     r'''
126:     Computes the Theil-Sen estimator for a set of points (x, y).
127: 
128:     `theilslopes` implements a method for robust linear regression.  It
129:     computes the slope as the median of all slopes between paired values.
130: 
131:     Parameters
132:     ----------
133:     y : array_like
134:         Dependent variable.
135:     x : array_like or None, optional
136:         Independent variable. If None, use ``arange(len(y))`` instead.
137:     alpha : float, optional
138:         Confidence degree between 0 and 1. Default is 95% confidence.
139:         Note that `alpha` is symmetric around 0.5, i.e. both 0.1 and 0.9 are
140:         interpreted as "find the 90% confidence interval".
141: 
142:     Returns
143:     -------
144:     medslope : float
145:         Theil slope.
146:     medintercept : float
147:         Intercept of the Theil line, as ``median(y) - medslope*median(x)``.
148:     lo_slope : float
149:         Lower bound of the confidence interval on `medslope`.
150:     up_slope : float
151:         Upper bound of the confidence interval on `medslope`.
152: 
153:     Notes
154:     -----
155:     The implementation of `theilslopes` follows [1]_. The intercept is
156:     not defined in [1]_, and here it is defined as ``median(y) -
157:     medslope*median(x)``, which is given in [3]_. Other definitions of
158:     the intercept exist in the literature. A confidence interval for
159:     the intercept is not given as this question is not addressed in
160:     [1]_.
161: 
162:     References
163:     ----------
164:     .. [1] P.K. Sen, "Estimates of the regression coefficient based on Kendall's tau",
165:            J. Am. Stat. Assoc., Vol. 63, pp. 1379-1389, 1968.
166:     .. [2] H. Theil, "A rank-invariant method of linear and polynomial
167:            regression analysis I, II and III",  Nederl. Akad. Wetensch., Proc.
168:            53:, pp. 386-392, pp. 521-525, pp. 1397-1412, 1950.
169:     .. [3] W.L. Conover, "Practical nonparametric statistics", 2nd ed.,
170:            John Wiley and Sons, New York, pp. 493.
171: 
172:     Examples
173:     --------
174:     >>> from scipy import stats
175:     >>> import matplotlib.pyplot as plt
176: 
177:     >>> x = np.linspace(-5, 5, num=150)
178:     >>> y = x + np.random.normal(size=x.size)
179:     >>> y[11:15] += 10  # add outliers
180:     >>> y[-5:] -= 7
181: 
182:     Compute the slope, intercept and 90% confidence interval.  For comparison,
183:     also compute the least-squares fit with `linregress`:
184: 
185:     >>> res = stats.theilslopes(y, x, 0.90)
186:     >>> lsq_res = stats.linregress(x, y)
187: 
188:     Plot the results. The Theil-Sen regression line is shown in red, with the
189:     dashed red lines illustrating the confidence interval of the slope (note
190:     that the dashed red lines are not the confidence interval of the regression
191:     as the confidence interval of the intercept is not included). The green
192:     line shows the least-squares fit for comparison.
193: 
194:     >>> fig = plt.figure()
195:     >>> ax = fig.add_subplot(111)
196:     >>> ax.plot(x, y, 'b.')
197:     >>> ax.plot(x, res[1] + res[0] * x, 'r-')
198:     >>> ax.plot(x, res[1] + res[2] * x, 'r--')
199:     >>> ax.plot(x, res[1] + res[3] * x, 'r--')
200:     >>> ax.plot(x, lsq_res[1] + lsq_res[0] * x, 'g-')
201:     >>> plt.show()
202: 
203:     '''
204:     # We copy both x and y so we can use _find_repeats.
205:     y = np.array(y).flatten()
206:     if x is None:
207:         x = np.arange(len(y), dtype=float)
208:     else:
209:         x = np.array(x, dtype=float).flatten()
210:         if len(x) != len(y):
211:             raise ValueError("Incompatible lengths ! (%s<>%s)" % (len(y), len(x)))
212: 
213:     # Compute sorted slopes only when deltax > 0
214:     deltax = x[:, np.newaxis] - x
215:     deltay = y[:, np.newaxis] - y
216:     slopes = deltay[deltax > 0] / deltax[deltax > 0]
217:     slopes.sort()
218:     medslope = np.median(slopes)
219:     medinter = np.median(y) - medslope * np.median(x)
220:     # Now compute confidence intervals
221:     if alpha > 0.5:
222:         alpha = 1. - alpha
223: 
224:     z = distributions.norm.ppf(alpha / 2.)
225:     # This implements (2.6) from Sen (1968)
226:     _, nxreps = _find_repeats(x)
227:     _, nyreps = _find_repeats(y)
228:     nt = len(slopes)       # N in Sen (1968)
229:     ny = len(y)            # n in Sen (1968)
230:     # Equation 2.6 in Sen (1968):
231:     sigsq = 1/18. * (ny * (ny-1) * (2*ny+5) -
232:                      np.sum(k * (k-1) * (2*k + 5) for k in nxreps) -
233:                      np.sum(k * (k-1) * (2*k + 5) for k in nyreps))
234:     # Find the confidence interval indices in `slopes`
235:     sigma = np.sqrt(sigsq)
236:     Ru = min(int(np.round((nt - z*sigma)/2.)), len(slopes)-1)
237:     Rl = max(int(np.round((nt + z*sigma)/2.)) - 1, 0)
238:     delta = slopes[[Rl, Ru]]
239:     return medslope, medinter, delta[0], delta[1]
240: 
241: 
242: def _find_repeats(arr):
243:     # This function assumes it may clobber its input.
244:     if len(arr) == 0:
245:         return np.array(0, np.float64), np.array(0, np.intp)
246: 
247:     # XXX This cast was previously needed for the Fortran implementation,
248:     # should we ditch it?
249:     arr = np.asarray(arr, np.float64).ravel()
250:     arr.sort()
251: 
252:     # Taken from NumPy 1.9's np.unique.
253:     change = np.concatenate(([True], arr[1:] != arr[:-1]))
254:     unique = arr[change]
255:     change_idx = np.concatenate(np.nonzero(change) + ([arr.size],))
256:     freq = np.diff(change_idx)
257:     atleast2 = freq > 1
258:     return unique[atleast2], freq[atleast2]
259: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from collections import namedtuple' statement (line 1)
try:
    from collections import namedtuple

except:
    namedtuple = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'collections', None, module_type_store, ['namedtuple'], [namedtuple])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_625809 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_625809) is not StypyTypeError):

    if (import_625809 != 'pyd_module'):
        __import__(import_625809)
        sys_modules_625810 = sys.modules[import_625809]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_625810.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_625809)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from scipy.stats import distributions' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/')
import_625811 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.stats')

if (type(import_625811) is not StypyTypeError):

    if (import_625811 != 'pyd_module'):
        __import__(import_625811)
        sys_modules_625812 = sys.modules[import_625811]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.stats', sys_modules_625812.module_type_store, module_type_store, ['distributions'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_625812, sys_modules_625812.module_type_store, module_type_store)
    else:
        from scipy.stats import distributions

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.stats', None, module_type_store, ['distributions'], [distributions])

else:
    # Assigning a type to the variable 'scipy.stats' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy.stats', import_625811)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/')


# Assigning a List to a Name (line 8):

# Assigning a List to a Name (line 8):
__all__ = ['_find_repeats', 'linregress', 'theilslopes']
module_type_store.set_exportable_members(['_find_repeats', 'linregress', 'theilslopes'])

# Obtaining an instance of the builtin type 'list' (line 8)
list_625813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
str_625814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', '_find_repeats')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_625813, str_625814)
# Adding element type (line 8)
str_625815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 28), 'str', 'linregress')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_625813, str_625815)
# Adding element type (line 8)
str_625816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 42), 'str', 'theilslopes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_625813, str_625816)

# Assigning a type to the variable '__all__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__all__', list_625813)

# Assigning a Call to a Name (line 10):

# Assigning a Call to a Name (line 10):

# Call to namedtuple(...): (line 10)
# Processing the call arguments (line 10)
str_625818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 30), 'str', 'LinregressResult')

# Obtaining an instance of the builtin type 'tuple' (line 10)
tuple_625819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 51), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 10)
# Adding element type (line 10)
str_625820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 51), 'str', 'slope')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 51), tuple_625819, str_625820)
# Adding element type (line 10)
str_625821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 60), 'str', 'intercept')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 51), tuple_625819, str_625821)
# Adding element type (line 10)
str_625822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 51), 'str', 'rvalue')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 51), tuple_625819, str_625822)
# Adding element type (line 10)
str_625823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 61), 'str', 'pvalue')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 51), tuple_625819, str_625823)
# Adding element type (line 10)
str_625824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 51), 'str', 'stderr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 51), tuple_625819, str_625824)

# Processing the call keyword arguments (line 10)
kwargs_625825 = {}
# Getting the type of 'namedtuple' (line 10)
namedtuple_625817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 19), 'namedtuple', False)
# Calling namedtuple(args, kwargs) (line 10)
namedtuple_call_result_625826 = invoke(stypy.reporting.localization.Localization(__file__, 10, 19), namedtuple_625817, *[str_625818, tuple_625819], **kwargs_625825)

# Assigning a type to the variable 'LinregressResult' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'LinregressResult', namedtuple_call_result_625826)

@norecursion
def linregress(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 14)
    None_625827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 20), 'None')
    defaults = [None_625827]
    # Create a new context for function 'linregress'
    module_type_store = module_type_store.open_function_context('linregress', 14, 0, False)
    
    # Passed parameters checking function
    linregress.stypy_localization = localization
    linregress.stypy_type_of_self = None
    linregress.stypy_type_store = module_type_store
    linregress.stypy_function_name = 'linregress'
    linregress.stypy_param_names_list = ['x', 'y']
    linregress.stypy_varargs_param_name = None
    linregress.stypy_kwargs_param_name = None
    linregress.stypy_call_defaults = defaults
    linregress.stypy_call_varargs = varargs
    linregress.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'linregress', ['x', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'linregress', localization, ['x', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'linregress(...)' code ##################

    str_625828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, (-1)), 'str', '\n    Calculate a linear least-squares regression for two sets of measurements.\n\n    Parameters\n    ----------\n    x, y : array_like\n        Two sets of measurements.  Both arrays should have the same length.\n        If only x is given (and y=None), then it must be a two-dimensional\n        array where one dimension has length 2.  The two sets of measurements\n        are then found by splitting the array along the length-2 dimension.\n\n    Returns\n    -------\n    slope : float\n        slope of the regression line\n    intercept : float\n        intercept of the regression line\n    rvalue : float\n        correlation coefficient\n    pvalue : float\n        two-sided p-value for a hypothesis test whose null hypothesis is\n        that the slope is zero, using Wald Test with t-distribution of\n        the test statistic.\n    stderr : float\n        Standard error of the estimated gradient.\n\n    See also\n    --------\n    :func:`scipy.optimize.curve_fit` : Use non-linear\n     least squares to fit a function to data.\n    :func:`scipy.optimize.leastsq` : Minimize the sum of\n     squares of a set of equations.\n\n    Examples\n    --------\n    >>> import matplotlib.pyplot as plt\n    >>> from scipy import stats\n    >>> np.random.seed(12345678)\n    >>> x = np.random.random(10)\n    >>> y = np.random.random(10)\n    >>> slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)\n\n    To get coefficient of determination (r_squared)\n\n    >>> print("r-squared:", r_value**2)\n    r-squared: 0.080402268539\n\n    Plot the data along with the fitted line\n\n    >>> plt.plot(x, y, \'o\', label=\'original data\')\n    >>> plt.plot(x, intercept + slope*x, \'r\', label=\'fitted line\')\n    >>> plt.legend()\n    >>> plt.show()\n\n    ')
    
    # Assigning a Num to a Name (line 70):
    
    # Assigning a Num to a Name (line 70):
    float_625829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 11), 'float')
    # Assigning a type to the variable 'TINY' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'TINY', float_625829)
    
    # Type idiom detected: calculating its left and rigth part (line 71)
    # Getting the type of 'y' (line 71)
    y_625830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 7), 'y')
    # Getting the type of 'None' (line 71)
    None_625831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'None')
    
    (may_be_625832, more_types_in_union_625833) = may_be_none(y_625830, None_625831)

    if may_be_625832:

        if more_types_in_union_625833:
            # Runtime conditional SSA (line 71)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to asarray(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'x' (line 72)
        x_625836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'x', False)
        # Processing the call keyword arguments (line 72)
        kwargs_625837 = {}
        # Getting the type of 'np' (line 72)
        np_625834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 72)
        asarray_625835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), np_625834, 'asarray')
        # Calling asarray(args, kwargs) (line 72)
        asarray_call_result_625838 = invoke(stypy.reporting.localization.Localization(__file__, 72, 12), asarray_625835, *[x_625836], **kwargs_625837)
        
        # Assigning a type to the variable 'x' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'x', asarray_call_result_625838)
        
        
        
        # Obtaining the type of the subscript
        int_625839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'int')
        # Getting the type of 'x' (line 73)
        x_625840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'x')
        # Obtaining the member 'shape' of a type (line 73)
        shape_625841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 11), x_625840, 'shape')
        # Obtaining the member '__getitem__' of a type (line 73)
        getitem___625842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 11), shape_625841, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
        subscript_call_result_625843 = invoke(stypy.reporting.localization.Localization(__file__, 73, 11), getitem___625842, int_625839)
        
        int_625844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 25), 'int')
        # Applying the binary operator '==' (line 73)
        result_eq_625845 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 11), '==', subscript_call_result_625843, int_625844)
        
        # Testing the type of an if condition (line 73)
        if_condition_625846 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 8), result_eq_625845)
        # Assigning a type to the variable 'if_condition_625846' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'if_condition_625846', if_condition_625846)
        # SSA begins for if statement (line 73)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Tuple (line 74):
        
        # Assigning a Subscript to a Name (line 74):
        
        # Obtaining the type of the subscript
        int_625847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 12), 'int')
        # Getting the type of 'x' (line 74)
        x_625848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'x')
        # Obtaining the member '__getitem__' of a type (line 74)
        getitem___625849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), x_625848, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 74)
        subscript_call_result_625850 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), getitem___625849, int_625847)
        
        # Assigning a type to the variable 'tuple_var_assignment_625797' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'tuple_var_assignment_625797', subscript_call_result_625850)
        
        # Assigning a Subscript to a Name (line 74):
        
        # Obtaining the type of the subscript
        int_625851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 12), 'int')
        # Getting the type of 'x' (line 74)
        x_625852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'x')
        # Obtaining the member '__getitem__' of a type (line 74)
        getitem___625853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), x_625852, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 74)
        subscript_call_result_625854 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), getitem___625853, int_625851)
        
        # Assigning a type to the variable 'tuple_var_assignment_625798' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'tuple_var_assignment_625798', subscript_call_result_625854)
        
        # Assigning a Name to a Name (line 74):
        # Getting the type of 'tuple_var_assignment_625797' (line 74)
        tuple_var_assignment_625797_625855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'tuple_var_assignment_625797')
        # Assigning a type to the variable 'x' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'x', tuple_var_assignment_625797_625855)
        
        # Assigning a Name to a Name (line 74):
        # Getting the type of 'tuple_var_assignment_625798' (line 74)
        tuple_var_assignment_625798_625856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'tuple_var_assignment_625798')
        # Assigning a type to the variable 'y' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'y', tuple_var_assignment_625798_625856)
        # SSA branch for the else part of an if statement (line 73)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        int_625857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 21), 'int')
        # Getting the type of 'x' (line 75)
        x_625858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 13), 'x')
        # Obtaining the member 'shape' of a type (line 75)
        shape_625859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 13), x_625858, 'shape')
        # Obtaining the member '__getitem__' of a type (line 75)
        getitem___625860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 13), shape_625859, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 75)
        subscript_call_result_625861 = invoke(stypy.reporting.localization.Localization(__file__, 75, 13), getitem___625860, int_625857)
        
        int_625862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 27), 'int')
        # Applying the binary operator '==' (line 75)
        result_eq_625863 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 13), '==', subscript_call_result_625861, int_625862)
        
        # Testing the type of an if condition (line 75)
        if_condition_625864 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 13), result_eq_625863)
        # Assigning a type to the variable 'if_condition_625864' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 13), 'if_condition_625864', if_condition_625864)
        # SSA begins for if statement (line 75)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Tuple (line 76):
        
        # Assigning a Subscript to a Name (line 76):
        
        # Obtaining the type of the subscript
        int_625865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 12), 'int')
        # Getting the type of 'x' (line 76)
        x_625866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'x')
        # Obtaining the member 'T' of a type (line 76)
        T_625867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 19), x_625866, 'T')
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___625868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), T_625867, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_625869 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), getitem___625868, int_625865)
        
        # Assigning a type to the variable 'tuple_var_assignment_625799' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'tuple_var_assignment_625799', subscript_call_result_625869)
        
        # Assigning a Subscript to a Name (line 76):
        
        # Obtaining the type of the subscript
        int_625870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 12), 'int')
        # Getting the type of 'x' (line 76)
        x_625871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'x')
        # Obtaining the member 'T' of a type (line 76)
        T_625872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 19), x_625871, 'T')
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___625873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), T_625872, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_625874 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), getitem___625873, int_625870)
        
        # Assigning a type to the variable 'tuple_var_assignment_625800' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'tuple_var_assignment_625800', subscript_call_result_625874)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'tuple_var_assignment_625799' (line 76)
        tuple_var_assignment_625799_625875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'tuple_var_assignment_625799')
        # Assigning a type to the variable 'x' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'x', tuple_var_assignment_625799_625875)
        
        # Assigning a Name to a Name (line 76):
        # Getting the type of 'tuple_var_assignment_625800' (line 76)
        tuple_var_assignment_625800_625876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'tuple_var_assignment_625800')
        # Assigning a type to the variable 'y' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'y', tuple_var_assignment_625800_625876)
        # SSA branch for the else part of an if statement (line 75)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 78):
        
        # Assigning a BinOp to a Name (line 78):
        str_625877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 19), 'str', 'If only `x` is given as input, it has to be of shape (2, N) or (N, 2), provided shape was %s')
        
        # Call to str(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'x' (line 79)
        x_625879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 67), 'x', False)
        # Obtaining the member 'shape' of a type (line 79)
        shape_625880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 67), x_625879, 'shape')
        # Processing the call keyword arguments (line 79)
        kwargs_625881 = {}
        # Getting the type of 'str' (line 79)
        str_625878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 63), 'str', False)
        # Calling str(args, kwargs) (line 79)
        str_call_result_625882 = invoke(stypy.reporting.localization.Localization(__file__, 79, 63), str_625878, *[shape_625880], **kwargs_625881)
        
        # Applying the binary operator '%' (line 78)
        result_mod_625883 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 19), '%', str_625877, str_call_result_625882)
        
        # Assigning a type to the variable 'msg' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'msg', result_mod_625883)
        
        # Call to ValueError(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'msg' (line 80)
        msg_625885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 29), 'msg', False)
        # Processing the call keyword arguments (line 80)
        kwargs_625886 = {}
        # Getting the type of 'ValueError' (line 80)
        ValueError_625884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 80)
        ValueError_call_result_625887 = invoke(stypy.reporting.localization.Localization(__file__, 80, 18), ValueError_625884, *[msg_625885], **kwargs_625886)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 80, 12), ValueError_call_result_625887, 'raise parameter', BaseException)
        # SSA join for if statement (line 75)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 73)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_625833:
            # Runtime conditional SSA for else branch (line 71)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_625832) or more_types_in_union_625833):
        
        # Assigning a Call to a Name (line 82):
        
        # Assigning a Call to a Name (line 82):
        
        # Call to asarray(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'x' (line 82)
        x_625890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'x', False)
        # Processing the call keyword arguments (line 82)
        kwargs_625891 = {}
        # Getting the type of 'np' (line 82)
        np_625888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 82)
        asarray_625889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), np_625888, 'asarray')
        # Calling asarray(args, kwargs) (line 82)
        asarray_call_result_625892 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), asarray_625889, *[x_625890], **kwargs_625891)
        
        # Assigning a type to the variable 'x' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'x', asarray_call_result_625892)
        
        # Assigning a Call to a Name (line 83):
        
        # Assigning a Call to a Name (line 83):
        
        # Call to asarray(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'y' (line 83)
        y_625895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 23), 'y', False)
        # Processing the call keyword arguments (line 83)
        kwargs_625896 = {}
        # Getting the type of 'np' (line 83)
        np_625893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 83)
        asarray_625894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), np_625893, 'asarray')
        # Calling asarray(args, kwargs) (line 83)
        asarray_call_result_625897 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), asarray_625894, *[y_625895], **kwargs_625896)
        
        # Assigning a type to the variable 'y' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'y', asarray_call_result_625897)

        if (may_be_625832 and more_types_in_union_625833):
            # SSA join for if statement (line 71)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'x' (line 85)
    x_625898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 7), 'x')
    # Obtaining the member 'size' of a type (line 85)
    size_625899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 7), x_625898, 'size')
    int_625900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 17), 'int')
    # Applying the binary operator '==' (line 85)
    result_eq_625901 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 7), '==', size_625899, int_625900)
    
    
    # Getting the type of 'y' (line 85)
    y_625902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 22), 'y')
    # Obtaining the member 'size' of a type (line 85)
    size_625903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 22), y_625902, 'size')
    int_625904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 32), 'int')
    # Applying the binary operator '==' (line 85)
    result_eq_625905 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 22), '==', size_625903, int_625904)
    
    # Applying the binary operator 'or' (line 85)
    result_or_keyword_625906 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 7), 'or', result_eq_625901, result_eq_625905)
    
    # Testing the type of an if condition (line 85)
    if_condition_625907 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 4), result_or_keyword_625906)
    # Assigning a type to the variable 'if_condition_625907' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'if_condition_625907', if_condition_625907)
    # SSA begins for if statement (line 85)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 86)
    # Processing the call arguments (line 86)
    str_625909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'str', 'Inputs must not be empty.')
    # Processing the call keyword arguments (line 86)
    kwargs_625910 = {}
    # Getting the type of 'ValueError' (line 86)
    ValueError_625908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 86)
    ValueError_call_result_625911 = invoke(stypy.reporting.localization.Localization(__file__, 86, 14), ValueError_625908, *[str_625909], **kwargs_625910)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 86, 8), ValueError_call_result_625911, 'raise parameter', BaseException)
    # SSA join for if statement (line 85)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 88):
    
    # Assigning a Call to a Name (line 88):
    
    # Call to len(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'x' (line 88)
    x_625913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'x', False)
    # Processing the call keyword arguments (line 88)
    kwargs_625914 = {}
    # Getting the type of 'len' (line 88)
    len_625912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'len', False)
    # Calling len(args, kwargs) (line 88)
    len_call_result_625915 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), len_625912, *[x_625913], **kwargs_625914)
    
    # Assigning a type to the variable 'n' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'n', len_call_result_625915)
    
    # Assigning a Call to a Name (line 89):
    
    # Assigning a Call to a Name (line 89):
    
    # Call to mean(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'x' (line 89)
    x_625918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'x', False)
    # Getting the type of 'None' (line 89)
    None_625919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 23), 'None', False)
    # Processing the call keyword arguments (line 89)
    kwargs_625920 = {}
    # Getting the type of 'np' (line 89)
    np_625916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'np', False)
    # Obtaining the member 'mean' of a type (line 89)
    mean_625917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), np_625916, 'mean')
    # Calling mean(args, kwargs) (line 89)
    mean_call_result_625921 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), mean_625917, *[x_625918, None_625919], **kwargs_625920)
    
    # Assigning a type to the variable 'xmean' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'xmean', mean_call_result_625921)
    
    # Assigning a Call to a Name (line 90):
    
    # Assigning a Call to a Name (line 90):
    
    # Call to mean(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'y' (line 90)
    y_625924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 20), 'y', False)
    # Getting the type of 'None' (line 90)
    None_625925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 23), 'None', False)
    # Processing the call keyword arguments (line 90)
    kwargs_625926 = {}
    # Getting the type of 'np' (line 90)
    np_625922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'np', False)
    # Obtaining the member 'mean' of a type (line 90)
    mean_625923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), np_625922, 'mean')
    # Calling mean(args, kwargs) (line 90)
    mean_call_result_625927 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), mean_625923, *[y_625924, None_625925], **kwargs_625926)
    
    # Assigning a type to the variable 'ymean' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'ymean', mean_call_result_625927)
    
    # Assigning a Attribute to a Tuple (line 93):
    
    # Assigning a Subscript to a Name (line 93):
    
    # Obtaining the type of the subscript
    int_625928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 4), 'int')
    
    # Call to cov(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'x' (line 93)
    x_625931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 38), 'x', False)
    # Getting the type of 'y' (line 93)
    y_625932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 41), 'y', False)
    # Processing the call keyword arguments (line 93)
    int_625933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 49), 'int')
    keyword_625934 = int_625933
    kwargs_625935 = {'bias': keyword_625934}
    # Getting the type of 'np' (line 93)
    np_625929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), 'np', False)
    # Obtaining the member 'cov' of a type (line 93)
    cov_625930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 31), np_625929, 'cov')
    # Calling cov(args, kwargs) (line 93)
    cov_call_result_625936 = invoke(stypy.reporting.localization.Localization(__file__, 93, 31), cov_625930, *[x_625931, y_625932], **kwargs_625935)
    
    # Obtaining the member 'flat' of a type (line 93)
    flat_625937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 31), cov_call_result_625936, 'flat')
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___625938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 4), flat_625937, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_625939 = invoke(stypy.reporting.localization.Localization(__file__, 93, 4), getitem___625938, int_625928)
    
    # Assigning a type to the variable 'tuple_var_assignment_625801' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'tuple_var_assignment_625801', subscript_call_result_625939)
    
    # Assigning a Subscript to a Name (line 93):
    
    # Obtaining the type of the subscript
    int_625940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 4), 'int')
    
    # Call to cov(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'x' (line 93)
    x_625943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 38), 'x', False)
    # Getting the type of 'y' (line 93)
    y_625944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 41), 'y', False)
    # Processing the call keyword arguments (line 93)
    int_625945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 49), 'int')
    keyword_625946 = int_625945
    kwargs_625947 = {'bias': keyword_625946}
    # Getting the type of 'np' (line 93)
    np_625941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), 'np', False)
    # Obtaining the member 'cov' of a type (line 93)
    cov_625942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 31), np_625941, 'cov')
    # Calling cov(args, kwargs) (line 93)
    cov_call_result_625948 = invoke(stypy.reporting.localization.Localization(__file__, 93, 31), cov_625942, *[x_625943, y_625944], **kwargs_625947)
    
    # Obtaining the member 'flat' of a type (line 93)
    flat_625949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 31), cov_call_result_625948, 'flat')
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___625950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 4), flat_625949, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_625951 = invoke(stypy.reporting.localization.Localization(__file__, 93, 4), getitem___625950, int_625940)
    
    # Assigning a type to the variable 'tuple_var_assignment_625802' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'tuple_var_assignment_625802', subscript_call_result_625951)
    
    # Assigning a Subscript to a Name (line 93):
    
    # Obtaining the type of the subscript
    int_625952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 4), 'int')
    
    # Call to cov(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'x' (line 93)
    x_625955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 38), 'x', False)
    # Getting the type of 'y' (line 93)
    y_625956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 41), 'y', False)
    # Processing the call keyword arguments (line 93)
    int_625957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 49), 'int')
    keyword_625958 = int_625957
    kwargs_625959 = {'bias': keyword_625958}
    # Getting the type of 'np' (line 93)
    np_625953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), 'np', False)
    # Obtaining the member 'cov' of a type (line 93)
    cov_625954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 31), np_625953, 'cov')
    # Calling cov(args, kwargs) (line 93)
    cov_call_result_625960 = invoke(stypy.reporting.localization.Localization(__file__, 93, 31), cov_625954, *[x_625955, y_625956], **kwargs_625959)
    
    # Obtaining the member 'flat' of a type (line 93)
    flat_625961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 31), cov_call_result_625960, 'flat')
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___625962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 4), flat_625961, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_625963 = invoke(stypy.reporting.localization.Localization(__file__, 93, 4), getitem___625962, int_625952)
    
    # Assigning a type to the variable 'tuple_var_assignment_625803' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'tuple_var_assignment_625803', subscript_call_result_625963)
    
    # Assigning a Subscript to a Name (line 93):
    
    # Obtaining the type of the subscript
    int_625964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 4), 'int')
    
    # Call to cov(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'x' (line 93)
    x_625967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 38), 'x', False)
    # Getting the type of 'y' (line 93)
    y_625968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 41), 'y', False)
    # Processing the call keyword arguments (line 93)
    int_625969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 49), 'int')
    keyword_625970 = int_625969
    kwargs_625971 = {'bias': keyword_625970}
    # Getting the type of 'np' (line 93)
    np_625965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), 'np', False)
    # Obtaining the member 'cov' of a type (line 93)
    cov_625966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 31), np_625965, 'cov')
    # Calling cov(args, kwargs) (line 93)
    cov_call_result_625972 = invoke(stypy.reporting.localization.Localization(__file__, 93, 31), cov_625966, *[x_625967, y_625968], **kwargs_625971)
    
    # Obtaining the member 'flat' of a type (line 93)
    flat_625973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 31), cov_call_result_625972, 'flat')
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___625974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 4), flat_625973, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_625975 = invoke(stypy.reporting.localization.Localization(__file__, 93, 4), getitem___625974, int_625964)
    
    # Assigning a type to the variable 'tuple_var_assignment_625804' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'tuple_var_assignment_625804', subscript_call_result_625975)
    
    # Assigning a Name to a Name (line 93):
    # Getting the type of 'tuple_var_assignment_625801' (line 93)
    tuple_var_assignment_625801_625976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'tuple_var_assignment_625801')
    # Assigning a type to the variable 'ssxm' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'ssxm', tuple_var_assignment_625801_625976)
    
    # Assigning a Name to a Name (line 93):
    # Getting the type of 'tuple_var_assignment_625802' (line 93)
    tuple_var_assignment_625802_625977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'tuple_var_assignment_625802')
    # Assigning a type to the variable 'ssxym' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 10), 'ssxym', tuple_var_assignment_625802_625977)
    
    # Assigning a Name to a Name (line 93):
    # Getting the type of 'tuple_var_assignment_625803' (line 93)
    tuple_var_assignment_625803_625978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'tuple_var_assignment_625803')
    # Assigning a type to the variable 'ssyxm' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'ssyxm', tuple_var_assignment_625803_625978)
    
    # Assigning a Name to a Name (line 93):
    # Getting the type of 'tuple_var_assignment_625804' (line 93)
    tuple_var_assignment_625804_625979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'tuple_var_assignment_625804')
    # Assigning a type to the variable 'ssym' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 24), 'ssym', tuple_var_assignment_625804_625979)
    
    # Assigning a Name to a Name (line 94):
    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'ssxym' (line 94)
    ssxym_625980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'ssxym')
    # Assigning a type to the variable 'r_num' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'r_num', ssxym_625980)
    
    # Assigning a Call to a Name (line 95):
    
    # Assigning a Call to a Name (line 95):
    
    # Call to sqrt(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'ssxm' (line 95)
    ssxm_625983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 20), 'ssxm', False)
    # Getting the type of 'ssym' (line 95)
    ssym_625984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 27), 'ssym', False)
    # Applying the binary operator '*' (line 95)
    result_mul_625985 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 20), '*', ssxm_625983, ssym_625984)
    
    # Processing the call keyword arguments (line 95)
    kwargs_625986 = {}
    # Getting the type of 'np' (line 95)
    np_625981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 95)
    sqrt_625982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), np_625981, 'sqrt')
    # Calling sqrt(args, kwargs) (line 95)
    sqrt_call_result_625987 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), sqrt_625982, *[result_mul_625985], **kwargs_625986)
    
    # Assigning a type to the variable 'r_den' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'r_den', sqrt_call_result_625987)
    
    
    # Getting the type of 'r_den' (line 96)
    r_den_625988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 7), 'r_den')
    float_625989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 16), 'float')
    # Applying the binary operator '==' (line 96)
    result_eq_625990 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 7), '==', r_den_625988, float_625989)
    
    # Testing the type of an if condition (line 96)
    if_condition_625991 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 4), result_eq_625990)
    # Assigning a type to the variable 'if_condition_625991' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'if_condition_625991', if_condition_625991)
    # SSA begins for if statement (line 96)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 97):
    
    # Assigning a Num to a Name (line 97):
    float_625992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 12), 'float')
    # Assigning a type to the variable 'r' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'r', float_625992)
    # SSA branch for the else part of an if statement (line 96)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 99):
    
    # Assigning a BinOp to a Name (line 99):
    # Getting the type of 'r_num' (line 99)
    r_num_625993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'r_num')
    # Getting the type of 'r_den' (line 99)
    r_den_625994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'r_den')
    # Applying the binary operator 'div' (line 99)
    result_div_625995 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 12), 'div', r_num_625993, r_den_625994)
    
    # Assigning a type to the variable 'r' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'r', result_div_625995)
    
    
    # Getting the type of 'r' (line 101)
    r_625996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'r')
    float_625997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 15), 'float')
    # Applying the binary operator '>' (line 101)
    result_gt_625998 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 11), '>', r_625996, float_625997)
    
    # Testing the type of an if condition (line 101)
    if_condition_625999 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 8), result_gt_625998)
    # Assigning a type to the variable 'if_condition_625999' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'if_condition_625999', if_condition_625999)
    # SSA begins for if statement (line 101)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 102):
    
    # Assigning a Num to a Name (line 102):
    float_626000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 16), 'float')
    # Assigning a type to the variable 'r' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'r', float_626000)
    # SSA branch for the else part of an if statement (line 101)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'r' (line 103)
    r_626001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 13), 'r')
    float_626002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 17), 'float')
    # Applying the binary operator '<' (line 103)
    result_lt_626003 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 13), '<', r_626001, float_626002)
    
    # Testing the type of an if condition (line 103)
    if_condition_626004 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 13), result_lt_626003)
    # Assigning a type to the variable 'if_condition_626004' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 13), 'if_condition_626004', if_condition_626004)
    # SSA begins for if statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 104):
    
    # Assigning a Num to a Name (line 104):
    float_626005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 16), 'float')
    # Assigning a type to the variable 'r' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'r', float_626005)
    # SSA join for if statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 101)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 96)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 106):
    
    # Assigning a BinOp to a Name (line 106):
    # Getting the type of 'n' (line 106)
    n_626006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 9), 'n')
    int_626007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 13), 'int')
    # Applying the binary operator '-' (line 106)
    result_sub_626008 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 9), '-', n_626006, int_626007)
    
    # Assigning a type to the variable 'df' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'df', result_sub_626008)
    
    # Assigning a BinOp to a Name (line 107):
    
    # Assigning a BinOp to a Name (line 107):
    # Getting the type of 'r_num' (line 107)
    r_num_626009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'r_num')
    # Getting the type of 'ssxm' (line 107)
    ssxm_626010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 20), 'ssxm')
    # Applying the binary operator 'div' (line 107)
    result_div_626011 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 12), 'div', r_num_626009, ssxm_626010)
    
    # Assigning a type to the variable 'slope' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'slope', result_div_626011)
    
    # Assigning a BinOp to a Name (line 108):
    
    # Assigning a BinOp to a Name (line 108):
    # Getting the type of 'ymean' (line 108)
    ymean_626012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'ymean')
    # Getting the type of 'slope' (line 108)
    slope_626013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'slope')
    # Getting the type of 'xmean' (line 108)
    xmean_626014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 30), 'xmean')
    # Applying the binary operator '*' (line 108)
    result_mul_626015 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 24), '*', slope_626013, xmean_626014)
    
    # Applying the binary operator '-' (line 108)
    result_sub_626016 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 16), '-', ymean_626012, result_mul_626015)
    
    # Assigning a type to the variable 'intercept' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'intercept', result_sub_626016)
    
    
    # Getting the type of 'n' (line 109)
    n_626017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 7), 'n')
    int_626018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 12), 'int')
    # Applying the binary operator '==' (line 109)
    result_eq_626019 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 7), '==', n_626017, int_626018)
    
    # Testing the type of an if condition (line 109)
    if_condition_626020 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 4), result_eq_626019)
    # Assigning a type to the variable 'if_condition_626020' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'if_condition_626020', if_condition_626020)
    # SSA begins for if statement (line 109)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_626021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 13), 'int')
    # Getting the type of 'y' (line 111)
    y_626022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'y')
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___626023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 11), y_626022, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_626024 = invoke(stypy.reporting.localization.Localization(__file__, 111, 11), getitem___626023, int_626021)
    
    
    # Obtaining the type of the subscript
    int_626025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 21), 'int')
    # Getting the type of 'y' (line 111)
    y_626026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'y')
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___626027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 19), y_626026, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_626028 = invoke(stypy.reporting.localization.Localization(__file__, 111, 19), getitem___626027, int_626025)
    
    # Applying the binary operator '==' (line 111)
    result_eq_626029 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 11), '==', subscript_call_result_626024, subscript_call_result_626028)
    
    # Testing the type of an if condition (line 111)
    if_condition_626030 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 8), result_eq_626029)
    # Assigning a type to the variable 'if_condition_626030' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'if_condition_626030', if_condition_626030)
    # SSA begins for if statement (line 111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 112):
    
    # Assigning a Num to a Name (line 112):
    float_626031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 19), 'float')
    # Assigning a type to the variable 'prob' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'prob', float_626031)
    # SSA branch for the else part of an if statement (line 111)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Name (line 114):
    
    # Assigning a Num to a Name (line 114):
    float_626032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 19), 'float')
    # Assigning a type to the variable 'prob' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'prob', float_626032)
    # SSA join for if statement (line 111)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 115):
    
    # Assigning a Num to a Name (line 115):
    float_626033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 19), 'float')
    # Assigning a type to the variable 'sterrest' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'sterrest', float_626033)
    # SSA branch for the else part of an if statement (line 109)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 117):
    
    # Assigning a BinOp to a Name (line 117):
    # Getting the type of 'r' (line 117)
    r_626034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'r')
    
    # Call to sqrt(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'df' (line 117)
    df_626037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 24), 'df', False)
    float_626038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 31), 'float')
    # Getting the type of 'r' (line 117)
    r_626039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 37), 'r', False)
    # Applying the binary operator '-' (line 117)
    result_sub_626040 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 31), '-', float_626038, r_626039)
    
    # Getting the type of 'TINY' (line 117)
    TINY_626041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 41), 'TINY', False)
    # Applying the binary operator '+' (line 117)
    result_add_626042 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 39), '+', result_sub_626040, TINY_626041)
    
    float_626043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 48), 'float')
    # Getting the type of 'r' (line 117)
    r_626044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 54), 'r', False)
    # Applying the binary operator '+' (line 117)
    result_add_626045 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 48), '+', float_626043, r_626044)
    
    # Getting the type of 'TINY' (line 117)
    TINY_626046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 58), 'TINY', False)
    # Applying the binary operator '+' (line 117)
    result_add_626047 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 56), '+', result_add_626045, TINY_626046)
    
    # Applying the binary operator '*' (line 117)
    result_mul_626048 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 30), '*', result_add_626042, result_add_626047)
    
    # Applying the binary operator 'div' (line 117)
    result_div_626049 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 24), 'div', df_626037, result_mul_626048)
    
    # Processing the call keyword arguments (line 117)
    kwargs_626050 = {}
    # Getting the type of 'np' (line 117)
    np_626035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 117)
    sqrt_626036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 16), np_626035, 'sqrt')
    # Calling sqrt(args, kwargs) (line 117)
    sqrt_call_result_626051 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), sqrt_626036, *[result_div_626049], **kwargs_626050)
    
    # Applying the binary operator '*' (line 117)
    result_mul_626052 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 12), '*', r_626034, sqrt_call_result_626051)
    
    # Assigning a type to the variable 't' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 't', result_mul_626052)
    
    # Assigning a BinOp to a Name (line 118):
    
    # Assigning a BinOp to a Name (line 118):
    int_626053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 15), 'int')
    
    # Call to sf(...): (line 118)
    # Processing the call arguments (line 118)
    
    # Call to abs(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 't' (line 118)
    t_626059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 45), 't', False)
    # Processing the call keyword arguments (line 118)
    kwargs_626060 = {}
    # Getting the type of 'np' (line 118)
    np_626057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 38), 'np', False)
    # Obtaining the member 'abs' of a type (line 118)
    abs_626058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 38), np_626057, 'abs')
    # Calling abs(args, kwargs) (line 118)
    abs_call_result_626061 = invoke(stypy.reporting.localization.Localization(__file__, 118, 38), abs_626058, *[t_626059], **kwargs_626060)
    
    # Getting the type of 'df' (line 118)
    df_626062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 49), 'df', False)
    # Processing the call keyword arguments (line 118)
    kwargs_626063 = {}
    # Getting the type of 'distributions' (line 118)
    distributions_626054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 19), 'distributions', False)
    # Obtaining the member 't' of a type (line 118)
    t_626055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 19), distributions_626054, 't')
    # Obtaining the member 'sf' of a type (line 118)
    sf_626056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 19), t_626055, 'sf')
    # Calling sf(args, kwargs) (line 118)
    sf_call_result_626064 = invoke(stypy.reporting.localization.Localization(__file__, 118, 19), sf_626056, *[abs_call_result_626061, df_626062], **kwargs_626063)
    
    # Applying the binary operator '*' (line 118)
    result_mul_626065 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 15), '*', int_626053, sf_call_result_626064)
    
    # Assigning a type to the variable 'prob' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'prob', result_mul_626065)
    
    # Assigning a Call to a Name (line 119):
    
    # Assigning a Call to a Name (line 119):
    
    # Call to sqrt(...): (line 119)
    # Processing the call arguments (line 119)
    int_626068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 28), 'int')
    # Getting the type of 'r' (line 119)
    r_626069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 32), 'r', False)
    int_626070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 35), 'int')
    # Applying the binary operator '**' (line 119)
    result_pow_626071 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 32), '**', r_626069, int_626070)
    
    # Applying the binary operator '-' (line 119)
    result_sub_626072 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 28), '-', int_626068, result_pow_626071)
    
    # Getting the type of 'ssym' (line 119)
    ssym_626073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 40), 'ssym', False)
    # Applying the binary operator '*' (line 119)
    result_mul_626074 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 27), '*', result_sub_626072, ssym_626073)
    
    # Getting the type of 'ssxm' (line 119)
    ssxm_626075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 47), 'ssxm', False)
    # Applying the binary operator 'div' (line 119)
    result_div_626076 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 45), 'div', result_mul_626074, ssxm_626075)
    
    # Getting the type of 'df' (line 119)
    df_626077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 54), 'df', False)
    # Applying the binary operator 'div' (line 119)
    result_div_626078 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 52), 'div', result_div_626076, df_626077)
    
    # Processing the call keyword arguments (line 119)
    kwargs_626079 = {}
    # Getting the type of 'np' (line 119)
    np_626066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 19), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 119)
    sqrt_626067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 19), np_626066, 'sqrt')
    # Calling sqrt(args, kwargs) (line 119)
    sqrt_call_result_626080 = invoke(stypy.reporting.localization.Localization(__file__, 119, 19), sqrt_626067, *[result_div_626078], **kwargs_626079)
    
    # Assigning a type to the variable 'sterrest' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'sterrest', sqrt_call_result_626080)
    # SSA join for if statement (line 109)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to LinregressResult(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'slope' (line 121)
    slope_626082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 28), 'slope', False)
    # Getting the type of 'intercept' (line 121)
    intercept_626083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 35), 'intercept', False)
    # Getting the type of 'r' (line 121)
    r_626084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 46), 'r', False)
    # Getting the type of 'prob' (line 121)
    prob_626085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 49), 'prob', False)
    # Getting the type of 'sterrest' (line 121)
    sterrest_626086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 55), 'sterrest', False)
    # Processing the call keyword arguments (line 121)
    kwargs_626087 = {}
    # Getting the type of 'LinregressResult' (line 121)
    LinregressResult_626081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'LinregressResult', False)
    # Calling LinregressResult(args, kwargs) (line 121)
    LinregressResult_call_result_626088 = invoke(stypy.reporting.localization.Localization(__file__, 121, 11), LinregressResult_626081, *[slope_626082, intercept_626083, r_626084, prob_626085, sterrest_626086], **kwargs_626087)
    
    # Assigning a type to the variable 'stypy_return_type' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type', LinregressResult_call_result_626088)
    
    # ################# End of 'linregress(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'linregress' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_626089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_626089)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'linregress'
    return stypy_return_type_626089

# Assigning a type to the variable 'linregress' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'linregress', linregress)

@norecursion
def theilslopes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 124)
    None_626090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 21), 'None')
    float_626091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 33), 'float')
    defaults = [None_626090, float_626091]
    # Create a new context for function 'theilslopes'
    module_type_store = module_type_store.open_function_context('theilslopes', 124, 0, False)
    
    # Passed parameters checking function
    theilslopes.stypy_localization = localization
    theilslopes.stypy_type_of_self = None
    theilslopes.stypy_type_store = module_type_store
    theilslopes.stypy_function_name = 'theilslopes'
    theilslopes.stypy_param_names_list = ['y', 'x', 'alpha']
    theilslopes.stypy_varargs_param_name = None
    theilslopes.stypy_kwargs_param_name = None
    theilslopes.stypy_call_defaults = defaults
    theilslopes.stypy_call_varargs = varargs
    theilslopes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'theilslopes', ['y', 'x', 'alpha'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'theilslopes', localization, ['y', 'x', 'alpha'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'theilslopes(...)' code ##################

    str_626092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, (-1)), 'str', '\n    Computes the Theil-Sen estimator for a set of points (x, y).\n\n    `theilslopes` implements a method for robust linear regression.  It\n    computes the slope as the median of all slopes between paired values.\n\n    Parameters\n    ----------\n    y : array_like\n        Dependent variable.\n    x : array_like or None, optional\n        Independent variable. If None, use ``arange(len(y))`` instead.\n    alpha : float, optional\n        Confidence degree between 0 and 1. Default is 95% confidence.\n        Note that `alpha` is symmetric around 0.5, i.e. both 0.1 and 0.9 are\n        interpreted as "find the 90% confidence interval".\n\n    Returns\n    -------\n    medslope : float\n        Theil slope.\n    medintercept : float\n        Intercept of the Theil line, as ``median(y) - medslope*median(x)``.\n    lo_slope : float\n        Lower bound of the confidence interval on `medslope`.\n    up_slope : float\n        Upper bound of the confidence interval on `medslope`.\n\n    Notes\n    -----\n    The implementation of `theilslopes` follows [1]_. The intercept is\n    not defined in [1]_, and here it is defined as ``median(y) -\n    medslope*median(x)``, which is given in [3]_. Other definitions of\n    the intercept exist in the literature. A confidence interval for\n    the intercept is not given as this question is not addressed in\n    [1]_.\n\n    References\n    ----------\n    .. [1] P.K. Sen, "Estimates of the regression coefficient based on Kendall\'s tau",\n           J. Am. Stat. Assoc., Vol. 63, pp. 1379-1389, 1968.\n    .. [2] H. Theil, "A rank-invariant method of linear and polynomial\n           regression analysis I, II and III",  Nederl. Akad. Wetensch., Proc.\n           53:, pp. 386-392, pp. 521-525, pp. 1397-1412, 1950.\n    .. [3] W.L. Conover, "Practical nonparametric statistics", 2nd ed.,\n           John Wiley and Sons, New York, pp. 493.\n\n    Examples\n    --------\n    >>> from scipy import stats\n    >>> import matplotlib.pyplot as plt\n\n    >>> x = np.linspace(-5, 5, num=150)\n    >>> y = x + np.random.normal(size=x.size)\n    >>> y[11:15] += 10  # add outliers\n    >>> y[-5:] -= 7\n\n    Compute the slope, intercept and 90% confidence interval.  For comparison,\n    also compute the least-squares fit with `linregress`:\n\n    >>> res = stats.theilslopes(y, x, 0.90)\n    >>> lsq_res = stats.linregress(x, y)\n\n    Plot the results. The Theil-Sen regression line is shown in red, with the\n    dashed red lines illustrating the confidence interval of the slope (note\n    that the dashed red lines are not the confidence interval of the regression\n    as the confidence interval of the intercept is not included). The green\n    line shows the least-squares fit for comparison.\n\n    >>> fig = plt.figure()\n    >>> ax = fig.add_subplot(111)\n    >>> ax.plot(x, y, \'b.\')\n    >>> ax.plot(x, res[1] + res[0] * x, \'r-\')\n    >>> ax.plot(x, res[1] + res[2] * x, \'r--\')\n    >>> ax.plot(x, res[1] + res[3] * x, \'r--\')\n    >>> ax.plot(x, lsq_res[1] + lsq_res[0] * x, \'g-\')\n    >>> plt.show()\n\n    ')
    
    # Assigning a Call to a Name (line 205):
    
    # Assigning a Call to a Name (line 205):
    
    # Call to flatten(...): (line 205)
    # Processing the call keyword arguments (line 205)
    kwargs_626099 = {}
    
    # Call to array(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'y' (line 205)
    y_626095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 17), 'y', False)
    # Processing the call keyword arguments (line 205)
    kwargs_626096 = {}
    # Getting the type of 'np' (line 205)
    np_626093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 205)
    array_626094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), np_626093, 'array')
    # Calling array(args, kwargs) (line 205)
    array_call_result_626097 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), array_626094, *[y_626095], **kwargs_626096)
    
    # Obtaining the member 'flatten' of a type (line 205)
    flatten_626098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), array_call_result_626097, 'flatten')
    # Calling flatten(args, kwargs) (line 205)
    flatten_call_result_626100 = invoke(stypy.reporting.localization.Localization(__file__, 205, 8), flatten_626098, *[], **kwargs_626099)
    
    # Assigning a type to the variable 'y' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'y', flatten_call_result_626100)
    
    # Type idiom detected: calculating its left and rigth part (line 206)
    # Getting the type of 'x' (line 206)
    x_626101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 7), 'x')
    # Getting the type of 'None' (line 206)
    None_626102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'None')
    
    (may_be_626103, more_types_in_union_626104) = may_be_none(x_626101, None_626102)

    if may_be_626103:

        if more_types_in_union_626104:
            # Runtime conditional SSA (line 206)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 207):
        
        # Assigning a Call to a Name (line 207):
        
        # Call to arange(...): (line 207)
        # Processing the call arguments (line 207)
        
        # Call to len(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'y' (line 207)
        y_626108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 26), 'y', False)
        # Processing the call keyword arguments (line 207)
        kwargs_626109 = {}
        # Getting the type of 'len' (line 207)
        len_626107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 22), 'len', False)
        # Calling len(args, kwargs) (line 207)
        len_call_result_626110 = invoke(stypy.reporting.localization.Localization(__file__, 207, 22), len_626107, *[y_626108], **kwargs_626109)
        
        # Processing the call keyword arguments (line 207)
        # Getting the type of 'float' (line 207)
        float_626111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 36), 'float', False)
        keyword_626112 = float_626111
        kwargs_626113 = {'dtype': keyword_626112}
        # Getting the type of 'np' (line 207)
        np_626105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 207)
        arange_626106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 12), np_626105, 'arange')
        # Calling arange(args, kwargs) (line 207)
        arange_call_result_626114 = invoke(stypy.reporting.localization.Localization(__file__, 207, 12), arange_626106, *[len_call_result_626110], **kwargs_626113)
        
        # Assigning a type to the variable 'x' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'x', arange_call_result_626114)

        if more_types_in_union_626104:
            # Runtime conditional SSA for else branch (line 206)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_626103) or more_types_in_union_626104):
        
        # Assigning a Call to a Name (line 209):
        
        # Assigning a Call to a Name (line 209):
        
        # Call to flatten(...): (line 209)
        # Processing the call keyword arguments (line 209)
        kwargs_626123 = {}
        
        # Call to array(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'x' (line 209)
        x_626117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 21), 'x', False)
        # Processing the call keyword arguments (line 209)
        # Getting the type of 'float' (line 209)
        float_626118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 30), 'float', False)
        keyword_626119 = float_626118
        kwargs_626120 = {'dtype': keyword_626119}
        # Getting the type of 'np' (line 209)
        np_626115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 209)
        array_626116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 12), np_626115, 'array')
        # Calling array(args, kwargs) (line 209)
        array_call_result_626121 = invoke(stypy.reporting.localization.Localization(__file__, 209, 12), array_626116, *[x_626117], **kwargs_626120)
        
        # Obtaining the member 'flatten' of a type (line 209)
        flatten_626122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 12), array_call_result_626121, 'flatten')
        # Calling flatten(args, kwargs) (line 209)
        flatten_call_result_626124 = invoke(stypy.reporting.localization.Localization(__file__, 209, 12), flatten_626122, *[], **kwargs_626123)
        
        # Assigning a type to the variable 'x' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'x', flatten_call_result_626124)
        
        
        
        # Call to len(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'x' (line 210)
        x_626126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 15), 'x', False)
        # Processing the call keyword arguments (line 210)
        kwargs_626127 = {}
        # Getting the type of 'len' (line 210)
        len_626125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 11), 'len', False)
        # Calling len(args, kwargs) (line 210)
        len_call_result_626128 = invoke(stypy.reporting.localization.Localization(__file__, 210, 11), len_626125, *[x_626126], **kwargs_626127)
        
        
        # Call to len(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'y' (line 210)
        y_626130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 25), 'y', False)
        # Processing the call keyword arguments (line 210)
        kwargs_626131 = {}
        # Getting the type of 'len' (line 210)
        len_626129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 21), 'len', False)
        # Calling len(args, kwargs) (line 210)
        len_call_result_626132 = invoke(stypy.reporting.localization.Localization(__file__, 210, 21), len_626129, *[y_626130], **kwargs_626131)
        
        # Applying the binary operator '!=' (line 210)
        result_ne_626133 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 11), '!=', len_call_result_626128, len_call_result_626132)
        
        # Testing the type of an if condition (line 210)
        if_condition_626134 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 8), result_ne_626133)
        # Assigning a type to the variable 'if_condition_626134' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'if_condition_626134', if_condition_626134)
        # SSA begins for if statement (line 210)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 211)
        # Processing the call arguments (line 211)
        str_626136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 29), 'str', 'Incompatible lengths ! (%s<>%s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 211)
        tuple_626137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 66), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 211)
        # Adding element type (line 211)
        
        # Call to len(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'y' (line 211)
        y_626139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 70), 'y', False)
        # Processing the call keyword arguments (line 211)
        kwargs_626140 = {}
        # Getting the type of 'len' (line 211)
        len_626138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 66), 'len', False)
        # Calling len(args, kwargs) (line 211)
        len_call_result_626141 = invoke(stypy.reporting.localization.Localization(__file__, 211, 66), len_626138, *[y_626139], **kwargs_626140)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 66), tuple_626137, len_call_result_626141)
        # Adding element type (line 211)
        
        # Call to len(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'x' (line 211)
        x_626143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 78), 'x', False)
        # Processing the call keyword arguments (line 211)
        kwargs_626144 = {}
        # Getting the type of 'len' (line 211)
        len_626142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 74), 'len', False)
        # Calling len(args, kwargs) (line 211)
        len_call_result_626145 = invoke(stypy.reporting.localization.Localization(__file__, 211, 74), len_626142, *[x_626143], **kwargs_626144)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 66), tuple_626137, len_call_result_626145)
        
        # Applying the binary operator '%' (line 211)
        result_mod_626146 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 29), '%', str_626136, tuple_626137)
        
        # Processing the call keyword arguments (line 211)
        kwargs_626147 = {}
        # Getting the type of 'ValueError' (line 211)
        ValueError_626135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 211)
        ValueError_call_result_626148 = invoke(stypy.reporting.localization.Localization(__file__, 211, 18), ValueError_626135, *[result_mod_626146], **kwargs_626147)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 211, 12), ValueError_call_result_626148, 'raise parameter', BaseException)
        # SSA join for if statement (line 210)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_626103 and more_types_in_union_626104):
            # SSA join for if statement (line 206)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 214):
    
    # Assigning a BinOp to a Name (line 214):
    
    # Obtaining the type of the subscript
    slice_626149 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 214, 13), None, None, None)
    # Getting the type of 'np' (line 214)
    np_626150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 18), 'np')
    # Obtaining the member 'newaxis' of a type (line 214)
    newaxis_626151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 18), np_626150, 'newaxis')
    # Getting the type of 'x' (line 214)
    x_626152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 13), 'x')
    # Obtaining the member '__getitem__' of a type (line 214)
    getitem___626153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 13), x_626152, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 214)
    subscript_call_result_626154 = invoke(stypy.reporting.localization.Localization(__file__, 214, 13), getitem___626153, (slice_626149, newaxis_626151))
    
    # Getting the type of 'x' (line 214)
    x_626155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 32), 'x')
    # Applying the binary operator '-' (line 214)
    result_sub_626156 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 13), '-', subscript_call_result_626154, x_626155)
    
    # Assigning a type to the variable 'deltax' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'deltax', result_sub_626156)
    
    # Assigning a BinOp to a Name (line 215):
    
    # Assigning a BinOp to a Name (line 215):
    
    # Obtaining the type of the subscript
    slice_626157 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 215, 13), None, None, None)
    # Getting the type of 'np' (line 215)
    np_626158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 18), 'np')
    # Obtaining the member 'newaxis' of a type (line 215)
    newaxis_626159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 18), np_626158, 'newaxis')
    # Getting the type of 'y' (line 215)
    y_626160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 13), 'y')
    # Obtaining the member '__getitem__' of a type (line 215)
    getitem___626161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 13), y_626160, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 215)
    subscript_call_result_626162 = invoke(stypy.reporting.localization.Localization(__file__, 215, 13), getitem___626161, (slice_626157, newaxis_626159))
    
    # Getting the type of 'y' (line 215)
    y_626163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 32), 'y')
    # Applying the binary operator '-' (line 215)
    result_sub_626164 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 13), '-', subscript_call_result_626162, y_626163)
    
    # Assigning a type to the variable 'deltay' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'deltay', result_sub_626164)
    
    # Assigning a BinOp to a Name (line 216):
    
    # Assigning a BinOp to a Name (line 216):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'deltax' (line 216)
    deltax_626165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 20), 'deltax')
    int_626166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 29), 'int')
    # Applying the binary operator '>' (line 216)
    result_gt_626167 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 20), '>', deltax_626165, int_626166)
    
    # Getting the type of 'deltay' (line 216)
    deltay_626168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 13), 'deltay')
    # Obtaining the member '__getitem__' of a type (line 216)
    getitem___626169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 13), deltay_626168, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 216)
    subscript_call_result_626170 = invoke(stypy.reporting.localization.Localization(__file__, 216, 13), getitem___626169, result_gt_626167)
    
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'deltax' (line 216)
    deltax_626171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 41), 'deltax')
    int_626172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 50), 'int')
    # Applying the binary operator '>' (line 216)
    result_gt_626173 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 41), '>', deltax_626171, int_626172)
    
    # Getting the type of 'deltax' (line 216)
    deltax_626174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 34), 'deltax')
    # Obtaining the member '__getitem__' of a type (line 216)
    getitem___626175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 34), deltax_626174, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 216)
    subscript_call_result_626176 = invoke(stypy.reporting.localization.Localization(__file__, 216, 34), getitem___626175, result_gt_626173)
    
    # Applying the binary operator 'div' (line 216)
    result_div_626177 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 13), 'div', subscript_call_result_626170, subscript_call_result_626176)
    
    # Assigning a type to the variable 'slopes' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'slopes', result_div_626177)
    
    # Call to sort(...): (line 217)
    # Processing the call keyword arguments (line 217)
    kwargs_626180 = {}
    # Getting the type of 'slopes' (line 217)
    slopes_626178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'slopes', False)
    # Obtaining the member 'sort' of a type (line 217)
    sort_626179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 4), slopes_626178, 'sort')
    # Calling sort(args, kwargs) (line 217)
    sort_call_result_626181 = invoke(stypy.reporting.localization.Localization(__file__, 217, 4), sort_626179, *[], **kwargs_626180)
    
    
    # Assigning a Call to a Name (line 218):
    
    # Assigning a Call to a Name (line 218):
    
    # Call to median(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'slopes' (line 218)
    slopes_626184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 25), 'slopes', False)
    # Processing the call keyword arguments (line 218)
    kwargs_626185 = {}
    # Getting the type of 'np' (line 218)
    np_626182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'np', False)
    # Obtaining the member 'median' of a type (line 218)
    median_626183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 15), np_626182, 'median')
    # Calling median(args, kwargs) (line 218)
    median_call_result_626186 = invoke(stypy.reporting.localization.Localization(__file__, 218, 15), median_626183, *[slopes_626184], **kwargs_626185)
    
    # Assigning a type to the variable 'medslope' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'medslope', median_call_result_626186)
    
    # Assigning a BinOp to a Name (line 219):
    
    # Assigning a BinOp to a Name (line 219):
    
    # Call to median(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'y' (line 219)
    y_626189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 25), 'y', False)
    # Processing the call keyword arguments (line 219)
    kwargs_626190 = {}
    # Getting the type of 'np' (line 219)
    np_626187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 15), 'np', False)
    # Obtaining the member 'median' of a type (line 219)
    median_626188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 15), np_626187, 'median')
    # Calling median(args, kwargs) (line 219)
    median_call_result_626191 = invoke(stypy.reporting.localization.Localization(__file__, 219, 15), median_626188, *[y_626189], **kwargs_626190)
    
    # Getting the type of 'medslope' (line 219)
    medslope_626192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), 'medslope')
    
    # Call to median(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'x' (line 219)
    x_626195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 51), 'x', False)
    # Processing the call keyword arguments (line 219)
    kwargs_626196 = {}
    # Getting the type of 'np' (line 219)
    np_626193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 41), 'np', False)
    # Obtaining the member 'median' of a type (line 219)
    median_626194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 41), np_626193, 'median')
    # Calling median(args, kwargs) (line 219)
    median_call_result_626197 = invoke(stypy.reporting.localization.Localization(__file__, 219, 41), median_626194, *[x_626195], **kwargs_626196)
    
    # Applying the binary operator '*' (line 219)
    result_mul_626198 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 30), '*', medslope_626192, median_call_result_626197)
    
    # Applying the binary operator '-' (line 219)
    result_sub_626199 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 15), '-', median_call_result_626191, result_mul_626198)
    
    # Assigning a type to the variable 'medinter' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'medinter', result_sub_626199)
    
    
    # Getting the type of 'alpha' (line 221)
    alpha_626200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 7), 'alpha')
    float_626201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 15), 'float')
    # Applying the binary operator '>' (line 221)
    result_gt_626202 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 7), '>', alpha_626200, float_626201)
    
    # Testing the type of an if condition (line 221)
    if_condition_626203 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 4), result_gt_626202)
    # Assigning a type to the variable 'if_condition_626203' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'if_condition_626203', if_condition_626203)
    # SSA begins for if statement (line 221)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 222):
    
    # Assigning a BinOp to a Name (line 222):
    float_626204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 16), 'float')
    # Getting the type of 'alpha' (line 222)
    alpha_626205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 21), 'alpha')
    # Applying the binary operator '-' (line 222)
    result_sub_626206 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 16), '-', float_626204, alpha_626205)
    
    # Assigning a type to the variable 'alpha' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'alpha', result_sub_626206)
    # SSA join for if statement (line 221)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 224):
    
    # Assigning a Call to a Name (line 224):
    
    # Call to ppf(...): (line 224)
    # Processing the call arguments (line 224)
    # Getting the type of 'alpha' (line 224)
    alpha_626210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 31), 'alpha', False)
    float_626211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 39), 'float')
    # Applying the binary operator 'div' (line 224)
    result_div_626212 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 31), 'div', alpha_626210, float_626211)
    
    # Processing the call keyword arguments (line 224)
    kwargs_626213 = {}
    # Getting the type of 'distributions' (line 224)
    distributions_626207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'distributions', False)
    # Obtaining the member 'norm' of a type (line 224)
    norm_626208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), distributions_626207, 'norm')
    # Obtaining the member 'ppf' of a type (line 224)
    ppf_626209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), norm_626208, 'ppf')
    # Calling ppf(args, kwargs) (line 224)
    ppf_call_result_626214 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), ppf_626209, *[result_div_626212], **kwargs_626213)
    
    # Assigning a type to the variable 'z' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'z', ppf_call_result_626214)
    
    # Assigning a Call to a Tuple (line 226):
    
    # Assigning a Subscript to a Name (line 226):
    
    # Obtaining the type of the subscript
    int_626215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 4), 'int')
    
    # Call to _find_repeats(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'x' (line 226)
    x_626217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 30), 'x', False)
    # Processing the call keyword arguments (line 226)
    kwargs_626218 = {}
    # Getting the type of '_find_repeats' (line 226)
    _find_repeats_626216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), '_find_repeats', False)
    # Calling _find_repeats(args, kwargs) (line 226)
    _find_repeats_call_result_626219 = invoke(stypy.reporting.localization.Localization(__file__, 226, 16), _find_repeats_626216, *[x_626217], **kwargs_626218)
    
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___626220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 4), _find_repeats_call_result_626219, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_626221 = invoke(stypy.reporting.localization.Localization(__file__, 226, 4), getitem___626220, int_626215)
    
    # Assigning a type to the variable 'tuple_var_assignment_625805' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_625805', subscript_call_result_626221)
    
    # Assigning a Subscript to a Name (line 226):
    
    # Obtaining the type of the subscript
    int_626222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 4), 'int')
    
    # Call to _find_repeats(...): (line 226)
    # Processing the call arguments (line 226)
    # Getting the type of 'x' (line 226)
    x_626224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 30), 'x', False)
    # Processing the call keyword arguments (line 226)
    kwargs_626225 = {}
    # Getting the type of '_find_repeats' (line 226)
    _find_repeats_626223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), '_find_repeats', False)
    # Calling _find_repeats(args, kwargs) (line 226)
    _find_repeats_call_result_626226 = invoke(stypy.reporting.localization.Localization(__file__, 226, 16), _find_repeats_626223, *[x_626224], **kwargs_626225)
    
    # Obtaining the member '__getitem__' of a type (line 226)
    getitem___626227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 4), _find_repeats_call_result_626226, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 226)
    subscript_call_result_626228 = invoke(stypy.reporting.localization.Localization(__file__, 226, 4), getitem___626227, int_626222)
    
    # Assigning a type to the variable 'tuple_var_assignment_625806' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_625806', subscript_call_result_626228)
    
    # Assigning a Name to a Name (line 226):
    # Getting the type of 'tuple_var_assignment_625805' (line 226)
    tuple_var_assignment_625805_626229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_625805')
    # Assigning a type to the variable '_' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), '_', tuple_var_assignment_625805_626229)
    
    # Assigning a Name to a Name (line 226):
    # Getting the type of 'tuple_var_assignment_625806' (line 226)
    tuple_var_assignment_625806_626230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'tuple_var_assignment_625806')
    # Assigning a type to the variable 'nxreps' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 7), 'nxreps', tuple_var_assignment_625806_626230)
    
    # Assigning a Call to a Tuple (line 227):
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_626231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 4), 'int')
    
    # Call to _find_repeats(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'y' (line 227)
    y_626233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 30), 'y', False)
    # Processing the call keyword arguments (line 227)
    kwargs_626234 = {}
    # Getting the type of '_find_repeats' (line 227)
    _find_repeats_626232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), '_find_repeats', False)
    # Calling _find_repeats(args, kwargs) (line 227)
    _find_repeats_call_result_626235 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), _find_repeats_626232, *[y_626233], **kwargs_626234)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___626236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 4), _find_repeats_call_result_626235, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_626237 = invoke(stypy.reporting.localization.Localization(__file__, 227, 4), getitem___626236, int_626231)
    
    # Assigning a type to the variable 'tuple_var_assignment_625807' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'tuple_var_assignment_625807', subscript_call_result_626237)
    
    # Assigning a Subscript to a Name (line 227):
    
    # Obtaining the type of the subscript
    int_626238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 4), 'int')
    
    # Call to _find_repeats(...): (line 227)
    # Processing the call arguments (line 227)
    # Getting the type of 'y' (line 227)
    y_626240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 30), 'y', False)
    # Processing the call keyword arguments (line 227)
    kwargs_626241 = {}
    # Getting the type of '_find_repeats' (line 227)
    _find_repeats_626239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), '_find_repeats', False)
    # Calling _find_repeats(args, kwargs) (line 227)
    _find_repeats_call_result_626242 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), _find_repeats_626239, *[y_626240], **kwargs_626241)
    
    # Obtaining the member '__getitem__' of a type (line 227)
    getitem___626243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 4), _find_repeats_call_result_626242, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 227)
    subscript_call_result_626244 = invoke(stypy.reporting.localization.Localization(__file__, 227, 4), getitem___626243, int_626238)
    
    # Assigning a type to the variable 'tuple_var_assignment_625808' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'tuple_var_assignment_625808', subscript_call_result_626244)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_625807' (line 227)
    tuple_var_assignment_625807_626245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'tuple_var_assignment_625807')
    # Assigning a type to the variable '_' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), '_', tuple_var_assignment_625807_626245)
    
    # Assigning a Name to a Name (line 227):
    # Getting the type of 'tuple_var_assignment_625808' (line 227)
    tuple_var_assignment_625808_626246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'tuple_var_assignment_625808')
    # Assigning a type to the variable 'nyreps' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 7), 'nyreps', tuple_var_assignment_625808_626246)
    
    # Assigning a Call to a Name (line 228):
    
    # Assigning a Call to a Name (line 228):
    
    # Call to len(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'slopes' (line 228)
    slopes_626248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 13), 'slopes', False)
    # Processing the call keyword arguments (line 228)
    kwargs_626249 = {}
    # Getting the type of 'len' (line 228)
    len_626247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 9), 'len', False)
    # Calling len(args, kwargs) (line 228)
    len_call_result_626250 = invoke(stypy.reporting.localization.Localization(__file__, 228, 9), len_626247, *[slopes_626248], **kwargs_626249)
    
    # Assigning a type to the variable 'nt' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'nt', len_call_result_626250)
    
    # Assigning a Call to a Name (line 229):
    
    # Assigning a Call to a Name (line 229):
    
    # Call to len(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'y' (line 229)
    y_626252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 13), 'y', False)
    # Processing the call keyword arguments (line 229)
    kwargs_626253 = {}
    # Getting the type of 'len' (line 229)
    len_626251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 9), 'len', False)
    # Calling len(args, kwargs) (line 229)
    len_call_result_626254 = invoke(stypy.reporting.localization.Localization(__file__, 229, 9), len_626251, *[y_626252], **kwargs_626253)
    
    # Assigning a type to the variable 'ny' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'ny', len_call_result_626254)
    
    # Assigning a BinOp to a Name (line 231):
    
    # Assigning a BinOp to a Name (line 231):
    int_626255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 12), 'int')
    float_626256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 14), 'float')
    # Applying the binary operator 'div' (line 231)
    result_div_626257 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 12), 'div', int_626255, float_626256)
    
    # Getting the type of 'ny' (line 231)
    ny_626258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 21), 'ny')
    # Getting the type of 'ny' (line 231)
    ny_626259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 27), 'ny')
    int_626260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 30), 'int')
    # Applying the binary operator '-' (line 231)
    result_sub_626261 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 27), '-', ny_626259, int_626260)
    
    # Applying the binary operator '*' (line 231)
    result_mul_626262 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 21), '*', ny_626258, result_sub_626261)
    
    int_626263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 36), 'int')
    # Getting the type of 'ny' (line 231)
    ny_626264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 38), 'ny')
    # Applying the binary operator '*' (line 231)
    result_mul_626265 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 36), '*', int_626263, ny_626264)
    
    int_626266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 41), 'int')
    # Applying the binary operator '+' (line 231)
    result_add_626267 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 36), '+', result_mul_626265, int_626266)
    
    # Applying the binary operator '*' (line 231)
    result_mul_626268 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 33), '*', result_mul_626262, result_add_626267)
    
    
    # Call to sum(...): (line 232)
    # Processing the call arguments (line 232)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 232, 28, True)
    # Calculating comprehension expression
    # Getting the type of 'nxreps' (line 232)
    nxreps_626282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 59), 'nxreps', False)
    comprehension_626283 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 28), nxreps_626282)
    # Assigning a type to the variable 'k' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 28), 'k', comprehension_626283)
    # Getting the type of 'k' (line 232)
    k_626271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 28), 'k', False)
    # Getting the type of 'k' (line 232)
    k_626272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 33), 'k', False)
    int_626273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 35), 'int')
    # Applying the binary operator '-' (line 232)
    result_sub_626274 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 33), '-', k_626272, int_626273)
    
    # Applying the binary operator '*' (line 232)
    result_mul_626275 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 28), '*', k_626271, result_sub_626274)
    
    int_626276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 41), 'int')
    # Getting the type of 'k' (line 232)
    k_626277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 43), 'k', False)
    # Applying the binary operator '*' (line 232)
    result_mul_626278 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 41), '*', int_626276, k_626277)
    
    int_626279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 47), 'int')
    # Applying the binary operator '+' (line 232)
    result_add_626280 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 41), '+', result_mul_626278, int_626279)
    
    # Applying the binary operator '*' (line 232)
    result_mul_626281 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 38), '*', result_mul_626275, result_add_626280)
    
    list_626284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 28), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 28), list_626284, result_mul_626281)
    # Processing the call keyword arguments (line 232)
    kwargs_626285 = {}
    # Getting the type of 'np' (line 232)
    np_626269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 21), 'np', False)
    # Obtaining the member 'sum' of a type (line 232)
    sum_626270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 21), np_626269, 'sum')
    # Calling sum(args, kwargs) (line 232)
    sum_call_result_626286 = invoke(stypy.reporting.localization.Localization(__file__, 232, 21), sum_626270, *[list_626284], **kwargs_626285)
    
    # Applying the binary operator '-' (line 231)
    result_sub_626287 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 21), '-', result_mul_626268, sum_call_result_626286)
    
    
    # Call to sum(...): (line 233)
    # Processing the call arguments (line 233)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 233, 28, True)
    # Calculating comprehension expression
    # Getting the type of 'nyreps' (line 233)
    nyreps_626301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 59), 'nyreps', False)
    comprehension_626302 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 28), nyreps_626301)
    # Assigning a type to the variable 'k' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 28), 'k', comprehension_626302)
    # Getting the type of 'k' (line 233)
    k_626290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 28), 'k', False)
    # Getting the type of 'k' (line 233)
    k_626291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 33), 'k', False)
    int_626292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 35), 'int')
    # Applying the binary operator '-' (line 233)
    result_sub_626293 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 33), '-', k_626291, int_626292)
    
    # Applying the binary operator '*' (line 233)
    result_mul_626294 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 28), '*', k_626290, result_sub_626293)
    
    int_626295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 41), 'int')
    # Getting the type of 'k' (line 233)
    k_626296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 43), 'k', False)
    # Applying the binary operator '*' (line 233)
    result_mul_626297 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 41), '*', int_626295, k_626296)
    
    int_626298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 47), 'int')
    # Applying the binary operator '+' (line 233)
    result_add_626299 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 41), '+', result_mul_626297, int_626298)
    
    # Applying the binary operator '*' (line 233)
    result_mul_626300 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 38), '*', result_mul_626294, result_add_626299)
    
    list_626303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 28), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 28), list_626303, result_mul_626300)
    # Processing the call keyword arguments (line 233)
    kwargs_626304 = {}
    # Getting the type of 'np' (line 233)
    np_626288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 21), 'np', False)
    # Obtaining the member 'sum' of a type (line 233)
    sum_626289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 21), np_626288, 'sum')
    # Calling sum(args, kwargs) (line 233)
    sum_call_result_626305 = invoke(stypy.reporting.localization.Localization(__file__, 233, 21), sum_626289, *[list_626303], **kwargs_626304)
    
    # Applying the binary operator '-' (line 232)
    result_sub_626306 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 67), '-', result_sub_626287, sum_call_result_626305)
    
    # Applying the binary operator '*' (line 231)
    result_mul_626307 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 18), '*', result_div_626257, result_sub_626306)
    
    # Assigning a type to the variable 'sigsq' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'sigsq', result_mul_626307)
    
    # Assigning a Call to a Name (line 235):
    
    # Assigning a Call to a Name (line 235):
    
    # Call to sqrt(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'sigsq' (line 235)
    sigsq_626310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 20), 'sigsq', False)
    # Processing the call keyword arguments (line 235)
    kwargs_626311 = {}
    # Getting the type of 'np' (line 235)
    np_626308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'np', False)
    # Obtaining the member 'sqrt' of a type (line 235)
    sqrt_626309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 12), np_626308, 'sqrt')
    # Calling sqrt(args, kwargs) (line 235)
    sqrt_call_result_626312 = invoke(stypy.reporting.localization.Localization(__file__, 235, 12), sqrt_626309, *[sigsq_626310], **kwargs_626311)
    
    # Assigning a type to the variable 'sigma' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'sigma', sqrt_call_result_626312)
    
    # Assigning a Call to a Name (line 236):
    
    # Assigning a Call to a Name (line 236):
    
    # Call to min(...): (line 236)
    # Processing the call arguments (line 236)
    
    # Call to int(...): (line 236)
    # Processing the call arguments (line 236)
    
    # Call to round(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'nt' (line 236)
    nt_626317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 27), 'nt', False)
    # Getting the type of 'z' (line 236)
    z_626318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 32), 'z', False)
    # Getting the type of 'sigma' (line 236)
    sigma_626319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 34), 'sigma', False)
    # Applying the binary operator '*' (line 236)
    result_mul_626320 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 32), '*', z_626318, sigma_626319)
    
    # Applying the binary operator '-' (line 236)
    result_sub_626321 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 27), '-', nt_626317, result_mul_626320)
    
    float_626322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 41), 'float')
    # Applying the binary operator 'div' (line 236)
    result_div_626323 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 26), 'div', result_sub_626321, float_626322)
    
    # Processing the call keyword arguments (line 236)
    kwargs_626324 = {}
    # Getting the type of 'np' (line 236)
    np_626315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 17), 'np', False)
    # Obtaining the member 'round' of a type (line 236)
    round_626316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 17), np_626315, 'round')
    # Calling round(args, kwargs) (line 236)
    round_call_result_626325 = invoke(stypy.reporting.localization.Localization(__file__, 236, 17), round_626316, *[result_div_626323], **kwargs_626324)
    
    # Processing the call keyword arguments (line 236)
    kwargs_626326 = {}
    # Getting the type of 'int' (line 236)
    int_626314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 13), 'int', False)
    # Calling int(args, kwargs) (line 236)
    int_call_result_626327 = invoke(stypy.reporting.localization.Localization(__file__, 236, 13), int_626314, *[round_call_result_626325], **kwargs_626326)
    
    
    # Call to len(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'slopes' (line 236)
    slopes_626329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 51), 'slopes', False)
    # Processing the call keyword arguments (line 236)
    kwargs_626330 = {}
    # Getting the type of 'len' (line 236)
    len_626328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 47), 'len', False)
    # Calling len(args, kwargs) (line 236)
    len_call_result_626331 = invoke(stypy.reporting.localization.Localization(__file__, 236, 47), len_626328, *[slopes_626329], **kwargs_626330)
    
    int_626332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 59), 'int')
    # Applying the binary operator '-' (line 236)
    result_sub_626333 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 47), '-', len_call_result_626331, int_626332)
    
    # Processing the call keyword arguments (line 236)
    kwargs_626334 = {}
    # Getting the type of 'min' (line 236)
    min_626313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 9), 'min', False)
    # Calling min(args, kwargs) (line 236)
    min_call_result_626335 = invoke(stypy.reporting.localization.Localization(__file__, 236, 9), min_626313, *[int_call_result_626327, result_sub_626333], **kwargs_626334)
    
    # Assigning a type to the variable 'Ru' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'Ru', min_call_result_626335)
    
    # Assigning a Call to a Name (line 237):
    
    # Assigning a Call to a Name (line 237):
    
    # Call to max(...): (line 237)
    # Processing the call arguments (line 237)
    
    # Call to int(...): (line 237)
    # Processing the call arguments (line 237)
    
    # Call to round(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'nt' (line 237)
    nt_626340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 27), 'nt', False)
    # Getting the type of 'z' (line 237)
    z_626341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 32), 'z', False)
    # Getting the type of 'sigma' (line 237)
    sigma_626342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 34), 'sigma', False)
    # Applying the binary operator '*' (line 237)
    result_mul_626343 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 32), '*', z_626341, sigma_626342)
    
    # Applying the binary operator '+' (line 237)
    result_add_626344 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 27), '+', nt_626340, result_mul_626343)
    
    float_626345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 41), 'float')
    # Applying the binary operator 'div' (line 237)
    result_div_626346 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 26), 'div', result_add_626344, float_626345)
    
    # Processing the call keyword arguments (line 237)
    kwargs_626347 = {}
    # Getting the type of 'np' (line 237)
    np_626338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 17), 'np', False)
    # Obtaining the member 'round' of a type (line 237)
    round_626339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 17), np_626338, 'round')
    # Calling round(args, kwargs) (line 237)
    round_call_result_626348 = invoke(stypy.reporting.localization.Localization(__file__, 237, 17), round_626339, *[result_div_626346], **kwargs_626347)
    
    # Processing the call keyword arguments (line 237)
    kwargs_626349 = {}
    # Getting the type of 'int' (line 237)
    int_626337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 13), 'int', False)
    # Calling int(args, kwargs) (line 237)
    int_call_result_626350 = invoke(stypy.reporting.localization.Localization(__file__, 237, 13), int_626337, *[round_call_result_626348], **kwargs_626349)
    
    int_626351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 48), 'int')
    # Applying the binary operator '-' (line 237)
    result_sub_626352 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 13), '-', int_call_result_626350, int_626351)
    
    int_626353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 51), 'int')
    # Processing the call keyword arguments (line 237)
    kwargs_626354 = {}
    # Getting the type of 'max' (line 237)
    max_626336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 9), 'max', False)
    # Calling max(args, kwargs) (line 237)
    max_call_result_626355 = invoke(stypy.reporting.localization.Localization(__file__, 237, 9), max_626336, *[result_sub_626352, int_626353], **kwargs_626354)
    
    # Assigning a type to the variable 'Rl' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'Rl', max_call_result_626355)
    
    # Assigning a Subscript to a Name (line 238):
    
    # Assigning a Subscript to a Name (line 238):
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'list' (line 238)
    list_626356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 238)
    # Adding element type (line 238)
    # Getting the type of 'Rl' (line 238)
    Rl_626357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 20), 'Rl')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 19), list_626356, Rl_626357)
    # Adding element type (line 238)
    # Getting the type of 'Ru' (line 238)
    Ru_626358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 24), 'Ru')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 19), list_626356, Ru_626358)
    
    # Getting the type of 'slopes' (line 238)
    slopes_626359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'slopes')
    # Obtaining the member '__getitem__' of a type (line 238)
    getitem___626360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), slopes_626359, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 238)
    subscript_call_result_626361 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), getitem___626360, list_626356)
    
    # Assigning a type to the variable 'delta' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'delta', subscript_call_result_626361)
    
    # Obtaining an instance of the builtin type 'tuple' (line 239)
    tuple_626362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 239)
    # Adding element type (line 239)
    # Getting the type of 'medslope' (line 239)
    medslope_626363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 11), 'medslope')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 11), tuple_626362, medslope_626363)
    # Adding element type (line 239)
    # Getting the type of 'medinter' (line 239)
    medinter_626364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 21), 'medinter')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 11), tuple_626362, medinter_626364)
    # Adding element type (line 239)
    
    # Obtaining the type of the subscript
    int_626365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 37), 'int')
    # Getting the type of 'delta' (line 239)
    delta_626366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 31), 'delta')
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___626367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 31), delta_626366, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_626368 = invoke(stypy.reporting.localization.Localization(__file__, 239, 31), getitem___626367, int_626365)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 11), tuple_626362, subscript_call_result_626368)
    # Adding element type (line 239)
    
    # Obtaining the type of the subscript
    int_626369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 47), 'int')
    # Getting the type of 'delta' (line 239)
    delta_626370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 41), 'delta')
    # Obtaining the member '__getitem__' of a type (line 239)
    getitem___626371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 41), delta_626370, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 239)
    subscript_call_result_626372 = invoke(stypy.reporting.localization.Localization(__file__, 239, 41), getitem___626371, int_626369)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 11), tuple_626362, subscript_call_result_626372)
    
    # Assigning a type to the variable 'stypy_return_type' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'stypy_return_type', tuple_626362)
    
    # ################# End of 'theilslopes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'theilslopes' in the type store
    # Getting the type of 'stypy_return_type' (line 124)
    stypy_return_type_626373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_626373)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'theilslopes'
    return stypy_return_type_626373

# Assigning a type to the variable 'theilslopes' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'theilslopes', theilslopes)

@norecursion
def _find_repeats(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_find_repeats'
    module_type_store = module_type_store.open_function_context('_find_repeats', 242, 0, False)
    
    # Passed parameters checking function
    _find_repeats.stypy_localization = localization
    _find_repeats.stypy_type_of_self = None
    _find_repeats.stypy_type_store = module_type_store
    _find_repeats.stypy_function_name = '_find_repeats'
    _find_repeats.stypy_param_names_list = ['arr']
    _find_repeats.stypy_varargs_param_name = None
    _find_repeats.stypy_kwargs_param_name = None
    _find_repeats.stypy_call_defaults = defaults
    _find_repeats.stypy_call_varargs = varargs
    _find_repeats.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_find_repeats', ['arr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_find_repeats', localization, ['arr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_find_repeats(...)' code ##################

    
    
    
    # Call to len(...): (line 244)
    # Processing the call arguments (line 244)
    # Getting the type of 'arr' (line 244)
    arr_626375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 11), 'arr', False)
    # Processing the call keyword arguments (line 244)
    kwargs_626376 = {}
    # Getting the type of 'len' (line 244)
    len_626374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 7), 'len', False)
    # Calling len(args, kwargs) (line 244)
    len_call_result_626377 = invoke(stypy.reporting.localization.Localization(__file__, 244, 7), len_626374, *[arr_626375], **kwargs_626376)
    
    int_626378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 19), 'int')
    # Applying the binary operator '==' (line 244)
    result_eq_626379 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 7), '==', len_call_result_626377, int_626378)
    
    # Testing the type of an if condition (line 244)
    if_condition_626380 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 244, 4), result_eq_626379)
    # Assigning a type to the variable 'if_condition_626380' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'if_condition_626380', if_condition_626380)
    # SSA begins for if statement (line 244)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 245)
    tuple_626381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 245)
    # Adding element type (line 245)
    
    # Call to array(...): (line 245)
    # Processing the call arguments (line 245)
    int_626384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 24), 'int')
    # Getting the type of 'np' (line 245)
    np_626385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 27), 'np', False)
    # Obtaining the member 'float64' of a type (line 245)
    float64_626386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 27), np_626385, 'float64')
    # Processing the call keyword arguments (line 245)
    kwargs_626387 = {}
    # Getting the type of 'np' (line 245)
    np_626382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'np', False)
    # Obtaining the member 'array' of a type (line 245)
    array_626383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 15), np_626382, 'array')
    # Calling array(args, kwargs) (line 245)
    array_call_result_626388 = invoke(stypy.reporting.localization.Localization(__file__, 245, 15), array_626383, *[int_626384, float64_626386], **kwargs_626387)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), tuple_626381, array_call_result_626388)
    # Adding element type (line 245)
    
    # Call to array(...): (line 245)
    # Processing the call arguments (line 245)
    int_626391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 49), 'int')
    # Getting the type of 'np' (line 245)
    np_626392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 52), 'np', False)
    # Obtaining the member 'intp' of a type (line 245)
    intp_626393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 52), np_626392, 'intp')
    # Processing the call keyword arguments (line 245)
    kwargs_626394 = {}
    # Getting the type of 'np' (line 245)
    np_626389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 40), 'np', False)
    # Obtaining the member 'array' of a type (line 245)
    array_626390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 40), np_626389, 'array')
    # Calling array(args, kwargs) (line 245)
    array_call_result_626395 = invoke(stypy.reporting.localization.Localization(__file__, 245, 40), array_626390, *[int_626391, intp_626393], **kwargs_626394)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), tuple_626381, array_call_result_626395)
    
    # Assigning a type to the variable 'stypy_return_type' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'stypy_return_type', tuple_626381)
    # SSA join for if statement (line 244)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 249):
    
    # Assigning a Call to a Name (line 249):
    
    # Call to ravel(...): (line 249)
    # Processing the call keyword arguments (line 249)
    kwargs_626404 = {}
    
    # Call to asarray(...): (line 249)
    # Processing the call arguments (line 249)
    # Getting the type of 'arr' (line 249)
    arr_626398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 21), 'arr', False)
    # Getting the type of 'np' (line 249)
    np_626399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 26), 'np', False)
    # Obtaining the member 'float64' of a type (line 249)
    float64_626400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 26), np_626399, 'float64')
    # Processing the call keyword arguments (line 249)
    kwargs_626401 = {}
    # Getting the type of 'np' (line 249)
    np_626396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 10), 'np', False)
    # Obtaining the member 'asarray' of a type (line 249)
    asarray_626397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 10), np_626396, 'asarray')
    # Calling asarray(args, kwargs) (line 249)
    asarray_call_result_626402 = invoke(stypy.reporting.localization.Localization(__file__, 249, 10), asarray_626397, *[arr_626398, float64_626400], **kwargs_626401)
    
    # Obtaining the member 'ravel' of a type (line 249)
    ravel_626403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 10), asarray_call_result_626402, 'ravel')
    # Calling ravel(args, kwargs) (line 249)
    ravel_call_result_626405 = invoke(stypy.reporting.localization.Localization(__file__, 249, 10), ravel_626403, *[], **kwargs_626404)
    
    # Assigning a type to the variable 'arr' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'arr', ravel_call_result_626405)
    
    # Call to sort(...): (line 250)
    # Processing the call keyword arguments (line 250)
    kwargs_626408 = {}
    # Getting the type of 'arr' (line 250)
    arr_626406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'arr', False)
    # Obtaining the member 'sort' of a type (line 250)
    sort_626407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 4), arr_626406, 'sort')
    # Calling sort(args, kwargs) (line 250)
    sort_call_result_626409 = invoke(stypy.reporting.localization.Localization(__file__, 250, 4), sort_626407, *[], **kwargs_626408)
    
    
    # Assigning a Call to a Name (line 253):
    
    # Assigning a Call to a Name (line 253):
    
    # Call to concatenate(...): (line 253)
    # Processing the call arguments (line 253)
    
    # Obtaining an instance of the builtin type 'tuple' (line 253)
    tuple_626412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 253)
    # Adding element type (line 253)
    
    # Obtaining an instance of the builtin type 'list' (line 253)
    list_626413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 253)
    # Adding element type (line 253)
    # Getting the type of 'True' (line 253)
    True_626414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 30), 'True', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 29), list_626413, True_626414)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 29), tuple_626412, list_626413)
    # Adding element type (line 253)
    
    
    # Obtaining the type of the subscript
    int_626415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 41), 'int')
    slice_626416 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 253, 37), int_626415, None, None)
    # Getting the type of 'arr' (line 253)
    arr_626417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 37), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___626418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 37), arr_626417, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_626419 = invoke(stypy.reporting.localization.Localization(__file__, 253, 37), getitem___626418, slice_626416)
    
    
    # Obtaining the type of the subscript
    int_626420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 53), 'int')
    slice_626421 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 253, 48), None, int_626420, None)
    # Getting the type of 'arr' (line 253)
    arr_626422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 48), 'arr', False)
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___626423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 48), arr_626422, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_626424 = invoke(stypy.reporting.localization.Localization(__file__, 253, 48), getitem___626423, slice_626421)
    
    # Applying the binary operator '!=' (line 253)
    result_ne_626425 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 37), '!=', subscript_call_result_626419, subscript_call_result_626424)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 29), tuple_626412, result_ne_626425)
    
    # Processing the call keyword arguments (line 253)
    kwargs_626426 = {}
    # Getting the type of 'np' (line 253)
    np_626410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 13), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 253)
    concatenate_626411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 13), np_626410, 'concatenate')
    # Calling concatenate(args, kwargs) (line 253)
    concatenate_call_result_626427 = invoke(stypy.reporting.localization.Localization(__file__, 253, 13), concatenate_626411, *[tuple_626412], **kwargs_626426)
    
    # Assigning a type to the variable 'change' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'change', concatenate_call_result_626427)
    
    # Assigning a Subscript to a Name (line 254):
    
    # Assigning a Subscript to a Name (line 254):
    
    # Obtaining the type of the subscript
    # Getting the type of 'change' (line 254)
    change_626428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 17), 'change')
    # Getting the type of 'arr' (line 254)
    arr_626429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 13), 'arr')
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___626430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 13), arr_626429, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_626431 = invoke(stypy.reporting.localization.Localization(__file__, 254, 13), getitem___626430, change_626428)
    
    # Assigning a type to the variable 'unique' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'unique', subscript_call_result_626431)
    
    # Assigning a Call to a Name (line 255):
    
    # Assigning a Call to a Name (line 255):
    
    # Call to concatenate(...): (line 255)
    # Processing the call arguments (line 255)
    
    # Call to nonzero(...): (line 255)
    # Processing the call arguments (line 255)
    # Getting the type of 'change' (line 255)
    change_626436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 43), 'change', False)
    # Processing the call keyword arguments (line 255)
    kwargs_626437 = {}
    # Getting the type of 'np' (line 255)
    np_626434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 32), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 255)
    nonzero_626435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 32), np_626434, 'nonzero')
    # Calling nonzero(args, kwargs) (line 255)
    nonzero_call_result_626438 = invoke(stypy.reporting.localization.Localization(__file__, 255, 32), nonzero_626435, *[change_626436], **kwargs_626437)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 255)
    tuple_626439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 54), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 255)
    # Adding element type (line 255)
    
    # Obtaining an instance of the builtin type 'list' (line 255)
    list_626440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 54), 'list')
    # Adding type elements to the builtin type 'list' instance (line 255)
    # Adding element type (line 255)
    # Getting the type of 'arr' (line 255)
    arr_626441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 55), 'arr', False)
    # Obtaining the member 'size' of a type (line 255)
    size_626442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 55), arr_626441, 'size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 54), list_626440, size_626442)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 54), tuple_626439, list_626440)
    
    # Applying the binary operator '+' (line 255)
    result_add_626443 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 32), '+', nonzero_call_result_626438, tuple_626439)
    
    # Processing the call keyword arguments (line 255)
    kwargs_626444 = {}
    # Getting the type of 'np' (line 255)
    np_626432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 17), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 255)
    concatenate_626433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 17), np_626432, 'concatenate')
    # Calling concatenate(args, kwargs) (line 255)
    concatenate_call_result_626445 = invoke(stypy.reporting.localization.Localization(__file__, 255, 17), concatenate_626433, *[result_add_626443], **kwargs_626444)
    
    # Assigning a type to the variable 'change_idx' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'change_idx', concatenate_call_result_626445)
    
    # Assigning a Call to a Name (line 256):
    
    # Assigning a Call to a Name (line 256):
    
    # Call to diff(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'change_idx' (line 256)
    change_idx_626448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 19), 'change_idx', False)
    # Processing the call keyword arguments (line 256)
    kwargs_626449 = {}
    # Getting the type of 'np' (line 256)
    np_626446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 11), 'np', False)
    # Obtaining the member 'diff' of a type (line 256)
    diff_626447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 11), np_626446, 'diff')
    # Calling diff(args, kwargs) (line 256)
    diff_call_result_626450 = invoke(stypy.reporting.localization.Localization(__file__, 256, 11), diff_626447, *[change_idx_626448], **kwargs_626449)
    
    # Assigning a type to the variable 'freq' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'freq', diff_call_result_626450)
    
    # Assigning a Compare to a Name (line 257):
    
    # Assigning a Compare to a Name (line 257):
    
    # Getting the type of 'freq' (line 257)
    freq_626451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 15), 'freq')
    int_626452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 22), 'int')
    # Applying the binary operator '>' (line 257)
    result_gt_626453 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 15), '>', freq_626451, int_626452)
    
    # Assigning a type to the variable 'atleast2' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'atleast2', result_gt_626453)
    
    # Obtaining an instance of the builtin type 'tuple' (line 258)
    tuple_626454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 258)
    # Adding element type (line 258)
    
    # Obtaining the type of the subscript
    # Getting the type of 'atleast2' (line 258)
    atleast2_626455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 18), 'atleast2')
    # Getting the type of 'unique' (line 258)
    unique_626456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 11), 'unique')
    # Obtaining the member '__getitem__' of a type (line 258)
    getitem___626457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 11), unique_626456, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 258)
    subscript_call_result_626458 = invoke(stypy.reporting.localization.Localization(__file__, 258, 11), getitem___626457, atleast2_626455)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 11), tuple_626454, subscript_call_result_626458)
    # Adding element type (line 258)
    
    # Obtaining the type of the subscript
    # Getting the type of 'atleast2' (line 258)
    atleast2_626459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 34), 'atleast2')
    # Getting the type of 'freq' (line 258)
    freq_626460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 29), 'freq')
    # Obtaining the member '__getitem__' of a type (line 258)
    getitem___626461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 29), freq_626460, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 258)
    subscript_call_result_626462 = invoke(stypy.reporting.localization.Localization(__file__, 258, 29), getitem___626461, atleast2_626459)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 11), tuple_626454, subscript_call_result_626462)
    
    # Assigning a type to the variable 'stypy_return_type' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'stypy_return_type', tuple_626454)
    
    # ################# End of '_find_repeats(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_find_repeats' in the type store
    # Getting the type of 'stypy_return_type' (line 242)
    stypy_return_type_626463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_626463)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_find_repeats'
    return stypy_return_type_626463

# Assigning a type to the variable '_find_repeats' (line 242)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 0), '_find_repeats', _find_repeats)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
