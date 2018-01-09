
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from scipy.linalg import lstsq
5: from math import factorial
6: from scipy.ndimage import convolve1d
7: from ._arraytools import axis_slice
8: 
9: 
10: def savgol_coeffs(window_length, polyorder, deriv=0, delta=1.0, pos=None,
11:                   use="conv"):
12:     '''Compute the coefficients for a 1-d Savitzky-Golay FIR filter.
13: 
14:     Parameters
15:     ----------
16:     window_length : int
17:         The length of the filter window (i.e. the number of coefficients).
18:         `window_length` must be an odd positive integer.
19:     polyorder : int
20:         The order of the polynomial used to fit the samples.
21:         `polyorder` must be less than `window_length`.
22:     deriv : int, optional
23:         The order of the derivative to compute.  This must be a
24:         nonnegative integer.  The default is 0, which means to filter
25:         the data without differentiating.
26:     delta : float, optional
27:         The spacing of the samples to which the filter will be applied.
28:         This is only used if deriv > 0.
29:     pos : int or None, optional
30:         If pos is not None, it specifies evaluation position within the
31:         window.  The default is the middle of the window.
32:     use : str, optional
33:         Either 'conv' or 'dot'.  This argument chooses the order of the
34:         coefficients.  The default is 'conv', which means that the
35:         coefficients are ordered to be used in a convolution.  With
36:         use='dot', the order is reversed, so the filter is applied by
37:         dotting the coefficients with the data set.
38: 
39:     Returns
40:     -------
41:     coeffs : 1-d ndarray
42:         The filter coefficients.
43: 
44:     References
45:     ----------
46:     A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of Data by
47:     Simplified Least Squares Procedures. Analytical Chemistry, 1964, 36 (8),
48:     pp 1627-1639.
49: 
50:     See Also
51:     --------
52:     savgol_filter
53: 
54:     Notes
55:     -----
56: 
57:     .. versionadded:: 0.14.0
58: 
59:     Examples
60:     --------
61:     >>> from scipy.signal import savgol_coeffs
62:     >>> savgol_coeffs(5, 2)
63:     array([-0.08571429,  0.34285714,  0.48571429,  0.34285714, -0.08571429])
64:     >>> savgol_coeffs(5, 2, deriv=1)
65:     array([  2.00000000e-01,   1.00000000e-01,   2.00607895e-16,
66:             -1.00000000e-01,  -2.00000000e-01])
67: 
68:     Note that use='dot' simply reverses the coefficients.
69: 
70:     >>> savgol_coeffs(5, 2, pos=3)
71:     array([ 0.25714286,  0.37142857,  0.34285714,  0.17142857, -0.14285714])
72:     >>> savgol_coeffs(5, 2, pos=3, use='dot')
73:     array([-0.14285714,  0.17142857,  0.34285714,  0.37142857,  0.25714286])
74: 
75:     `x` contains data from the parabola x = t**2, sampled at
76:     t = -1, 0, 1, 2, 3.  `c` holds the coefficients that will compute the
77:     derivative at the last position.  When dotted with `x` the result should
78:     be 6.
79: 
80:     >>> x = np.array([1, 0, 1, 4, 9])
81:     >>> c = savgol_coeffs(5, 2, pos=4, deriv=1, use='dot')
82:     >>> c.dot(x)
83:     6.0000000000000018
84:     '''
85: 
86:     # An alternative method for finding the coefficients when deriv=0 is
87:     #    t = np.arange(window_length)
88:     #    unit = (t == pos).astype(int)
89:     #    coeffs = np.polyval(np.polyfit(t, unit, polyorder), t)
90:     # The method implemented here is faster.
91: 
92:     # To recreate the table of sample coefficients shown in the chapter on
93:     # the Savitzy-Golay filter in the Numerical Recipes book, use
94:     #    window_length = nL + nR + 1
95:     #    pos = nL + 1
96:     #    c = savgol_coeffs(window_length, M, pos=pos, use='dot')
97: 
98:     if polyorder >= window_length:
99:         raise ValueError("polyorder must be less than window_length.")
100: 
101:     halflen, rem = divmod(window_length, 2)
102: 
103:     if rem == 0:
104:         raise ValueError("window_length must be odd.")
105: 
106:     if pos is None:
107:         pos = halflen
108: 
109:     if not (0 <= pos < window_length):
110:         raise ValueError("pos must be nonnegative and less than "
111:                          "window_length.")
112: 
113:     if use not in ['conv', 'dot']:
114:         raise ValueError("`use` must be 'conv' or 'dot'")
115: 
116:     # Form the design matrix A.  The columns of A are powers of the integers
117:     # from -pos to window_length - pos - 1.  The powers (i.e. rows) range
118:     # from 0 to polyorder.  (That is, A is a vandermonde matrix, but not
119:     # necessarily square.)
120:     x = np.arange(-pos, window_length - pos, dtype=float)
121:     if use == "conv":
122:         # Reverse so that result can be used in a convolution.
123:         x = x[::-1]
124: 
125:     order = np.arange(polyorder + 1).reshape(-1, 1)
126:     A = x ** order
127: 
128:     # y determines which order derivative is returned.
129:     y = np.zeros(polyorder + 1)
130:     # The coefficient assigned to y[deriv] scales the result to take into
131:     # account the order of the derivative and the sample spacing.
132:     y[deriv] = factorial(deriv) / (delta ** deriv)
133: 
134:     # Find the least-squares solution of A*c = y
135:     coeffs, _, _, _ = lstsq(A, y)
136: 
137:     return coeffs
138: 
139: 
140: def _polyder(p, m):
141:     '''Differentiate polynomials represented with coefficients.
142: 
143:     p must be a 1D or 2D array.  In the 2D case, each column gives
144:     the coefficients of a polynomial; the first row holds the coefficients
145:     associated with the highest power.  m must be a nonnegative integer.
146:     (numpy.polyder doesn't handle the 2D case.)
147:     '''
148: 
149:     if m == 0:
150:         result = p
151:     else:
152:         n = len(p)
153:         if n <= m:
154:             result = np.zeros_like(p[:1, ...])
155:         else:
156:             dp = p[:-m].copy()
157:             for k in range(m):
158:                 rng = np.arange(n - k - 1, m - k - 1, -1)
159:                 dp *= rng.reshape((n - m,) + (1,) * (p.ndim - 1))
160:             result = dp
161:     return result
162: 
163: 
164: def _fit_edge(x, window_start, window_stop, interp_start, interp_stop,
165:               axis, polyorder, deriv, delta, y):
166:     '''
167:     Given an n-d array `x` and the specification of a slice of `x` from
168:     `window_start` to `window_stop` along `axis`, create an interpolating
169:     polynomial of each 1-d slice, and evaluate that polynomial in the slice
170:     from `interp_start` to `interp_stop`.  Put the result into the
171:     corresponding slice of `y`.
172:     '''
173: 
174:     # Get the edge into a (window_length, -1) array.
175:     x_edge = axis_slice(x, start=window_start, stop=window_stop, axis=axis)
176:     if axis == 0 or axis == -x.ndim:
177:         xx_edge = x_edge
178:         swapped = False
179:     else:
180:         xx_edge = x_edge.swapaxes(axis, 0)
181:         swapped = True
182:     xx_edge = xx_edge.reshape(xx_edge.shape[0], -1)
183: 
184:     # Fit the edges.  poly_coeffs has shape (polyorder + 1, -1),
185:     # where '-1' is the same as in xx_edge.
186:     poly_coeffs = np.polyfit(np.arange(0, window_stop - window_start),
187:                              xx_edge, polyorder)
188: 
189:     if deriv > 0:
190:         poly_coeffs = _polyder(poly_coeffs, deriv)
191: 
192:     # Compute the interpolated values for the edge.
193:     i = np.arange(interp_start - window_start, interp_stop - window_start)
194:     values = np.polyval(poly_coeffs, i.reshape(-1, 1)) / (delta ** deriv)
195: 
196:     # Now put the values into the appropriate slice of y.
197:     # First reshape values to match y.
198:     shp = list(y.shape)
199:     shp[0], shp[axis] = shp[axis], shp[0]
200:     values = values.reshape(interp_stop - interp_start, *shp[1:])
201:     if swapped:
202:         values = values.swapaxes(0, axis)
203:     # Get a view of the data to be replaced by values.
204:     y_edge = axis_slice(y, start=interp_start, stop=interp_stop, axis=axis)
205:     y_edge[...] = values
206: 
207: 
208: def _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, y):
209:     '''
210:     Use polynomial interpolation of x at the low and high ends of the axis
211:     to fill in the halflen values in y.
212: 
213:     This function just calls _fit_edge twice, once for each end of the axis.
214:     '''
215:     halflen = window_length // 2
216:     _fit_edge(x, 0, window_length, 0, halflen, axis,
217:               polyorder, deriv, delta, y)
218:     n = x.shape[axis]
219:     _fit_edge(x, n - window_length, n, n - halflen, n, axis,
220:               polyorder, deriv, delta, y)
221: 
222: 
223: def savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0,
224:                   axis=-1, mode='interp', cval=0.0):
225:     ''' Apply a Savitzky-Golay filter to an array.
226: 
227:     This is a 1-d filter.  If `x`  has dimension greater than 1, `axis`
228:     determines the axis along which the filter is applied.
229: 
230:     Parameters
231:     ----------
232:     x : array_like
233:         The data to be filtered.  If `x` is not a single or double precision
234:         floating point array, it will be converted to type `numpy.float64`
235:         before filtering.
236:     window_length : int
237:         The length of the filter window (i.e. the number of coefficients).
238:         `window_length` must be a positive odd integer. If `mode` is 'interp',
239:         `window_length` must be less than or equal to the size of `x`.
240:     polyorder : int
241:         The order of the polynomial used to fit the samples.
242:         `polyorder` must be less than `window_length`.
243:     deriv : int, optional
244:         The order of the derivative to compute.  This must be a
245:         nonnegative integer.  The default is 0, which means to filter
246:         the data without differentiating.
247:     delta : float, optional
248:         The spacing of the samples to which the filter will be applied.
249:         This is only used if deriv > 0.  Default is 1.0.
250:     axis : int, optional
251:         The axis of the array `x` along which the filter is to be applied.
252:         Default is -1.
253:     mode : str, optional
254:         Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
255:         determines the type of extension to use for the padded signal to
256:         which the filter is applied.  When `mode` is 'constant', the padding
257:         value is given by `cval`.  See the Notes for more details on 'mirror',
258:         'constant', 'wrap', and 'nearest'.
259:         When the 'interp' mode is selected (the default), no extension
260:         is used.  Instead, a degree `polyorder` polynomial is fit to the
261:         last `window_length` values of the edges, and this polynomial is
262:         used to evaluate the last `window_length // 2` output values.
263:     cval : scalar, optional
264:         Value to fill past the edges of the input if `mode` is 'constant'.
265:         Default is 0.0.
266: 
267:     Returns
268:     -------
269:     y : ndarray, same shape as `x`
270:         The filtered data.
271: 
272:     See Also
273:     --------
274:     savgol_coeffs
275: 
276:     Notes
277:     -----
278:     Details on the `mode` options:
279: 
280:         'mirror':
281:             Repeats the values at the edges in reverse order.  The value
282:             closest to the edge is not included.
283:         'nearest':
284:             The extension contains the nearest input value.
285:         'constant':
286:             The extension contains the value given by the `cval` argument.
287:         'wrap':
288:             The extension contains the values from the other end of the array.
289: 
290:     For example, if the input is [1, 2, 3, 4, 5, 6, 7, 8], and
291:     `window_length` is 7, the following shows the extended data for
292:     the various `mode` options (assuming `cval` is 0)::
293: 
294:         mode       |   Ext   |         Input          |   Ext
295:         -----------+---------+------------------------+---------
296:         'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5
297:         'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
298:         'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0
299:         'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3
300: 
301:     .. versionadded:: 0.14.0
302: 
303:     Examples
304:     --------
305:     >>> from scipy.signal import savgol_filter
306:     >>> np.set_printoptions(precision=2)  # For compact display.
307:     >>> x = np.array([2, 2, 5, 2, 1, 0, 1, 4, 9])
308: 
309:     Filter with a window length of 5 and a degree 2 polynomial.  Use
310:     the defaults for all other parameters.
311: 
312:     >>> savgol_filter(x, 5, 2)
313:     array([ 1.66,  3.17,  3.54,  2.86,  0.66,  0.17,  1.  ,  4.  ,  9.  ])
314: 
315:     Note that the last five values in x are samples of a parabola, so
316:     when mode='interp' (the default) is used with polyorder=2, the last
317:     three values are unchanged.  Compare that to, for example,
318:     `mode='nearest'`:
319: 
320:     >>> savgol_filter(x, 5, 2, mode='nearest')
321:     array([ 1.74,  3.03,  3.54,  2.86,  0.66,  0.17,  1.  ,  4.6 ,  7.97])
322: 
323:     '''
324:     if mode not in ["mirror", "constant", "nearest", "interp", "wrap"]:
325:         raise ValueError("mode must be 'mirror', 'constant', 'nearest' "
326:                          "'wrap' or 'interp'.")
327: 
328:     x = np.asarray(x)
329:     # Ensure that x is either single or double precision floating point.
330:     if x.dtype != np.float64 and x.dtype != np.float32:
331:         x = x.astype(np.float64)
332: 
333:     coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)
334: 
335:     if mode == "interp":
336:         if window_length > x.size:
337:             raise ValueError("If mode is 'interp', window_length must be less "
338:                              "than or equal to the size of x.")
339: 
340:         # Do not pad.  Instead, for the elements within `window_length // 2`
341:         # of the ends of the sequence, use the polynomial that is fitted to
342:         # the last `window_length` elements.
343:         y = convolve1d(x, coeffs, axis=axis, mode="constant")
344:         _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, y)
345:     else:
346:         # Any mode other than 'interp' is passed on to ndimage.convolve1d.
347:         y = convolve1d(x, coeffs, axis=axis, mode=mode, cval=cval)
348: 
349:     return y
350: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288115 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_288115) is not StypyTypeError):

    if (import_288115 != 'pyd_module'):
        __import__(import_288115)
        sys_modules_288116 = sys.modules[import_288115]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_288116.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_288115)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy.linalg import lstsq' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288117 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.linalg')

if (type(import_288117) is not StypyTypeError):

    if (import_288117 != 'pyd_module'):
        __import__(import_288117)
        sys_modules_288118 = sys.modules[import_288117]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.linalg', sys_modules_288118.module_type_store, module_type_store, ['lstsq'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_288118, sys_modules_288118.module_type_store, module_type_store)
    else:
        from scipy.linalg import lstsq

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.linalg', None, module_type_store, ['lstsq'], [lstsq])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy.linalg', import_288117)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from math import factorial' statement (line 5)
try:
    from math import factorial

except:
    factorial = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'math', None, module_type_store, ['factorial'], [factorial])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.ndimage import convolve1d' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288119 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.ndimage')

if (type(import_288119) is not StypyTypeError):

    if (import_288119 != 'pyd_module'):
        __import__(import_288119)
        sys_modules_288120 = sys.modules[import_288119]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.ndimage', sys_modules_288120.module_type_store, module_type_store, ['convolve1d'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_288120, sys_modules_288120.module_type_store, module_type_store)
    else:
        from scipy.ndimage import convolve1d

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.ndimage', None, module_type_store, ['convolve1d'], [convolve1d])

else:
    # Assigning a type to the variable 'scipy.ndimage' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.ndimage', import_288119)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.signal._arraytools import axis_slice' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_288121 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.signal._arraytools')

if (type(import_288121) is not StypyTypeError):

    if (import_288121 != 'pyd_module'):
        __import__(import_288121)
        sys_modules_288122 = sys.modules[import_288121]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.signal._arraytools', sys_modules_288122.module_type_store, module_type_store, ['axis_slice'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_288122, sys_modules_288122.module_type_store, module_type_store)
    else:
        from scipy.signal._arraytools import axis_slice

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.signal._arraytools', None, module_type_store, ['axis_slice'], [axis_slice])

else:
    # Assigning a type to the variable 'scipy.signal._arraytools' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.signal._arraytools', import_288121)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')


@norecursion
def savgol_coeffs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_288123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 50), 'int')
    float_288124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 59), 'float')
    # Getting the type of 'None' (line 10)
    None_288125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 68), 'None')
    str_288126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 22), 'str', 'conv')
    defaults = [int_288123, float_288124, None_288125, str_288126]
    # Create a new context for function 'savgol_coeffs'
    module_type_store = module_type_store.open_function_context('savgol_coeffs', 10, 0, False)
    
    # Passed parameters checking function
    savgol_coeffs.stypy_localization = localization
    savgol_coeffs.stypy_type_of_self = None
    savgol_coeffs.stypy_type_store = module_type_store
    savgol_coeffs.stypy_function_name = 'savgol_coeffs'
    savgol_coeffs.stypy_param_names_list = ['window_length', 'polyorder', 'deriv', 'delta', 'pos', 'use']
    savgol_coeffs.stypy_varargs_param_name = None
    savgol_coeffs.stypy_kwargs_param_name = None
    savgol_coeffs.stypy_call_defaults = defaults
    savgol_coeffs.stypy_call_varargs = varargs
    savgol_coeffs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'savgol_coeffs', ['window_length', 'polyorder', 'deriv', 'delta', 'pos', 'use'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'savgol_coeffs', localization, ['window_length', 'polyorder', 'deriv', 'delta', 'pos', 'use'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'savgol_coeffs(...)' code ##################

    str_288127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, (-1)), 'str', "Compute the coefficients for a 1-d Savitzky-Golay FIR filter.\n\n    Parameters\n    ----------\n    window_length : int\n        The length of the filter window (i.e. the number of coefficients).\n        `window_length` must be an odd positive integer.\n    polyorder : int\n        The order of the polynomial used to fit the samples.\n        `polyorder` must be less than `window_length`.\n    deriv : int, optional\n        The order of the derivative to compute.  This must be a\n        nonnegative integer.  The default is 0, which means to filter\n        the data without differentiating.\n    delta : float, optional\n        The spacing of the samples to which the filter will be applied.\n        This is only used if deriv > 0.\n    pos : int or None, optional\n        If pos is not None, it specifies evaluation position within the\n        window.  The default is the middle of the window.\n    use : str, optional\n        Either 'conv' or 'dot'.  This argument chooses the order of the\n        coefficients.  The default is 'conv', which means that the\n        coefficients are ordered to be used in a convolution.  With\n        use='dot', the order is reversed, so the filter is applied by\n        dotting the coefficients with the data set.\n\n    Returns\n    -------\n    coeffs : 1-d ndarray\n        The filter coefficients.\n\n    References\n    ----------\n    A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of Data by\n    Simplified Least Squares Procedures. Analytical Chemistry, 1964, 36 (8),\n    pp 1627-1639.\n\n    See Also\n    --------\n    savgol_filter\n\n    Notes\n    -----\n\n    .. versionadded:: 0.14.0\n\n    Examples\n    --------\n    >>> from scipy.signal import savgol_coeffs\n    >>> savgol_coeffs(5, 2)\n    array([-0.08571429,  0.34285714,  0.48571429,  0.34285714, -0.08571429])\n    >>> savgol_coeffs(5, 2, deriv=1)\n    array([  2.00000000e-01,   1.00000000e-01,   2.00607895e-16,\n            -1.00000000e-01,  -2.00000000e-01])\n\n    Note that use='dot' simply reverses the coefficients.\n\n    >>> savgol_coeffs(5, 2, pos=3)\n    array([ 0.25714286,  0.37142857,  0.34285714,  0.17142857, -0.14285714])\n    >>> savgol_coeffs(5, 2, pos=3, use='dot')\n    array([-0.14285714,  0.17142857,  0.34285714,  0.37142857,  0.25714286])\n\n    `x` contains data from the parabola x = t**2, sampled at\n    t = -1, 0, 1, 2, 3.  `c` holds the coefficients that will compute the\n    derivative at the last position.  When dotted with `x` the result should\n    be 6.\n\n    >>> x = np.array([1, 0, 1, 4, 9])\n    >>> c = savgol_coeffs(5, 2, pos=4, deriv=1, use='dot')\n    >>> c.dot(x)\n    6.0000000000000018\n    ")
    
    
    # Getting the type of 'polyorder' (line 98)
    polyorder_288128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 7), 'polyorder')
    # Getting the type of 'window_length' (line 98)
    window_length_288129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'window_length')
    # Applying the binary operator '>=' (line 98)
    result_ge_288130 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 7), '>=', polyorder_288128, window_length_288129)
    
    # Testing the type of an if condition (line 98)
    if_condition_288131 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 4), result_ge_288130)
    # Assigning a type to the variable 'if_condition_288131' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'if_condition_288131', if_condition_288131)
    # SSA begins for if statement (line 98)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 99)
    # Processing the call arguments (line 99)
    str_288133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 25), 'str', 'polyorder must be less than window_length.')
    # Processing the call keyword arguments (line 99)
    kwargs_288134 = {}
    # Getting the type of 'ValueError' (line 99)
    ValueError_288132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 99)
    ValueError_call_result_288135 = invoke(stypy.reporting.localization.Localization(__file__, 99, 14), ValueError_288132, *[str_288133], **kwargs_288134)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 99, 8), ValueError_call_result_288135, 'raise parameter', BaseException)
    # SSA join for if statement (line 98)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 101):
    
    # Assigning a Subscript to a Name (line 101):
    
    # Obtaining the type of the subscript
    int_288136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 4), 'int')
    
    # Call to divmod(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'window_length' (line 101)
    window_length_288138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 26), 'window_length', False)
    int_288139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 41), 'int')
    # Processing the call keyword arguments (line 101)
    kwargs_288140 = {}
    # Getting the type of 'divmod' (line 101)
    divmod_288137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'divmod', False)
    # Calling divmod(args, kwargs) (line 101)
    divmod_call_result_288141 = invoke(stypy.reporting.localization.Localization(__file__, 101, 19), divmod_288137, *[window_length_288138, int_288139], **kwargs_288140)
    
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___288142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 4), divmod_call_result_288141, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_288143 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), getitem___288142, int_288136)
    
    # Assigning a type to the variable 'tuple_var_assignment_288107' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'tuple_var_assignment_288107', subscript_call_result_288143)
    
    # Assigning a Subscript to a Name (line 101):
    
    # Obtaining the type of the subscript
    int_288144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 4), 'int')
    
    # Call to divmod(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'window_length' (line 101)
    window_length_288146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 26), 'window_length', False)
    int_288147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 41), 'int')
    # Processing the call keyword arguments (line 101)
    kwargs_288148 = {}
    # Getting the type of 'divmod' (line 101)
    divmod_288145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'divmod', False)
    # Calling divmod(args, kwargs) (line 101)
    divmod_call_result_288149 = invoke(stypy.reporting.localization.Localization(__file__, 101, 19), divmod_288145, *[window_length_288146, int_288147], **kwargs_288148)
    
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___288150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 4), divmod_call_result_288149, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_288151 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), getitem___288150, int_288144)
    
    # Assigning a type to the variable 'tuple_var_assignment_288108' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'tuple_var_assignment_288108', subscript_call_result_288151)
    
    # Assigning a Name to a Name (line 101):
    # Getting the type of 'tuple_var_assignment_288107' (line 101)
    tuple_var_assignment_288107_288152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'tuple_var_assignment_288107')
    # Assigning a type to the variable 'halflen' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'halflen', tuple_var_assignment_288107_288152)
    
    # Assigning a Name to a Name (line 101):
    # Getting the type of 'tuple_var_assignment_288108' (line 101)
    tuple_var_assignment_288108_288153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'tuple_var_assignment_288108')
    # Assigning a type to the variable 'rem' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 13), 'rem', tuple_var_assignment_288108_288153)
    
    
    # Getting the type of 'rem' (line 103)
    rem_288154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 7), 'rem')
    int_288155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 14), 'int')
    # Applying the binary operator '==' (line 103)
    result_eq_288156 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 7), '==', rem_288154, int_288155)
    
    # Testing the type of an if condition (line 103)
    if_condition_288157 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 4), result_eq_288156)
    # Assigning a type to the variable 'if_condition_288157' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'if_condition_288157', if_condition_288157)
    # SSA begins for if statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 104)
    # Processing the call arguments (line 104)
    str_288159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 25), 'str', 'window_length must be odd.')
    # Processing the call keyword arguments (line 104)
    kwargs_288160 = {}
    # Getting the type of 'ValueError' (line 104)
    ValueError_288158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 104)
    ValueError_call_result_288161 = invoke(stypy.reporting.localization.Localization(__file__, 104, 14), ValueError_288158, *[str_288159], **kwargs_288160)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 104, 8), ValueError_call_result_288161, 'raise parameter', BaseException)
    # SSA join for if statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 106)
    # Getting the type of 'pos' (line 106)
    pos_288162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 7), 'pos')
    # Getting the type of 'None' (line 106)
    None_288163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 'None')
    
    (may_be_288164, more_types_in_union_288165) = may_be_none(pos_288162, None_288163)

    if may_be_288164:

        if more_types_in_union_288165:
            # Runtime conditional SSA (line 106)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 107):
        
        # Assigning a Name to a Name (line 107):
        # Getting the type of 'halflen' (line 107)
        halflen_288166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 14), 'halflen')
        # Assigning a type to the variable 'pos' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'pos', halflen_288166)

        if more_types_in_union_288165:
            # SSA join for if statement (line 106)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    int_288167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 12), 'int')
    # Getting the type of 'pos' (line 109)
    pos_288168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 17), 'pos')
    # Applying the binary operator '<=' (line 109)
    result_le_288169 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 12), '<=', int_288167, pos_288168)
    # Getting the type of 'window_length' (line 109)
    window_length_288170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 23), 'window_length')
    # Applying the binary operator '<' (line 109)
    result_lt_288171 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 12), '<', pos_288168, window_length_288170)
    # Applying the binary operator '&' (line 109)
    result_and__288172 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 12), '&', result_le_288169, result_lt_288171)
    
    # Applying the 'not' unary operator (line 109)
    result_not__288173 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 7), 'not', result_and__288172)
    
    # Testing the type of an if condition (line 109)
    if_condition_288174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 4), result_not__288173)
    # Assigning a type to the variable 'if_condition_288174' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'if_condition_288174', if_condition_288174)
    # SSA begins for if statement (line 109)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 110)
    # Processing the call arguments (line 110)
    str_288176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 25), 'str', 'pos must be nonnegative and less than window_length.')
    # Processing the call keyword arguments (line 110)
    kwargs_288177 = {}
    # Getting the type of 'ValueError' (line 110)
    ValueError_288175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 110)
    ValueError_call_result_288178 = invoke(stypy.reporting.localization.Localization(__file__, 110, 14), ValueError_288175, *[str_288176], **kwargs_288177)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 110, 8), ValueError_call_result_288178, 'raise parameter', BaseException)
    # SSA join for if statement (line 109)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'use' (line 113)
    use_288179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 7), 'use')
    
    # Obtaining an instance of the builtin type 'list' (line 113)
    list_288180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 113)
    # Adding element type (line 113)
    str_288181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 19), 'str', 'conv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 18), list_288180, str_288181)
    # Adding element type (line 113)
    str_288182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 27), 'str', 'dot')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 18), list_288180, str_288182)
    
    # Applying the binary operator 'notin' (line 113)
    result_contains_288183 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 7), 'notin', use_288179, list_288180)
    
    # Testing the type of an if condition (line 113)
    if_condition_288184 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 4), result_contains_288183)
    # Assigning a type to the variable 'if_condition_288184' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'if_condition_288184', if_condition_288184)
    # SSA begins for if statement (line 113)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 114)
    # Processing the call arguments (line 114)
    str_288186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 25), 'str', "`use` must be 'conv' or 'dot'")
    # Processing the call keyword arguments (line 114)
    kwargs_288187 = {}
    # Getting the type of 'ValueError' (line 114)
    ValueError_288185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 114)
    ValueError_call_result_288188 = invoke(stypy.reporting.localization.Localization(__file__, 114, 14), ValueError_288185, *[str_288186], **kwargs_288187)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 114, 8), ValueError_call_result_288188, 'raise parameter', BaseException)
    # SSA join for if statement (line 113)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 120):
    
    # Assigning a Call to a Name (line 120):
    
    # Call to arange(...): (line 120)
    # Processing the call arguments (line 120)
    
    # Getting the type of 'pos' (line 120)
    pos_288191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'pos', False)
    # Applying the 'usub' unary operator (line 120)
    result___neg___288192 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 18), 'usub', pos_288191)
    
    # Getting the type of 'window_length' (line 120)
    window_length_288193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 24), 'window_length', False)
    # Getting the type of 'pos' (line 120)
    pos_288194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 40), 'pos', False)
    # Applying the binary operator '-' (line 120)
    result_sub_288195 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 24), '-', window_length_288193, pos_288194)
    
    # Processing the call keyword arguments (line 120)
    # Getting the type of 'float' (line 120)
    float_288196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 51), 'float', False)
    keyword_288197 = float_288196
    kwargs_288198 = {'dtype': keyword_288197}
    # Getting the type of 'np' (line 120)
    np_288189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 120)
    arange_288190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), np_288189, 'arange')
    # Calling arange(args, kwargs) (line 120)
    arange_call_result_288199 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), arange_288190, *[result___neg___288192, result_sub_288195], **kwargs_288198)
    
    # Assigning a type to the variable 'x' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'x', arange_call_result_288199)
    
    
    # Getting the type of 'use' (line 121)
    use_288200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 7), 'use')
    str_288201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 14), 'str', 'conv')
    # Applying the binary operator '==' (line 121)
    result_eq_288202 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 7), '==', use_288200, str_288201)
    
    # Testing the type of an if condition (line 121)
    if_condition_288203 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 4), result_eq_288202)
    # Assigning a type to the variable 'if_condition_288203' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'if_condition_288203', if_condition_288203)
    # SSA begins for if statement (line 121)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 123):
    
    # Assigning a Subscript to a Name (line 123):
    
    # Obtaining the type of the subscript
    int_288204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 16), 'int')
    slice_288205 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 123, 12), None, None, int_288204)
    # Getting the type of 'x' (line 123)
    x_288206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'x')
    # Obtaining the member '__getitem__' of a type (line 123)
    getitem___288207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), x_288206, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 123)
    subscript_call_result_288208 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), getitem___288207, slice_288205)
    
    # Assigning a type to the variable 'x' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'x', subscript_call_result_288208)
    # SSA join for if statement (line 121)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 125):
    
    # Assigning a Call to a Name (line 125):
    
    # Call to reshape(...): (line 125)
    # Processing the call arguments (line 125)
    int_288217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 45), 'int')
    int_288218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 49), 'int')
    # Processing the call keyword arguments (line 125)
    kwargs_288219 = {}
    
    # Call to arange(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'polyorder' (line 125)
    polyorder_288211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 22), 'polyorder', False)
    int_288212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 34), 'int')
    # Applying the binary operator '+' (line 125)
    result_add_288213 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 22), '+', polyorder_288211, int_288212)
    
    # Processing the call keyword arguments (line 125)
    kwargs_288214 = {}
    # Getting the type of 'np' (line 125)
    np_288209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'np', False)
    # Obtaining the member 'arange' of a type (line 125)
    arange_288210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), np_288209, 'arange')
    # Calling arange(args, kwargs) (line 125)
    arange_call_result_288215 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), arange_288210, *[result_add_288213], **kwargs_288214)
    
    # Obtaining the member 'reshape' of a type (line 125)
    reshape_288216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 12), arange_call_result_288215, 'reshape')
    # Calling reshape(args, kwargs) (line 125)
    reshape_call_result_288220 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), reshape_288216, *[int_288217, int_288218], **kwargs_288219)
    
    # Assigning a type to the variable 'order' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'order', reshape_call_result_288220)
    
    # Assigning a BinOp to a Name (line 126):
    
    # Assigning a BinOp to a Name (line 126):
    # Getting the type of 'x' (line 126)
    x_288221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'x')
    # Getting the type of 'order' (line 126)
    order_288222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 13), 'order')
    # Applying the binary operator '**' (line 126)
    result_pow_288223 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 8), '**', x_288221, order_288222)
    
    # Assigning a type to the variable 'A' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'A', result_pow_288223)
    
    # Assigning a Call to a Name (line 129):
    
    # Assigning a Call to a Name (line 129):
    
    # Call to zeros(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'polyorder' (line 129)
    polyorder_288226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 17), 'polyorder', False)
    int_288227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 29), 'int')
    # Applying the binary operator '+' (line 129)
    result_add_288228 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 17), '+', polyorder_288226, int_288227)
    
    # Processing the call keyword arguments (line 129)
    kwargs_288229 = {}
    # Getting the type of 'np' (line 129)
    np_288224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'np', False)
    # Obtaining the member 'zeros' of a type (line 129)
    zeros_288225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), np_288224, 'zeros')
    # Calling zeros(args, kwargs) (line 129)
    zeros_call_result_288230 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), zeros_288225, *[result_add_288228], **kwargs_288229)
    
    # Assigning a type to the variable 'y' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'y', zeros_call_result_288230)
    
    # Assigning a BinOp to a Subscript (line 132):
    
    # Assigning a BinOp to a Subscript (line 132):
    
    # Call to factorial(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'deriv' (line 132)
    deriv_288232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 25), 'deriv', False)
    # Processing the call keyword arguments (line 132)
    kwargs_288233 = {}
    # Getting the type of 'factorial' (line 132)
    factorial_288231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'factorial', False)
    # Calling factorial(args, kwargs) (line 132)
    factorial_call_result_288234 = invoke(stypy.reporting.localization.Localization(__file__, 132, 15), factorial_288231, *[deriv_288232], **kwargs_288233)
    
    # Getting the type of 'delta' (line 132)
    delta_288235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 35), 'delta')
    # Getting the type of 'deriv' (line 132)
    deriv_288236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 44), 'deriv')
    # Applying the binary operator '**' (line 132)
    result_pow_288237 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 35), '**', delta_288235, deriv_288236)
    
    # Applying the binary operator 'div' (line 132)
    result_div_288238 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 15), 'div', factorial_call_result_288234, result_pow_288237)
    
    # Getting the type of 'y' (line 132)
    y_288239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'y')
    # Getting the type of 'deriv' (line 132)
    deriv_288240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 6), 'deriv')
    # Storing an element on a container (line 132)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 4), y_288239, (deriv_288240, result_div_288238))
    
    # Assigning a Call to a Tuple (line 135):
    
    # Assigning a Subscript to a Name (line 135):
    
    # Obtaining the type of the subscript
    int_288241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 4), 'int')
    
    # Call to lstsq(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'A' (line 135)
    A_288243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 28), 'A', False)
    # Getting the type of 'y' (line 135)
    y_288244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 31), 'y', False)
    # Processing the call keyword arguments (line 135)
    kwargs_288245 = {}
    # Getting the type of 'lstsq' (line 135)
    lstsq_288242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 22), 'lstsq', False)
    # Calling lstsq(args, kwargs) (line 135)
    lstsq_call_result_288246 = invoke(stypy.reporting.localization.Localization(__file__, 135, 22), lstsq_288242, *[A_288243, y_288244], **kwargs_288245)
    
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___288247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 4), lstsq_call_result_288246, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_288248 = invoke(stypy.reporting.localization.Localization(__file__, 135, 4), getitem___288247, int_288241)
    
    # Assigning a type to the variable 'tuple_var_assignment_288109' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'tuple_var_assignment_288109', subscript_call_result_288248)
    
    # Assigning a Subscript to a Name (line 135):
    
    # Obtaining the type of the subscript
    int_288249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 4), 'int')
    
    # Call to lstsq(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'A' (line 135)
    A_288251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 28), 'A', False)
    # Getting the type of 'y' (line 135)
    y_288252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 31), 'y', False)
    # Processing the call keyword arguments (line 135)
    kwargs_288253 = {}
    # Getting the type of 'lstsq' (line 135)
    lstsq_288250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 22), 'lstsq', False)
    # Calling lstsq(args, kwargs) (line 135)
    lstsq_call_result_288254 = invoke(stypy.reporting.localization.Localization(__file__, 135, 22), lstsq_288250, *[A_288251, y_288252], **kwargs_288253)
    
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___288255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 4), lstsq_call_result_288254, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_288256 = invoke(stypy.reporting.localization.Localization(__file__, 135, 4), getitem___288255, int_288249)
    
    # Assigning a type to the variable 'tuple_var_assignment_288110' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'tuple_var_assignment_288110', subscript_call_result_288256)
    
    # Assigning a Subscript to a Name (line 135):
    
    # Obtaining the type of the subscript
    int_288257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 4), 'int')
    
    # Call to lstsq(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'A' (line 135)
    A_288259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 28), 'A', False)
    # Getting the type of 'y' (line 135)
    y_288260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 31), 'y', False)
    # Processing the call keyword arguments (line 135)
    kwargs_288261 = {}
    # Getting the type of 'lstsq' (line 135)
    lstsq_288258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 22), 'lstsq', False)
    # Calling lstsq(args, kwargs) (line 135)
    lstsq_call_result_288262 = invoke(stypy.reporting.localization.Localization(__file__, 135, 22), lstsq_288258, *[A_288259, y_288260], **kwargs_288261)
    
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___288263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 4), lstsq_call_result_288262, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_288264 = invoke(stypy.reporting.localization.Localization(__file__, 135, 4), getitem___288263, int_288257)
    
    # Assigning a type to the variable 'tuple_var_assignment_288111' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'tuple_var_assignment_288111', subscript_call_result_288264)
    
    # Assigning a Subscript to a Name (line 135):
    
    # Obtaining the type of the subscript
    int_288265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 4), 'int')
    
    # Call to lstsq(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'A' (line 135)
    A_288267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 28), 'A', False)
    # Getting the type of 'y' (line 135)
    y_288268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 31), 'y', False)
    # Processing the call keyword arguments (line 135)
    kwargs_288269 = {}
    # Getting the type of 'lstsq' (line 135)
    lstsq_288266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 22), 'lstsq', False)
    # Calling lstsq(args, kwargs) (line 135)
    lstsq_call_result_288270 = invoke(stypy.reporting.localization.Localization(__file__, 135, 22), lstsq_288266, *[A_288267, y_288268], **kwargs_288269)
    
    # Obtaining the member '__getitem__' of a type (line 135)
    getitem___288271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 4), lstsq_call_result_288270, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 135)
    subscript_call_result_288272 = invoke(stypy.reporting.localization.Localization(__file__, 135, 4), getitem___288271, int_288265)
    
    # Assigning a type to the variable 'tuple_var_assignment_288112' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'tuple_var_assignment_288112', subscript_call_result_288272)
    
    # Assigning a Name to a Name (line 135):
    # Getting the type of 'tuple_var_assignment_288109' (line 135)
    tuple_var_assignment_288109_288273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'tuple_var_assignment_288109')
    # Assigning a type to the variable 'coeffs' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'coeffs', tuple_var_assignment_288109_288273)
    
    # Assigning a Name to a Name (line 135):
    # Getting the type of 'tuple_var_assignment_288110' (line 135)
    tuple_var_assignment_288110_288274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'tuple_var_assignment_288110')
    # Assigning a type to the variable '_' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), '_', tuple_var_assignment_288110_288274)
    
    # Assigning a Name to a Name (line 135):
    # Getting the type of 'tuple_var_assignment_288111' (line 135)
    tuple_var_assignment_288111_288275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'tuple_var_assignment_288111')
    # Assigning a type to the variable '_' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), '_', tuple_var_assignment_288111_288275)
    
    # Assigning a Name to a Name (line 135):
    # Getting the type of 'tuple_var_assignment_288112' (line 135)
    tuple_var_assignment_288112_288276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'tuple_var_assignment_288112')
    # Assigning a type to the variable '_' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), '_', tuple_var_assignment_288112_288276)
    # Getting the type of 'coeffs' (line 137)
    coeffs_288277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'coeffs')
    # Assigning a type to the variable 'stypy_return_type' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type', coeffs_288277)
    
    # ################# End of 'savgol_coeffs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'savgol_coeffs' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_288278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288278)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'savgol_coeffs'
    return stypy_return_type_288278

# Assigning a type to the variable 'savgol_coeffs' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'savgol_coeffs', savgol_coeffs)

@norecursion
def _polyder(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_polyder'
    module_type_store = module_type_store.open_function_context('_polyder', 140, 0, False)
    
    # Passed parameters checking function
    _polyder.stypy_localization = localization
    _polyder.stypy_type_of_self = None
    _polyder.stypy_type_store = module_type_store
    _polyder.stypy_function_name = '_polyder'
    _polyder.stypy_param_names_list = ['p', 'm']
    _polyder.stypy_varargs_param_name = None
    _polyder.stypy_kwargs_param_name = None
    _polyder.stypy_call_defaults = defaults
    _polyder.stypy_call_varargs = varargs
    _polyder.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_polyder', ['p', 'm'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_polyder', localization, ['p', 'm'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_polyder(...)' code ##################

    str_288279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, (-1)), 'str', "Differentiate polynomials represented with coefficients.\n\n    p must be a 1D or 2D array.  In the 2D case, each column gives\n    the coefficients of a polynomial; the first row holds the coefficients\n    associated with the highest power.  m must be a nonnegative integer.\n    (numpy.polyder doesn't handle the 2D case.)\n    ")
    
    
    # Getting the type of 'm' (line 149)
    m_288280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 7), 'm')
    int_288281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 12), 'int')
    # Applying the binary operator '==' (line 149)
    result_eq_288282 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 7), '==', m_288280, int_288281)
    
    # Testing the type of an if condition (line 149)
    if_condition_288283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 4), result_eq_288282)
    # Assigning a type to the variable 'if_condition_288283' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'if_condition_288283', if_condition_288283)
    # SSA begins for if statement (line 149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 150):
    
    # Assigning a Name to a Name (line 150):
    # Getting the type of 'p' (line 150)
    p_288284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 17), 'p')
    # Assigning a type to the variable 'result' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'result', p_288284)
    # SSA branch for the else part of an if statement (line 149)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 152):
    
    # Assigning a Call to a Name (line 152):
    
    # Call to len(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'p' (line 152)
    p_288286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'p', False)
    # Processing the call keyword arguments (line 152)
    kwargs_288287 = {}
    # Getting the type of 'len' (line 152)
    len_288285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'len', False)
    # Calling len(args, kwargs) (line 152)
    len_call_result_288288 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), len_288285, *[p_288286], **kwargs_288287)
    
    # Assigning a type to the variable 'n' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'n', len_call_result_288288)
    
    
    # Getting the type of 'n' (line 153)
    n_288289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'n')
    # Getting the type of 'm' (line 153)
    m_288290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'm')
    # Applying the binary operator '<=' (line 153)
    result_le_288291 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 11), '<=', n_288289, m_288290)
    
    # Testing the type of an if condition (line 153)
    if_condition_288292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 8), result_le_288291)
    # Assigning a type to the variable 'if_condition_288292' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'if_condition_288292', if_condition_288292)
    # SSA begins for if statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 154):
    
    # Assigning a Call to a Name (line 154):
    
    # Call to zeros_like(...): (line 154)
    # Processing the call arguments (line 154)
    
    # Obtaining the type of the subscript
    int_288295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 38), 'int')
    slice_288296 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 154, 35), None, int_288295, None)
    Ellipsis_288297 = Ellipsis
    # Getting the type of 'p' (line 154)
    p_288298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 35), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 154)
    getitem___288299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 35), p_288298, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
    subscript_call_result_288300 = invoke(stypy.reporting.localization.Localization(__file__, 154, 35), getitem___288299, (slice_288296, Ellipsis_288297))
    
    # Processing the call keyword arguments (line 154)
    kwargs_288301 = {}
    # Getting the type of 'np' (line 154)
    np_288293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 21), 'np', False)
    # Obtaining the member 'zeros_like' of a type (line 154)
    zeros_like_288294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 21), np_288293, 'zeros_like')
    # Calling zeros_like(args, kwargs) (line 154)
    zeros_like_call_result_288302 = invoke(stypy.reporting.localization.Localization(__file__, 154, 21), zeros_like_288294, *[subscript_call_result_288300], **kwargs_288301)
    
    # Assigning a type to the variable 'result' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'result', zeros_like_call_result_288302)
    # SSA branch for the else part of an if statement (line 153)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 156):
    
    # Assigning a Call to a Name (line 156):
    
    # Call to copy(...): (line 156)
    # Processing the call keyword arguments (line 156)
    kwargs_288310 = {}
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'm' (line 156)
    m_288303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'm', False)
    # Applying the 'usub' unary operator (line 156)
    result___neg___288304 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 20), 'usub', m_288303)
    
    slice_288305 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 156, 17), None, result___neg___288304, None)
    # Getting the type of 'p' (line 156)
    p_288306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 17), 'p', False)
    # Obtaining the member '__getitem__' of a type (line 156)
    getitem___288307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 17), p_288306, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 156)
    subscript_call_result_288308 = invoke(stypy.reporting.localization.Localization(__file__, 156, 17), getitem___288307, slice_288305)
    
    # Obtaining the member 'copy' of a type (line 156)
    copy_288309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 17), subscript_call_result_288308, 'copy')
    # Calling copy(args, kwargs) (line 156)
    copy_call_result_288311 = invoke(stypy.reporting.localization.Localization(__file__, 156, 17), copy_288309, *[], **kwargs_288310)
    
    # Assigning a type to the variable 'dp' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'dp', copy_call_result_288311)
    
    
    # Call to range(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'm' (line 157)
    m_288313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 27), 'm', False)
    # Processing the call keyword arguments (line 157)
    kwargs_288314 = {}
    # Getting the type of 'range' (line 157)
    range_288312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'range', False)
    # Calling range(args, kwargs) (line 157)
    range_call_result_288315 = invoke(stypy.reporting.localization.Localization(__file__, 157, 21), range_288312, *[m_288313], **kwargs_288314)
    
    # Testing the type of a for loop iterable (line 157)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 157, 12), range_call_result_288315)
    # Getting the type of the for loop variable (line 157)
    for_loop_var_288316 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 157, 12), range_call_result_288315)
    # Assigning a type to the variable 'k' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'k', for_loop_var_288316)
    # SSA begins for a for statement (line 157)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 158):
    
    # Assigning a Call to a Name (line 158):
    
    # Call to arange(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'n' (line 158)
    n_288319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 32), 'n', False)
    # Getting the type of 'k' (line 158)
    k_288320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 36), 'k', False)
    # Applying the binary operator '-' (line 158)
    result_sub_288321 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 32), '-', n_288319, k_288320)
    
    int_288322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 40), 'int')
    # Applying the binary operator '-' (line 158)
    result_sub_288323 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 38), '-', result_sub_288321, int_288322)
    
    # Getting the type of 'm' (line 158)
    m_288324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 43), 'm', False)
    # Getting the type of 'k' (line 158)
    k_288325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 47), 'k', False)
    # Applying the binary operator '-' (line 158)
    result_sub_288326 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 43), '-', m_288324, k_288325)
    
    int_288327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 51), 'int')
    # Applying the binary operator '-' (line 158)
    result_sub_288328 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 49), '-', result_sub_288326, int_288327)
    
    int_288329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 54), 'int')
    # Processing the call keyword arguments (line 158)
    kwargs_288330 = {}
    # Getting the type of 'np' (line 158)
    np_288317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 22), 'np', False)
    # Obtaining the member 'arange' of a type (line 158)
    arange_288318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 22), np_288317, 'arange')
    # Calling arange(args, kwargs) (line 158)
    arange_call_result_288331 = invoke(stypy.reporting.localization.Localization(__file__, 158, 22), arange_288318, *[result_sub_288323, result_sub_288328, int_288329], **kwargs_288330)
    
    # Assigning a type to the variable 'rng' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'rng', arange_call_result_288331)
    
    # Getting the type of 'dp' (line 159)
    dp_288332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'dp')
    
    # Call to reshape(...): (line 159)
    # Processing the call arguments (line 159)
    
    # Obtaining an instance of the builtin type 'tuple' (line 159)
    tuple_288335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 159)
    # Adding element type (line 159)
    # Getting the type of 'n' (line 159)
    n_288336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 35), 'n', False)
    # Getting the type of 'm' (line 159)
    m_288337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 39), 'm', False)
    # Applying the binary operator '-' (line 159)
    result_sub_288338 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 35), '-', n_288336, m_288337)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 35), tuple_288335, result_sub_288338)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 159)
    tuple_288339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 159)
    # Adding element type (line 159)
    int_288340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 46), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 46), tuple_288339, int_288340)
    
    # Getting the type of 'p' (line 159)
    p_288341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 53), 'p', False)
    # Obtaining the member 'ndim' of a type (line 159)
    ndim_288342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 53), p_288341, 'ndim')
    int_288343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 62), 'int')
    # Applying the binary operator '-' (line 159)
    result_sub_288344 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 53), '-', ndim_288342, int_288343)
    
    # Applying the binary operator '*' (line 159)
    result_mul_288345 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 45), '*', tuple_288339, result_sub_288344)
    
    # Applying the binary operator '+' (line 159)
    result_add_288346 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 34), '+', tuple_288335, result_mul_288345)
    
    # Processing the call keyword arguments (line 159)
    kwargs_288347 = {}
    # Getting the type of 'rng' (line 159)
    rng_288333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'rng', False)
    # Obtaining the member 'reshape' of a type (line 159)
    reshape_288334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 22), rng_288333, 'reshape')
    # Calling reshape(args, kwargs) (line 159)
    reshape_call_result_288348 = invoke(stypy.reporting.localization.Localization(__file__, 159, 22), reshape_288334, *[result_add_288346], **kwargs_288347)
    
    # Applying the binary operator '*=' (line 159)
    result_imul_288349 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 16), '*=', dp_288332, reshape_call_result_288348)
    # Assigning a type to the variable 'dp' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'dp', result_imul_288349)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 160):
    
    # Assigning a Name to a Name (line 160):
    # Getting the type of 'dp' (line 160)
    dp_288350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 21), 'dp')
    # Assigning a type to the variable 'result' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'result', dp_288350)
    # SSA join for if statement (line 153)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 149)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 161)
    result_288351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type', result_288351)
    
    # ################# End of '_polyder(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_polyder' in the type store
    # Getting the type of 'stypy_return_type' (line 140)
    stypy_return_type_288352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288352)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_polyder'
    return stypy_return_type_288352

# Assigning a type to the variable '_polyder' (line 140)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), '_polyder', _polyder)

@norecursion
def _fit_edge(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fit_edge'
    module_type_store = module_type_store.open_function_context('_fit_edge', 164, 0, False)
    
    # Passed parameters checking function
    _fit_edge.stypy_localization = localization
    _fit_edge.stypy_type_of_self = None
    _fit_edge.stypy_type_store = module_type_store
    _fit_edge.stypy_function_name = '_fit_edge'
    _fit_edge.stypy_param_names_list = ['x', 'window_start', 'window_stop', 'interp_start', 'interp_stop', 'axis', 'polyorder', 'deriv', 'delta', 'y']
    _fit_edge.stypy_varargs_param_name = None
    _fit_edge.stypy_kwargs_param_name = None
    _fit_edge.stypy_call_defaults = defaults
    _fit_edge.stypy_call_varargs = varargs
    _fit_edge.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fit_edge', ['x', 'window_start', 'window_stop', 'interp_start', 'interp_stop', 'axis', 'polyorder', 'deriv', 'delta', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fit_edge', localization, ['x', 'window_start', 'window_stop', 'interp_start', 'interp_stop', 'axis', 'polyorder', 'deriv', 'delta', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fit_edge(...)' code ##################

    str_288353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, (-1)), 'str', '\n    Given an n-d array `x` and the specification of a slice of `x` from\n    `window_start` to `window_stop` along `axis`, create an interpolating\n    polynomial of each 1-d slice, and evaluate that polynomial in the slice\n    from `interp_start` to `interp_stop`.  Put the result into the\n    corresponding slice of `y`.\n    ')
    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to axis_slice(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'x' (line 175)
    x_288355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 24), 'x', False)
    # Processing the call keyword arguments (line 175)
    # Getting the type of 'window_start' (line 175)
    window_start_288356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 33), 'window_start', False)
    keyword_288357 = window_start_288356
    # Getting the type of 'window_stop' (line 175)
    window_stop_288358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 52), 'window_stop', False)
    keyword_288359 = window_stop_288358
    # Getting the type of 'axis' (line 175)
    axis_288360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 70), 'axis', False)
    keyword_288361 = axis_288360
    kwargs_288362 = {'start': keyword_288357, 'stop': keyword_288359, 'axis': keyword_288361}
    # Getting the type of 'axis_slice' (line 175)
    axis_slice_288354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 13), 'axis_slice', False)
    # Calling axis_slice(args, kwargs) (line 175)
    axis_slice_call_result_288363 = invoke(stypy.reporting.localization.Localization(__file__, 175, 13), axis_slice_288354, *[x_288355], **kwargs_288362)
    
    # Assigning a type to the variable 'x_edge' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'x_edge', axis_slice_call_result_288363)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'axis' (line 176)
    axis_288364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 7), 'axis')
    int_288365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 15), 'int')
    # Applying the binary operator '==' (line 176)
    result_eq_288366 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 7), '==', axis_288364, int_288365)
    
    
    # Getting the type of 'axis' (line 176)
    axis_288367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 20), 'axis')
    
    # Getting the type of 'x' (line 176)
    x_288368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 29), 'x')
    # Obtaining the member 'ndim' of a type (line 176)
    ndim_288369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 29), x_288368, 'ndim')
    # Applying the 'usub' unary operator (line 176)
    result___neg___288370 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 28), 'usub', ndim_288369)
    
    # Applying the binary operator '==' (line 176)
    result_eq_288371 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 20), '==', axis_288367, result___neg___288370)
    
    # Applying the binary operator 'or' (line 176)
    result_or_keyword_288372 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 7), 'or', result_eq_288366, result_eq_288371)
    
    # Testing the type of an if condition (line 176)
    if_condition_288373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 4), result_or_keyword_288372)
    # Assigning a type to the variable 'if_condition_288373' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'if_condition_288373', if_condition_288373)
    # SSA begins for if statement (line 176)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 177):
    
    # Assigning a Name to a Name (line 177):
    # Getting the type of 'x_edge' (line 177)
    x_edge_288374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 18), 'x_edge')
    # Assigning a type to the variable 'xx_edge' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'xx_edge', x_edge_288374)
    
    # Assigning a Name to a Name (line 178):
    
    # Assigning a Name to a Name (line 178):
    # Getting the type of 'False' (line 178)
    False_288375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 18), 'False')
    # Assigning a type to the variable 'swapped' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'swapped', False_288375)
    # SSA branch for the else part of an if statement (line 176)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 180):
    
    # Assigning a Call to a Name (line 180):
    
    # Call to swapaxes(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'axis' (line 180)
    axis_288378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 34), 'axis', False)
    int_288379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 40), 'int')
    # Processing the call keyword arguments (line 180)
    kwargs_288380 = {}
    # Getting the type of 'x_edge' (line 180)
    x_edge_288376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 18), 'x_edge', False)
    # Obtaining the member 'swapaxes' of a type (line 180)
    swapaxes_288377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 18), x_edge_288376, 'swapaxes')
    # Calling swapaxes(args, kwargs) (line 180)
    swapaxes_call_result_288381 = invoke(stypy.reporting.localization.Localization(__file__, 180, 18), swapaxes_288377, *[axis_288378, int_288379], **kwargs_288380)
    
    # Assigning a type to the variable 'xx_edge' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'xx_edge', swapaxes_call_result_288381)
    
    # Assigning a Name to a Name (line 181):
    
    # Assigning a Name to a Name (line 181):
    # Getting the type of 'True' (line 181)
    True_288382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 18), 'True')
    # Assigning a type to the variable 'swapped' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'swapped', True_288382)
    # SSA join for if statement (line 176)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 182):
    
    # Assigning a Call to a Name (line 182):
    
    # Call to reshape(...): (line 182)
    # Processing the call arguments (line 182)
    
    # Obtaining the type of the subscript
    int_288385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 44), 'int')
    # Getting the type of 'xx_edge' (line 182)
    xx_edge_288386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 30), 'xx_edge', False)
    # Obtaining the member 'shape' of a type (line 182)
    shape_288387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 30), xx_edge_288386, 'shape')
    # Obtaining the member '__getitem__' of a type (line 182)
    getitem___288388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 30), shape_288387, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 182)
    subscript_call_result_288389 = invoke(stypy.reporting.localization.Localization(__file__, 182, 30), getitem___288388, int_288385)
    
    int_288390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 48), 'int')
    # Processing the call keyword arguments (line 182)
    kwargs_288391 = {}
    # Getting the type of 'xx_edge' (line 182)
    xx_edge_288383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 14), 'xx_edge', False)
    # Obtaining the member 'reshape' of a type (line 182)
    reshape_288384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 14), xx_edge_288383, 'reshape')
    # Calling reshape(args, kwargs) (line 182)
    reshape_call_result_288392 = invoke(stypy.reporting.localization.Localization(__file__, 182, 14), reshape_288384, *[subscript_call_result_288389, int_288390], **kwargs_288391)
    
    # Assigning a type to the variable 'xx_edge' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'xx_edge', reshape_call_result_288392)
    
    # Assigning a Call to a Name (line 186):
    
    # Assigning a Call to a Name (line 186):
    
    # Call to polyfit(...): (line 186)
    # Processing the call arguments (line 186)
    
    # Call to arange(...): (line 186)
    # Processing the call arguments (line 186)
    int_288397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 39), 'int')
    # Getting the type of 'window_stop' (line 186)
    window_stop_288398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 42), 'window_stop', False)
    # Getting the type of 'window_start' (line 186)
    window_start_288399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 56), 'window_start', False)
    # Applying the binary operator '-' (line 186)
    result_sub_288400 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 42), '-', window_stop_288398, window_start_288399)
    
    # Processing the call keyword arguments (line 186)
    kwargs_288401 = {}
    # Getting the type of 'np' (line 186)
    np_288395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 29), 'np', False)
    # Obtaining the member 'arange' of a type (line 186)
    arange_288396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 29), np_288395, 'arange')
    # Calling arange(args, kwargs) (line 186)
    arange_call_result_288402 = invoke(stypy.reporting.localization.Localization(__file__, 186, 29), arange_288396, *[int_288397, result_sub_288400], **kwargs_288401)
    
    # Getting the type of 'xx_edge' (line 187)
    xx_edge_288403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 29), 'xx_edge', False)
    # Getting the type of 'polyorder' (line 187)
    polyorder_288404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 38), 'polyorder', False)
    # Processing the call keyword arguments (line 186)
    kwargs_288405 = {}
    # Getting the type of 'np' (line 186)
    np_288393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 18), 'np', False)
    # Obtaining the member 'polyfit' of a type (line 186)
    polyfit_288394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 18), np_288393, 'polyfit')
    # Calling polyfit(args, kwargs) (line 186)
    polyfit_call_result_288406 = invoke(stypy.reporting.localization.Localization(__file__, 186, 18), polyfit_288394, *[arange_call_result_288402, xx_edge_288403, polyorder_288404], **kwargs_288405)
    
    # Assigning a type to the variable 'poly_coeffs' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'poly_coeffs', polyfit_call_result_288406)
    
    
    # Getting the type of 'deriv' (line 189)
    deriv_288407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 7), 'deriv')
    int_288408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 15), 'int')
    # Applying the binary operator '>' (line 189)
    result_gt_288409 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 7), '>', deriv_288407, int_288408)
    
    # Testing the type of an if condition (line 189)
    if_condition_288410 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 4), result_gt_288409)
    # Assigning a type to the variable 'if_condition_288410' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'if_condition_288410', if_condition_288410)
    # SSA begins for if statement (line 189)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 190):
    
    # Assigning a Call to a Name (line 190):
    
    # Call to _polyder(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'poly_coeffs' (line 190)
    poly_coeffs_288412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 31), 'poly_coeffs', False)
    # Getting the type of 'deriv' (line 190)
    deriv_288413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 44), 'deriv', False)
    # Processing the call keyword arguments (line 190)
    kwargs_288414 = {}
    # Getting the type of '_polyder' (line 190)
    _polyder_288411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 22), '_polyder', False)
    # Calling _polyder(args, kwargs) (line 190)
    _polyder_call_result_288415 = invoke(stypy.reporting.localization.Localization(__file__, 190, 22), _polyder_288411, *[poly_coeffs_288412, deriv_288413], **kwargs_288414)
    
    # Assigning a type to the variable 'poly_coeffs' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'poly_coeffs', _polyder_call_result_288415)
    # SSA join for if statement (line 189)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 193):
    
    # Assigning a Call to a Name (line 193):
    
    # Call to arange(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'interp_start' (line 193)
    interp_start_288418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 18), 'interp_start', False)
    # Getting the type of 'window_start' (line 193)
    window_start_288419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 33), 'window_start', False)
    # Applying the binary operator '-' (line 193)
    result_sub_288420 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 18), '-', interp_start_288418, window_start_288419)
    
    # Getting the type of 'interp_stop' (line 193)
    interp_stop_288421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 47), 'interp_stop', False)
    # Getting the type of 'window_start' (line 193)
    window_start_288422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 61), 'window_start', False)
    # Applying the binary operator '-' (line 193)
    result_sub_288423 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 47), '-', interp_stop_288421, window_start_288422)
    
    # Processing the call keyword arguments (line 193)
    kwargs_288424 = {}
    # Getting the type of 'np' (line 193)
    np_288416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'np', False)
    # Obtaining the member 'arange' of a type (line 193)
    arange_288417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 8), np_288416, 'arange')
    # Calling arange(args, kwargs) (line 193)
    arange_call_result_288425 = invoke(stypy.reporting.localization.Localization(__file__, 193, 8), arange_288417, *[result_sub_288420, result_sub_288423], **kwargs_288424)
    
    # Assigning a type to the variable 'i' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'i', arange_call_result_288425)
    
    # Assigning a BinOp to a Name (line 194):
    
    # Assigning a BinOp to a Name (line 194):
    
    # Call to polyval(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 'poly_coeffs' (line 194)
    poly_coeffs_288428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 24), 'poly_coeffs', False)
    
    # Call to reshape(...): (line 194)
    # Processing the call arguments (line 194)
    int_288431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 47), 'int')
    int_288432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 51), 'int')
    # Processing the call keyword arguments (line 194)
    kwargs_288433 = {}
    # Getting the type of 'i' (line 194)
    i_288429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 37), 'i', False)
    # Obtaining the member 'reshape' of a type (line 194)
    reshape_288430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 37), i_288429, 'reshape')
    # Calling reshape(args, kwargs) (line 194)
    reshape_call_result_288434 = invoke(stypy.reporting.localization.Localization(__file__, 194, 37), reshape_288430, *[int_288431, int_288432], **kwargs_288433)
    
    # Processing the call keyword arguments (line 194)
    kwargs_288435 = {}
    # Getting the type of 'np' (line 194)
    np_288426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 13), 'np', False)
    # Obtaining the member 'polyval' of a type (line 194)
    polyval_288427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 13), np_288426, 'polyval')
    # Calling polyval(args, kwargs) (line 194)
    polyval_call_result_288436 = invoke(stypy.reporting.localization.Localization(__file__, 194, 13), polyval_288427, *[poly_coeffs_288428, reshape_call_result_288434], **kwargs_288435)
    
    # Getting the type of 'delta' (line 194)
    delta_288437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 58), 'delta')
    # Getting the type of 'deriv' (line 194)
    deriv_288438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 67), 'deriv')
    # Applying the binary operator '**' (line 194)
    result_pow_288439 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 58), '**', delta_288437, deriv_288438)
    
    # Applying the binary operator 'div' (line 194)
    result_div_288440 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 13), 'div', polyval_call_result_288436, result_pow_288439)
    
    # Assigning a type to the variable 'values' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'values', result_div_288440)
    
    # Assigning a Call to a Name (line 198):
    
    # Assigning a Call to a Name (line 198):
    
    # Call to list(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'y' (line 198)
    y_288442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'y', False)
    # Obtaining the member 'shape' of a type (line 198)
    shape_288443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 15), y_288442, 'shape')
    # Processing the call keyword arguments (line 198)
    kwargs_288444 = {}
    # Getting the type of 'list' (line 198)
    list_288441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 10), 'list', False)
    # Calling list(args, kwargs) (line 198)
    list_call_result_288445 = invoke(stypy.reporting.localization.Localization(__file__, 198, 10), list_288441, *[shape_288443], **kwargs_288444)
    
    # Assigning a type to the variable 'shp' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'shp', list_call_result_288445)
    
    # Assigning a Tuple to a Tuple (line 199):
    
    # Assigning a Subscript to a Name (line 199):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 199)
    axis_288446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'axis')
    # Getting the type of 'shp' (line 199)
    shp_288447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 24), 'shp')
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___288448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 24), shp_288447, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_288449 = invoke(stypy.reporting.localization.Localization(__file__, 199, 24), getitem___288448, axis_288446)
    
    # Assigning a type to the variable 'tuple_assignment_288113' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'tuple_assignment_288113', subscript_call_result_288449)
    
    # Assigning a Subscript to a Name (line 199):
    
    # Obtaining the type of the subscript
    int_288450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 39), 'int')
    # Getting the type of 'shp' (line 199)
    shp_288451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 35), 'shp')
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___288452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 35), shp_288451, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_288453 = invoke(stypy.reporting.localization.Localization(__file__, 199, 35), getitem___288452, int_288450)
    
    # Assigning a type to the variable 'tuple_assignment_288114' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'tuple_assignment_288114', subscript_call_result_288453)
    
    # Assigning a Name to a Subscript (line 199):
    # Getting the type of 'tuple_assignment_288113' (line 199)
    tuple_assignment_288113_288454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'tuple_assignment_288113')
    # Getting the type of 'shp' (line 199)
    shp_288455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'shp')
    int_288456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 8), 'int')
    # Storing an element on a container (line 199)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 4), shp_288455, (int_288456, tuple_assignment_288113_288454))
    
    # Assigning a Name to a Subscript (line 199):
    # Getting the type of 'tuple_assignment_288114' (line 199)
    tuple_assignment_288114_288457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'tuple_assignment_288114')
    # Getting the type of 'shp' (line 199)
    shp_288458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'shp')
    # Getting the type of 'axis' (line 199)
    axis_288459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'axis')
    # Storing an element on a container (line 199)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 12), shp_288458, (axis_288459, tuple_assignment_288114_288457))
    
    # Assigning a Call to a Name (line 200):
    
    # Assigning a Call to a Name (line 200):
    
    # Call to reshape(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'interp_stop' (line 200)
    interp_stop_288462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 28), 'interp_stop', False)
    # Getting the type of 'interp_start' (line 200)
    interp_start_288463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 42), 'interp_start', False)
    # Applying the binary operator '-' (line 200)
    result_sub_288464 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 28), '-', interp_stop_288462, interp_start_288463)
    
    
    # Obtaining the type of the subscript
    int_288465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 61), 'int')
    slice_288466 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 200, 57), int_288465, None, None)
    # Getting the type of 'shp' (line 200)
    shp_288467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 57), 'shp', False)
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___288468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 57), shp_288467, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_288469 = invoke(stypy.reporting.localization.Localization(__file__, 200, 57), getitem___288468, slice_288466)
    
    # Processing the call keyword arguments (line 200)
    kwargs_288470 = {}
    # Getting the type of 'values' (line 200)
    values_288460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 13), 'values', False)
    # Obtaining the member 'reshape' of a type (line 200)
    reshape_288461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 13), values_288460, 'reshape')
    # Calling reshape(args, kwargs) (line 200)
    reshape_call_result_288471 = invoke(stypy.reporting.localization.Localization(__file__, 200, 13), reshape_288461, *[result_sub_288464, subscript_call_result_288469], **kwargs_288470)
    
    # Assigning a type to the variable 'values' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'values', reshape_call_result_288471)
    
    # Getting the type of 'swapped' (line 201)
    swapped_288472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 7), 'swapped')
    # Testing the type of an if condition (line 201)
    if_condition_288473 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 4), swapped_288472)
    # Assigning a type to the variable 'if_condition_288473' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'if_condition_288473', if_condition_288473)
    # SSA begins for if statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 202):
    
    # Assigning a Call to a Name (line 202):
    
    # Call to swapaxes(...): (line 202)
    # Processing the call arguments (line 202)
    int_288476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 33), 'int')
    # Getting the type of 'axis' (line 202)
    axis_288477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 36), 'axis', False)
    # Processing the call keyword arguments (line 202)
    kwargs_288478 = {}
    # Getting the type of 'values' (line 202)
    values_288474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 17), 'values', False)
    # Obtaining the member 'swapaxes' of a type (line 202)
    swapaxes_288475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 17), values_288474, 'swapaxes')
    # Calling swapaxes(args, kwargs) (line 202)
    swapaxes_call_result_288479 = invoke(stypy.reporting.localization.Localization(__file__, 202, 17), swapaxes_288475, *[int_288476, axis_288477], **kwargs_288478)
    
    # Assigning a type to the variable 'values' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'values', swapaxes_call_result_288479)
    # SSA join for if statement (line 201)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 204):
    
    # Assigning a Call to a Name (line 204):
    
    # Call to axis_slice(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'y' (line 204)
    y_288481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 24), 'y', False)
    # Processing the call keyword arguments (line 204)
    # Getting the type of 'interp_start' (line 204)
    interp_start_288482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 33), 'interp_start', False)
    keyword_288483 = interp_start_288482
    # Getting the type of 'interp_stop' (line 204)
    interp_stop_288484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 52), 'interp_stop', False)
    keyword_288485 = interp_stop_288484
    # Getting the type of 'axis' (line 204)
    axis_288486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 70), 'axis', False)
    keyword_288487 = axis_288486
    kwargs_288488 = {'start': keyword_288483, 'stop': keyword_288485, 'axis': keyword_288487}
    # Getting the type of 'axis_slice' (line 204)
    axis_slice_288480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 13), 'axis_slice', False)
    # Calling axis_slice(args, kwargs) (line 204)
    axis_slice_call_result_288489 = invoke(stypy.reporting.localization.Localization(__file__, 204, 13), axis_slice_288480, *[y_288481], **kwargs_288488)
    
    # Assigning a type to the variable 'y_edge' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'y_edge', axis_slice_call_result_288489)
    
    # Assigning a Name to a Subscript (line 205):
    
    # Assigning a Name to a Subscript (line 205):
    # Getting the type of 'values' (line 205)
    values_288490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 18), 'values')
    # Getting the type of 'y_edge' (line 205)
    y_edge_288491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'y_edge')
    Ellipsis_288492 = Ellipsis
    # Storing an element on a container (line 205)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 4), y_edge_288491, (Ellipsis_288492, values_288490))
    
    # ################# End of '_fit_edge(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fit_edge' in the type store
    # Getting the type of 'stypy_return_type' (line 164)
    stypy_return_type_288493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288493)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fit_edge'
    return stypy_return_type_288493

# Assigning a type to the variable '_fit_edge' (line 164)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 0), '_fit_edge', _fit_edge)

@norecursion
def _fit_edges_polyfit(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_fit_edges_polyfit'
    module_type_store = module_type_store.open_function_context('_fit_edges_polyfit', 208, 0, False)
    
    # Passed parameters checking function
    _fit_edges_polyfit.stypy_localization = localization
    _fit_edges_polyfit.stypy_type_of_self = None
    _fit_edges_polyfit.stypy_type_store = module_type_store
    _fit_edges_polyfit.stypy_function_name = '_fit_edges_polyfit'
    _fit_edges_polyfit.stypy_param_names_list = ['x', 'window_length', 'polyorder', 'deriv', 'delta', 'axis', 'y']
    _fit_edges_polyfit.stypy_varargs_param_name = None
    _fit_edges_polyfit.stypy_kwargs_param_name = None
    _fit_edges_polyfit.stypy_call_defaults = defaults
    _fit_edges_polyfit.stypy_call_varargs = varargs
    _fit_edges_polyfit.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_fit_edges_polyfit', ['x', 'window_length', 'polyorder', 'deriv', 'delta', 'axis', 'y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_fit_edges_polyfit', localization, ['x', 'window_length', 'polyorder', 'deriv', 'delta', 'axis', 'y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_fit_edges_polyfit(...)' code ##################

    str_288494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, (-1)), 'str', '\n    Use polynomial interpolation of x at the low and high ends of the axis\n    to fill in the halflen values in y.\n\n    This function just calls _fit_edge twice, once for each end of the axis.\n    ')
    
    # Assigning a BinOp to a Name (line 215):
    
    # Assigning a BinOp to a Name (line 215):
    # Getting the type of 'window_length' (line 215)
    window_length_288495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 14), 'window_length')
    int_288496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 31), 'int')
    # Applying the binary operator '//' (line 215)
    result_floordiv_288497 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 14), '//', window_length_288495, int_288496)
    
    # Assigning a type to the variable 'halflen' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'halflen', result_floordiv_288497)
    
    # Call to _fit_edge(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'x' (line 216)
    x_288499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 14), 'x', False)
    int_288500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 17), 'int')
    # Getting the type of 'window_length' (line 216)
    window_length_288501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 20), 'window_length', False)
    int_288502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 35), 'int')
    # Getting the type of 'halflen' (line 216)
    halflen_288503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 38), 'halflen', False)
    # Getting the type of 'axis' (line 216)
    axis_288504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 47), 'axis', False)
    # Getting the type of 'polyorder' (line 217)
    polyorder_288505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 14), 'polyorder', False)
    # Getting the type of 'deriv' (line 217)
    deriv_288506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 25), 'deriv', False)
    # Getting the type of 'delta' (line 217)
    delta_288507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 32), 'delta', False)
    # Getting the type of 'y' (line 217)
    y_288508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 39), 'y', False)
    # Processing the call keyword arguments (line 216)
    kwargs_288509 = {}
    # Getting the type of '_fit_edge' (line 216)
    _fit_edge_288498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), '_fit_edge', False)
    # Calling _fit_edge(args, kwargs) (line 216)
    _fit_edge_call_result_288510 = invoke(stypy.reporting.localization.Localization(__file__, 216, 4), _fit_edge_288498, *[x_288499, int_288500, window_length_288501, int_288502, halflen_288503, axis_288504, polyorder_288505, deriv_288506, delta_288507, y_288508], **kwargs_288509)
    
    
    # Assigning a Subscript to a Name (line 218):
    
    # Assigning a Subscript to a Name (line 218):
    
    # Obtaining the type of the subscript
    # Getting the type of 'axis' (line 218)
    axis_288511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'axis')
    # Getting the type of 'x' (line 218)
    x_288512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'x')
    # Obtaining the member 'shape' of a type (line 218)
    shape_288513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), x_288512, 'shape')
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___288514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), shape_288513, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_288515 = invoke(stypy.reporting.localization.Localization(__file__, 218, 8), getitem___288514, axis_288511)
    
    # Assigning a type to the variable 'n' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'n', subscript_call_result_288515)
    
    # Call to _fit_edge(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'x' (line 219)
    x_288517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 14), 'x', False)
    # Getting the type of 'n' (line 219)
    n_288518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 17), 'n', False)
    # Getting the type of 'window_length' (line 219)
    window_length_288519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'window_length', False)
    # Applying the binary operator '-' (line 219)
    result_sub_288520 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 17), '-', n_288518, window_length_288519)
    
    # Getting the type of 'n' (line 219)
    n_288521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 36), 'n', False)
    # Getting the type of 'n' (line 219)
    n_288522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 39), 'n', False)
    # Getting the type of 'halflen' (line 219)
    halflen_288523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 43), 'halflen', False)
    # Applying the binary operator '-' (line 219)
    result_sub_288524 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 39), '-', n_288522, halflen_288523)
    
    # Getting the type of 'n' (line 219)
    n_288525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 52), 'n', False)
    # Getting the type of 'axis' (line 219)
    axis_288526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 55), 'axis', False)
    # Getting the type of 'polyorder' (line 220)
    polyorder_288527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 14), 'polyorder', False)
    # Getting the type of 'deriv' (line 220)
    deriv_288528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 25), 'deriv', False)
    # Getting the type of 'delta' (line 220)
    delta_288529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 32), 'delta', False)
    # Getting the type of 'y' (line 220)
    y_288530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 39), 'y', False)
    # Processing the call keyword arguments (line 219)
    kwargs_288531 = {}
    # Getting the type of '_fit_edge' (line 219)
    _fit_edge_288516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), '_fit_edge', False)
    # Calling _fit_edge(args, kwargs) (line 219)
    _fit_edge_call_result_288532 = invoke(stypy.reporting.localization.Localization(__file__, 219, 4), _fit_edge_288516, *[x_288517, result_sub_288520, n_288521, result_sub_288524, n_288525, axis_288526, polyorder_288527, deriv_288528, delta_288529, y_288530], **kwargs_288531)
    
    
    # ################# End of '_fit_edges_polyfit(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_fit_edges_polyfit' in the type store
    # Getting the type of 'stypy_return_type' (line 208)
    stypy_return_type_288533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288533)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_fit_edges_polyfit'
    return stypy_return_type_288533

# Assigning a type to the variable '_fit_edges_polyfit' (line 208)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), '_fit_edges_polyfit', _fit_edges_polyfit)

@norecursion
def savgol_filter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_288534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 53), 'int')
    float_288535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 62), 'float')
    int_288536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 23), 'int')
    str_288537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 32), 'str', 'interp')
    float_288538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 47), 'float')
    defaults = [int_288534, float_288535, int_288536, str_288537, float_288538]
    # Create a new context for function 'savgol_filter'
    module_type_store = module_type_store.open_function_context('savgol_filter', 223, 0, False)
    
    # Passed parameters checking function
    savgol_filter.stypy_localization = localization
    savgol_filter.stypy_type_of_self = None
    savgol_filter.stypy_type_store = module_type_store
    savgol_filter.stypy_function_name = 'savgol_filter'
    savgol_filter.stypy_param_names_list = ['x', 'window_length', 'polyorder', 'deriv', 'delta', 'axis', 'mode', 'cval']
    savgol_filter.stypy_varargs_param_name = None
    savgol_filter.stypy_kwargs_param_name = None
    savgol_filter.stypy_call_defaults = defaults
    savgol_filter.stypy_call_varargs = varargs
    savgol_filter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'savgol_filter', ['x', 'window_length', 'polyorder', 'deriv', 'delta', 'axis', 'mode', 'cval'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'savgol_filter', localization, ['x', 'window_length', 'polyorder', 'deriv', 'delta', 'axis', 'mode', 'cval'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'savgol_filter(...)' code ##################

    str_288539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, (-1)), 'str', " Apply a Savitzky-Golay filter to an array.\n\n    This is a 1-d filter.  If `x`  has dimension greater than 1, `axis`\n    determines the axis along which the filter is applied.\n\n    Parameters\n    ----------\n    x : array_like\n        The data to be filtered.  If `x` is not a single or double precision\n        floating point array, it will be converted to type `numpy.float64`\n        before filtering.\n    window_length : int\n        The length of the filter window (i.e. the number of coefficients).\n        `window_length` must be a positive odd integer. If `mode` is 'interp',\n        `window_length` must be less than or equal to the size of `x`.\n    polyorder : int\n        The order of the polynomial used to fit the samples.\n        `polyorder` must be less than `window_length`.\n    deriv : int, optional\n        The order of the derivative to compute.  This must be a\n        nonnegative integer.  The default is 0, which means to filter\n        the data without differentiating.\n    delta : float, optional\n        The spacing of the samples to which the filter will be applied.\n        This is only used if deriv > 0.  Default is 1.0.\n    axis : int, optional\n        The axis of the array `x` along which the filter is to be applied.\n        Default is -1.\n    mode : str, optional\n        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This\n        determines the type of extension to use for the padded signal to\n        which the filter is applied.  When `mode` is 'constant', the padding\n        value is given by `cval`.  See the Notes for more details on 'mirror',\n        'constant', 'wrap', and 'nearest'.\n        When the 'interp' mode is selected (the default), no extension\n        is used.  Instead, a degree `polyorder` polynomial is fit to the\n        last `window_length` values of the edges, and this polynomial is\n        used to evaluate the last `window_length // 2` output values.\n    cval : scalar, optional\n        Value to fill past the edges of the input if `mode` is 'constant'.\n        Default is 0.0.\n\n    Returns\n    -------\n    y : ndarray, same shape as `x`\n        The filtered data.\n\n    See Also\n    --------\n    savgol_coeffs\n\n    Notes\n    -----\n    Details on the `mode` options:\n\n        'mirror':\n            Repeats the values at the edges in reverse order.  The value\n            closest to the edge is not included.\n        'nearest':\n            The extension contains the nearest input value.\n        'constant':\n            The extension contains the value given by the `cval` argument.\n        'wrap':\n            The extension contains the values from the other end of the array.\n\n    For example, if the input is [1, 2, 3, 4, 5, 6, 7, 8], and\n    `window_length` is 7, the following shows the extended data for\n    the various `mode` options (assuming `cval` is 0)::\n\n        mode       |   Ext   |         Input          |   Ext\n        -----------+---------+------------------------+---------\n        'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5\n        'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8\n        'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0\n        'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3\n\n    .. versionadded:: 0.14.0\n\n    Examples\n    --------\n    >>> from scipy.signal import savgol_filter\n    >>> np.set_printoptions(precision=2)  # For compact display.\n    >>> x = np.array([2, 2, 5, 2, 1, 0, 1, 4, 9])\n\n    Filter with a window length of 5 and a degree 2 polynomial.  Use\n    the defaults for all other parameters.\n\n    >>> savgol_filter(x, 5, 2)\n    array([ 1.66,  3.17,  3.54,  2.86,  0.66,  0.17,  1.  ,  4.  ,  9.  ])\n\n    Note that the last five values in x are samples of a parabola, so\n    when mode='interp' (the default) is used with polyorder=2, the last\n    three values are unchanged.  Compare that to, for example,\n    `mode='nearest'`:\n\n    >>> savgol_filter(x, 5, 2, mode='nearest')\n    array([ 1.74,  3.03,  3.54,  2.86,  0.66,  0.17,  1.  ,  4.6 ,  7.97])\n\n    ")
    
    
    # Getting the type of 'mode' (line 324)
    mode_288540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 7), 'mode')
    
    # Obtaining an instance of the builtin type 'list' (line 324)
    list_288541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 324)
    # Adding element type (line 324)
    str_288542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 20), 'str', 'mirror')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 19), list_288541, str_288542)
    # Adding element type (line 324)
    str_288543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 30), 'str', 'constant')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 19), list_288541, str_288543)
    # Adding element type (line 324)
    str_288544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 42), 'str', 'nearest')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 19), list_288541, str_288544)
    # Adding element type (line 324)
    str_288545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 53), 'str', 'interp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 19), list_288541, str_288545)
    # Adding element type (line 324)
    str_288546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 63), 'str', 'wrap')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 19), list_288541, str_288546)
    
    # Applying the binary operator 'notin' (line 324)
    result_contains_288547 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 7), 'notin', mode_288540, list_288541)
    
    # Testing the type of an if condition (line 324)
    if_condition_288548 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 4), result_contains_288547)
    # Assigning a type to the variable 'if_condition_288548' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'if_condition_288548', if_condition_288548)
    # SSA begins for if statement (line 324)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 325)
    # Processing the call arguments (line 325)
    str_288550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 25), 'str', "mode must be 'mirror', 'constant', 'nearest' 'wrap' or 'interp'.")
    # Processing the call keyword arguments (line 325)
    kwargs_288551 = {}
    # Getting the type of 'ValueError' (line 325)
    ValueError_288549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 325)
    ValueError_call_result_288552 = invoke(stypy.reporting.localization.Localization(__file__, 325, 14), ValueError_288549, *[str_288550], **kwargs_288551)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 325, 8), ValueError_call_result_288552, 'raise parameter', BaseException)
    # SSA join for if statement (line 324)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 328):
    
    # Assigning a Call to a Name (line 328):
    
    # Call to asarray(...): (line 328)
    # Processing the call arguments (line 328)
    # Getting the type of 'x' (line 328)
    x_288555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 19), 'x', False)
    # Processing the call keyword arguments (line 328)
    kwargs_288556 = {}
    # Getting the type of 'np' (line 328)
    np_288553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'np', False)
    # Obtaining the member 'asarray' of a type (line 328)
    asarray_288554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 8), np_288553, 'asarray')
    # Calling asarray(args, kwargs) (line 328)
    asarray_call_result_288557 = invoke(stypy.reporting.localization.Localization(__file__, 328, 8), asarray_288554, *[x_288555], **kwargs_288556)
    
    # Assigning a type to the variable 'x' (line 328)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'x', asarray_call_result_288557)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'x' (line 330)
    x_288558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 7), 'x')
    # Obtaining the member 'dtype' of a type (line 330)
    dtype_288559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 7), x_288558, 'dtype')
    # Getting the type of 'np' (line 330)
    np_288560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 18), 'np')
    # Obtaining the member 'float64' of a type (line 330)
    float64_288561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 18), np_288560, 'float64')
    # Applying the binary operator '!=' (line 330)
    result_ne_288562 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 7), '!=', dtype_288559, float64_288561)
    
    
    # Getting the type of 'x' (line 330)
    x_288563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 33), 'x')
    # Obtaining the member 'dtype' of a type (line 330)
    dtype_288564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 33), x_288563, 'dtype')
    # Getting the type of 'np' (line 330)
    np_288565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 44), 'np')
    # Obtaining the member 'float32' of a type (line 330)
    float32_288566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 44), np_288565, 'float32')
    # Applying the binary operator '!=' (line 330)
    result_ne_288567 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 33), '!=', dtype_288564, float32_288566)
    
    # Applying the binary operator 'and' (line 330)
    result_and_keyword_288568 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 7), 'and', result_ne_288562, result_ne_288567)
    
    # Testing the type of an if condition (line 330)
    if_condition_288569 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 4), result_and_keyword_288568)
    # Assigning a type to the variable 'if_condition_288569' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'if_condition_288569', if_condition_288569)
    # SSA begins for if statement (line 330)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 331):
    
    # Assigning a Call to a Name (line 331):
    
    # Call to astype(...): (line 331)
    # Processing the call arguments (line 331)
    # Getting the type of 'np' (line 331)
    np_288572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 21), 'np', False)
    # Obtaining the member 'float64' of a type (line 331)
    float64_288573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 21), np_288572, 'float64')
    # Processing the call keyword arguments (line 331)
    kwargs_288574 = {}
    # Getting the type of 'x' (line 331)
    x_288570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'x', False)
    # Obtaining the member 'astype' of a type (line 331)
    astype_288571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 12), x_288570, 'astype')
    # Calling astype(args, kwargs) (line 331)
    astype_call_result_288575 = invoke(stypy.reporting.localization.Localization(__file__, 331, 12), astype_288571, *[float64_288573], **kwargs_288574)
    
    # Assigning a type to the variable 'x' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'x', astype_call_result_288575)
    # SSA join for if statement (line 330)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 333):
    
    # Assigning a Call to a Name (line 333):
    
    # Call to savgol_coeffs(...): (line 333)
    # Processing the call arguments (line 333)
    # Getting the type of 'window_length' (line 333)
    window_length_288577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 27), 'window_length', False)
    # Getting the type of 'polyorder' (line 333)
    polyorder_288578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 42), 'polyorder', False)
    # Processing the call keyword arguments (line 333)
    # Getting the type of 'deriv' (line 333)
    deriv_288579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 59), 'deriv', False)
    keyword_288580 = deriv_288579
    # Getting the type of 'delta' (line 333)
    delta_288581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 72), 'delta', False)
    keyword_288582 = delta_288581
    kwargs_288583 = {'deriv': keyword_288580, 'delta': keyword_288582}
    # Getting the type of 'savgol_coeffs' (line 333)
    savgol_coeffs_288576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 13), 'savgol_coeffs', False)
    # Calling savgol_coeffs(args, kwargs) (line 333)
    savgol_coeffs_call_result_288584 = invoke(stypy.reporting.localization.Localization(__file__, 333, 13), savgol_coeffs_288576, *[window_length_288577, polyorder_288578], **kwargs_288583)
    
    # Assigning a type to the variable 'coeffs' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'coeffs', savgol_coeffs_call_result_288584)
    
    
    # Getting the type of 'mode' (line 335)
    mode_288585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 7), 'mode')
    str_288586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 15), 'str', 'interp')
    # Applying the binary operator '==' (line 335)
    result_eq_288587 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 7), '==', mode_288585, str_288586)
    
    # Testing the type of an if condition (line 335)
    if_condition_288588 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 335, 4), result_eq_288587)
    # Assigning a type to the variable 'if_condition_288588' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'if_condition_288588', if_condition_288588)
    # SSA begins for if statement (line 335)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'window_length' (line 336)
    window_length_288589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 11), 'window_length')
    # Getting the type of 'x' (line 336)
    x_288590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 27), 'x')
    # Obtaining the member 'size' of a type (line 336)
    size_288591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 27), x_288590, 'size')
    # Applying the binary operator '>' (line 336)
    result_gt_288592 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 11), '>', window_length_288589, size_288591)
    
    # Testing the type of an if condition (line 336)
    if_condition_288593 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 8), result_gt_288592)
    # Assigning a type to the variable 'if_condition_288593' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'if_condition_288593', if_condition_288593)
    # SSA begins for if statement (line 336)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 337)
    # Processing the call arguments (line 337)
    str_288595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 29), 'str', "If mode is 'interp', window_length must be less than or equal to the size of x.")
    # Processing the call keyword arguments (line 337)
    kwargs_288596 = {}
    # Getting the type of 'ValueError' (line 337)
    ValueError_288594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 337)
    ValueError_call_result_288597 = invoke(stypy.reporting.localization.Localization(__file__, 337, 18), ValueError_288594, *[str_288595], **kwargs_288596)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 337, 12), ValueError_call_result_288597, 'raise parameter', BaseException)
    # SSA join for if statement (line 336)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 343):
    
    # Assigning a Call to a Name (line 343):
    
    # Call to convolve1d(...): (line 343)
    # Processing the call arguments (line 343)
    # Getting the type of 'x' (line 343)
    x_288599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 23), 'x', False)
    # Getting the type of 'coeffs' (line 343)
    coeffs_288600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 26), 'coeffs', False)
    # Processing the call keyword arguments (line 343)
    # Getting the type of 'axis' (line 343)
    axis_288601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 39), 'axis', False)
    keyword_288602 = axis_288601
    str_288603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 50), 'str', 'constant')
    keyword_288604 = str_288603
    kwargs_288605 = {'mode': keyword_288604, 'axis': keyword_288602}
    # Getting the type of 'convolve1d' (line 343)
    convolve1d_288598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'convolve1d', False)
    # Calling convolve1d(args, kwargs) (line 343)
    convolve1d_call_result_288606 = invoke(stypy.reporting.localization.Localization(__file__, 343, 12), convolve1d_288598, *[x_288599, coeffs_288600], **kwargs_288605)
    
    # Assigning a type to the variable 'y' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'y', convolve1d_call_result_288606)
    
    # Call to _fit_edges_polyfit(...): (line 344)
    # Processing the call arguments (line 344)
    # Getting the type of 'x' (line 344)
    x_288608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 27), 'x', False)
    # Getting the type of 'window_length' (line 344)
    window_length_288609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 30), 'window_length', False)
    # Getting the type of 'polyorder' (line 344)
    polyorder_288610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 45), 'polyorder', False)
    # Getting the type of 'deriv' (line 344)
    deriv_288611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 56), 'deriv', False)
    # Getting the type of 'delta' (line 344)
    delta_288612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 63), 'delta', False)
    # Getting the type of 'axis' (line 344)
    axis_288613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 70), 'axis', False)
    # Getting the type of 'y' (line 344)
    y_288614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 76), 'y', False)
    # Processing the call keyword arguments (line 344)
    kwargs_288615 = {}
    # Getting the type of '_fit_edges_polyfit' (line 344)
    _fit_edges_polyfit_288607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), '_fit_edges_polyfit', False)
    # Calling _fit_edges_polyfit(args, kwargs) (line 344)
    _fit_edges_polyfit_call_result_288616 = invoke(stypy.reporting.localization.Localization(__file__, 344, 8), _fit_edges_polyfit_288607, *[x_288608, window_length_288609, polyorder_288610, deriv_288611, delta_288612, axis_288613, y_288614], **kwargs_288615)
    
    # SSA branch for the else part of an if statement (line 335)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 347):
    
    # Assigning a Call to a Name (line 347):
    
    # Call to convolve1d(...): (line 347)
    # Processing the call arguments (line 347)
    # Getting the type of 'x' (line 347)
    x_288618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 23), 'x', False)
    # Getting the type of 'coeffs' (line 347)
    coeffs_288619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 26), 'coeffs', False)
    # Processing the call keyword arguments (line 347)
    # Getting the type of 'axis' (line 347)
    axis_288620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 39), 'axis', False)
    keyword_288621 = axis_288620
    # Getting the type of 'mode' (line 347)
    mode_288622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 50), 'mode', False)
    keyword_288623 = mode_288622
    # Getting the type of 'cval' (line 347)
    cval_288624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 61), 'cval', False)
    keyword_288625 = cval_288624
    kwargs_288626 = {'cval': keyword_288625, 'mode': keyword_288623, 'axis': keyword_288621}
    # Getting the type of 'convolve1d' (line 347)
    convolve1d_288617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'convolve1d', False)
    # Calling convolve1d(args, kwargs) (line 347)
    convolve1d_call_result_288627 = invoke(stypy.reporting.localization.Localization(__file__, 347, 12), convolve1d_288617, *[x_288618, coeffs_288619], **kwargs_288626)
    
    # Assigning a type to the variable 'y' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'y', convolve1d_call_result_288627)
    # SSA join for if statement (line 335)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'y' (line 349)
    y_288628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 11), 'y')
    # Assigning a type to the variable 'stypy_return_type' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'stypy_return_type', y_288628)
    
    # ################# End of 'savgol_filter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'savgol_filter' in the type store
    # Getting the type of 'stypy_return_type' (line 223)
    stypy_return_type_288629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_288629)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'savgol_filter'
    return stypy_return_type_288629

# Assigning a type to the variable 'savgol_filter' (line 223)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 0), 'savgol_filter', savgol_filter)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
