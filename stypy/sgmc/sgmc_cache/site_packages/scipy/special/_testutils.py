
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import os
4: 
5: from distutils.version import LooseVersion
6: 
7: import functools
8: 
9: import numpy as np
10: from numpy.testing import assert_
11: import pytest
12: 
13: import scipy.special as sc
14: 
15: __all__ = ['with_special_errors', 'assert_tol_equal', 'assert_func_equal',
16:            'FuncData']
17: 
18: 
19: #------------------------------------------------------------------------------
20: # Check if a module is present to be used in tests
21: #------------------------------------------------------------------------------
22: 
23: class MissingModule(object):
24:     def __init__(self, name):
25:         self.name = name
26: 
27: 
28: def check_version(module, min_ver):
29:     if type(module) == MissingModule:
30:         return pytest.mark.skip(reason="{} is not installed".format(module.name))
31:     return pytest.mark.skipif(LooseVersion(module.__version__) < LooseVersion(min_ver),
32:                               reason="{} version >= {} required".format(module.__name__, min_ver))
33: 
34: 
35: #------------------------------------------------------------------------------
36: # Enable convergence and loss of precision warnings -- turn off one by one
37: #------------------------------------------------------------------------------
38: 
39: def with_special_errors(func):
40:     '''
41:     Enable special function errors (such as underflow, overflow,
42:     loss of precision, etc.)
43:     '''
44:     @functools.wraps(func)
45:     def wrapper(*a, **kw):
46:         with sc.errstate(all='raise'):
47:             res = func(*a, **kw)
48:         return res
49:     return wrapper
50: 
51: 
52: #------------------------------------------------------------------------------
53: # Comparing function values at many data points at once, with helpful
54: #------------------------------------------------------------------------------
55: 
56: def assert_tol_equal(a, b, rtol=1e-7, atol=0, err_msg='', verbose=True):
57:     '''Assert that `a` and `b` are equal to tolerance ``atol + rtol*abs(b)``'''
58:     def compare(x, y):
59:         return np.allclose(x, y, rtol=rtol, atol=atol)
60:     a, b = np.asanyarray(a), np.asanyarray(b)
61:     header = 'Not equal to tolerance rtol=%g, atol=%g' % (rtol, atol)
62:     np.testing.utils.assert_array_compare(compare, a, b, err_msg=str(err_msg),
63:                                           verbose=verbose, header=header)
64: 
65: 
66: #------------------------------------------------------------------------------
67: # Comparing function values at many data points at once, with helpful
68: # error reports
69: #------------------------------------------------------------------------------
70: 
71: def assert_func_equal(func, results, points, rtol=None, atol=None,
72:                       param_filter=None, knownfailure=None,
73:                       vectorized=True, dtype=None, nan_ok=False,
74:                       ignore_inf_sign=False, distinguish_nan_and_inf=True):
75:     if hasattr(points, 'next'):
76:         # it's a generator
77:         points = list(points)
78: 
79:     points = np.asarray(points)
80:     if points.ndim == 1:
81:         points = points[:,None]
82:     nparams = points.shape[1]
83: 
84:     if hasattr(results, '__name__'):
85:         # function
86:         data = points
87:         result_columns = None
88:         result_func = results
89:     else:
90:         # dataset
91:         data = np.c_[points, results]
92:         result_columns = list(range(nparams, data.shape[1]))
93:         result_func = None
94: 
95:     fdata = FuncData(func, data, list(range(nparams)),
96:                      result_columns=result_columns, result_func=result_func,
97:                      rtol=rtol, atol=atol, param_filter=param_filter,
98:                      knownfailure=knownfailure, nan_ok=nan_ok, vectorized=vectorized,
99:                      ignore_inf_sign=ignore_inf_sign,
100:                      distinguish_nan_and_inf=distinguish_nan_and_inf)
101:     fdata.check()
102: 
103: 
104: class FuncData(object):
105:     '''
106:     Data set for checking a special function.
107: 
108:     Parameters
109:     ----------
110:     func : function
111:         Function to test
112:     filename : str
113:         Input file name
114:     param_columns : int or tuple of ints
115:         Columns indices in which the parameters to `func` lie.
116:         Can be imaginary integers to indicate that the parameter
117:         should be cast to complex.
118:     result_columns : int or tuple of ints, optional
119:         Column indices for expected results from `func`.
120:     result_func : callable, optional
121:         Function to call to obtain results.
122:     rtol : float, optional
123:         Required relative tolerance. Default is 5*eps.
124:     atol : float, optional
125:         Required absolute tolerance. Default is 5*tiny.
126:     param_filter : function, or tuple of functions/Nones, optional
127:         Filter functions to exclude some parameter ranges.
128:         If omitted, no filtering is done.
129:     knownfailure : str, optional
130:         Known failure error message to raise when the test is run.
131:         If omitted, no exception is raised.
132:     nan_ok : bool, optional
133:         If nan is always an accepted result.
134:     vectorized : bool, optional
135:         Whether all functions passed in are vectorized.
136:     ignore_inf_sign : bool, optional
137:         Whether to ignore signs of infinities.
138:         (Doesn't matter for complex-valued functions.)
139:     distinguish_nan_and_inf : bool, optional
140:         If True, treat numbers which contain nans or infs as as
141:         equal. Sets ignore_inf_sign to be True.
142: 
143:     '''
144: 
145:     def __init__(self, func, data, param_columns, result_columns=None,
146:                  result_func=None, rtol=None, atol=None, param_filter=None,
147:                  knownfailure=None, dataname=None, nan_ok=False, vectorized=True,
148:                  ignore_inf_sign=False, distinguish_nan_and_inf=True):
149:         self.func = func
150:         self.data = data
151:         self.dataname = dataname
152:         if not hasattr(param_columns, '__len__'):
153:             param_columns = (param_columns,)
154:         self.param_columns = tuple(param_columns)
155:         if result_columns is not None:
156:             if not hasattr(result_columns, '__len__'):
157:                 result_columns = (result_columns,)
158:             self.result_columns = tuple(result_columns)
159:             if result_func is not None:
160:                 raise ValueError("Only result_func or result_columns should be provided")
161:         elif result_func is not None:
162:             self.result_columns = None
163:         else:
164:             raise ValueError("Either result_func or result_columns should be provided")
165:         self.result_func = result_func
166:         self.rtol = rtol
167:         self.atol = atol
168:         if not hasattr(param_filter, '__len__'):
169:             param_filter = (param_filter,)
170:         self.param_filter = param_filter
171:         self.knownfailure = knownfailure
172:         self.nan_ok = nan_ok
173:         self.vectorized = vectorized
174:         self.ignore_inf_sign = ignore_inf_sign
175:         self.distinguish_nan_and_inf = distinguish_nan_and_inf
176:         if not self.distinguish_nan_and_inf:
177:             self.ignore_inf_sign = True
178: 
179:     def get_tolerances(self, dtype):
180:         if not np.issubdtype(dtype, np.inexact):
181:             dtype = np.dtype(float)
182:         info = np.finfo(dtype)
183:         rtol, atol = self.rtol, self.atol
184:         if rtol is None:
185:             rtol = 5*info.eps
186:         if atol is None:
187:             atol = 5*info.tiny
188:         return rtol, atol
189: 
190:     def check(self, data=None, dtype=None):
191:         '''Check the special function against the data.'''
192: 
193:         if self.knownfailure:
194:             pytest.xfail(reason=self.knownfailure)
195: 
196:         if data is None:
197:             data = self.data
198: 
199:         if dtype is None:
200:             dtype = data.dtype
201:         else:
202:             data = data.astype(dtype)
203: 
204:         rtol, atol = self.get_tolerances(dtype)
205: 
206:         # Apply given filter functions
207:         if self.param_filter:
208:             param_mask = np.ones((data.shape[0],), np.bool_)
209:             for j, filter in zip(self.param_columns, self.param_filter):
210:                 if filter:
211:                     param_mask &= list(filter(data[:,j]))
212:             data = data[param_mask]
213: 
214:         # Pick parameters from the correct columns
215:         params = []
216:         for j in self.param_columns:
217:             if np.iscomplexobj(j):
218:                 j = int(j.imag)
219:                 params.append(data[:,j].astype(complex))
220:             else:
221:                 params.append(data[:,j])
222: 
223:         # Helper for evaluating results
224:         def eval_func_at_params(func, skip_mask=None):
225:             if self.vectorized:
226:                 got = func(*params)
227:             else:
228:                 got = []
229:                 for j in range(len(params[0])):
230:                     if skip_mask is not None and skip_mask[j]:
231:                         got.append(np.nan)
232:                         continue
233:                     got.append(func(*tuple([params[i][j] for i in range(len(params))])))
234:                 got = np.asarray(got)
235:             if not isinstance(got, tuple):
236:                 got = (got,)
237:             return got
238: 
239:         # Evaluate function to be tested
240:         got = eval_func_at_params(self.func)
241: 
242:         # Grab the correct results
243:         if self.result_columns is not None:
244:             # Correct results passed in with the data
245:             wanted = tuple([data[:,icol] for icol in self.result_columns])
246:         else:
247:             # Function producing correct results passed in
248:             skip_mask = None
249:             if self.nan_ok and len(got) == 1:
250:                 # Don't spend time evaluating what doesn't need to be evaluated
251:                 skip_mask = np.isnan(got[0])
252:             wanted = eval_func_at_params(self.result_func, skip_mask=skip_mask)
253: 
254:         # Check the validity of each output returned
255:         assert_(len(got) == len(wanted))
256: 
257:         for output_num, (x, y) in enumerate(zip(got, wanted)):
258:             if np.issubdtype(x.dtype, np.complexfloating) or self.ignore_inf_sign:
259:                 pinf_x = np.isinf(x)
260:                 pinf_y = np.isinf(y)
261:                 minf_x = np.isinf(x)
262:                 minf_y = np.isinf(y)
263:             else:
264:                 pinf_x = np.isposinf(x)
265:                 pinf_y = np.isposinf(y)
266:                 minf_x = np.isneginf(x)
267:                 minf_y = np.isneginf(y)
268:             nan_x = np.isnan(x)
269:             nan_y = np.isnan(y)
270: 
271:             olderr = np.seterr(all='ignore')
272:             try:
273:                 abs_y = np.absolute(y)
274:                 abs_y[~np.isfinite(abs_y)] = 0
275:                 diff = np.absolute(x - y)
276:                 diff[~np.isfinite(diff)] = 0
277: 
278:                 rdiff = diff / np.absolute(y)
279:                 rdiff[~np.isfinite(rdiff)] = 0
280:             finally:
281:                 np.seterr(**olderr)
282: 
283:             tol_mask = (diff <= atol + rtol*abs_y)
284:             pinf_mask = (pinf_x == pinf_y)
285:             minf_mask = (minf_x == minf_y)
286: 
287:             nan_mask = (nan_x == nan_y)
288: 
289:             bad_j = ~(tol_mask & pinf_mask & minf_mask & nan_mask)
290: 
291:             point_count = bad_j.size
292:             if self.nan_ok:
293:                 bad_j &= ~nan_x
294:                 bad_j &= ~nan_y
295:                 point_count -= (nan_x | nan_y).sum()
296: 
297:             if not self.distinguish_nan_and_inf and not self.nan_ok:
298:                 # If nan's are okay we've already covered all these cases
299:                 inf_x = np.isinf(x)
300:                 inf_y = np.isinf(y)
301:                 both_nonfinite = (inf_x & nan_y) | (nan_x & inf_y)
302:                 bad_j &= ~both_nonfinite
303:                 point_count -= both_nonfinite.sum()
304: 
305:             if np.any(bad_j):
306:                 # Some bad results: inform what, where, and how bad
307:                 msg = [""]
308:                 msg.append("Max |adiff|: %g" % diff.max())
309:                 msg.append("Max |rdiff|: %g" % rdiff.max())
310:                 msg.append("Bad results (%d out of %d) for the following points (in output %d):"
311:                            % (np.sum(bad_j), point_count, output_num,))
312:                 for j in np.where(bad_j)[0]:
313:                     j = int(j)
314:                     fmt = lambda x: "%30s" % np.array2string(x[j], precision=18)
315:                     a = "  ".join(map(fmt, params))
316:                     b = "  ".join(map(fmt, got))
317:                     c = "  ".join(map(fmt, wanted))
318:                     d = fmt(rdiff)
319:                     msg.append("%s => %s != %s  (rdiff %s)" % (a, b, c, d))
320:                 assert_(False, "\n".join(msg))
321: 
322:     def __repr__(self):
323:         '''Pretty-printing, esp. for Nose output'''
324:         if np.any(list(map(np.iscomplexobj, self.param_columns))):
325:             is_complex = " (complex)"
326:         else:
327:             is_complex = ""
328:         if self.dataname:
329:             return "<Data for %s%s: %s>" % (self.func.__name__, is_complex,
330:                                             os.path.basename(self.dataname))
331:         else:
332:             return "<Data for %s%s>" % (self.func.__name__, is_complex)
333: 

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

# 'from distutils.version import LooseVersion' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_511016 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.version')

if (type(import_511016) is not StypyTypeError):

    if (import_511016 != 'pyd_module'):
        __import__(import_511016)
        sys_modules_511017 = sys.modules[import_511016]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.version', sys_modules_511017.module_type_store, module_type_store, ['LooseVersion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_511017, sys_modules_511017.module_type_store, module_type_store)
    else:
        from distutils.version import LooseVersion

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.version', None, module_type_store, ['LooseVersion'], [LooseVersion])

else:
    # Assigning a type to the variable 'distutils.version' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.version', import_511016)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import functools' statement (line 7)
import functools

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'functools', functools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import numpy' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_511018 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy')

if (type(import_511018) is not StypyTypeError):

    if (import_511018 != 'pyd_module'):
        __import__(import_511018)
        sys_modules_511019 = sys.modules[import_511018]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', sys_modules_511019.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy', import_511018)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy.testing import assert_' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_511020 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing')

if (type(import_511020) is not StypyTypeError):

    if (import_511020 != 'pyd_module'):
        __import__(import_511020)
        sys_modules_511021 = sys.modules[import_511020]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing', sys_modules_511021.module_type_store, module_type_store, ['assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_511021, sys_modules_511021.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing', None, module_type_store, ['assert_'], [assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing', import_511020)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import pytest' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_511022 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest')

if (type(import_511022) is not StypyTypeError):

    if (import_511022 != 'pyd_module'):
        __import__(import_511022)
        sys_modules_511023 = sys.modules[import_511022]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest', sys_modules_511023.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'pytest', import_511022)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import scipy.special' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_511024 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.special')

if (type(import_511024) is not StypyTypeError):

    if (import_511024 != 'pyd_module'):
        __import__(import_511024)
        sys_modules_511025 = sys.modules[import_511024]
        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'sc', sys_modules_511025.module_type_store, module_type_store)
    else:
        import scipy.special as sc

        import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'sc', scipy.special, module_type_store)

else:
    # Assigning a type to the variable 'scipy.special' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.special', import_511024)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')


# Assigning a List to a Name (line 15):

# Assigning a List to a Name (line 15):
__all__ = ['with_special_errors', 'assert_tol_equal', 'assert_func_equal', 'FuncData']
module_type_store.set_exportable_members(['with_special_errors', 'assert_tol_equal', 'assert_func_equal', 'FuncData'])

# Obtaining an instance of the builtin type 'list' (line 15)
list_511026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
str_511027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'str', 'with_special_errors')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_511026, str_511027)
# Adding element type (line 15)
str_511028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 34), 'str', 'assert_tol_equal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_511026, str_511028)
# Adding element type (line 15)
str_511029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 54), 'str', 'assert_func_equal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_511026, str_511029)
# Adding element type (line 15)
str_511030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'str', 'FuncData')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_511026, str_511030)

# Assigning a type to the variable '__all__' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), '__all__', list_511026)
# Declaration of the 'MissingModule' class

class MissingModule(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MissingModule.__init__', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 25):
        
        # Assigning a Name to a Attribute (line 25):
        # Getting the type of 'name' (line 25)
        name_511031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'name')
        # Getting the type of 'self' (line 25)
        self_511032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member 'name' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_511032, 'name', name_511031)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'MissingModule' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'MissingModule', MissingModule)

@norecursion
def check_version(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_version'
    module_type_store = module_type_store.open_function_context('check_version', 28, 0, False)
    
    # Passed parameters checking function
    check_version.stypy_localization = localization
    check_version.stypy_type_of_self = None
    check_version.stypy_type_store = module_type_store
    check_version.stypy_function_name = 'check_version'
    check_version.stypy_param_names_list = ['module', 'min_ver']
    check_version.stypy_varargs_param_name = None
    check_version.stypy_kwargs_param_name = None
    check_version.stypy_call_defaults = defaults
    check_version.stypy_call_varargs = varargs
    check_version.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_version', ['module', 'min_ver'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_version', localization, ['module', 'min_ver'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_version(...)' code ##################

    
    
    
    # Call to type(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'module' (line 29)
    module_511034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'module', False)
    # Processing the call keyword arguments (line 29)
    kwargs_511035 = {}
    # Getting the type of 'type' (line 29)
    type_511033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 7), 'type', False)
    # Calling type(args, kwargs) (line 29)
    type_call_result_511036 = invoke(stypy.reporting.localization.Localization(__file__, 29, 7), type_511033, *[module_511034], **kwargs_511035)
    
    # Getting the type of 'MissingModule' (line 29)
    MissingModule_511037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'MissingModule')
    # Applying the binary operator '==' (line 29)
    result_eq_511038 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 7), '==', type_call_result_511036, MissingModule_511037)
    
    # Testing the type of an if condition (line 29)
    if_condition_511039 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 4), result_eq_511038)
    # Assigning a type to the variable 'if_condition_511039' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'if_condition_511039', if_condition_511039)
    # SSA begins for if statement (line 29)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to skip(...): (line 30)
    # Processing the call keyword arguments (line 30)
    
    # Call to format(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'module' (line 30)
    module_511045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 68), 'module', False)
    # Obtaining the member 'name' of a type (line 30)
    name_511046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 68), module_511045, 'name')
    # Processing the call keyword arguments (line 30)
    kwargs_511047 = {}
    str_511043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 39), 'str', '{} is not installed')
    # Obtaining the member 'format' of a type (line 30)
    format_511044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 39), str_511043, 'format')
    # Calling format(args, kwargs) (line 30)
    format_call_result_511048 = invoke(stypy.reporting.localization.Localization(__file__, 30, 39), format_511044, *[name_511046], **kwargs_511047)
    
    keyword_511049 = format_call_result_511048
    kwargs_511050 = {'reason': keyword_511049}
    # Getting the type of 'pytest' (line 30)
    pytest_511040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'pytest', False)
    # Obtaining the member 'mark' of a type (line 30)
    mark_511041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 15), pytest_511040, 'mark')
    # Obtaining the member 'skip' of a type (line 30)
    skip_511042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 15), mark_511041, 'skip')
    # Calling skip(args, kwargs) (line 30)
    skip_call_result_511051 = invoke(stypy.reporting.localization.Localization(__file__, 30, 15), skip_511042, *[], **kwargs_511050)
    
    # Assigning a type to the variable 'stypy_return_type' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type', skip_call_result_511051)
    # SSA join for if statement (line 29)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to skipif(...): (line 31)
    # Processing the call arguments (line 31)
    
    
    # Call to LooseVersion(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'module' (line 31)
    module_511056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 43), 'module', False)
    # Obtaining the member '__version__' of a type (line 31)
    version___511057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 43), module_511056, '__version__')
    # Processing the call keyword arguments (line 31)
    kwargs_511058 = {}
    # Getting the type of 'LooseVersion' (line 31)
    LooseVersion_511055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'LooseVersion', False)
    # Calling LooseVersion(args, kwargs) (line 31)
    LooseVersion_call_result_511059 = invoke(stypy.reporting.localization.Localization(__file__, 31, 30), LooseVersion_511055, *[version___511057], **kwargs_511058)
    
    
    # Call to LooseVersion(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'min_ver' (line 31)
    min_ver_511061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 78), 'min_ver', False)
    # Processing the call keyword arguments (line 31)
    kwargs_511062 = {}
    # Getting the type of 'LooseVersion' (line 31)
    LooseVersion_511060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 65), 'LooseVersion', False)
    # Calling LooseVersion(args, kwargs) (line 31)
    LooseVersion_call_result_511063 = invoke(stypy.reporting.localization.Localization(__file__, 31, 65), LooseVersion_511060, *[min_ver_511061], **kwargs_511062)
    
    # Applying the binary operator '<' (line 31)
    result_lt_511064 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 30), '<', LooseVersion_call_result_511059, LooseVersion_call_result_511063)
    
    # Processing the call keyword arguments (line 31)
    
    # Call to format(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'module' (line 32)
    module_511067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 72), 'module', False)
    # Obtaining the member '__name__' of a type (line 32)
    name___511068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 72), module_511067, '__name__')
    # Getting the type of 'min_ver' (line 32)
    min_ver_511069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 89), 'min_ver', False)
    # Processing the call keyword arguments (line 32)
    kwargs_511070 = {}
    str_511065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 37), 'str', '{} version >= {} required')
    # Obtaining the member 'format' of a type (line 32)
    format_511066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 37), str_511065, 'format')
    # Calling format(args, kwargs) (line 32)
    format_call_result_511071 = invoke(stypy.reporting.localization.Localization(__file__, 32, 37), format_511066, *[name___511068, min_ver_511069], **kwargs_511070)
    
    keyword_511072 = format_call_result_511071
    kwargs_511073 = {'reason': keyword_511072}
    # Getting the type of 'pytest' (line 31)
    pytest_511052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'pytest', False)
    # Obtaining the member 'mark' of a type (line 31)
    mark_511053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 11), pytest_511052, 'mark')
    # Obtaining the member 'skipif' of a type (line 31)
    skipif_511054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 11), mark_511053, 'skipif')
    # Calling skipif(args, kwargs) (line 31)
    skipif_call_result_511074 = invoke(stypy.reporting.localization.Localization(__file__, 31, 11), skipif_511054, *[result_lt_511064], **kwargs_511073)
    
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type', skipif_call_result_511074)
    
    # ################# End of 'check_version(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_version' in the type store
    # Getting the type of 'stypy_return_type' (line 28)
    stypy_return_type_511075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_511075)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_version'
    return stypy_return_type_511075

# Assigning a type to the variable 'check_version' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'check_version', check_version)

@norecursion
def with_special_errors(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'with_special_errors'
    module_type_store = module_type_store.open_function_context('with_special_errors', 39, 0, False)
    
    # Passed parameters checking function
    with_special_errors.stypy_localization = localization
    with_special_errors.stypy_type_of_self = None
    with_special_errors.stypy_type_store = module_type_store
    with_special_errors.stypy_function_name = 'with_special_errors'
    with_special_errors.stypy_param_names_list = ['func']
    with_special_errors.stypy_varargs_param_name = None
    with_special_errors.stypy_kwargs_param_name = None
    with_special_errors.stypy_call_defaults = defaults
    with_special_errors.stypy_call_varargs = varargs
    with_special_errors.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'with_special_errors', ['func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'with_special_errors', localization, ['func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'with_special_errors(...)' code ##################

    str_511076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', '\n    Enable special function errors (such as underflow, overflow,\n    loss of precision, etc.)\n    ')

    @norecursion
    def wrapper(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'wrapper'
        module_type_store = module_type_store.open_function_context('wrapper', 44, 4, False)
        
        # Passed parameters checking function
        wrapper.stypy_localization = localization
        wrapper.stypy_type_of_self = None
        wrapper.stypy_type_store = module_type_store
        wrapper.stypy_function_name = 'wrapper'
        wrapper.stypy_param_names_list = []
        wrapper.stypy_varargs_param_name = 'a'
        wrapper.stypy_kwargs_param_name = 'kw'
        wrapper.stypy_call_defaults = defaults
        wrapper.stypy_call_varargs = varargs
        wrapper.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'wrapper', [], 'a', 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'wrapper', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'wrapper(...)' code ##################

        
        # Call to errstate(...): (line 46)
        # Processing the call keyword arguments (line 46)
        str_511079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 29), 'str', 'raise')
        keyword_511080 = str_511079
        kwargs_511081 = {'all': keyword_511080}
        # Getting the type of 'sc' (line 46)
        sc_511077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'sc', False)
        # Obtaining the member 'errstate' of a type (line 46)
        errstate_511078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 13), sc_511077, 'errstate')
        # Calling errstate(args, kwargs) (line 46)
        errstate_call_result_511082 = invoke(stypy.reporting.localization.Localization(__file__, 46, 13), errstate_511078, *[], **kwargs_511081)
        
        with_511083 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 46, 13), errstate_call_result_511082, 'with parameter', '__enter__', '__exit__')

        if with_511083:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 46)
            enter___511084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 13), errstate_call_result_511082, '__enter__')
            with_enter_511085 = invoke(stypy.reporting.localization.Localization(__file__, 46, 13), enter___511084)
            
            # Assigning a Call to a Name (line 47):
            
            # Assigning a Call to a Name (line 47):
            
            # Call to func(...): (line 47)
            # Getting the type of 'a' (line 47)
            a_511087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'a', False)
            # Processing the call keyword arguments (line 47)
            # Getting the type of 'kw' (line 47)
            kw_511088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 29), 'kw', False)
            kwargs_511089 = {'kw_511088': kw_511088}
            # Getting the type of 'func' (line 47)
            func_511086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 18), 'func', False)
            # Calling func(args, kwargs) (line 47)
            func_call_result_511090 = invoke(stypy.reporting.localization.Localization(__file__, 47, 18), func_511086, *[a_511087], **kwargs_511089)
            
            # Assigning a type to the variable 'res' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'res', func_call_result_511090)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 46)
            exit___511091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 13), errstate_call_result_511082, '__exit__')
            with_exit_511092 = invoke(stypy.reporting.localization.Localization(__file__, 46, 13), exit___511091, None, None, None)

        # Getting the type of 'res' (line 48)
        res_511093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'res')
        # Assigning a type to the variable 'stypy_return_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', res_511093)
        
        # ################# End of 'wrapper(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'wrapper' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_511094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_511094)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'wrapper'
        return stypy_return_type_511094

    # Assigning a type to the variable 'wrapper' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'wrapper', wrapper)
    # Getting the type of 'wrapper' (line 49)
    wrapper_511095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'wrapper')
    # Assigning a type to the variable 'stypy_return_type' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type', wrapper_511095)
    
    # ################# End of 'with_special_errors(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'with_special_errors' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_511096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_511096)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'with_special_errors'
    return stypy_return_type_511096

# Assigning a type to the variable 'with_special_errors' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'with_special_errors', with_special_errors)

@norecursion
def assert_tol_equal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    float_511097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 32), 'float')
    int_511098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 43), 'int')
    str_511099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 54), 'str', '')
    # Getting the type of 'True' (line 56)
    True_511100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 66), 'True')
    defaults = [float_511097, int_511098, str_511099, True_511100]
    # Create a new context for function 'assert_tol_equal'
    module_type_store = module_type_store.open_function_context('assert_tol_equal', 56, 0, False)
    
    # Passed parameters checking function
    assert_tol_equal.stypy_localization = localization
    assert_tol_equal.stypy_type_of_self = None
    assert_tol_equal.stypy_type_store = module_type_store
    assert_tol_equal.stypy_function_name = 'assert_tol_equal'
    assert_tol_equal.stypy_param_names_list = ['a', 'b', 'rtol', 'atol', 'err_msg', 'verbose']
    assert_tol_equal.stypy_varargs_param_name = None
    assert_tol_equal.stypy_kwargs_param_name = None
    assert_tol_equal.stypy_call_defaults = defaults
    assert_tol_equal.stypy_call_varargs = varargs
    assert_tol_equal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_tol_equal', ['a', 'b', 'rtol', 'atol', 'err_msg', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_tol_equal', localization, ['a', 'b', 'rtol', 'atol', 'err_msg', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_tol_equal(...)' code ##################

    str_511101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 4), 'str', 'Assert that `a` and `b` are equal to tolerance ``atol + rtol*abs(b)``')

    @norecursion
    def compare(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'compare'
        module_type_store = module_type_store.open_function_context('compare', 58, 4, False)
        
        # Passed parameters checking function
        compare.stypy_localization = localization
        compare.stypy_type_of_self = None
        compare.stypy_type_store = module_type_store
        compare.stypy_function_name = 'compare'
        compare.stypy_param_names_list = ['x', 'y']
        compare.stypy_varargs_param_name = None
        compare.stypy_kwargs_param_name = None
        compare.stypy_call_defaults = defaults
        compare.stypy_call_varargs = varargs
        compare.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'compare', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'compare', localization, ['x', 'y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'compare(...)' code ##################

        
        # Call to allclose(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'x' (line 59)
        x_511104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 27), 'x', False)
        # Getting the type of 'y' (line 59)
        y_511105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 30), 'y', False)
        # Processing the call keyword arguments (line 59)
        # Getting the type of 'rtol' (line 59)
        rtol_511106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 38), 'rtol', False)
        keyword_511107 = rtol_511106
        # Getting the type of 'atol' (line 59)
        atol_511108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 49), 'atol', False)
        keyword_511109 = atol_511108
        kwargs_511110 = {'rtol': keyword_511107, 'atol': keyword_511109}
        # Getting the type of 'np' (line 59)
        np_511102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'np', False)
        # Obtaining the member 'allclose' of a type (line 59)
        allclose_511103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 15), np_511102, 'allclose')
        # Calling allclose(args, kwargs) (line 59)
        allclose_call_result_511111 = invoke(stypy.reporting.localization.Localization(__file__, 59, 15), allclose_511103, *[x_511104, y_511105], **kwargs_511110)
        
        # Assigning a type to the variable 'stypy_return_type' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type', allclose_call_result_511111)
        
        # ################# End of 'compare(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'compare' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_511112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_511112)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'compare'
        return stypy_return_type_511112

    # Assigning a type to the variable 'compare' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'compare', compare)
    
    # Assigning a Tuple to a Tuple (line 60):
    
    # Assigning a Call to a Name (line 60):
    
    # Call to asanyarray(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'a' (line 60)
    a_511115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'a', False)
    # Processing the call keyword arguments (line 60)
    kwargs_511116 = {}
    # Getting the type of 'np' (line 60)
    np_511113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 60)
    asanyarray_511114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 11), np_511113, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 60)
    asanyarray_call_result_511117 = invoke(stypy.reporting.localization.Localization(__file__, 60, 11), asanyarray_511114, *[a_511115], **kwargs_511116)
    
    # Assigning a type to the variable 'tuple_assignment_511010' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'tuple_assignment_511010', asanyarray_call_result_511117)
    
    # Assigning a Call to a Name (line 60):
    
    # Call to asanyarray(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'b' (line 60)
    b_511120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 43), 'b', False)
    # Processing the call keyword arguments (line 60)
    kwargs_511121 = {}
    # Getting the type of 'np' (line 60)
    np_511118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 60)
    asanyarray_511119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 29), np_511118, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 60)
    asanyarray_call_result_511122 = invoke(stypy.reporting.localization.Localization(__file__, 60, 29), asanyarray_511119, *[b_511120], **kwargs_511121)
    
    # Assigning a type to the variable 'tuple_assignment_511011' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'tuple_assignment_511011', asanyarray_call_result_511122)
    
    # Assigning a Name to a Name (line 60):
    # Getting the type of 'tuple_assignment_511010' (line 60)
    tuple_assignment_511010_511123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'tuple_assignment_511010')
    # Assigning a type to the variable 'a' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'a', tuple_assignment_511010_511123)
    
    # Assigning a Name to a Name (line 60):
    # Getting the type of 'tuple_assignment_511011' (line 60)
    tuple_assignment_511011_511124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'tuple_assignment_511011')
    # Assigning a type to the variable 'b' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 7), 'b', tuple_assignment_511011_511124)
    
    # Assigning a BinOp to a Name (line 61):
    
    # Assigning a BinOp to a Name (line 61):
    str_511125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 13), 'str', 'Not equal to tolerance rtol=%g, atol=%g')
    
    # Obtaining an instance of the builtin type 'tuple' (line 61)
    tuple_511126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 58), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 61)
    # Adding element type (line 61)
    # Getting the type of 'rtol' (line 61)
    rtol_511127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 58), 'rtol')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 58), tuple_511126, rtol_511127)
    # Adding element type (line 61)
    # Getting the type of 'atol' (line 61)
    atol_511128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 64), 'atol')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 58), tuple_511126, atol_511128)
    
    # Applying the binary operator '%' (line 61)
    result_mod_511129 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 13), '%', str_511125, tuple_511126)
    
    # Assigning a type to the variable 'header' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'header', result_mod_511129)
    
    # Call to assert_array_compare(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'compare' (line 62)
    compare_511134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 42), 'compare', False)
    # Getting the type of 'a' (line 62)
    a_511135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 51), 'a', False)
    # Getting the type of 'b' (line 62)
    b_511136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 54), 'b', False)
    # Processing the call keyword arguments (line 62)
    
    # Call to str(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'err_msg' (line 62)
    err_msg_511138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 69), 'err_msg', False)
    # Processing the call keyword arguments (line 62)
    kwargs_511139 = {}
    # Getting the type of 'str' (line 62)
    str_511137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 65), 'str', False)
    # Calling str(args, kwargs) (line 62)
    str_call_result_511140 = invoke(stypy.reporting.localization.Localization(__file__, 62, 65), str_511137, *[err_msg_511138], **kwargs_511139)
    
    keyword_511141 = str_call_result_511140
    # Getting the type of 'verbose' (line 63)
    verbose_511142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 50), 'verbose', False)
    keyword_511143 = verbose_511142
    # Getting the type of 'header' (line 63)
    header_511144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 66), 'header', False)
    keyword_511145 = header_511144
    kwargs_511146 = {'header': keyword_511145, 'err_msg': keyword_511141, 'verbose': keyword_511143}
    # Getting the type of 'np' (line 62)
    np_511130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'np', False)
    # Obtaining the member 'testing' of a type (line 62)
    testing_511131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 4), np_511130, 'testing')
    # Obtaining the member 'utils' of a type (line 62)
    utils_511132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 4), testing_511131, 'utils')
    # Obtaining the member 'assert_array_compare' of a type (line 62)
    assert_array_compare_511133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 4), utils_511132, 'assert_array_compare')
    # Calling assert_array_compare(args, kwargs) (line 62)
    assert_array_compare_call_result_511147 = invoke(stypy.reporting.localization.Localization(__file__, 62, 4), assert_array_compare_511133, *[compare_511134, a_511135, b_511136], **kwargs_511146)
    
    
    # ################# End of 'assert_tol_equal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_tol_equal' in the type store
    # Getting the type of 'stypy_return_type' (line 56)
    stypy_return_type_511148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_511148)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_tol_equal'
    return stypy_return_type_511148

# Assigning a type to the variable 'assert_tol_equal' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'assert_tol_equal', assert_tol_equal)

@norecursion
def assert_func_equal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 71)
    None_511149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 50), 'None')
    # Getting the type of 'None' (line 71)
    None_511150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 61), 'None')
    # Getting the type of 'None' (line 72)
    None_511151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 35), 'None')
    # Getting the type of 'None' (line 72)
    None_511152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 54), 'None')
    # Getting the type of 'True' (line 73)
    True_511153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 33), 'True')
    # Getting the type of 'None' (line 73)
    None_511154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 45), 'None')
    # Getting the type of 'False' (line 73)
    False_511155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 58), 'False')
    # Getting the type of 'False' (line 74)
    False_511156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 38), 'False')
    # Getting the type of 'True' (line 74)
    True_511157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 69), 'True')
    defaults = [None_511149, None_511150, None_511151, None_511152, True_511153, None_511154, False_511155, False_511156, True_511157]
    # Create a new context for function 'assert_func_equal'
    module_type_store = module_type_store.open_function_context('assert_func_equal', 71, 0, False)
    
    # Passed parameters checking function
    assert_func_equal.stypy_localization = localization
    assert_func_equal.stypy_type_of_self = None
    assert_func_equal.stypy_type_store = module_type_store
    assert_func_equal.stypy_function_name = 'assert_func_equal'
    assert_func_equal.stypy_param_names_list = ['func', 'results', 'points', 'rtol', 'atol', 'param_filter', 'knownfailure', 'vectorized', 'dtype', 'nan_ok', 'ignore_inf_sign', 'distinguish_nan_and_inf']
    assert_func_equal.stypy_varargs_param_name = None
    assert_func_equal.stypy_kwargs_param_name = None
    assert_func_equal.stypy_call_defaults = defaults
    assert_func_equal.stypy_call_varargs = varargs
    assert_func_equal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_func_equal', ['func', 'results', 'points', 'rtol', 'atol', 'param_filter', 'knownfailure', 'vectorized', 'dtype', 'nan_ok', 'ignore_inf_sign', 'distinguish_nan_and_inf'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_func_equal', localization, ['func', 'results', 'points', 'rtol', 'atol', 'param_filter', 'knownfailure', 'vectorized', 'dtype', 'nan_ok', 'ignore_inf_sign', 'distinguish_nan_and_inf'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_func_equal(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 75)
    str_511158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 23), 'str', 'next')
    # Getting the type of 'points' (line 75)
    points_511159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'points')
    
    (may_be_511160, more_types_in_union_511161) = may_provide_member(str_511158, points_511159)

    if may_be_511160:

        if more_types_in_union_511161:
            # Runtime conditional SSA (line 75)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'points' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'points', remove_not_member_provider_from_union(points_511159, 'next'))
        
        # Assigning a Call to a Name (line 77):
        
        # Assigning a Call to a Name (line 77):
        
        # Call to list(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'points' (line 77)
        points_511163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 22), 'points', False)
        # Processing the call keyword arguments (line 77)
        kwargs_511164 = {}
        # Getting the type of 'list' (line 77)
        list_511162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 17), 'list', False)
        # Calling list(args, kwargs) (line 77)
        list_call_result_511165 = invoke(stypy.reporting.localization.Localization(__file__, 77, 17), list_511162, *[points_511163], **kwargs_511164)
        
        # Assigning a type to the variable 'points' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'points', list_call_result_511165)

        if more_types_in_union_511161:
            # SSA join for if statement (line 75)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 79):
    
    # Assigning a Call to a Name (line 79):
    
    # Call to asarray(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'points' (line 79)
    points_511168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'points', False)
    # Processing the call keyword arguments (line 79)
    kwargs_511169 = {}
    # Getting the type of 'np' (line 79)
    np_511166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 13), 'np', False)
    # Obtaining the member 'asarray' of a type (line 79)
    asarray_511167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 13), np_511166, 'asarray')
    # Calling asarray(args, kwargs) (line 79)
    asarray_call_result_511170 = invoke(stypy.reporting.localization.Localization(__file__, 79, 13), asarray_511167, *[points_511168], **kwargs_511169)
    
    # Assigning a type to the variable 'points' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'points', asarray_call_result_511170)
    
    
    # Getting the type of 'points' (line 80)
    points_511171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 7), 'points')
    # Obtaining the member 'ndim' of a type (line 80)
    ndim_511172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 7), points_511171, 'ndim')
    int_511173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 22), 'int')
    # Applying the binary operator '==' (line 80)
    result_eq_511174 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 7), '==', ndim_511172, int_511173)
    
    # Testing the type of an if condition (line 80)
    if_condition_511175 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 4), result_eq_511174)
    # Assigning a type to the variable 'if_condition_511175' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'if_condition_511175', if_condition_511175)
    # SSA begins for if statement (line 80)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 81):
    
    # Assigning a Subscript to a Name (line 81):
    
    # Obtaining the type of the subscript
    slice_511176 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 81, 17), None, None, None)
    # Getting the type of 'None' (line 81)
    None_511177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 26), 'None')
    # Getting the type of 'points' (line 81)
    points_511178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'points')
    # Obtaining the member '__getitem__' of a type (line 81)
    getitem___511179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 17), points_511178, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 81)
    subscript_call_result_511180 = invoke(stypy.reporting.localization.Localization(__file__, 81, 17), getitem___511179, (slice_511176, None_511177))
    
    # Assigning a type to the variable 'points' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'points', subscript_call_result_511180)
    # SSA join for if statement (line 80)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 82):
    
    # Assigning a Subscript to a Name (line 82):
    
    # Obtaining the type of the subscript
    int_511181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 27), 'int')
    # Getting the type of 'points' (line 82)
    points_511182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 14), 'points')
    # Obtaining the member 'shape' of a type (line 82)
    shape_511183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 14), points_511182, 'shape')
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___511184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 14), shape_511183, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_511185 = invoke(stypy.reporting.localization.Localization(__file__, 82, 14), getitem___511184, int_511181)
    
    # Assigning a type to the variable 'nparams' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'nparams', subscript_call_result_511185)
    
    # Type idiom detected: calculating its left and rigth part (line 84)
    str_511186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 24), 'str', '__name__')
    # Getting the type of 'results' (line 84)
    results_511187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'results')
    
    (may_be_511188, more_types_in_union_511189) = may_provide_member(str_511186, results_511187)

    if may_be_511188:

        if more_types_in_union_511189:
            # Runtime conditional SSA (line 84)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'results' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'results', remove_not_member_provider_from_union(results_511187, '__name__'))
        
        # Assigning a Name to a Name (line 86):
        
        # Assigning a Name to a Name (line 86):
        # Getting the type of 'points' (line 86)
        points_511190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'points')
        # Assigning a type to the variable 'data' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'data', points_511190)
        
        # Assigning a Name to a Name (line 87):
        
        # Assigning a Name to a Name (line 87):
        # Getting the type of 'None' (line 87)
        None_511191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 25), 'None')
        # Assigning a type to the variable 'result_columns' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'result_columns', None_511191)
        
        # Assigning a Name to a Name (line 88):
        
        # Assigning a Name to a Name (line 88):
        # Getting the type of 'results' (line 88)
        results_511192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 22), 'results')
        # Assigning a type to the variable 'result_func' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'result_func', results_511192)

        if more_types_in_union_511189:
            # Runtime conditional SSA for else branch (line 84)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_511188) or more_types_in_union_511189):
        # Assigning a type to the variable 'results' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'results', remove_member_provider_from_union(results_511187, '__name__'))
        
        # Assigning a Subscript to a Name (line 91):
        
        # Assigning a Subscript to a Name (line 91):
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 91)
        tuple_511193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 91)
        # Adding element type (line 91)
        # Getting the type of 'points' (line 91)
        points_511194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'points')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 21), tuple_511193, points_511194)
        # Adding element type (line 91)
        # Getting the type of 'results' (line 91)
        results_511195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'results')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 21), tuple_511193, results_511195)
        
        # Getting the type of 'np' (line 91)
        np_511196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'np')
        # Obtaining the member 'c_' of a type (line 91)
        c__511197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 15), np_511196, 'c_')
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___511198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 15), c__511197, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_511199 = invoke(stypy.reporting.localization.Localization(__file__, 91, 15), getitem___511198, tuple_511193)
        
        # Assigning a type to the variable 'data' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'data', subscript_call_result_511199)
        
        # Assigning a Call to a Name (line 92):
        
        # Assigning a Call to a Name (line 92):
        
        # Call to list(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Call to range(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'nparams' (line 92)
        nparams_511202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 36), 'nparams', False)
        
        # Obtaining the type of the subscript
        int_511203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 56), 'int')
        # Getting the type of 'data' (line 92)
        data_511204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 45), 'data', False)
        # Obtaining the member 'shape' of a type (line 92)
        shape_511205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 45), data_511204, 'shape')
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___511206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 45), shape_511205, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_511207 = invoke(stypy.reporting.localization.Localization(__file__, 92, 45), getitem___511206, int_511203)
        
        # Processing the call keyword arguments (line 92)
        kwargs_511208 = {}
        # Getting the type of 'range' (line 92)
        range_511201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 30), 'range', False)
        # Calling range(args, kwargs) (line 92)
        range_call_result_511209 = invoke(stypy.reporting.localization.Localization(__file__, 92, 30), range_511201, *[nparams_511202, subscript_call_result_511207], **kwargs_511208)
        
        # Processing the call keyword arguments (line 92)
        kwargs_511210 = {}
        # Getting the type of 'list' (line 92)
        list_511200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'list', False)
        # Calling list(args, kwargs) (line 92)
        list_call_result_511211 = invoke(stypy.reporting.localization.Localization(__file__, 92, 25), list_511200, *[range_call_result_511209], **kwargs_511210)
        
        # Assigning a type to the variable 'result_columns' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'result_columns', list_call_result_511211)
        
        # Assigning a Name to a Name (line 93):
        
        # Assigning a Name to a Name (line 93):
        # Getting the type of 'None' (line 93)
        None_511212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 22), 'None')
        # Assigning a type to the variable 'result_func' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'result_func', None_511212)

        if (may_be_511188 and more_types_in_union_511189):
            # SSA join for if statement (line 84)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 95):
    
    # Assigning a Call to a Name (line 95):
    
    # Call to FuncData(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'func' (line 95)
    func_511214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'func', False)
    # Getting the type of 'data' (line 95)
    data_511215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 27), 'data', False)
    
    # Call to list(...): (line 95)
    # Processing the call arguments (line 95)
    
    # Call to range(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'nparams' (line 95)
    nparams_511218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 44), 'nparams', False)
    # Processing the call keyword arguments (line 95)
    kwargs_511219 = {}
    # Getting the type of 'range' (line 95)
    range_511217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 38), 'range', False)
    # Calling range(args, kwargs) (line 95)
    range_call_result_511220 = invoke(stypy.reporting.localization.Localization(__file__, 95, 38), range_511217, *[nparams_511218], **kwargs_511219)
    
    # Processing the call keyword arguments (line 95)
    kwargs_511221 = {}
    # Getting the type of 'list' (line 95)
    list_511216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 33), 'list', False)
    # Calling list(args, kwargs) (line 95)
    list_call_result_511222 = invoke(stypy.reporting.localization.Localization(__file__, 95, 33), list_511216, *[range_call_result_511220], **kwargs_511221)
    
    # Processing the call keyword arguments (line 95)
    # Getting the type of 'result_columns' (line 96)
    result_columns_511223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 36), 'result_columns', False)
    keyword_511224 = result_columns_511223
    # Getting the type of 'result_func' (line 96)
    result_func_511225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 64), 'result_func', False)
    keyword_511226 = result_func_511225
    # Getting the type of 'rtol' (line 97)
    rtol_511227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 26), 'rtol', False)
    keyword_511228 = rtol_511227
    # Getting the type of 'atol' (line 97)
    atol_511229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 37), 'atol', False)
    keyword_511230 = atol_511229
    # Getting the type of 'param_filter' (line 97)
    param_filter_511231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 56), 'param_filter', False)
    keyword_511232 = param_filter_511231
    # Getting the type of 'knownfailure' (line 98)
    knownfailure_511233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 34), 'knownfailure', False)
    keyword_511234 = knownfailure_511233
    # Getting the type of 'nan_ok' (line 98)
    nan_ok_511235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 55), 'nan_ok', False)
    keyword_511236 = nan_ok_511235
    # Getting the type of 'vectorized' (line 98)
    vectorized_511237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 74), 'vectorized', False)
    keyword_511238 = vectorized_511237
    # Getting the type of 'ignore_inf_sign' (line 99)
    ignore_inf_sign_511239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 37), 'ignore_inf_sign', False)
    keyword_511240 = ignore_inf_sign_511239
    # Getting the type of 'distinguish_nan_and_inf' (line 100)
    distinguish_nan_and_inf_511241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 45), 'distinguish_nan_and_inf', False)
    keyword_511242 = distinguish_nan_and_inf_511241
    kwargs_511243 = {'param_filter': keyword_511232, 'nan_ok': keyword_511236, 'knownfailure': keyword_511234, 'distinguish_nan_and_inf': keyword_511242, 'vectorized': keyword_511238, 'rtol': keyword_511228, 'ignore_inf_sign': keyword_511240, 'atol': keyword_511230, 'result_func': keyword_511226, 'result_columns': keyword_511224}
    # Getting the type of 'FuncData' (line 95)
    FuncData_511213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'FuncData', False)
    # Calling FuncData(args, kwargs) (line 95)
    FuncData_call_result_511244 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), FuncData_511213, *[func_511214, data_511215, list_call_result_511222], **kwargs_511243)
    
    # Assigning a type to the variable 'fdata' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'fdata', FuncData_call_result_511244)
    
    # Call to check(...): (line 101)
    # Processing the call keyword arguments (line 101)
    kwargs_511247 = {}
    # Getting the type of 'fdata' (line 101)
    fdata_511245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'fdata', False)
    # Obtaining the member 'check' of a type (line 101)
    check_511246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 4), fdata_511245, 'check')
    # Calling check(args, kwargs) (line 101)
    check_call_result_511248 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), check_511246, *[], **kwargs_511247)
    
    
    # ################# End of 'assert_func_equal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_func_equal' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_511249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_511249)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_func_equal'
    return stypy_return_type_511249

# Assigning a type to the variable 'assert_func_equal' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'assert_func_equal', assert_func_equal)
# Declaration of the 'FuncData' class

class FuncData(object, ):
    str_511250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, (-1)), 'str', "\n    Data set for checking a special function.\n\n    Parameters\n    ----------\n    func : function\n        Function to test\n    filename : str\n        Input file name\n    param_columns : int or tuple of ints\n        Columns indices in which the parameters to `func` lie.\n        Can be imaginary integers to indicate that the parameter\n        should be cast to complex.\n    result_columns : int or tuple of ints, optional\n        Column indices for expected results from `func`.\n    result_func : callable, optional\n        Function to call to obtain results.\n    rtol : float, optional\n        Required relative tolerance. Default is 5*eps.\n    atol : float, optional\n        Required absolute tolerance. Default is 5*tiny.\n    param_filter : function, or tuple of functions/Nones, optional\n        Filter functions to exclude some parameter ranges.\n        If omitted, no filtering is done.\n    knownfailure : str, optional\n        Known failure error message to raise when the test is run.\n        If omitted, no exception is raised.\n    nan_ok : bool, optional\n        If nan is always an accepted result.\n    vectorized : bool, optional\n        Whether all functions passed in are vectorized.\n    ignore_inf_sign : bool, optional\n        Whether to ignore signs of infinities.\n        (Doesn't matter for complex-valued functions.)\n    distinguish_nan_and_inf : bool, optional\n        If True, treat numbers which contain nans or infs as as\n        equal. Sets ignore_inf_sign to be True.\n\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 145)
        None_511251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 65), 'None')
        # Getting the type of 'None' (line 146)
        None_511252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'None')
        # Getting the type of 'None' (line 146)
        None_511253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 40), 'None')
        # Getting the type of 'None' (line 146)
        None_511254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 51), 'None')
        # Getting the type of 'None' (line 146)
        None_511255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 70), 'None')
        # Getting the type of 'None' (line 147)
        None_511256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 30), 'None')
        # Getting the type of 'None' (line 147)
        None_511257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 45), 'None')
        # Getting the type of 'False' (line 147)
        False_511258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 58), 'False')
        # Getting the type of 'True' (line 147)
        True_511259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 76), 'True')
        # Getting the type of 'False' (line 148)
        False_511260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 33), 'False')
        # Getting the type of 'True' (line 148)
        True_511261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 64), 'True')
        defaults = [None_511251, None_511252, None_511253, None_511254, None_511255, None_511256, None_511257, False_511258, True_511259, False_511260, True_511261]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FuncData.__init__', ['func', 'data', 'param_columns', 'result_columns', 'result_func', 'rtol', 'atol', 'param_filter', 'knownfailure', 'dataname', 'nan_ok', 'vectorized', 'ignore_inf_sign', 'distinguish_nan_and_inf'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['func', 'data', 'param_columns', 'result_columns', 'result_func', 'rtol', 'atol', 'param_filter', 'knownfailure', 'dataname', 'nan_ok', 'vectorized', 'ignore_inf_sign', 'distinguish_nan_and_inf'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 149):
        
        # Assigning a Name to a Attribute (line 149):
        # Getting the type of 'func' (line 149)
        func_511262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'func')
        # Getting the type of 'self' (line 149)
        self_511263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'self')
        # Setting the type of the member 'func' of a type (line 149)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), self_511263, 'func', func_511262)
        
        # Assigning a Name to a Attribute (line 150):
        
        # Assigning a Name to a Attribute (line 150):
        # Getting the type of 'data' (line 150)
        data_511264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 20), 'data')
        # Getting the type of 'self' (line 150)
        self_511265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'self')
        # Setting the type of the member 'data' of a type (line 150)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), self_511265, 'data', data_511264)
        
        # Assigning a Name to a Attribute (line 151):
        
        # Assigning a Name to a Attribute (line 151):
        # Getting the type of 'dataname' (line 151)
        dataname_511266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'dataname')
        # Getting the type of 'self' (line 151)
        self_511267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'self')
        # Setting the type of the member 'dataname' of a type (line 151)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), self_511267, 'dataname', dataname_511266)
        
        # Type idiom detected: calculating its left and rigth part (line 152)
        str_511268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 38), 'str', '__len__')
        # Getting the type of 'param_columns' (line 152)
        param_columns_511269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 23), 'param_columns')
        
        (may_be_511270, more_types_in_union_511271) = may_not_provide_member(str_511268, param_columns_511269)

        if may_be_511270:

            if more_types_in_union_511271:
                # Runtime conditional SSA (line 152)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'param_columns' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'param_columns', remove_member_provider_from_union(param_columns_511269, '__len__'))
            
            # Assigning a Tuple to a Name (line 153):
            
            # Assigning a Tuple to a Name (line 153):
            
            # Obtaining an instance of the builtin type 'tuple' (line 153)
            tuple_511272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 29), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 153)
            # Adding element type (line 153)
            # Getting the type of 'param_columns' (line 153)
            param_columns_511273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 29), 'param_columns')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 29), tuple_511272, param_columns_511273)
            
            # Assigning a type to the variable 'param_columns' (line 153)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'param_columns', tuple_511272)

            if more_types_in_union_511271:
                # SSA join for if statement (line 152)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Attribute (line 154):
        
        # Assigning a Call to a Attribute (line 154):
        
        # Call to tuple(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'param_columns' (line 154)
        param_columns_511275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 35), 'param_columns', False)
        # Processing the call keyword arguments (line 154)
        kwargs_511276 = {}
        # Getting the type of 'tuple' (line 154)
        tuple_511274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 29), 'tuple', False)
        # Calling tuple(args, kwargs) (line 154)
        tuple_call_result_511277 = invoke(stypy.reporting.localization.Localization(__file__, 154, 29), tuple_511274, *[param_columns_511275], **kwargs_511276)
        
        # Getting the type of 'self' (line 154)
        self_511278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'self')
        # Setting the type of the member 'param_columns' of a type (line 154)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), self_511278, 'param_columns', tuple_call_result_511277)
        
        # Type idiom detected: calculating its left and rigth part (line 155)
        # Getting the type of 'result_columns' (line 155)
        result_columns_511279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'result_columns')
        # Getting the type of 'None' (line 155)
        None_511280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 33), 'None')
        
        (may_be_511281, more_types_in_union_511282) = may_not_be_none(result_columns_511279, None_511280)

        if may_be_511281:

            if more_types_in_union_511282:
                # Runtime conditional SSA (line 155)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 156)
            str_511283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 43), 'str', '__len__')
            # Getting the type of 'result_columns' (line 156)
            result_columns_511284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 27), 'result_columns')
            
            (may_be_511285, more_types_in_union_511286) = may_not_provide_member(str_511283, result_columns_511284)

            if may_be_511285:

                if more_types_in_union_511286:
                    # Runtime conditional SSA (line 156)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'result_columns' (line 156)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'result_columns', remove_member_provider_from_union(result_columns_511284, '__len__'))
                
                # Assigning a Tuple to a Name (line 157):
                
                # Assigning a Tuple to a Name (line 157):
                
                # Obtaining an instance of the builtin type 'tuple' (line 157)
                tuple_511287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 34), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 157)
                # Adding element type (line 157)
                # Getting the type of 'result_columns' (line 157)
                result_columns_511288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 34), 'result_columns')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 34), tuple_511287, result_columns_511288)
                
                # Assigning a type to the variable 'result_columns' (line 157)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'result_columns', tuple_511287)

                if more_types_in_union_511286:
                    # SSA join for if statement (line 156)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Call to a Attribute (line 158):
            
            # Assigning a Call to a Attribute (line 158):
            
            # Call to tuple(...): (line 158)
            # Processing the call arguments (line 158)
            # Getting the type of 'result_columns' (line 158)
            result_columns_511290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 40), 'result_columns', False)
            # Processing the call keyword arguments (line 158)
            kwargs_511291 = {}
            # Getting the type of 'tuple' (line 158)
            tuple_511289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 34), 'tuple', False)
            # Calling tuple(args, kwargs) (line 158)
            tuple_call_result_511292 = invoke(stypy.reporting.localization.Localization(__file__, 158, 34), tuple_511289, *[result_columns_511290], **kwargs_511291)
            
            # Getting the type of 'self' (line 158)
            self_511293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'self')
            # Setting the type of the member 'result_columns' of a type (line 158)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 12), self_511293, 'result_columns', tuple_call_result_511292)
            
            # Type idiom detected: calculating its left and rigth part (line 159)
            # Getting the type of 'result_func' (line 159)
            result_func_511294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'result_func')
            # Getting the type of 'None' (line 159)
            None_511295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 34), 'None')
            
            (may_be_511296, more_types_in_union_511297) = may_not_be_none(result_func_511294, None_511295)

            if may_be_511296:

                if more_types_in_union_511297:
                    # Runtime conditional SSA (line 159)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to ValueError(...): (line 160)
                # Processing the call arguments (line 160)
                str_511299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 33), 'str', 'Only result_func or result_columns should be provided')
                # Processing the call keyword arguments (line 160)
                kwargs_511300 = {}
                # Getting the type of 'ValueError' (line 160)
                ValueError_511298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 22), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 160)
                ValueError_call_result_511301 = invoke(stypy.reporting.localization.Localization(__file__, 160, 22), ValueError_511298, *[str_511299], **kwargs_511300)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 160, 16), ValueError_call_result_511301, 'raise parameter', BaseException)

                if more_types_in_union_511297:
                    # SSA join for if statement (line 159)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_511282:
                # Runtime conditional SSA for else branch (line 155)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_511281) or more_types_in_union_511282):
            
            # Type idiom detected: calculating its left and rigth part (line 161)
            # Getting the type of 'result_func' (line 161)
            result_func_511302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 13), 'result_func')
            # Getting the type of 'None' (line 161)
            None_511303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 32), 'None')
            
            (may_be_511304, more_types_in_union_511305) = may_not_be_none(result_func_511302, None_511303)

            if may_be_511304:

                if more_types_in_union_511305:
                    # Runtime conditional SSA (line 161)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Name to a Attribute (line 162):
                
                # Assigning a Name to a Attribute (line 162):
                # Getting the type of 'None' (line 162)
                None_511306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 34), 'None')
                # Getting the type of 'self' (line 162)
                self_511307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'self')
                # Setting the type of the member 'result_columns' of a type (line 162)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), self_511307, 'result_columns', None_511306)

                if more_types_in_union_511305:
                    # Runtime conditional SSA for else branch (line 161)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_511304) or more_types_in_union_511305):
                
                # Call to ValueError(...): (line 164)
                # Processing the call arguments (line 164)
                str_511309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 29), 'str', 'Either result_func or result_columns should be provided')
                # Processing the call keyword arguments (line 164)
                kwargs_511310 = {}
                # Getting the type of 'ValueError' (line 164)
                ValueError_511308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 18), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 164)
                ValueError_call_result_511311 = invoke(stypy.reporting.localization.Localization(__file__, 164, 18), ValueError_511308, *[str_511309], **kwargs_511310)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 164, 12), ValueError_call_result_511311, 'raise parameter', BaseException)

                if (may_be_511304 and more_types_in_union_511305):
                    # SSA join for if statement (line 161)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_511281 and more_types_in_union_511282):
                # SSA join for if statement (line 155)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 165):
        
        # Assigning a Name to a Attribute (line 165):
        # Getting the type of 'result_func' (line 165)
        result_func_511312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 27), 'result_func')
        # Getting the type of 'self' (line 165)
        self_511313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'self')
        # Setting the type of the member 'result_func' of a type (line 165)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), self_511313, 'result_func', result_func_511312)
        
        # Assigning a Name to a Attribute (line 166):
        
        # Assigning a Name to a Attribute (line 166):
        # Getting the type of 'rtol' (line 166)
        rtol_511314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 20), 'rtol')
        # Getting the type of 'self' (line 166)
        self_511315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'self')
        # Setting the type of the member 'rtol' of a type (line 166)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), self_511315, 'rtol', rtol_511314)
        
        # Assigning a Name to a Attribute (line 167):
        
        # Assigning a Name to a Attribute (line 167):
        # Getting the type of 'atol' (line 167)
        atol_511316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 20), 'atol')
        # Getting the type of 'self' (line 167)
        self_511317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'self')
        # Setting the type of the member 'atol' of a type (line 167)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), self_511317, 'atol', atol_511316)
        
        # Type idiom detected: calculating its left and rigth part (line 168)
        str_511318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 37), 'str', '__len__')
        # Getting the type of 'param_filter' (line 168)
        param_filter_511319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 23), 'param_filter')
        
        (may_be_511320, more_types_in_union_511321) = may_not_provide_member(str_511318, param_filter_511319)

        if may_be_511320:

            if more_types_in_union_511321:
                # Runtime conditional SSA (line 168)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'param_filter' (line 168)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'param_filter', remove_member_provider_from_union(param_filter_511319, '__len__'))
            
            # Assigning a Tuple to a Name (line 169):
            
            # Assigning a Tuple to a Name (line 169):
            
            # Obtaining an instance of the builtin type 'tuple' (line 169)
            tuple_511322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 28), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 169)
            # Adding element type (line 169)
            # Getting the type of 'param_filter' (line 169)
            param_filter_511323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 28), 'param_filter')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 28), tuple_511322, param_filter_511323)
            
            # Assigning a type to the variable 'param_filter' (line 169)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'param_filter', tuple_511322)

            if more_types_in_union_511321:
                # SSA join for if statement (line 168)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 170):
        
        # Assigning a Name to a Attribute (line 170):
        # Getting the type of 'param_filter' (line 170)
        param_filter_511324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 28), 'param_filter')
        # Getting the type of 'self' (line 170)
        self_511325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'self')
        # Setting the type of the member 'param_filter' of a type (line 170)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), self_511325, 'param_filter', param_filter_511324)
        
        # Assigning a Name to a Attribute (line 171):
        
        # Assigning a Name to a Attribute (line 171):
        # Getting the type of 'knownfailure' (line 171)
        knownfailure_511326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 28), 'knownfailure')
        # Getting the type of 'self' (line 171)
        self_511327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'self')
        # Setting the type of the member 'knownfailure' of a type (line 171)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), self_511327, 'knownfailure', knownfailure_511326)
        
        # Assigning a Name to a Attribute (line 172):
        
        # Assigning a Name to a Attribute (line 172):
        # Getting the type of 'nan_ok' (line 172)
        nan_ok_511328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 22), 'nan_ok')
        # Getting the type of 'self' (line 172)
        self_511329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'self')
        # Setting the type of the member 'nan_ok' of a type (line 172)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), self_511329, 'nan_ok', nan_ok_511328)
        
        # Assigning a Name to a Attribute (line 173):
        
        # Assigning a Name to a Attribute (line 173):
        # Getting the type of 'vectorized' (line 173)
        vectorized_511330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 26), 'vectorized')
        # Getting the type of 'self' (line 173)
        self_511331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'self')
        # Setting the type of the member 'vectorized' of a type (line 173)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), self_511331, 'vectorized', vectorized_511330)
        
        # Assigning a Name to a Attribute (line 174):
        
        # Assigning a Name to a Attribute (line 174):
        # Getting the type of 'ignore_inf_sign' (line 174)
        ignore_inf_sign_511332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 31), 'ignore_inf_sign')
        # Getting the type of 'self' (line 174)
        self_511333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'self')
        # Setting the type of the member 'ignore_inf_sign' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), self_511333, 'ignore_inf_sign', ignore_inf_sign_511332)
        
        # Assigning a Name to a Attribute (line 175):
        
        # Assigning a Name to a Attribute (line 175):
        # Getting the type of 'distinguish_nan_and_inf' (line 175)
        distinguish_nan_and_inf_511334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 39), 'distinguish_nan_and_inf')
        # Getting the type of 'self' (line 175)
        self_511335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'self')
        # Setting the type of the member 'distinguish_nan_and_inf' of a type (line 175)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), self_511335, 'distinguish_nan_and_inf', distinguish_nan_and_inf_511334)
        
        
        # Getting the type of 'self' (line 176)
        self_511336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'self')
        # Obtaining the member 'distinguish_nan_and_inf' of a type (line 176)
        distinguish_nan_and_inf_511337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 15), self_511336, 'distinguish_nan_and_inf')
        # Applying the 'not' unary operator (line 176)
        result_not__511338 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 11), 'not', distinguish_nan_and_inf_511337)
        
        # Testing the type of an if condition (line 176)
        if_condition_511339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 8), result_not__511338)
        # Assigning a type to the variable 'if_condition_511339' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'if_condition_511339', if_condition_511339)
        # SSA begins for if statement (line 176)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 177):
        
        # Assigning a Name to a Attribute (line 177):
        # Getting the type of 'True' (line 177)
        True_511340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 35), 'True')
        # Getting the type of 'self' (line 177)
        self_511341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'self')
        # Setting the type of the member 'ignore_inf_sign' of a type (line 177)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 12), self_511341, 'ignore_inf_sign', True_511340)
        # SSA join for if statement (line 176)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_tolerances(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_tolerances'
        module_type_store = module_type_store.open_function_context('get_tolerances', 179, 4, False)
        # Assigning a type to the variable 'self' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FuncData.get_tolerances.__dict__.__setitem__('stypy_localization', localization)
        FuncData.get_tolerances.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FuncData.get_tolerances.__dict__.__setitem__('stypy_type_store', module_type_store)
        FuncData.get_tolerances.__dict__.__setitem__('stypy_function_name', 'FuncData.get_tolerances')
        FuncData.get_tolerances.__dict__.__setitem__('stypy_param_names_list', ['dtype'])
        FuncData.get_tolerances.__dict__.__setitem__('stypy_varargs_param_name', None)
        FuncData.get_tolerances.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FuncData.get_tolerances.__dict__.__setitem__('stypy_call_defaults', defaults)
        FuncData.get_tolerances.__dict__.__setitem__('stypy_call_varargs', varargs)
        FuncData.get_tolerances.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FuncData.get_tolerances.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FuncData.get_tolerances', ['dtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_tolerances', localization, ['dtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_tolerances(...)' code ##################

        
        
        
        # Call to issubdtype(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'dtype' (line 180)
        dtype_511344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 29), 'dtype', False)
        # Getting the type of 'np' (line 180)
        np_511345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 36), 'np', False)
        # Obtaining the member 'inexact' of a type (line 180)
        inexact_511346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 36), np_511345, 'inexact')
        # Processing the call keyword arguments (line 180)
        kwargs_511347 = {}
        # Getting the type of 'np' (line 180)
        np_511342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 15), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 180)
        issubdtype_511343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 15), np_511342, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 180)
        issubdtype_call_result_511348 = invoke(stypy.reporting.localization.Localization(__file__, 180, 15), issubdtype_511343, *[dtype_511344, inexact_511346], **kwargs_511347)
        
        # Applying the 'not' unary operator (line 180)
        result_not__511349 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 11), 'not', issubdtype_call_result_511348)
        
        # Testing the type of an if condition (line 180)
        if_condition_511350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 8), result_not__511349)
        # Assigning a type to the variable 'if_condition_511350' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'if_condition_511350', if_condition_511350)
        # SSA begins for if statement (line 180)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 181):
        
        # Assigning a Call to a Name (line 181):
        
        # Call to dtype(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'float' (line 181)
        float_511353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 29), 'float', False)
        # Processing the call keyword arguments (line 181)
        kwargs_511354 = {}
        # Getting the type of 'np' (line 181)
        np_511351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 20), 'np', False)
        # Obtaining the member 'dtype' of a type (line 181)
        dtype_511352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 20), np_511351, 'dtype')
        # Calling dtype(args, kwargs) (line 181)
        dtype_call_result_511355 = invoke(stypy.reporting.localization.Localization(__file__, 181, 20), dtype_511352, *[float_511353], **kwargs_511354)
        
        # Assigning a type to the variable 'dtype' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'dtype', dtype_call_result_511355)
        # SSA join for if statement (line 180)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 182):
        
        # Assigning a Call to a Name (line 182):
        
        # Call to finfo(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'dtype' (line 182)
        dtype_511358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 24), 'dtype', False)
        # Processing the call keyword arguments (line 182)
        kwargs_511359 = {}
        # Getting the type of 'np' (line 182)
        np_511356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'np', False)
        # Obtaining the member 'finfo' of a type (line 182)
        finfo_511357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 15), np_511356, 'finfo')
        # Calling finfo(args, kwargs) (line 182)
        finfo_call_result_511360 = invoke(stypy.reporting.localization.Localization(__file__, 182, 15), finfo_511357, *[dtype_511358], **kwargs_511359)
        
        # Assigning a type to the variable 'info' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'info', finfo_call_result_511360)
        
        # Assigning a Tuple to a Tuple (line 183):
        
        # Assigning a Attribute to a Name (line 183):
        # Getting the type of 'self' (line 183)
        self_511361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 21), 'self')
        # Obtaining the member 'rtol' of a type (line 183)
        rtol_511362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 21), self_511361, 'rtol')
        # Assigning a type to the variable 'tuple_assignment_511012' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'tuple_assignment_511012', rtol_511362)
        
        # Assigning a Attribute to a Name (line 183):
        # Getting the type of 'self' (line 183)
        self_511363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 32), 'self')
        # Obtaining the member 'atol' of a type (line 183)
        atol_511364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 32), self_511363, 'atol')
        # Assigning a type to the variable 'tuple_assignment_511013' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'tuple_assignment_511013', atol_511364)
        
        # Assigning a Name to a Name (line 183):
        # Getting the type of 'tuple_assignment_511012' (line 183)
        tuple_assignment_511012_511365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'tuple_assignment_511012')
        # Assigning a type to the variable 'rtol' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'rtol', tuple_assignment_511012_511365)
        
        # Assigning a Name to a Name (line 183):
        # Getting the type of 'tuple_assignment_511013' (line 183)
        tuple_assignment_511013_511366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'tuple_assignment_511013')
        # Assigning a type to the variable 'atol' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 14), 'atol', tuple_assignment_511013_511366)
        
        # Type idiom detected: calculating its left and rigth part (line 184)
        # Getting the type of 'rtol' (line 184)
        rtol_511367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 'rtol')
        # Getting the type of 'None' (line 184)
        None_511368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 19), 'None')
        
        (may_be_511369, more_types_in_union_511370) = may_be_none(rtol_511367, None_511368)

        if may_be_511369:

            if more_types_in_union_511370:
                # Runtime conditional SSA (line 184)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 185):
            
            # Assigning a BinOp to a Name (line 185):
            int_511371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 19), 'int')
            # Getting the type of 'info' (line 185)
            info_511372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 21), 'info')
            # Obtaining the member 'eps' of a type (line 185)
            eps_511373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 21), info_511372, 'eps')
            # Applying the binary operator '*' (line 185)
            result_mul_511374 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 19), '*', int_511371, eps_511373)
            
            # Assigning a type to the variable 'rtol' (line 185)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'rtol', result_mul_511374)

            if more_types_in_union_511370:
                # SSA join for if statement (line 184)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 186)
        # Getting the type of 'atol' (line 186)
        atol_511375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 11), 'atol')
        # Getting the type of 'None' (line 186)
        None_511376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 19), 'None')
        
        (may_be_511377, more_types_in_union_511378) = may_be_none(atol_511375, None_511376)

        if may_be_511377:

            if more_types_in_union_511378:
                # Runtime conditional SSA (line 186)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 187):
            
            # Assigning a BinOp to a Name (line 187):
            int_511379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 19), 'int')
            # Getting the type of 'info' (line 187)
            info_511380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 21), 'info')
            # Obtaining the member 'tiny' of a type (line 187)
            tiny_511381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 21), info_511380, 'tiny')
            # Applying the binary operator '*' (line 187)
            result_mul_511382 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 19), '*', int_511379, tiny_511381)
            
            # Assigning a type to the variable 'atol' (line 187)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'atol', result_mul_511382)

            if more_types_in_union_511378:
                # SSA join for if statement (line 186)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Obtaining an instance of the builtin type 'tuple' (line 188)
        tuple_511383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 188)
        # Adding element type (line 188)
        # Getting the type of 'rtol' (line 188)
        rtol_511384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'rtol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 15), tuple_511383, rtol_511384)
        # Adding element type (line 188)
        # Getting the type of 'atol' (line 188)
        atol_511385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 21), 'atol')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 188, 15), tuple_511383, atol_511385)
        
        # Assigning a type to the variable 'stypy_return_type' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'stypy_return_type', tuple_511383)
        
        # ################# End of 'get_tolerances(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_tolerances' in the type store
        # Getting the type of 'stypy_return_type' (line 179)
        stypy_return_type_511386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_511386)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_tolerances'
        return stypy_return_type_511386


    @norecursion
    def check(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 190)
        None_511387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 25), 'None')
        # Getting the type of 'None' (line 190)
        None_511388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 37), 'None')
        defaults = [None_511387, None_511388]
        # Create a new context for function 'check'
        module_type_store = module_type_store.open_function_context('check', 190, 4, False)
        # Assigning a type to the variable 'self' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FuncData.check.__dict__.__setitem__('stypy_localization', localization)
        FuncData.check.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FuncData.check.__dict__.__setitem__('stypy_type_store', module_type_store)
        FuncData.check.__dict__.__setitem__('stypy_function_name', 'FuncData.check')
        FuncData.check.__dict__.__setitem__('stypy_param_names_list', ['data', 'dtype'])
        FuncData.check.__dict__.__setitem__('stypy_varargs_param_name', None)
        FuncData.check.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FuncData.check.__dict__.__setitem__('stypy_call_defaults', defaults)
        FuncData.check.__dict__.__setitem__('stypy_call_varargs', varargs)
        FuncData.check.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FuncData.check.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FuncData.check', ['data', 'dtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check', localization, ['data', 'dtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check(...)' code ##################

        str_511389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 8), 'str', 'Check the special function against the data.')
        
        # Getting the type of 'self' (line 193)
        self_511390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 'self')
        # Obtaining the member 'knownfailure' of a type (line 193)
        knownfailure_511391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 11), self_511390, 'knownfailure')
        # Testing the type of an if condition (line 193)
        if_condition_511392 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 8), knownfailure_511391)
        # Assigning a type to the variable 'if_condition_511392' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'if_condition_511392', if_condition_511392)
        # SSA begins for if statement (line 193)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to xfail(...): (line 194)
        # Processing the call keyword arguments (line 194)
        # Getting the type of 'self' (line 194)
        self_511395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 32), 'self', False)
        # Obtaining the member 'knownfailure' of a type (line 194)
        knownfailure_511396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 32), self_511395, 'knownfailure')
        keyword_511397 = knownfailure_511396
        kwargs_511398 = {'reason': keyword_511397}
        # Getting the type of 'pytest' (line 194)
        pytest_511393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'pytest', False)
        # Obtaining the member 'xfail' of a type (line 194)
        xfail_511394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), pytest_511393, 'xfail')
        # Calling xfail(args, kwargs) (line 194)
        xfail_call_result_511399 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), xfail_511394, *[], **kwargs_511398)
        
        # SSA join for if statement (line 193)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 196)
        # Getting the type of 'data' (line 196)
        data_511400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 11), 'data')
        # Getting the type of 'None' (line 196)
        None_511401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 19), 'None')
        
        (may_be_511402, more_types_in_union_511403) = may_be_none(data_511400, None_511401)

        if may_be_511402:

            if more_types_in_union_511403:
                # Runtime conditional SSA (line 196)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 197):
            
            # Assigning a Attribute to a Name (line 197):
            # Getting the type of 'self' (line 197)
            self_511404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'self')
            # Obtaining the member 'data' of a type (line 197)
            data_511405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 19), self_511404, 'data')
            # Assigning a type to the variable 'data' (line 197)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'data', data_511405)

            if more_types_in_union_511403:
                # SSA join for if statement (line 196)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 199)
        # Getting the type of 'dtype' (line 199)
        dtype_511406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'dtype')
        # Getting the type of 'None' (line 199)
        None_511407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'None')
        
        (may_be_511408, more_types_in_union_511409) = may_be_none(dtype_511406, None_511407)

        if may_be_511408:

            if more_types_in_union_511409:
                # Runtime conditional SSA (line 199)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 200):
            
            # Assigning a Attribute to a Name (line 200):
            # Getting the type of 'data' (line 200)
            data_511410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'data')
            # Obtaining the member 'dtype' of a type (line 200)
            dtype_511411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 20), data_511410, 'dtype')
            # Assigning a type to the variable 'dtype' (line 200)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'dtype', dtype_511411)

            if more_types_in_union_511409:
                # Runtime conditional SSA for else branch (line 199)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_511408) or more_types_in_union_511409):
            
            # Assigning a Call to a Name (line 202):
            
            # Assigning a Call to a Name (line 202):
            
            # Call to astype(...): (line 202)
            # Processing the call arguments (line 202)
            # Getting the type of 'dtype' (line 202)
            dtype_511414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 31), 'dtype', False)
            # Processing the call keyword arguments (line 202)
            kwargs_511415 = {}
            # Getting the type of 'data' (line 202)
            data_511412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 19), 'data', False)
            # Obtaining the member 'astype' of a type (line 202)
            astype_511413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 19), data_511412, 'astype')
            # Calling astype(args, kwargs) (line 202)
            astype_call_result_511416 = invoke(stypy.reporting.localization.Localization(__file__, 202, 19), astype_511413, *[dtype_511414], **kwargs_511415)
            
            # Assigning a type to the variable 'data' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'data', astype_call_result_511416)

            if (may_be_511408 and more_types_in_union_511409):
                # SSA join for if statement (line 199)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Tuple (line 204):
        
        # Assigning a Subscript to a Name (line 204):
        
        # Obtaining the type of the subscript
        int_511417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 8), 'int')
        
        # Call to get_tolerances(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'dtype' (line 204)
        dtype_511420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 41), 'dtype', False)
        # Processing the call keyword arguments (line 204)
        kwargs_511421 = {}
        # Getting the type of 'self' (line 204)
        self_511418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'self', False)
        # Obtaining the member 'get_tolerances' of a type (line 204)
        get_tolerances_511419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 21), self_511418, 'get_tolerances')
        # Calling get_tolerances(args, kwargs) (line 204)
        get_tolerances_call_result_511422 = invoke(stypy.reporting.localization.Localization(__file__, 204, 21), get_tolerances_511419, *[dtype_511420], **kwargs_511421)
        
        # Obtaining the member '__getitem__' of a type (line 204)
        getitem___511423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), get_tolerances_call_result_511422, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 204)
        subscript_call_result_511424 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), getitem___511423, int_511417)
        
        # Assigning a type to the variable 'tuple_var_assignment_511014' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'tuple_var_assignment_511014', subscript_call_result_511424)
        
        # Assigning a Subscript to a Name (line 204):
        
        # Obtaining the type of the subscript
        int_511425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 8), 'int')
        
        # Call to get_tolerances(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'dtype' (line 204)
        dtype_511428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 41), 'dtype', False)
        # Processing the call keyword arguments (line 204)
        kwargs_511429 = {}
        # Getting the type of 'self' (line 204)
        self_511426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 21), 'self', False)
        # Obtaining the member 'get_tolerances' of a type (line 204)
        get_tolerances_511427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 21), self_511426, 'get_tolerances')
        # Calling get_tolerances(args, kwargs) (line 204)
        get_tolerances_call_result_511430 = invoke(stypy.reporting.localization.Localization(__file__, 204, 21), get_tolerances_511427, *[dtype_511428], **kwargs_511429)
        
        # Obtaining the member '__getitem__' of a type (line 204)
        getitem___511431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), get_tolerances_call_result_511430, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 204)
        subscript_call_result_511432 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), getitem___511431, int_511425)
        
        # Assigning a type to the variable 'tuple_var_assignment_511015' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'tuple_var_assignment_511015', subscript_call_result_511432)
        
        # Assigning a Name to a Name (line 204):
        # Getting the type of 'tuple_var_assignment_511014' (line 204)
        tuple_var_assignment_511014_511433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'tuple_var_assignment_511014')
        # Assigning a type to the variable 'rtol' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'rtol', tuple_var_assignment_511014_511433)
        
        # Assigning a Name to a Name (line 204):
        # Getting the type of 'tuple_var_assignment_511015' (line 204)
        tuple_var_assignment_511015_511434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'tuple_var_assignment_511015')
        # Assigning a type to the variable 'atol' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 14), 'atol', tuple_var_assignment_511015_511434)
        
        # Getting the type of 'self' (line 207)
        self_511435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 11), 'self')
        # Obtaining the member 'param_filter' of a type (line 207)
        param_filter_511436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 11), self_511435, 'param_filter')
        # Testing the type of an if condition (line 207)
        if_condition_511437 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 8), param_filter_511436)
        # Assigning a type to the variable 'if_condition_511437' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'if_condition_511437', if_condition_511437)
        # SSA begins for if statement (line 207)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 208):
        
        # Assigning a Call to a Name (line 208):
        
        # Call to ones(...): (line 208)
        # Processing the call arguments (line 208)
        
        # Obtaining an instance of the builtin type 'tuple' (line 208)
        tuple_511440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 208)
        # Adding element type (line 208)
        
        # Obtaining the type of the subscript
        int_511441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 45), 'int')
        # Getting the type of 'data' (line 208)
        data_511442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 34), 'data', False)
        # Obtaining the member 'shape' of a type (line 208)
        shape_511443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 34), data_511442, 'shape')
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___511444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 34), shape_511443, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_511445 = invoke(stypy.reporting.localization.Localization(__file__, 208, 34), getitem___511444, int_511441)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 34), tuple_511440, subscript_call_result_511445)
        
        # Getting the type of 'np' (line 208)
        np_511446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 51), 'np', False)
        # Obtaining the member 'bool_' of a type (line 208)
        bool__511447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 51), np_511446, 'bool_')
        # Processing the call keyword arguments (line 208)
        kwargs_511448 = {}
        # Getting the type of 'np' (line 208)
        np_511438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 25), 'np', False)
        # Obtaining the member 'ones' of a type (line 208)
        ones_511439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 25), np_511438, 'ones')
        # Calling ones(args, kwargs) (line 208)
        ones_call_result_511449 = invoke(stypy.reporting.localization.Localization(__file__, 208, 25), ones_511439, *[tuple_511440, bool__511447], **kwargs_511448)
        
        # Assigning a type to the variable 'param_mask' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'param_mask', ones_call_result_511449)
        
        
        # Call to zip(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'self' (line 209)
        self_511451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 33), 'self', False)
        # Obtaining the member 'param_columns' of a type (line 209)
        param_columns_511452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 33), self_511451, 'param_columns')
        # Getting the type of 'self' (line 209)
        self_511453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 53), 'self', False)
        # Obtaining the member 'param_filter' of a type (line 209)
        param_filter_511454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 53), self_511453, 'param_filter')
        # Processing the call keyword arguments (line 209)
        kwargs_511455 = {}
        # Getting the type of 'zip' (line 209)
        zip_511450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 29), 'zip', False)
        # Calling zip(args, kwargs) (line 209)
        zip_call_result_511456 = invoke(stypy.reporting.localization.Localization(__file__, 209, 29), zip_511450, *[param_columns_511452, param_filter_511454], **kwargs_511455)
        
        # Testing the type of a for loop iterable (line 209)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 209, 12), zip_call_result_511456)
        # Getting the type of the for loop variable (line 209)
        for_loop_var_511457 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 209, 12), zip_call_result_511456)
        # Assigning a type to the variable 'j' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 12), for_loop_var_511457))
        # Assigning a type to the variable 'filter' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'filter', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 12), for_loop_var_511457))
        # SSA begins for a for statement (line 209)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'filter' (line 210)
        filter_511458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 19), 'filter')
        # Testing the type of an if condition (line 210)
        if_condition_511459 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 16), filter_511458)
        # Assigning a type to the variable 'if_condition_511459' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'if_condition_511459', if_condition_511459)
        # SSA begins for if statement (line 210)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'param_mask' (line 211)
        param_mask_511460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'param_mask')
        
        # Call to list(...): (line 211)
        # Processing the call arguments (line 211)
        
        # Call to filter(...): (line 211)
        # Processing the call arguments (line 211)
        
        # Obtaining the type of the subscript
        slice_511463 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 211, 46), None, None, None)
        # Getting the type of 'j' (line 211)
        j_511464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 53), 'j', False)
        # Getting the type of 'data' (line 211)
        data_511465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 46), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 211)
        getitem___511466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 46), data_511465, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 211)
        subscript_call_result_511467 = invoke(stypy.reporting.localization.Localization(__file__, 211, 46), getitem___511466, (slice_511463, j_511464))
        
        # Processing the call keyword arguments (line 211)
        kwargs_511468 = {}
        # Getting the type of 'filter' (line 211)
        filter_511462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 39), 'filter', False)
        # Calling filter(args, kwargs) (line 211)
        filter_call_result_511469 = invoke(stypy.reporting.localization.Localization(__file__, 211, 39), filter_511462, *[subscript_call_result_511467], **kwargs_511468)
        
        # Processing the call keyword arguments (line 211)
        kwargs_511470 = {}
        # Getting the type of 'list' (line 211)
        list_511461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 34), 'list', False)
        # Calling list(args, kwargs) (line 211)
        list_call_result_511471 = invoke(stypy.reporting.localization.Localization(__file__, 211, 34), list_511461, *[filter_call_result_511469], **kwargs_511470)
        
        # Applying the binary operator '&=' (line 211)
        result_iand_511472 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 20), '&=', param_mask_511460, list_call_result_511471)
        # Assigning a type to the variable 'param_mask' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'param_mask', result_iand_511472)
        
        # SSA join for if statement (line 210)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 212):
        
        # Assigning a Subscript to a Name (line 212):
        
        # Obtaining the type of the subscript
        # Getting the type of 'param_mask' (line 212)
        param_mask_511473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 24), 'param_mask')
        # Getting the type of 'data' (line 212)
        data_511474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 19), 'data')
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___511475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 19), data_511474, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_511476 = invoke(stypy.reporting.localization.Localization(__file__, 212, 19), getitem___511475, param_mask_511473)
        
        # Assigning a type to the variable 'data' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'data', subscript_call_result_511476)
        # SSA join for if statement (line 207)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 215):
        
        # Assigning a List to a Name (line 215):
        
        # Obtaining an instance of the builtin type 'list' (line 215)
        list_511477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 215)
        
        # Assigning a type to the variable 'params' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'params', list_511477)
        
        # Getting the type of 'self' (line 216)
        self_511478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 17), 'self')
        # Obtaining the member 'param_columns' of a type (line 216)
        param_columns_511479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 17), self_511478, 'param_columns')
        # Testing the type of a for loop iterable (line 216)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 216, 8), param_columns_511479)
        # Getting the type of the for loop variable (line 216)
        for_loop_var_511480 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 216, 8), param_columns_511479)
        # Assigning a type to the variable 'j' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'j', for_loop_var_511480)
        # SSA begins for a for statement (line 216)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to iscomplexobj(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'j' (line 217)
        j_511483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 31), 'j', False)
        # Processing the call keyword arguments (line 217)
        kwargs_511484 = {}
        # Getting the type of 'np' (line 217)
        np_511481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'np', False)
        # Obtaining the member 'iscomplexobj' of a type (line 217)
        iscomplexobj_511482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 15), np_511481, 'iscomplexobj')
        # Calling iscomplexobj(args, kwargs) (line 217)
        iscomplexobj_call_result_511485 = invoke(stypy.reporting.localization.Localization(__file__, 217, 15), iscomplexobj_511482, *[j_511483], **kwargs_511484)
        
        # Testing the type of an if condition (line 217)
        if_condition_511486 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 12), iscomplexobj_call_result_511485)
        # Assigning a type to the variable 'if_condition_511486' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'if_condition_511486', if_condition_511486)
        # SSA begins for if statement (line 217)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 218):
        
        # Assigning a Call to a Name (line 218):
        
        # Call to int(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'j' (line 218)
        j_511488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 24), 'j', False)
        # Obtaining the member 'imag' of a type (line 218)
        imag_511489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 24), j_511488, 'imag')
        # Processing the call keyword arguments (line 218)
        kwargs_511490 = {}
        # Getting the type of 'int' (line 218)
        int_511487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 20), 'int', False)
        # Calling int(args, kwargs) (line 218)
        int_call_result_511491 = invoke(stypy.reporting.localization.Localization(__file__, 218, 20), int_511487, *[imag_511489], **kwargs_511490)
        
        # Assigning a type to the variable 'j' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'j', int_call_result_511491)
        
        # Call to append(...): (line 219)
        # Processing the call arguments (line 219)
        
        # Call to astype(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'complex' (line 219)
        complex_511500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 47), 'complex', False)
        # Processing the call keyword arguments (line 219)
        kwargs_511501 = {}
        
        # Obtaining the type of the subscript
        slice_511494 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 219, 30), None, None, None)
        # Getting the type of 'j' (line 219)
        j_511495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 37), 'j', False)
        # Getting the type of 'data' (line 219)
        data_511496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 30), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 219)
        getitem___511497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 30), data_511496, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 219)
        subscript_call_result_511498 = invoke(stypy.reporting.localization.Localization(__file__, 219, 30), getitem___511497, (slice_511494, j_511495))
        
        # Obtaining the member 'astype' of a type (line 219)
        astype_511499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 30), subscript_call_result_511498, 'astype')
        # Calling astype(args, kwargs) (line 219)
        astype_call_result_511502 = invoke(stypy.reporting.localization.Localization(__file__, 219, 30), astype_511499, *[complex_511500], **kwargs_511501)
        
        # Processing the call keyword arguments (line 219)
        kwargs_511503 = {}
        # Getting the type of 'params' (line 219)
        params_511492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'params', False)
        # Obtaining the member 'append' of a type (line 219)
        append_511493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 16), params_511492, 'append')
        # Calling append(args, kwargs) (line 219)
        append_call_result_511504 = invoke(stypy.reporting.localization.Localization(__file__, 219, 16), append_511493, *[astype_call_result_511502], **kwargs_511503)
        
        # SSA branch for the else part of an if statement (line 217)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 221)
        # Processing the call arguments (line 221)
        
        # Obtaining the type of the subscript
        slice_511507 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 221, 30), None, None, None)
        # Getting the type of 'j' (line 221)
        j_511508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 37), 'j', False)
        # Getting the type of 'data' (line 221)
        data_511509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 30), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___511510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 30), data_511509, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_511511 = invoke(stypy.reporting.localization.Localization(__file__, 221, 30), getitem___511510, (slice_511507, j_511508))
        
        # Processing the call keyword arguments (line 221)
        kwargs_511512 = {}
        # Getting the type of 'params' (line 221)
        params_511505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'params', False)
        # Obtaining the member 'append' of a type (line 221)
        append_511506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 16), params_511505, 'append')
        # Calling append(args, kwargs) (line 221)
        append_call_result_511513 = invoke(stypy.reporting.localization.Localization(__file__, 221, 16), append_511506, *[subscript_call_result_511511], **kwargs_511512)
        
        # SSA join for if statement (line 217)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def eval_func_at_params(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'None' (line 224)
            None_511514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 48), 'None')
            defaults = [None_511514]
            # Create a new context for function 'eval_func_at_params'
            module_type_store = module_type_store.open_function_context('eval_func_at_params', 224, 8, False)
            
            # Passed parameters checking function
            eval_func_at_params.stypy_localization = localization
            eval_func_at_params.stypy_type_of_self = None
            eval_func_at_params.stypy_type_store = module_type_store
            eval_func_at_params.stypy_function_name = 'eval_func_at_params'
            eval_func_at_params.stypy_param_names_list = ['func', 'skip_mask']
            eval_func_at_params.stypy_varargs_param_name = None
            eval_func_at_params.stypy_kwargs_param_name = None
            eval_func_at_params.stypy_call_defaults = defaults
            eval_func_at_params.stypy_call_varargs = varargs
            eval_func_at_params.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'eval_func_at_params', ['func', 'skip_mask'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'eval_func_at_params', localization, ['func', 'skip_mask'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'eval_func_at_params(...)' code ##################

            
            # Getting the type of 'self' (line 225)
            self_511515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 15), 'self')
            # Obtaining the member 'vectorized' of a type (line 225)
            vectorized_511516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 15), self_511515, 'vectorized')
            # Testing the type of an if condition (line 225)
            if_condition_511517 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 12), vectorized_511516)
            # Assigning a type to the variable 'if_condition_511517' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'if_condition_511517', if_condition_511517)
            # SSA begins for if statement (line 225)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 226):
            
            # Assigning a Call to a Name (line 226):
            
            # Call to func(...): (line 226)
            # Getting the type of 'params' (line 226)
            params_511519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 28), 'params', False)
            # Processing the call keyword arguments (line 226)
            kwargs_511520 = {}
            # Getting the type of 'func' (line 226)
            func_511518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 22), 'func', False)
            # Calling func(args, kwargs) (line 226)
            func_call_result_511521 = invoke(stypy.reporting.localization.Localization(__file__, 226, 22), func_511518, *[params_511519], **kwargs_511520)
            
            # Assigning a type to the variable 'got' (line 226)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'got', func_call_result_511521)
            # SSA branch for the else part of an if statement (line 225)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a List to a Name (line 228):
            
            # Assigning a List to a Name (line 228):
            
            # Obtaining an instance of the builtin type 'list' (line 228)
            list_511522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 22), 'list')
            # Adding type elements to the builtin type 'list' instance (line 228)
            
            # Assigning a type to the variable 'got' (line 228)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'got', list_511522)
            
            
            # Call to range(...): (line 229)
            # Processing the call arguments (line 229)
            
            # Call to len(...): (line 229)
            # Processing the call arguments (line 229)
            
            # Obtaining the type of the subscript
            int_511525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 42), 'int')
            # Getting the type of 'params' (line 229)
            params_511526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 35), 'params', False)
            # Obtaining the member '__getitem__' of a type (line 229)
            getitem___511527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 35), params_511526, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 229)
            subscript_call_result_511528 = invoke(stypy.reporting.localization.Localization(__file__, 229, 35), getitem___511527, int_511525)
            
            # Processing the call keyword arguments (line 229)
            kwargs_511529 = {}
            # Getting the type of 'len' (line 229)
            len_511524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 31), 'len', False)
            # Calling len(args, kwargs) (line 229)
            len_call_result_511530 = invoke(stypy.reporting.localization.Localization(__file__, 229, 31), len_511524, *[subscript_call_result_511528], **kwargs_511529)
            
            # Processing the call keyword arguments (line 229)
            kwargs_511531 = {}
            # Getting the type of 'range' (line 229)
            range_511523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 25), 'range', False)
            # Calling range(args, kwargs) (line 229)
            range_call_result_511532 = invoke(stypy.reporting.localization.Localization(__file__, 229, 25), range_511523, *[len_call_result_511530], **kwargs_511531)
            
            # Testing the type of a for loop iterable (line 229)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 229, 16), range_call_result_511532)
            # Getting the type of the for loop variable (line 229)
            for_loop_var_511533 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 229, 16), range_call_result_511532)
            # Assigning a type to the variable 'j' (line 229)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 16), 'j', for_loop_var_511533)
            # SSA begins for a for statement (line 229)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'skip_mask' (line 230)
            skip_mask_511534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 23), 'skip_mask')
            # Getting the type of 'None' (line 230)
            None_511535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 40), 'None')
            # Applying the binary operator 'isnot' (line 230)
            result_is_not_511536 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 23), 'isnot', skip_mask_511534, None_511535)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 230)
            j_511537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 59), 'j')
            # Getting the type of 'skip_mask' (line 230)
            skip_mask_511538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 49), 'skip_mask')
            # Obtaining the member '__getitem__' of a type (line 230)
            getitem___511539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 49), skip_mask_511538, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 230)
            subscript_call_result_511540 = invoke(stypy.reporting.localization.Localization(__file__, 230, 49), getitem___511539, j_511537)
            
            # Applying the binary operator 'and' (line 230)
            result_and_keyword_511541 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 23), 'and', result_is_not_511536, subscript_call_result_511540)
            
            # Testing the type of an if condition (line 230)
            if_condition_511542 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 20), result_and_keyword_511541)
            # Assigning a type to the variable 'if_condition_511542' (line 230)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 20), 'if_condition_511542', if_condition_511542)
            # SSA begins for if statement (line 230)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 231)
            # Processing the call arguments (line 231)
            # Getting the type of 'np' (line 231)
            np_511545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 35), 'np', False)
            # Obtaining the member 'nan' of a type (line 231)
            nan_511546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 35), np_511545, 'nan')
            # Processing the call keyword arguments (line 231)
            kwargs_511547 = {}
            # Getting the type of 'got' (line 231)
            got_511543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 24), 'got', False)
            # Obtaining the member 'append' of a type (line 231)
            append_511544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 24), got_511543, 'append')
            # Calling append(args, kwargs) (line 231)
            append_call_result_511548 = invoke(stypy.reporting.localization.Localization(__file__, 231, 24), append_511544, *[nan_511546], **kwargs_511547)
            
            # SSA join for if statement (line 230)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to append(...): (line 233)
            # Processing the call arguments (line 233)
            
            # Call to func(...): (line 233)
            
            # Call to tuple(...): (line 233)
            # Processing the call arguments (line 233)
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Call to range(...): (line 233)
            # Processing the call arguments (line 233)
            
            # Call to len(...): (line 233)
            # Processing the call arguments (line 233)
            # Getting the type of 'params' (line 233)
            params_511562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 76), 'params', False)
            # Processing the call keyword arguments (line 233)
            kwargs_511563 = {}
            # Getting the type of 'len' (line 233)
            len_511561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 72), 'len', False)
            # Calling len(args, kwargs) (line 233)
            len_call_result_511564 = invoke(stypy.reporting.localization.Localization(__file__, 233, 72), len_511561, *[params_511562], **kwargs_511563)
            
            # Processing the call keyword arguments (line 233)
            kwargs_511565 = {}
            # Getting the type of 'range' (line 233)
            range_511560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 66), 'range', False)
            # Calling range(args, kwargs) (line 233)
            range_call_result_511566 = invoke(stypy.reporting.localization.Localization(__file__, 233, 66), range_511560, *[len_call_result_511564], **kwargs_511565)
            
            comprehension_511567 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 44), range_call_result_511566)
            # Assigning a type to the variable 'i' (line 233)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 44), 'i', comprehension_511567)
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 233)
            j_511553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 54), 'j', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 233)
            i_511554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 51), 'i', False)
            # Getting the type of 'params' (line 233)
            params_511555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 44), 'params', False)
            # Obtaining the member '__getitem__' of a type (line 233)
            getitem___511556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 44), params_511555, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 233)
            subscript_call_result_511557 = invoke(stypy.reporting.localization.Localization(__file__, 233, 44), getitem___511556, i_511554)
            
            # Obtaining the member '__getitem__' of a type (line 233)
            getitem___511558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 44), subscript_call_result_511557, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 233)
            subscript_call_result_511559 = invoke(stypy.reporting.localization.Localization(__file__, 233, 44), getitem___511558, j_511553)
            
            list_511568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 44), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 44), list_511568, subscript_call_result_511559)
            # Processing the call keyword arguments (line 233)
            kwargs_511569 = {}
            # Getting the type of 'tuple' (line 233)
            tuple_511552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 37), 'tuple', False)
            # Calling tuple(args, kwargs) (line 233)
            tuple_call_result_511570 = invoke(stypy.reporting.localization.Localization(__file__, 233, 37), tuple_511552, *[list_511568], **kwargs_511569)
            
            # Processing the call keyword arguments (line 233)
            kwargs_511571 = {}
            # Getting the type of 'func' (line 233)
            func_511551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 31), 'func', False)
            # Calling func(args, kwargs) (line 233)
            func_call_result_511572 = invoke(stypy.reporting.localization.Localization(__file__, 233, 31), func_511551, *[tuple_call_result_511570], **kwargs_511571)
            
            # Processing the call keyword arguments (line 233)
            kwargs_511573 = {}
            # Getting the type of 'got' (line 233)
            got_511549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 20), 'got', False)
            # Obtaining the member 'append' of a type (line 233)
            append_511550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 20), got_511549, 'append')
            # Calling append(args, kwargs) (line 233)
            append_call_result_511574 = invoke(stypy.reporting.localization.Localization(__file__, 233, 20), append_511550, *[func_call_result_511572], **kwargs_511573)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 234):
            
            # Assigning a Call to a Name (line 234):
            
            # Call to asarray(...): (line 234)
            # Processing the call arguments (line 234)
            # Getting the type of 'got' (line 234)
            got_511577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 33), 'got', False)
            # Processing the call keyword arguments (line 234)
            kwargs_511578 = {}
            # Getting the type of 'np' (line 234)
            np_511575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 22), 'np', False)
            # Obtaining the member 'asarray' of a type (line 234)
            asarray_511576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 22), np_511575, 'asarray')
            # Calling asarray(args, kwargs) (line 234)
            asarray_call_result_511579 = invoke(stypy.reporting.localization.Localization(__file__, 234, 22), asarray_511576, *[got_511577], **kwargs_511578)
            
            # Assigning a type to the variable 'got' (line 234)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'got', asarray_call_result_511579)
            # SSA join for if statement (line 225)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Type idiom detected: calculating its left and rigth part (line 235)
            # Getting the type of 'tuple' (line 235)
            tuple_511580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 35), 'tuple')
            # Getting the type of 'got' (line 235)
            got_511581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 30), 'got')
            
            (may_be_511582, more_types_in_union_511583) = may_not_be_subtype(tuple_511580, got_511581)

            if may_be_511582:

                if more_types_in_union_511583:
                    # Runtime conditional SSA (line 235)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'got' (line 235)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'got', remove_subtype_from_union(got_511581, tuple))
                
                # Assigning a Tuple to a Name (line 236):
                
                # Assigning a Tuple to a Name (line 236):
                
                # Obtaining an instance of the builtin type 'tuple' (line 236)
                tuple_511584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 236)
                # Adding element type (line 236)
                # Getting the type of 'got' (line 236)
                got_511585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 23), 'got')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 23), tuple_511584, got_511585)
                
                # Assigning a type to the variable 'got' (line 236)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'got', tuple_511584)

                if more_types_in_union_511583:
                    # SSA join for if statement (line 235)
                    module_type_store = module_type_store.join_ssa_context()


            
            # Getting the type of 'got' (line 237)
            got_511586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 19), 'got')
            # Assigning a type to the variable 'stypy_return_type' (line 237)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'stypy_return_type', got_511586)
            
            # ################# End of 'eval_func_at_params(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'eval_func_at_params' in the type store
            # Getting the type of 'stypy_return_type' (line 224)
            stypy_return_type_511587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_511587)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'eval_func_at_params'
            return stypy_return_type_511587

        # Assigning a type to the variable 'eval_func_at_params' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'eval_func_at_params', eval_func_at_params)
        
        # Assigning a Call to a Name (line 240):
        
        # Assigning a Call to a Name (line 240):
        
        # Call to eval_func_at_params(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'self' (line 240)
        self_511589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 34), 'self', False)
        # Obtaining the member 'func' of a type (line 240)
        func_511590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 34), self_511589, 'func')
        # Processing the call keyword arguments (line 240)
        kwargs_511591 = {}
        # Getting the type of 'eval_func_at_params' (line 240)
        eval_func_at_params_511588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 14), 'eval_func_at_params', False)
        # Calling eval_func_at_params(args, kwargs) (line 240)
        eval_func_at_params_call_result_511592 = invoke(stypy.reporting.localization.Localization(__file__, 240, 14), eval_func_at_params_511588, *[func_511590], **kwargs_511591)
        
        # Assigning a type to the variable 'got' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'got', eval_func_at_params_call_result_511592)
        
        
        # Getting the type of 'self' (line 243)
        self_511593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 11), 'self')
        # Obtaining the member 'result_columns' of a type (line 243)
        result_columns_511594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 11), self_511593, 'result_columns')
        # Getting the type of 'None' (line 243)
        None_511595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 38), 'None')
        # Applying the binary operator 'isnot' (line 243)
        result_is_not_511596 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 11), 'isnot', result_columns_511594, None_511595)
        
        # Testing the type of an if condition (line 243)
        if_condition_511597 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 8), result_is_not_511596)
        # Assigning a type to the variable 'if_condition_511597' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'if_condition_511597', if_condition_511597)
        # SSA begins for if statement (line 243)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 245):
        
        # Assigning a Call to a Name (line 245):
        
        # Call to tuple(...): (line 245)
        # Processing the call arguments (line 245)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 245)
        self_511604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 53), 'self', False)
        # Obtaining the member 'result_columns' of a type (line 245)
        result_columns_511605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 53), self_511604, 'result_columns')
        comprehension_511606 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 28), result_columns_511605)
        # Assigning a type to the variable 'icol' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 28), 'icol', comprehension_511606)
        
        # Obtaining the type of the subscript
        slice_511599 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 245, 28), None, None, None)
        # Getting the type of 'icol' (line 245)
        icol_511600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 35), 'icol', False)
        # Getting the type of 'data' (line 245)
        data_511601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 28), 'data', False)
        # Obtaining the member '__getitem__' of a type (line 245)
        getitem___511602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 28), data_511601, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 245)
        subscript_call_result_511603 = invoke(stypy.reporting.localization.Localization(__file__, 245, 28), getitem___511602, (slice_511599, icol_511600))
        
        list_511607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 28), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 28), list_511607, subscript_call_result_511603)
        # Processing the call keyword arguments (line 245)
        kwargs_511608 = {}
        # Getting the type of 'tuple' (line 245)
        tuple_511598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 21), 'tuple', False)
        # Calling tuple(args, kwargs) (line 245)
        tuple_call_result_511609 = invoke(stypy.reporting.localization.Localization(__file__, 245, 21), tuple_511598, *[list_511607], **kwargs_511608)
        
        # Assigning a type to the variable 'wanted' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'wanted', tuple_call_result_511609)
        # SSA branch for the else part of an if statement (line 243)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 248):
        
        # Assigning a Name to a Name (line 248):
        # Getting the type of 'None' (line 248)
        None_511610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 24), 'None')
        # Assigning a type to the variable 'skip_mask' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'skip_mask', None_511610)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 249)
        self_511611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 15), 'self')
        # Obtaining the member 'nan_ok' of a type (line 249)
        nan_ok_511612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 15), self_511611, 'nan_ok')
        
        
        # Call to len(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'got' (line 249)
        got_511614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 35), 'got', False)
        # Processing the call keyword arguments (line 249)
        kwargs_511615 = {}
        # Getting the type of 'len' (line 249)
        len_511613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 31), 'len', False)
        # Calling len(args, kwargs) (line 249)
        len_call_result_511616 = invoke(stypy.reporting.localization.Localization(__file__, 249, 31), len_511613, *[got_511614], **kwargs_511615)
        
        int_511617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 43), 'int')
        # Applying the binary operator '==' (line 249)
        result_eq_511618 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 31), '==', len_call_result_511616, int_511617)
        
        # Applying the binary operator 'and' (line 249)
        result_and_keyword_511619 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 15), 'and', nan_ok_511612, result_eq_511618)
        
        # Testing the type of an if condition (line 249)
        if_condition_511620 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 12), result_and_keyword_511619)
        # Assigning a type to the variable 'if_condition_511620' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'if_condition_511620', if_condition_511620)
        # SSA begins for if statement (line 249)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 251):
        
        # Assigning a Call to a Name (line 251):
        
        # Call to isnan(...): (line 251)
        # Processing the call arguments (line 251)
        
        # Obtaining the type of the subscript
        int_511623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 41), 'int')
        # Getting the type of 'got' (line 251)
        got_511624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 37), 'got', False)
        # Obtaining the member '__getitem__' of a type (line 251)
        getitem___511625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 37), got_511624, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 251)
        subscript_call_result_511626 = invoke(stypy.reporting.localization.Localization(__file__, 251, 37), getitem___511625, int_511623)
        
        # Processing the call keyword arguments (line 251)
        kwargs_511627 = {}
        # Getting the type of 'np' (line 251)
        np_511621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 28), 'np', False)
        # Obtaining the member 'isnan' of a type (line 251)
        isnan_511622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 28), np_511621, 'isnan')
        # Calling isnan(args, kwargs) (line 251)
        isnan_call_result_511628 = invoke(stypy.reporting.localization.Localization(__file__, 251, 28), isnan_511622, *[subscript_call_result_511626], **kwargs_511627)
        
        # Assigning a type to the variable 'skip_mask' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'skip_mask', isnan_call_result_511628)
        # SSA join for if statement (line 249)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 252):
        
        # Assigning a Call to a Name (line 252):
        
        # Call to eval_func_at_params(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'self' (line 252)
        self_511630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 41), 'self', False)
        # Obtaining the member 'result_func' of a type (line 252)
        result_func_511631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 41), self_511630, 'result_func')
        # Processing the call keyword arguments (line 252)
        # Getting the type of 'skip_mask' (line 252)
        skip_mask_511632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 69), 'skip_mask', False)
        keyword_511633 = skip_mask_511632
        kwargs_511634 = {'skip_mask': keyword_511633}
        # Getting the type of 'eval_func_at_params' (line 252)
        eval_func_at_params_511629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 21), 'eval_func_at_params', False)
        # Calling eval_func_at_params(args, kwargs) (line 252)
        eval_func_at_params_call_result_511635 = invoke(stypy.reporting.localization.Localization(__file__, 252, 21), eval_func_at_params_511629, *[result_func_511631], **kwargs_511634)
        
        # Assigning a type to the variable 'wanted' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'wanted', eval_func_at_params_call_result_511635)
        # SSA join for if statement (line 243)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 255)
        # Processing the call arguments (line 255)
        
        
        # Call to len(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'got' (line 255)
        got_511638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'got', False)
        # Processing the call keyword arguments (line 255)
        kwargs_511639 = {}
        # Getting the type of 'len' (line 255)
        len_511637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'len', False)
        # Calling len(args, kwargs) (line 255)
        len_call_result_511640 = invoke(stypy.reporting.localization.Localization(__file__, 255, 16), len_511637, *[got_511638], **kwargs_511639)
        
        
        # Call to len(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'wanted' (line 255)
        wanted_511642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 32), 'wanted', False)
        # Processing the call keyword arguments (line 255)
        kwargs_511643 = {}
        # Getting the type of 'len' (line 255)
        len_511641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 28), 'len', False)
        # Calling len(args, kwargs) (line 255)
        len_call_result_511644 = invoke(stypy.reporting.localization.Localization(__file__, 255, 28), len_511641, *[wanted_511642], **kwargs_511643)
        
        # Applying the binary operator '==' (line 255)
        result_eq_511645 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 16), '==', len_call_result_511640, len_call_result_511644)
        
        # Processing the call keyword arguments (line 255)
        kwargs_511646 = {}
        # Getting the type of 'assert_' (line 255)
        assert__511636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 255)
        assert__call_result_511647 = invoke(stypy.reporting.localization.Localization(__file__, 255, 8), assert__511636, *[result_eq_511645], **kwargs_511646)
        
        
        
        # Call to enumerate(...): (line 257)
        # Processing the call arguments (line 257)
        
        # Call to zip(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'got' (line 257)
        got_511650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 48), 'got', False)
        # Getting the type of 'wanted' (line 257)
        wanted_511651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 53), 'wanted', False)
        # Processing the call keyword arguments (line 257)
        kwargs_511652 = {}
        # Getting the type of 'zip' (line 257)
        zip_511649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 44), 'zip', False)
        # Calling zip(args, kwargs) (line 257)
        zip_call_result_511653 = invoke(stypy.reporting.localization.Localization(__file__, 257, 44), zip_511649, *[got_511650, wanted_511651], **kwargs_511652)
        
        # Processing the call keyword arguments (line 257)
        kwargs_511654 = {}
        # Getting the type of 'enumerate' (line 257)
        enumerate_511648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 34), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 257)
        enumerate_call_result_511655 = invoke(stypy.reporting.localization.Localization(__file__, 257, 34), enumerate_511648, *[zip_call_result_511653], **kwargs_511654)
        
        # Testing the type of a for loop iterable (line 257)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 257, 8), enumerate_call_result_511655)
        # Getting the type of the for loop variable (line 257)
        for_loop_var_511656 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 257, 8), enumerate_call_result_511655)
        # Assigning a type to the variable 'output_num' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'output_num', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), for_loop_var_511656))
        # Assigning a type to the variable 'x' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'x', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), for_loop_var_511656))
        # Assigning a type to the variable 'y' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'y', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 8), for_loop_var_511656))
        # SSA begins for a for statement (line 257)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Call to issubdtype(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'x' (line 258)
        x_511659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 29), 'x', False)
        # Obtaining the member 'dtype' of a type (line 258)
        dtype_511660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 29), x_511659, 'dtype')
        # Getting the type of 'np' (line 258)
        np_511661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 38), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 258)
        complexfloating_511662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 38), np_511661, 'complexfloating')
        # Processing the call keyword arguments (line 258)
        kwargs_511663 = {}
        # Getting the type of 'np' (line 258)
        np_511657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 258)
        issubdtype_511658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 15), np_511657, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 258)
        issubdtype_call_result_511664 = invoke(stypy.reporting.localization.Localization(__file__, 258, 15), issubdtype_511658, *[dtype_511660, complexfloating_511662], **kwargs_511663)
        
        # Getting the type of 'self' (line 258)
        self_511665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 61), 'self')
        # Obtaining the member 'ignore_inf_sign' of a type (line 258)
        ignore_inf_sign_511666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 61), self_511665, 'ignore_inf_sign')
        # Applying the binary operator 'or' (line 258)
        result_or_keyword_511667 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 15), 'or', issubdtype_call_result_511664, ignore_inf_sign_511666)
        
        # Testing the type of an if condition (line 258)
        if_condition_511668 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 12), result_or_keyword_511667)
        # Assigning a type to the variable 'if_condition_511668' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'if_condition_511668', if_condition_511668)
        # SSA begins for if statement (line 258)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 259):
        
        # Assigning a Call to a Name (line 259):
        
        # Call to isinf(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'x' (line 259)
        x_511671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 34), 'x', False)
        # Processing the call keyword arguments (line 259)
        kwargs_511672 = {}
        # Getting the type of 'np' (line 259)
        np_511669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 25), 'np', False)
        # Obtaining the member 'isinf' of a type (line 259)
        isinf_511670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 25), np_511669, 'isinf')
        # Calling isinf(args, kwargs) (line 259)
        isinf_call_result_511673 = invoke(stypy.reporting.localization.Localization(__file__, 259, 25), isinf_511670, *[x_511671], **kwargs_511672)
        
        # Assigning a type to the variable 'pinf_x' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'pinf_x', isinf_call_result_511673)
        
        # Assigning a Call to a Name (line 260):
        
        # Assigning a Call to a Name (line 260):
        
        # Call to isinf(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'y' (line 260)
        y_511676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 34), 'y', False)
        # Processing the call keyword arguments (line 260)
        kwargs_511677 = {}
        # Getting the type of 'np' (line 260)
        np_511674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 25), 'np', False)
        # Obtaining the member 'isinf' of a type (line 260)
        isinf_511675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 25), np_511674, 'isinf')
        # Calling isinf(args, kwargs) (line 260)
        isinf_call_result_511678 = invoke(stypy.reporting.localization.Localization(__file__, 260, 25), isinf_511675, *[y_511676], **kwargs_511677)
        
        # Assigning a type to the variable 'pinf_y' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'pinf_y', isinf_call_result_511678)
        
        # Assigning a Call to a Name (line 261):
        
        # Assigning a Call to a Name (line 261):
        
        # Call to isinf(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'x' (line 261)
        x_511681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 34), 'x', False)
        # Processing the call keyword arguments (line 261)
        kwargs_511682 = {}
        # Getting the type of 'np' (line 261)
        np_511679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 25), 'np', False)
        # Obtaining the member 'isinf' of a type (line 261)
        isinf_511680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 25), np_511679, 'isinf')
        # Calling isinf(args, kwargs) (line 261)
        isinf_call_result_511683 = invoke(stypy.reporting.localization.Localization(__file__, 261, 25), isinf_511680, *[x_511681], **kwargs_511682)
        
        # Assigning a type to the variable 'minf_x' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 16), 'minf_x', isinf_call_result_511683)
        
        # Assigning a Call to a Name (line 262):
        
        # Assigning a Call to a Name (line 262):
        
        # Call to isinf(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'y' (line 262)
        y_511686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 34), 'y', False)
        # Processing the call keyword arguments (line 262)
        kwargs_511687 = {}
        # Getting the type of 'np' (line 262)
        np_511684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 25), 'np', False)
        # Obtaining the member 'isinf' of a type (line 262)
        isinf_511685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 25), np_511684, 'isinf')
        # Calling isinf(args, kwargs) (line 262)
        isinf_call_result_511688 = invoke(stypy.reporting.localization.Localization(__file__, 262, 25), isinf_511685, *[y_511686], **kwargs_511687)
        
        # Assigning a type to the variable 'minf_y' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 16), 'minf_y', isinf_call_result_511688)
        # SSA branch for the else part of an if statement (line 258)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 264):
        
        # Assigning a Call to a Name (line 264):
        
        # Call to isposinf(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'x' (line 264)
        x_511691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 37), 'x', False)
        # Processing the call keyword arguments (line 264)
        kwargs_511692 = {}
        # Getting the type of 'np' (line 264)
        np_511689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 25), 'np', False)
        # Obtaining the member 'isposinf' of a type (line 264)
        isposinf_511690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 25), np_511689, 'isposinf')
        # Calling isposinf(args, kwargs) (line 264)
        isposinf_call_result_511693 = invoke(stypy.reporting.localization.Localization(__file__, 264, 25), isposinf_511690, *[x_511691], **kwargs_511692)
        
        # Assigning a type to the variable 'pinf_x' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 'pinf_x', isposinf_call_result_511693)
        
        # Assigning a Call to a Name (line 265):
        
        # Assigning a Call to a Name (line 265):
        
        # Call to isposinf(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'y' (line 265)
        y_511696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 37), 'y', False)
        # Processing the call keyword arguments (line 265)
        kwargs_511697 = {}
        # Getting the type of 'np' (line 265)
        np_511694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 25), 'np', False)
        # Obtaining the member 'isposinf' of a type (line 265)
        isposinf_511695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 25), np_511694, 'isposinf')
        # Calling isposinf(args, kwargs) (line 265)
        isposinf_call_result_511698 = invoke(stypy.reporting.localization.Localization(__file__, 265, 25), isposinf_511695, *[y_511696], **kwargs_511697)
        
        # Assigning a type to the variable 'pinf_y' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 16), 'pinf_y', isposinf_call_result_511698)
        
        # Assigning a Call to a Name (line 266):
        
        # Assigning a Call to a Name (line 266):
        
        # Call to isneginf(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'x' (line 266)
        x_511701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 37), 'x', False)
        # Processing the call keyword arguments (line 266)
        kwargs_511702 = {}
        # Getting the type of 'np' (line 266)
        np_511699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 25), 'np', False)
        # Obtaining the member 'isneginf' of a type (line 266)
        isneginf_511700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 25), np_511699, 'isneginf')
        # Calling isneginf(args, kwargs) (line 266)
        isneginf_call_result_511703 = invoke(stypy.reporting.localization.Localization(__file__, 266, 25), isneginf_511700, *[x_511701], **kwargs_511702)
        
        # Assigning a type to the variable 'minf_x' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'minf_x', isneginf_call_result_511703)
        
        # Assigning a Call to a Name (line 267):
        
        # Assigning a Call to a Name (line 267):
        
        # Call to isneginf(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'y' (line 267)
        y_511706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 37), 'y', False)
        # Processing the call keyword arguments (line 267)
        kwargs_511707 = {}
        # Getting the type of 'np' (line 267)
        np_511704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 25), 'np', False)
        # Obtaining the member 'isneginf' of a type (line 267)
        isneginf_511705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 25), np_511704, 'isneginf')
        # Calling isneginf(args, kwargs) (line 267)
        isneginf_call_result_511708 = invoke(stypy.reporting.localization.Localization(__file__, 267, 25), isneginf_511705, *[y_511706], **kwargs_511707)
        
        # Assigning a type to the variable 'minf_y' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 16), 'minf_y', isneginf_call_result_511708)
        # SSA join for if statement (line 258)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 268):
        
        # Assigning a Call to a Name (line 268):
        
        # Call to isnan(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'x' (line 268)
        x_511711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 29), 'x', False)
        # Processing the call keyword arguments (line 268)
        kwargs_511712 = {}
        # Getting the type of 'np' (line 268)
        np_511709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 20), 'np', False)
        # Obtaining the member 'isnan' of a type (line 268)
        isnan_511710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 20), np_511709, 'isnan')
        # Calling isnan(args, kwargs) (line 268)
        isnan_call_result_511713 = invoke(stypy.reporting.localization.Localization(__file__, 268, 20), isnan_511710, *[x_511711], **kwargs_511712)
        
        # Assigning a type to the variable 'nan_x' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'nan_x', isnan_call_result_511713)
        
        # Assigning a Call to a Name (line 269):
        
        # Assigning a Call to a Name (line 269):
        
        # Call to isnan(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'y' (line 269)
        y_511716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 29), 'y', False)
        # Processing the call keyword arguments (line 269)
        kwargs_511717 = {}
        # Getting the type of 'np' (line 269)
        np_511714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 20), 'np', False)
        # Obtaining the member 'isnan' of a type (line 269)
        isnan_511715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 20), np_511714, 'isnan')
        # Calling isnan(args, kwargs) (line 269)
        isnan_call_result_511718 = invoke(stypy.reporting.localization.Localization(__file__, 269, 20), isnan_511715, *[y_511716], **kwargs_511717)
        
        # Assigning a type to the variable 'nan_y' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'nan_y', isnan_call_result_511718)
        
        # Assigning a Call to a Name (line 271):
        
        # Assigning a Call to a Name (line 271):
        
        # Call to seterr(...): (line 271)
        # Processing the call keyword arguments (line 271)
        str_511721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 35), 'str', 'ignore')
        keyword_511722 = str_511721
        kwargs_511723 = {'all': keyword_511722}
        # Getting the type of 'np' (line 271)
        np_511719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 21), 'np', False)
        # Obtaining the member 'seterr' of a type (line 271)
        seterr_511720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 21), np_511719, 'seterr')
        # Calling seterr(args, kwargs) (line 271)
        seterr_call_result_511724 = invoke(stypy.reporting.localization.Localization(__file__, 271, 21), seterr_511720, *[], **kwargs_511723)
        
        # Assigning a type to the variable 'olderr' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'olderr', seterr_call_result_511724)
        
        # Try-finally block (line 272)
        
        # Assigning a Call to a Name (line 273):
        
        # Assigning a Call to a Name (line 273):
        
        # Call to absolute(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'y' (line 273)
        y_511727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 36), 'y', False)
        # Processing the call keyword arguments (line 273)
        kwargs_511728 = {}
        # Getting the type of 'np' (line 273)
        np_511725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 24), 'np', False)
        # Obtaining the member 'absolute' of a type (line 273)
        absolute_511726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 24), np_511725, 'absolute')
        # Calling absolute(args, kwargs) (line 273)
        absolute_call_result_511729 = invoke(stypy.reporting.localization.Localization(__file__, 273, 24), absolute_511726, *[y_511727], **kwargs_511728)
        
        # Assigning a type to the variable 'abs_y' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 16), 'abs_y', absolute_call_result_511729)
        
        # Assigning a Num to a Subscript (line 274):
        
        # Assigning a Num to a Subscript (line 274):
        int_511730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 45), 'int')
        # Getting the type of 'abs_y' (line 274)
        abs_y_511731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'abs_y')
        
        
        # Call to isfinite(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'abs_y' (line 274)
        abs_y_511734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 35), 'abs_y', False)
        # Processing the call keyword arguments (line 274)
        kwargs_511735 = {}
        # Getting the type of 'np' (line 274)
        np_511732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 23), 'np', False)
        # Obtaining the member 'isfinite' of a type (line 274)
        isfinite_511733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 23), np_511732, 'isfinite')
        # Calling isfinite(args, kwargs) (line 274)
        isfinite_call_result_511736 = invoke(stypy.reporting.localization.Localization(__file__, 274, 23), isfinite_511733, *[abs_y_511734], **kwargs_511735)
        
        # Applying the '~' unary operator (line 274)
        result_inv_511737 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 22), '~', isfinite_call_result_511736)
        
        # Storing an element on a container (line 274)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 16), abs_y_511731, (result_inv_511737, int_511730))
        
        # Assigning a Call to a Name (line 275):
        
        # Assigning a Call to a Name (line 275):
        
        # Call to absolute(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'x' (line 275)
        x_511740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 35), 'x', False)
        # Getting the type of 'y' (line 275)
        y_511741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 39), 'y', False)
        # Applying the binary operator '-' (line 275)
        result_sub_511742 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 35), '-', x_511740, y_511741)
        
        # Processing the call keyword arguments (line 275)
        kwargs_511743 = {}
        # Getting the type of 'np' (line 275)
        np_511738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 23), 'np', False)
        # Obtaining the member 'absolute' of a type (line 275)
        absolute_511739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 23), np_511738, 'absolute')
        # Calling absolute(args, kwargs) (line 275)
        absolute_call_result_511744 = invoke(stypy.reporting.localization.Localization(__file__, 275, 23), absolute_511739, *[result_sub_511742], **kwargs_511743)
        
        # Assigning a type to the variable 'diff' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'diff', absolute_call_result_511744)
        
        # Assigning a Num to a Subscript (line 276):
        
        # Assigning a Num to a Subscript (line 276):
        int_511745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 43), 'int')
        # Getting the type of 'diff' (line 276)
        diff_511746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'diff')
        
        
        # Call to isfinite(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 'diff' (line 276)
        diff_511749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 34), 'diff', False)
        # Processing the call keyword arguments (line 276)
        kwargs_511750 = {}
        # Getting the type of 'np' (line 276)
        np_511747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 22), 'np', False)
        # Obtaining the member 'isfinite' of a type (line 276)
        isfinite_511748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 22), np_511747, 'isfinite')
        # Calling isfinite(args, kwargs) (line 276)
        isfinite_call_result_511751 = invoke(stypy.reporting.localization.Localization(__file__, 276, 22), isfinite_511748, *[diff_511749], **kwargs_511750)
        
        # Applying the '~' unary operator (line 276)
        result_inv_511752 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 21), '~', isfinite_call_result_511751)
        
        # Storing an element on a container (line 276)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 16), diff_511746, (result_inv_511752, int_511745))
        
        # Assigning a BinOp to a Name (line 278):
        
        # Assigning a BinOp to a Name (line 278):
        # Getting the type of 'diff' (line 278)
        diff_511753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 24), 'diff')
        
        # Call to absolute(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'y' (line 278)
        y_511756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 43), 'y', False)
        # Processing the call keyword arguments (line 278)
        kwargs_511757 = {}
        # Getting the type of 'np' (line 278)
        np_511754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 31), 'np', False)
        # Obtaining the member 'absolute' of a type (line 278)
        absolute_511755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 31), np_511754, 'absolute')
        # Calling absolute(args, kwargs) (line 278)
        absolute_call_result_511758 = invoke(stypy.reporting.localization.Localization(__file__, 278, 31), absolute_511755, *[y_511756], **kwargs_511757)
        
        # Applying the binary operator 'div' (line 278)
        result_div_511759 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 24), 'div', diff_511753, absolute_call_result_511758)
        
        # Assigning a type to the variable 'rdiff' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'rdiff', result_div_511759)
        
        # Assigning a Num to a Subscript (line 279):
        
        # Assigning a Num to a Subscript (line 279):
        int_511760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 45), 'int')
        # Getting the type of 'rdiff' (line 279)
        rdiff_511761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 16), 'rdiff')
        
        
        # Call to isfinite(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'rdiff' (line 279)
        rdiff_511764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 35), 'rdiff', False)
        # Processing the call keyword arguments (line 279)
        kwargs_511765 = {}
        # Getting the type of 'np' (line 279)
        np_511762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 23), 'np', False)
        # Obtaining the member 'isfinite' of a type (line 279)
        isfinite_511763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 23), np_511762, 'isfinite')
        # Calling isfinite(args, kwargs) (line 279)
        isfinite_call_result_511766 = invoke(stypy.reporting.localization.Localization(__file__, 279, 23), isfinite_511763, *[rdiff_511764], **kwargs_511765)
        
        # Applying the '~' unary operator (line 279)
        result_inv_511767 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 22), '~', isfinite_call_result_511766)
        
        # Storing an element on a container (line 279)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 16), rdiff_511761, (result_inv_511767, int_511760))
        
        # finally branch of the try-finally block (line 272)
        
        # Call to seterr(...): (line 281)
        # Processing the call keyword arguments (line 281)
        # Getting the type of 'olderr' (line 281)
        olderr_511770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 28), 'olderr', False)
        kwargs_511771 = {'olderr_511770': olderr_511770}
        # Getting the type of 'np' (line 281)
        np_511768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'np', False)
        # Obtaining the member 'seterr' of a type (line 281)
        seterr_511769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 16), np_511768, 'seterr')
        # Calling seterr(args, kwargs) (line 281)
        seterr_call_result_511772 = invoke(stypy.reporting.localization.Localization(__file__, 281, 16), seterr_511769, *[], **kwargs_511771)
        
        
        
        # Assigning a Compare to a Name (line 283):
        
        # Assigning a Compare to a Name (line 283):
        
        # Getting the type of 'diff' (line 283)
        diff_511773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 24), 'diff')
        # Getting the type of 'atol' (line 283)
        atol_511774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 32), 'atol')
        # Getting the type of 'rtol' (line 283)
        rtol_511775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 39), 'rtol')
        # Getting the type of 'abs_y' (line 283)
        abs_y_511776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 44), 'abs_y')
        # Applying the binary operator '*' (line 283)
        result_mul_511777 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 39), '*', rtol_511775, abs_y_511776)
        
        # Applying the binary operator '+' (line 283)
        result_add_511778 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 32), '+', atol_511774, result_mul_511777)
        
        # Applying the binary operator '<=' (line 283)
        result_le_511779 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 24), '<=', diff_511773, result_add_511778)
        
        # Assigning a type to the variable 'tol_mask' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'tol_mask', result_le_511779)
        
        # Assigning a Compare to a Name (line 284):
        
        # Assigning a Compare to a Name (line 284):
        
        # Getting the type of 'pinf_x' (line 284)
        pinf_x_511780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 25), 'pinf_x')
        # Getting the type of 'pinf_y' (line 284)
        pinf_y_511781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 35), 'pinf_y')
        # Applying the binary operator '==' (line 284)
        result_eq_511782 = python_operator(stypy.reporting.localization.Localization(__file__, 284, 25), '==', pinf_x_511780, pinf_y_511781)
        
        # Assigning a type to the variable 'pinf_mask' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'pinf_mask', result_eq_511782)
        
        # Assigning a Compare to a Name (line 285):
        
        # Assigning a Compare to a Name (line 285):
        
        # Getting the type of 'minf_x' (line 285)
        minf_x_511783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 25), 'minf_x')
        # Getting the type of 'minf_y' (line 285)
        minf_y_511784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 35), 'minf_y')
        # Applying the binary operator '==' (line 285)
        result_eq_511785 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 25), '==', minf_x_511783, minf_y_511784)
        
        # Assigning a type to the variable 'minf_mask' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'minf_mask', result_eq_511785)
        
        # Assigning a Compare to a Name (line 287):
        
        # Assigning a Compare to a Name (line 287):
        
        # Getting the type of 'nan_x' (line 287)
        nan_x_511786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 24), 'nan_x')
        # Getting the type of 'nan_y' (line 287)
        nan_y_511787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 33), 'nan_y')
        # Applying the binary operator '==' (line 287)
        result_eq_511788 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 24), '==', nan_x_511786, nan_y_511787)
        
        # Assigning a type to the variable 'nan_mask' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'nan_mask', result_eq_511788)
        
        # Assigning a UnaryOp to a Name (line 289):
        
        # Assigning a UnaryOp to a Name (line 289):
        
        # Getting the type of 'tol_mask' (line 289)
        tol_mask_511789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 22), 'tol_mask')
        # Getting the type of 'pinf_mask' (line 289)
        pinf_mask_511790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 33), 'pinf_mask')
        # Applying the binary operator '&' (line 289)
        result_and__511791 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 22), '&', tol_mask_511789, pinf_mask_511790)
        
        # Getting the type of 'minf_mask' (line 289)
        minf_mask_511792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 45), 'minf_mask')
        # Applying the binary operator '&' (line 289)
        result_and__511793 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 43), '&', result_and__511791, minf_mask_511792)
        
        # Getting the type of 'nan_mask' (line 289)
        nan_mask_511794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 57), 'nan_mask')
        # Applying the binary operator '&' (line 289)
        result_and__511795 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 55), '&', result_and__511793, nan_mask_511794)
        
        # Applying the '~' unary operator (line 289)
        result_inv_511796 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 20), '~', result_and__511795)
        
        # Assigning a type to the variable 'bad_j' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'bad_j', result_inv_511796)
        
        # Assigning a Attribute to a Name (line 291):
        
        # Assigning a Attribute to a Name (line 291):
        # Getting the type of 'bad_j' (line 291)
        bad_j_511797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 26), 'bad_j')
        # Obtaining the member 'size' of a type (line 291)
        size_511798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 26), bad_j_511797, 'size')
        # Assigning a type to the variable 'point_count' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'point_count', size_511798)
        
        # Getting the type of 'self' (line 292)
        self_511799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 15), 'self')
        # Obtaining the member 'nan_ok' of a type (line 292)
        nan_ok_511800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 15), self_511799, 'nan_ok')
        # Testing the type of an if condition (line 292)
        if_condition_511801 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 12), nan_ok_511800)
        # Assigning a type to the variable 'if_condition_511801' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'if_condition_511801', if_condition_511801)
        # SSA begins for if statement (line 292)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'bad_j' (line 293)
        bad_j_511802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 16), 'bad_j')
        
        # Getting the type of 'nan_x' (line 293)
        nan_x_511803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 26), 'nan_x')
        # Applying the '~' unary operator (line 293)
        result_inv_511804 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 25), '~', nan_x_511803)
        
        # Applying the binary operator '&=' (line 293)
        result_iand_511805 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 16), '&=', bad_j_511802, result_inv_511804)
        # Assigning a type to the variable 'bad_j' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 16), 'bad_j', result_iand_511805)
        
        
        # Getting the type of 'bad_j' (line 294)
        bad_j_511806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 16), 'bad_j')
        
        # Getting the type of 'nan_y' (line 294)
        nan_y_511807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 26), 'nan_y')
        # Applying the '~' unary operator (line 294)
        result_inv_511808 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 25), '~', nan_y_511807)
        
        # Applying the binary operator '&=' (line 294)
        result_iand_511809 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 16), '&=', bad_j_511806, result_inv_511808)
        # Assigning a type to the variable 'bad_j' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 16), 'bad_j', result_iand_511809)
        
        
        # Getting the type of 'point_count' (line 295)
        point_count_511810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'point_count')
        
        # Call to sum(...): (line 295)
        # Processing the call keyword arguments (line 295)
        kwargs_511815 = {}
        # Getting the type of 'nan_x' (line 295)
        nan_x_511811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 32), 'nan_x', False)
        # Getting the type of 'nan_y' (line 295)
        nan_y_511812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 40), 'nan_y', False)
        # Applying the binary operator '|' (line 295)
        result_or__511813 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 32), '|', nan_x_511811, nan_y_511812)
        
        # Obtaining the member 'sum' of a type (line 295)
        sum_511814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 32), result_or__511813, 'sum')
        # Calling sum(args, kwargs) (line 295)
        sum_call_result_511816 = invoke(stypy.reporting.localization.Localization(__file__, 295, 32), sum_511814, *[], **kwargs_511815)
        
        # Applying the binary operator '-=' (line 295)
        result_isub_511817 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 16), '-=', point_count_511810, sum_call_result_511816)
        # Assigning a type to the variable 'point_count' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'point_count', result_isub_511817)
        
        # SSA join for if statement (line 292)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 297)
        self_511818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 19), 'self')
        # Obtaining the member 'distinguish_nan_and_inf' of a type (line 297)
        distinguish_nan_and_inf_511819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 19), self_511818, 'distinguish_nan_and_inf')
        # Applying the 'not' unary operator (line 297)
        result_not__511820 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 15), 'not', distinguish_nan_and_inf_511819)
        
        
        # Getting the type of 'self' (line 297)
        self_511821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 56), 'self')
        # Obtaining the member 'nan_ok' of a type (line 297)
        nan_ok_511822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 56), self_511821, 'nan_ok')
        # Applying the 'not' unary operator (line 297)
        result_not__511823 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 52), 'not', nan_ok_511822)
        
        # Applying the binary operator 'and' (line 297)
        result_and_keyword_511824 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 15), 'and', result_not__511820, result_not__511823)
        
        # Testing the type of an if condition (line 297)
        if_condition_511825 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 12), result_and_keyword_511824)
        # Assigning a type to the variable 'if_condition_511825' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'if_condition_511825', if_condition_511825)
        # SSA begins for if statement (line 297)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 299):
        
        # Assigning a Call to a Name (line 299):
        
        # Call to isinf(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'x' (line 299)
        x_511828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 33), 'x', False)
        # Processing the call keyword arguments (line 299)
        kwargs_511829 = {}
        # Getting the type of 'np' (line 299)
        np_511826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 24), 'np', False)
        # Obtaining the member 'isinf' of a type (line 299)
        isinf_511827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 24), np_511826, 'isinf')
        # Calling isinf(args, kwargs) (line 299)
        isinf_call_result_511830 = invoke(stypy.reporting.localization.Localization(__file__, 299, 24), isinf_511827, *[x_511828], **kwargs_511829)
        
        # Assigning a type to the variable 'inf_x' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 'inf_x', isinf_call_result_511830)
        
        # Assigning a Call to a Name (line 300):
        
        # Assigning a Call to a Name (line 300):
        
        # Call to isinf(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'y' (line 300)
        y_511833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 33), 'y', False)
        # Processing the call keyword arguments (line 300)
        kwargs_511834 = {}
        # Getting the type of 'np' (line 300)
        np_511831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 24), 'np', False)
        # Obtaining the member 'isinf' of a type (line 300)
        isinf_511832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 24), np_511831, 'isinf')
        # Calling isinf(args, kwargs) (line 300)
        isinf_call_result_511835 = invoke(stypy.reporting.localization.Localization(__file__, 300, 24), isinf_511832, *[y_511833], **kwargs_511834)
        
        # Assigning a type to the variable 'inf_y' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'inf_y', isinf_call_result_511835)
        
        # Assigning a BinOp to a Name (line 301):
        
        # Assigning a BinOp to a Name (line 301):
        # Getting the type of 'inf_x' (line 301)
        inf_x_511836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 34), 'inf_x')
        # Getting the type of 'nan_y' (line 301)
        nan_y_511837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 42), 'nan_y')
        # Applying the binary operator '&' (line 301)
        result_and__511838 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 34), '&', inf_x_511836, nan_y_511837)
        
        # Getting the type of 'nan_x' (line 301)
        nan_x_511839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 52), 'nan_x')
        # Getting the type of 'inf_y' (line 301)
        inf_y_511840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 60), 'inf_y')
        # Applying the binary operator '&' (line 301)
        result_and__511841 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 52), '&', nan_x_511839, inf_y_511840)
        
        # Applying the binary operator '|' (line 301)
        result_or__511842 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 33), '|', result_and__511838, result_and__511841)
        
        # Assigning a type to the variable 'both_nonfinite' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'both_nonfinite', result_or__511842)
        
        # Getting the type of 'bad_j' (line 302)
        bad_j_511843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'bad_j')
        
        # Getting the type of 'both_nonfinite' (line 302)
        both_nonfinite_511844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 26), 'both_nonfinite')
        # Applying the '~' unary operator (line 302)
        result_inv_511845 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 25), '~', both_nonfinite_511844)
        
        # Applying the binary operator '&=' (line 302)
        result_iand_511846 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 16), '&=', bad_j_511843, result_inv_511845)
        # Assigning a type to the variable 'bad_j' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'bad_j', result_iand_511846)
        
        
        # Getting the type of 'point_count' (line 303)
        point_count_511847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'point_count')
        
        # Call to sum(...): (line 303)
        # Processing the call keyword arguments (line 303)
        kwargs_511850 = {}
        # Getting the type of 'both_nonfinite' (line 303)
        both_nonfinite_511848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 31), 'both_nonfinite', False)
        # Obtaining the member 'sum' of a type (line 303)
        sum_511849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 31), both_nonfinite_511848, 'sum')
        # Calling sum(args, kwargs) (line 303)
        sum_call_result_511851 = invoke(stypy.reporting.localization.Localization(__file__, 303, 31), sum_511849, *[], **kwargs_511850)
        
        # Applying the binary operator '-=' (line 303)
        result_isub_511852 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 16), '-=', point_count_511847, sum_call_result_511851)
        # Assigning a type to the variable 'point_count' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'point_count', result_isub_511852)
        
        # SSA join for if statement (line 297)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to any(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 'bad_j' (line 305)
        bad_j_511855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 22), 'bad_j', False)
        # Processing the call keyword arguments (line 305)
        kwargs_511856 = {}
        # Getting the type of 'np' (line 305)
        np_511853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 15), 'np', False)
        # Obtaining the member 'any' of a type (line 305)
        any_511854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 15), np_511853, 'any')
        # Calling any(args, kwargs) (line 305)
        any_call_result_511857 = invoke(stypy.reporting.localization.Localization(__file__, 305, 15), any_511854, *[bad_j_511855], **kwargs_511856)
        
        # Testing the type of an if condition (line 305)
        if_condition_511858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 12), any_call_result_511857)
        # Assigning a type to the variable 'if_condition_511858' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'if_condition_511858', if_condition_511858)
        # SSA begins for if statement (line 305)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 307):
        
        # Assigning a List to a Name (line 307):
        
        # Obtaining an instance of the builtin type 'list' (line 307)
        list_511859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 307)
        # Adding element type (line 307)
        str_511860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 23), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 22), list_511859, str_511860)
        
        # Assigning a type to the variable 'msg' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 16), 'msg', list_511859)
        
        # Call to append(...): (line 308)
        # Processing the call arguments (line 308)
        str_511863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 27), 'str', 'Max |adiff|: %g')
        
        # Call to max(...): (line 308)
        # Processing the call keyword arguments (line 308)
        kwargs_511866 = {}
        # Getting the type of 'diff' (line 308)
        diff_511864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 47), 'diff', False)
        # Obtaining the member 'max' of a type (line 308)
        max_511865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 47), diff_511864, 'max')
        # Calling max(args, kwargs) (line 308)
        max_call_result_511867 = invoke(stypy.reporting.localization.Localization(__file__, 308, 47), max_511865, *[], **kwargs_511866)
        
        # Applying the binary operator '%' (line 308)
        result_mod_511868 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 27), '%', str_511863, max_call_result_511867)
        
        # Processing the call keyword arguments (line 308)
        kwargs_511869 = {}
        # Getting the type of 'msg' (line 308)
        msg_511861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'msg', False)
        # Obtaining the member 'append' of a type (line 308)
        append_511862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 16), msg_511861, 'append')
        # Calling append(args, kwargs) (line 308)
        append_call_result_511870 = invoke(stypy.reporting.localization.Localization(__file__, 308, 16), append_511862, *[result_mod_511868], **kwargs_511869)
        
        
        # Call to append(...): (line 309)
        # Processing the call arguments (line 309)
        str_511873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 27), 'str', 'Max |rdiff|: %g')
        
        # Call to max(...): (line 309)
        # Processing the call keyword arguments (line 309)
        kwargs_511876 = {}
        # Getting the type of 'rdiff' (line 309)
        rdiff_511874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 47), 'rdiff', False)
        # Obtaining the member 'max' of a type (line 309)
        max_511875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 47), rdiff_511874, 'max')
        # Calling max(args, kwargs) (line 309)
        max_call_result_511877 = invoke(stypy.reporting.localization.Localization(__file__, 309, 47), max_511875, *[], **kwargs_511876)
        
        # Applying the binary operator '%' (line 309)
        result_mod_511878 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 27), '%', str_511873, max_call_result_511877)
        
        # Processing the call keyword arguments (line 309)
        kwargs_511879 = {}
        # Getting the type of 'msg' (line 309)
        msg_511871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'msg', False)
        # Obtaining the member 'append' of a type (line 309)
        append_511872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 16), msg_511871, 'append')
        # Calling append(args, kwargs) (line 309)
        append_call_result_511880 = invoke(stypy.reporting.localization.Localization(__file__, 309, 16), append_511872, *[result_mod_511878], **kwargs_511879)
        
        
        # Call to append(...): (line 310)
        # Processing the call arguments (line 310)
        str_511883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 27), 'str', 'Bad results (%d out of %d) for the following points (in output %d):')
        
        # Obtaining an instance of the builtin type 'tuple' (line 311)
        tuple_511884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 311)
        # Adding element type (line 311)
        
        # Call to sum(...): (line 311)
        # Processing the call arguments (line 311)
        # Getting the type of 'bad_j' (line 311)
        bad_j_511887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 37), 'bad_j', False)
        # Processing the call keyword arguments (line 311)
        kwargs_511888 = {}
        # Getting the type of 'np' (line 311)
        np_511885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 30), 'np', False)
        # Obtaining the member 'sum' of a type (line 311)
        sum_511886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 30), np_511885, 'sum')
        # Calling sum(args, kwargs) (line 311)
        sum_call_result_511889 = invoke(stypy.reporting.localization.Localization(__file__, 311, 30), sum_511886, *[bad_j_511887], **kwargs_511888)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 30), tuple_511884, sum_call_result_511889)
        # Adding element type (line 311)
        # Getting the type of 'point_count' (line 311)
        point_count_511890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 45), 'point_count', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 30), tuple_511884, point_count_511890)
        # Adding element type (line 311)
        # Getting the type of 'output_num' (line 311)
        output_num_511891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 58), 'output_num', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 30), tuple_511884, output_num_511891)
        
        # Applying the binary operator '%' (line 310)
        result_mod_511892 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 27), '%', str_511883, tuple_511884)
        
        # Processing the call keyword arguments (line 310)
        kwargs_511893 = {}
        # Getting the type of 'msg' (line 310)
        msg_511881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 16), 'msg', False)
        # Obtaining the member 'append' of a type (line 310)
        append_511882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 16), msg_511881, 'append')
        # Calling append(args, kwargs) (line 310)
        append_call_result_511894 = invoke(stypy.reporting.localization.Localization(__file__, 310, 16), append_511882, *[result_mod_511892], **kwargs_511893)
        
        
        
        # Obtaining the type of the subscript
        int_511895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 41), 'int')
        
        # Call to where(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'bad_j' (line 312)
        bad_j_511898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 34), 'bad_j', False)
        # Processing the call keyword arguments (line 312)
        kwargs_511899 = {}
        # Getting the type of 'np' (line 312)
        np_511896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 25), 'np', False)
        # Obtaining the member 'where' of a type (line 312)
        where_511897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 25), np_511896, 'where')
        # Calling where(args, kwargs) (line 312)
        where_call_result_511900 = invoke(stypy.reporting.localization.Localization(__file__, 312, 25), where_511897, *[bad_j_511898], **kwargs_511899)
        
        # Obtaining the member '__getitem__' of a type (line 312)
        getitem___511901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 25), where_call_result_511900, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 312)
        subscript_call_result_511902 = invoke(stypy.reporting.localization.Localization(__file__, 312, 25), getitem___511901, int_511895)
        
        # Testing the type of a for loop iterable (line 312)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 312, 16), subscript_call_result_511902)
        # Getting the type of the for loop variable (line 312)
        for_loop_var_511903 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 312, 16), subscript_call_result_511902)
        # Assigning a type to the variable 'j' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 16), 'j', for_loop_var_511903)
        # SSA begins for a for statement (line 312)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 313):
        
        # Assigning a Call to a Name (line 313):
        
        # Call to int(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 'j' (line 313)
        j_511905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 28), 'j', False)
        # Processing the call keyword arguments (line 313)
        kwargs_511906 = {}
        # Getting the type of 'int' (line 313)
        int_511904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 24), 'int', False)
        # Calling int(args, kwargs) (line 313)
        int_call_result_511907 = invoke(stypy.reporting.localization.Localization(__file__, 313, 24), int_511904, *[j_511905], **kwargs_511906)
        
        # Assigning a type to the variable 'j' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 20), 'j', int_call_result_511907)
        
        # Assigning a Lambda to a Name (line 314):
        
        # Assigning a Lambda to a Name (line 314):

        @norecursion
        def _stypy_temp_lambda_310(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_310'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_310', 314, 26, True)
            # Passed parameters checking function
            _stypy_temp_lambda_310.stypy_localization = localization
            _stypy_temp_lambda_310.stypy_type_of_self = None
            _stypy_temp_lambda_310.stypy_type_store = module_type_store
            _stypy_temp_lambda_310.stypy_function_name = '_stypy_temp_lambda_310'
            _stypy_temp_lambda_310.stypy_param_names_list = ['x']
            _stypy_temp_lambda_310.stypy_varargs_param_name = None
            _stypy_temp_lambda_310.stypy_kwargs_param_name = None
            _stypy_temp_lambda_310.stypy_call_defaults = defaults
            _stypy_temp_lambda_310.stypy_call_varargs = varargs
            _stypy_temp_lambda_310.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_310', ['x'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_310', ['x'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            str_511908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 36), 'str', '%30s')
            
            # Call to array2string(...): (line 314)
            # Processing the call arguments (line 314)
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 314)
            j_511911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 63), 'j', False)
            # Getting the type of 'x' (line 314)
            x_511912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 61), 'x', False)
            # Obtaining the member '__getitem__' of a type (line 314)
            getitem___511913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 61), x_511912, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 314)
            subscript_call_result_511914 = invoke(stypy.reporting.localization.Localization(__file__, 314, 61), getitem___511913, j_511911)
            
            # Processing the call keyword arguments (line 314)
            int_511915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 77), 'int')
            keyword_511916 = int_511915
            kwargs_511917 = {'precision': keyword_511916}
            # Getting the type of 'np' (line 314)
            np_511909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 45), 'np', False)
            # Obtaining the member 'array2string' of a type (line 314)
            array2string_511910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 45), np_511909, 'array2string')
            # Calling array2string(args, kwargs) (line 314)
            array2string_call_result_511918 = invoke(stypy.reporting.localization.Localization(__file__, 314, 45), array2string_511910, *[subscript_call_result_511914], **kwargs_511917)
            
            # Applying the binary operator '%' (line 314)
            result_mod_511919 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 36), '%', str_511908, array2string_call_result_511918)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 314)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 26), 'stypy_return_type', result_mod_511919)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_310' in the type store
            # Getting the type of 'stypy_return_type' (line 314)
            stypy_return_type_511920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 26), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_511920)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_310'
            return stypy_return_type_511920

        # Assigning a type to the variable '_stypy_temp_lambda_310' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 26), '_stypy_temp_lambda_310', _stypy_temp_lambda_310)
        # Getting the type of '_stypy_temp_lambda_310' (line 314)
        _stypy_temp_lambda_310_511921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 26), '_stypy_temp_lambda_310')
        # Assigning a type to the variable 'fmt' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 20), 'fmt', _stypy_temp_lambda_310_511921)
        
        # Assigning a Call to a Name (line 315):
        
        # Assigning a Call to a Name (line 315):
        
        # Call to join(...): (line 315)
        # Processing the call arguments (line 315)
        
        # Call to map(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'fmt' (line 315)
        fmt_511925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 38), 'fmt', False)
        # Getting the type of 'params' (line 315)
        params_511926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 43), 'params', False)
        # Processing the call keyword arguments (line 315)
        kwargs_511927 = {}
        # Getting the type of 'map' (line 315)
        map_511924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 34), 'map', False)
        # Calling map(args, kwargs) (line 315)
        map_call_result_511928 = invoke(stypy.reporting.localization.Localization(__file__, 315, 34), map_511924, *[fmt_511925, params_511926], **kwargs_511927)
        
        # Processing the call keyword arguments (line 315)
        kwargs_511929 = {}
        str_511922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 24), 'str', '  ')
        # Obtaining the member 'join' of a type (line 315)
        join_511923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 24), str_511922, 'join')
        # Calling join(args, kwargs) (line 315)
        join_call_result_511930 = invoke(stypy.reporting.localization.Localization(__file__, 315, 24), join_511923, *[map_call_result_511928], **kwargs_511929)
        
        # Assigning a type to the variable 'a' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 20), 'a', join_call_result_511930)
        
        # Assigning a Call to a Name (line 316):
        
        # Assigning a Call to a Name (line 316):
        
        # Call to join(...): (line 316)
        # Processing the call arguments (line 316)
        
        # Call to map(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'fmt' (line 316)
        fmt_511934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 38), 'fmt', False)
        # Getting the type of 'got' (line 316)
        got_511935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 43), 'got', False)
        # Processing the call keyword arguments (line 316)
        kwargs_511936 = {}
        # Getting the type of 'map' (line 316)
        map_511933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 34), 'map', False)
        # Calling map(args, kwargs) (line 316)
        map_call_result_511937 = invoke(stypy.reporting.localization.Localization(__file__, 316, 34), map_511933, *[fmt_511934, got_511935], **kwargs_511936)
        
        # Processing the call keyword arguments (line 316)
        kwargs_511938 = {}
        str_511931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 24), 'str', '  ')
        # Obtaining the member 'join' of a type (line 316)
        join_511932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 24), str_511931, 'join')
        # Calling join(args, kwargs) (line 316)
        join_call_result_511939 = invoke(stypy.reporting.localization.Localization(__file__, 316, 24), join_511932, *[map_call_result_511937], **kwargs_511938)
        
        # Assigning a type to the variable 'b' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 20), 'b', join_call_result_511939)
        
        # Assigning a Call to a Name (line 317):
        
        # Assigning a Call to a Name (line 317):
        
        # Call to join(...): (line 317)
        # Processing the call arguments (line 317)
        
        # Call to map(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'fmt' (line 317)
        fmt_511943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 38), 'fmt', False)
        # Getting the type of 'wanted' (line 317)
        wanted_511944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 43), 'wanted', False)
        # Processing the call keyword arguments (line 317)
        kwargs_511945 = {}
        # Getting the type of 'map' (line 317)
        map_511942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 34), 'map', False)
        # Calling map(args, kwargs) (line 317)
        map_call_result_511946 = invoke(stypy.reporting.localization.Localization(__file__, 317, 34), map_511942, *[fmt_511943, wanted_511944], **kwargs_511945)
        
        # Processing the call keyword arguments (line 317)
        kwargs_511947 = {}
        str_511940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 24), 'str', '  ')
        # Obtaining the member 'join' of a type (line 317)
        join_511941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 24), str_511940, 'join')
        # Calling join(args, kwargs) (line 317)
        join_call_result_511948 = invoke(stypy.reporting.localization.Localization(__file__, 317, 24), join_511941, *[map_call_result_511946], **kwargs_511947)
        
        # Assigning a type to the variable 'c' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'c', join_call_result_511948)
        
        # Assigning a Call to a Name (line 318):
        
        # Assigning a Call to a Name (line 318):
        
        # Call to fmt(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'rdiff' (line 318)
        rdiff_511950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 28), 'rdiff', False)
        # Processing the call keyword arguments (line 318)
        kwargs_511951 = {}
        # Getting the type of 'fmt' (line 318)
        fmt_511949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 24), 'fmt', False)
        # Calling fmt(args, kwargs) (line 318)
        fmt_call_result_511952 = invoke(stypy.reporting.localization.Localization(__file__, 318, 24), fmt_511949, *[rdiff_511950], **kwargs_511951)
        
        # Assigning a type to the variable 'd' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 20), 'd', fmt_call_result_511952)
        
        # Call to append(...): (line 319)
        # Processing the call arguments (line 319)
        str_511955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 31), 'str', '%s => %s != %s  (rdiff %s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 319)
        tuple_511956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 63), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 319)
        # Adding element type (line 319)
        # Getting the type of 'a' (line 319)
        a_511957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 63), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 63), tuple_511956, a_511957)
        # Adding element type (line 319)
        # Getting the type of 'b' (line 319)
        b_511958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 66), 'b', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 63), tuple_511956, b_511958)
        # Adding element type (line 319)
        # Getting the type of 'c' (line 319)
        c_511959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 69), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 63), tuple_511956, c_511959)
        # Adding element type (line 319)
        # Getting the type of 'd' (line 319)
        d_511960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 72), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 63), tuple_511956, d_511960)
        
        # Applying the binary operator '%' (line 319)
        result_mod_511961 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 31), '%', str_511955, tuple_511956)
        
        # Processing the call keyword arguments (line 319)
        kwargs_511962 = {}
        # Getting the type of 'msg' (line 319)
        msg_511953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 20), 'msg', False)
        # Obtaining the member 'append' of a type (line 319)
        append_511954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 20), msg_511953, 'append')
        # Calling append(args, kwargs) (line 319)
        append_call_result_511963 = invoke(stypy.reporting.localization.Localization(__file__, 319, 20), append_511954, *[result_mod_511961], **kwargs_511962)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'False' (line 320)
        False_511965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 24), 'False', False)
        
        # Call to join(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'msg' (line 320)
        msg_511968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 41), 'msg', False)
        # Processing the call keyword arguments (line 320)
        kwargs_511969 = {}
        str_511966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 31), 'str', '\n')
        # Obtaining the member 'join' of a type (line 320)
        join_511967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 31), str_511966, 'join')
        # Calling join(args, kwargs) (line 320)
        join_call_result_511970 = invoke(stypy.reporting.localization.Localization(__file__, 320, 31), join_511967, *[msg_511968], **kwargs_511969)
        
        # Processing the call keyword arguments (line 320)
        kwargs_511971 = {}
        # Getting the type of 'assert_' (line 320)
        assert__511964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 16), 'assert_', False)
        # Calling assert_(args, kwargs) (line 320)
        assert__call_result_511972 = invoke(stypy.reporting.localization.Localization(__file__, 320, 16), assert__511964, *[False_511965, join_call_result_511970], **kwargs_511971)
        
        # SSA join for if statement (line 305)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check' in the type store
        # Getting the type of 'stypy_return_type' (line 190)
        stypy_return_type_511973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_511973)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check'
        return stypy_return_type_511973


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 322, 4, False)
        # Assigning a type to the variable 'self' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FuncData.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        FuncData.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FuncData.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FuncData.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'FuncData.stypy__repr__')
        FuncData.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        FuncData.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FuncData.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FuncData.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FuncData.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FuncData.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FuncData.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FuncData.stypy__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        str_511974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 8), 'str', 'Pretty-printing, esp. for Nose output')
        
        
        # Call to any(...): (line 324)
        # Processing the call arguments (line 324)
        
        # Call to list(...): (line 324)
        # Processing the call arguments (line 324)
        
        # Call to map(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 'np' (line 324)
        np_511979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 27), 'np', False)
        # Obtaining the member 'iscomplexobj' of a type (line 324)
        iscomplexobj_511980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 27), np_511979, 'iscomplexobj')
        # Getting the type of 'self' (line 324)
        self_511981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 44), 'self', False)
        # Obtaining the member 'param_columns' of a type (line 324)
        param_columns_511982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 44), self_511981, 'param_columns')
        # Processing the call keyword arguments (line 324)
        kwargs_511983 = {}
        # Getting the type of 'map' (line 324)
        map_511978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 23), 'map', False)
        # Calling map(args, kwargs) (line 324)
        map_call_result_511984 = invoke(stypy.reporting.localization.Localization(__file__, 324, 23), map_511978, *[iscomplexobj_511980, param_columns_511982], **kwargs_511983)
        
        # Processing the call keyword arguments (line 324)
        kwargs_511985 = {}
        # Getting the type of 'list' (line 324)
        list_511977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 18), 'list', False)
        # Calling list(args, kwargs) (line 324)
        list_call_result_511986 = invoke(stypy.reporting.localization.Localization(__file__, 324, 18), list_511977, *[map_call_result_511984], **kwargs_511985)
        
        # Processing the call keyword arguments (line 324)
        kwargs_511987 = {}
        # Getting the type of 'np' (line 324)
        np_511975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 11), 'np', False)
        # Obtaining the member 'any' of a type (line 324)
        any_511976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 11), np_511975, 'any')
        # Calling any(args, kwargs) (line 324)
        any_call_result_511988 = invoke(stypy.reporting.localization.Localization(__file__, 324, 11), any_511976, *[list_call_result_511986], **kwargs_511987)
        
        # Testing the type of an if condition (line 324)
        if_condition_511989 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 8), any_call_result_511988)
        # Assigning a type to the variable 'if_condition_511989' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'if_condition_511989', if_condition_511989)
        # SSA begins for if statement (line 324)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 325):
        
        # Assigning a Str to a Name (line 325):
        str_511990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 25), 'str', ' (complex)')
        # Assigning a type to the variable 'is_complex' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'is_complex', str_511990)
        # SSA branch for the else part of an if statement (line 324)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 327):
        
        # Assigning a Str to a Name (line 327):
        str_511991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 25), 'str', '')
        # Assigning a type to the variable 'is_complex' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'is_complex', str_511991)
        # SSA join for if statement (line 324)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 328)
        self_511992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 11), 'self')
        # Obtaining the member 'dataname' of a type (line 328)
        dataname_511993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 11), self_511992, 'dataname')
        # Testing the type of an if condition (line 328)
        if_condition_511994 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 328, 8), dataname_511993)
        # Assigning a type to the variable 'if_condition_511994' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'if_condition_511994', if_condition_511994)
        # SSA begins for if statement (line 328)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_511995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 19), 'str', '<Data for %s%s: %s>')
        
        # Obtaining an instance of the builtin type 'tuple' (line 329)
        tuple_511996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 329)
        # Adding element type (line 329)
        # Getting the type of 'self' (line 329)
        self_511997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 44), 'self')
        # Obtaining the member 'func' of a type (line 329)
        func_511998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 44), self_511997, 'func')
        # Obtaining the member '__name__' of a type (line 329)
        name___511999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 44), func_511998, '__name__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 44), tuple_511996, name___511999)
        # Adding element type (line 329)
        # Getting the type of 'is_complex' (line 329)
        is_complex_512000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 64), 'is_complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 44), tuple_511996, is_complex_512000)
        # Adding element type (line 329)
        
        # Call to basename(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 'self' (line 330)
        self_512004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 61), 'self', False)
        # Obtaining the member 'dataname' of a type (line 330)
        dataname_512005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 61), self_512004, 'dataname')
        # Processing the call keyword arguments (line 330)
        kwargs_512006 = {}
        # Getting the type of 'os' (line 330)
        os_512001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 44), 'os', False)
        # Obtaining the member 'path' of a type (line 330)
        path_512002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 44), os_512001, 'path')
        # Obtaining the member 'basename' of a type (line 330)
        basename_512003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 44), path_512002, 'basename')
        # Calling basename(args, kwargs) (line 330)
        basename_call_result_512007 = invoke(stypy.reporting.localization.Localization(__file__, 330, 44), basename_512003, *[dataname_512005], **kwargs_512006)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 44), tuple_511996, basename_call_result_512007)
        
        # Applying the binary operator '%' (line 329)
        result_mod_512008 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 19), '%', str_511995, tuple_511996)
        
        # Assigning a type to the variable 'stypy_return_type' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'stypy_return_type', result_mod_512008)
        # SSA branch for the else part of an if statement (line 328)
        module_type_store.open_ssa_branch('else')
        str_512009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 19), 'str', '<Data for %s%s>')
        
        # Obtaining an instance of the builtin type 'tuple' (line 332)
        tuple_512010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 332)
        # Adding element type (line 332)
        # Getting the type of 'self' (line 332)
        self_512011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 40), 'self')
        # Obtaining the member 'func' of a type (line 332)
        func_512012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 40), self_512011, 'func')
        # Obtaining the member '__name__' of a type (line 332)
        name___512013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 40), func_512012, '__name__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 40), tuple_512010, name___512013)
        # Adding element type (line 332)
        # Getting the type of 'is_complex' (line 332)
        is_complex_512014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 60), 'is_complex')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 40), tuple_512010, is_complex_512014)
        
        # Applying the binary operator '%' (line 332)
        result_mod_512015 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 19), '%', str_512009, tuple_512010)
        
        # Assigning a type to the variable 'stypy_return_type' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'stypy_return_type', result_mod_512015)
        # SSA join for if statement (line 328)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 322)
        stypy_return_type_512016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_512016)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_512016


# Assigning a type to the variable 'FuncData' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'FuncData', FuncData)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
