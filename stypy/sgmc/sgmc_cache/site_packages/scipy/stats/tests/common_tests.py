
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import pickle
4: 
5: import numpy as np
6: import numpy.testing as npt
7: from numpy.testing import assert_allclose, assert_equal
8: from scipy._lib._numpy_compat import suppress_warnings
9: from pytest import raises as assert_raises
10: 
11: import numpy.ma.testutils as ma_npt
12: 
13: from scipy._lib._util import getargspec_no_self as _getargspec
14: from scipy import stats
15: 
16: 
17: def check_named_results(res, attributes, ma=False):
18:     for i, attr in enumerate(attributes):
19:         if ma:
20:             ma_npt.assert_equal(res[i], getattr(res, attr))
21:         else:
22:             npt.assert_equal(res[i], getattr(res, attr))
23: 
24: 
25: def check_normalization(distfn, args, distname):
26:     norm_moment = distfn.moment(0, *args)
27:     npt.assert_allclose(norm_moment, 1.0)
28: 
29:     # this is a temporary plug: either ncf or expect is problematic;
30:     # best be marked as a knownfail, but I've no clue how to do it.
31:     if distname == "ncf":
32:         atol, rtol = 1e-5, 0
33:     else:
34:         atol, rtol = 1e-7, 1e-7
35: 
36:     normalization_expect = distfn.expect(lambda x: 1, args=args)
37:     npt.assert_allclose(normalization_expect, 1.0, atol=atol, rtol=rtol,
38:             err_msg=distname, verbose=True)
39: 
40:     normalization_cdf = distfn.cdf(distfn.b, *args)
41:     npt.assert_allclose(normalization_cdf, 1.0)
42: 
43: 
44: def check_moment(distfn, arg, m, v, msg):
45:     m1 = distfn.moment(1, *arg)
46:     m2 = distfn.moment(2, *arg)
47:     if not np.isinf(m):
48:         npt.assert_almost_equal(m1, m, decimal=10, err_msg=msg +
49:                             ' - 1st moment')
50:     else:                     # or np.isnan(m1),
51:         npt.assert_(np.isinf(m1),
52:                msg + ' - 1st moment -infinite, m1=%s' % str(m1))
53: 
54:     if not np.isinf(v):
55:         npt.assert_almost_equal(m2 - m1 * m1, v, decimal=10, err_msg=msg +
56:                             ' - 2ndt moment')
57:     else:                     # or np.isnan(m2),
58:         npt.assert_(np.isinf(m2),
59:                msg + ' - 2nd moment -infinite, m2=%s' % str(m2))
60: 
61: 
62: def check_mean_expect(distfn, arg, m, msg):
63:     if np.isfinite(m):
64:         m1 = distfn.expect(lambda x: x, arg)
65:         npt.assert_almost_equal(m1, m, decimal=5, err_msg=msg +
66:                             ' - 1st moment (expect)')
67: 
68: 
69: def check_var_expect(distfn, arg, m, v, msg):
70:     if np.isfinite(v):
71:         m2 = distfn.expect(lambda x: x*x, arg)
72:         npt.assert_almost_equal(m2, v + m*m, decimal=5, err_msg=msg +
73:                             ' - 2st moment (expect)')
74: 
75: 
76: def check_skew_expect(distfn, arg, m, v, s, msg):
77:     if np.isfinite(s):
78:         m3e = distfn.expect(lambda x: np.power(x-m, 3), arg)
79:         npt.assert_almost_equal(m3e, s * np.power(v, 1.5),
80:                 decimal=5, err_msg=msg + ' - skew')
81:     else:
82:         npt.assert_(np.isnan(s))
83: 
84: 
85: def check_kurt_expect(distfn, arg, m, v, k, msg):
86:     if np.isfinite(k):
87:         m4e = distfn.expect(lambda x: np.power(x-m, 4), arg)
88:         npt.assert_allclose(m4e, (k + 3.) * np.power(v, 2), atol=1e-5, rtol=1e-5,
89:                 err_msg=msg + ' - kurtosis')
90:     else:
91:         npt.assert_(np.isnan(k))
92: 
93: 
94: def check_entropy(distfn, arg, msg):
95:     ent = distfn.entropy(*arg)
96:     npt.assert_(not np.isnan(ent), msg + 'test Entropy is nan')
97: 
98: 
99: def check_private_entropy(distfn, args, superclass):
100:     # compare a generic _entropy with the distribution-specific implementation
101:     npt.assert_allclose(distfn._entropy(*args),
102:                         superclass._entropy(distfn, *args))
103: 
104: 
105: def check_edge_support(distfn, args):
106:     # Make sure that x=self.a and self.b are handled correctly.
107:     x = [distfn.a, distfn.b]
108:     if isinstance(distfn, stats.rv_discrete):
109:         x = [distfn.a - 1, distfn.b]
110: 
111:     npt.assert_equal(distfn.cdf(x, *args), [0.0, 1.0])
112:     npt.assert_equal(distfn.sf(x, *args), [1.0, 0.0])
113: 
114:     if distfn.name not in ('skellam', 'dlaplace'):
115:         # with a = -inf, log(0) generates warnings
116:         npt.assert_equal(distfn.logcdf(x, *args), [-np.inf, 0.0])
117:         npt.assert_equal(distfn.logsf(x, *args), [0.0, -np.inf])
118: 
119:     npt.assert_equal(distfn.ppf([0.0, 1.0], *args), x)
120:     npt.assert_equal(distfn.isf([0.0, 1.0], *args), x[::-1])
121: 
122:     # out-of-bounds for isf & ppf
123:     npt.assert_(np.isnan(distfn.isf([-1, 2], *args)).all())
124:     npt.assert_(np.isnan(distfn.ppf([-1, 2], *args)).all())
125: 
126: 
127: def check_named_args(distfn, x, shape_args, defaults, meths):
128:     ## Check calling w/ named arguments.
129: 
130:     # check consistency of shapes, numargs and _parse signature
131:     signature = _getargspec(distfn._parse_args)
132:     npt.assert_(signature.varargs is None)
133:     npt.assert_(signature.keywords is None)
134:     npt.assert_(list(signature.defaults) == list(defaults))
135: 
136:     shape_argnames = signature.args[:-len(defaults)]  # a, b, loc=0, scale=1
137:     if distfn.shapes:
138:         shapes_ = distfn.shapes.replace(',', ' ').split()
139:     else:
140:         shapes_ = ''
141:     npt.assert_(len(shapes_) == distfn.numargs)
142:     npt.assert_(len(shapes_) == len(shape_argnames))
143: 
144:     # check calling w/ named arguments
145:     shape_args = list(shape_args)
146: 
147:     vals = [meth(x, *shape_args) for meth in meths]
148:     npt.assert_(np.all(np.isfinite(vals)))
149: 
150:     names, a, k = shape_argnames[:], shape_args[:], {}
151:     while names:
152:         k.update({names.pop(): a.pop()})
153:         v = [meth(x, *a, **k) for meth in meths]
154:         npt.assert_array_equal(vals, v)
155:         if 'n' not in k.keys():
156:             # `n` is first parameter of moment(), so can't be used as named arg
157:             npt.assert_equal(distfn.moment(1, *a, **k),
158:                              distfn.moment(1, *shape_args))
159: 
160:     # unknown arguments should not go through:
161:     k.update({'kaboom': 42})
162:     assert_raises(TypeError, distfn.cdf, x, **k)
163: 
164: 
165: def check_random_state_property(distfn, args):
166:     # check the random_state attribute of a distribution *instance*
167: 
168:     # This test fiddles with distfn.random_state. This breaks other tests,
169:     # hence need to save it and then restore.
170:     rndm = distfn.random_state
171: 
172:     # baseline: this relies on the global state
173:     np.random.seed(1234)
174:     distfn.random_state = None
175:     r0 = distfn.rvs(*args, size=8)
176: 
177:     # use an explicit instance-level random_state
178:     distfn.random_state = 1234
179:     r1 = distfn.rvs(*args, size=8)
180:     npt.assert_equal(r0, r1)
181: 
182:     distfn.random_state = np.random.RandomState(1234)
183:     r2 = distfn.rvs(*args, size=8)
184:     npt.assert_equal(r0, r2)
185: 
186:     # can override the instance-level random_state for an individual .rvs call
187:     distfn.random_state = 2
188:     orig_state = distfn.random_state.get_state()
189: 
190:     r3 = distfn.rvs(*args, size=8, random_state=np.random.RandomState(1234))
191:     npt.assert_equal(r0, r3)
192: 
193:     # ... and that does not alter the instance-level random_state!
194:     npt.assert_equal(distfn.random_state.get_state(), orig_state)
195: 
196:     # finally, restore the random_state
197:     distfn.random_state = rndm
198: 
199: 
200: def check_meth_dtype(distfn, arg, meths):
201:     q0 = [0.25, 0.5, 0.75]
202:     x0 = distfn.ppf(q0, *arg)
203:     x_cast = [x0.astype(tp) for tp in
204:                         (np.int_, np.float16, np.float32, np.float64)]
205: 
206:     for x in x_cast:
207:         # casting may have clipped the values, exclude those
208:         distfn._argcheck(*arg)
209:         x = x[(distfn.a < x) & (x < distfn.b)]
210:         for meth in meths:
211:             val = meth(x, *arg)
212:             npt.assert_(val.dtype == np.float_)
213: 
214: 
215: def check_ppf_dtype(distfn, arg):
216:     q0 = np.asarray([0.25, 0.5, 0.75])
217:     q_cast = [q0.astype(tp) for tp in (np.float16, np.float32, np.float64)]
218:     for q in q_cast:
219:         for meth in [distfn.ppf, distfn.isf]:
220:             val = meth(q, *arg)
221:             npt.assert_(val.dtype == np.float_)
222: 
223: 
224: def check_cmplx_deriv(distfn, arg):
225:     # Distributions allow complex arguments.
226:     def deriv(f, x, *arg):
227:         x = np.asarray(x)
228:         h = 1e-10
229:         return (f(x + h*1j, *arg)/h).imag
230: 
231:     x0 = distfn.ppf([0.25, 0.51, 0.75], *arg)
232:     x_cast = [x0.astype(tp) for tp in
233:                         (np.int_, np.float16, np.float32, np.float64)]
234: 
235:     for x in x_cast:
236:         # casting may have clipped the values, exclude those
237:         distfn._argcheck(*arg)
238:         x = x[(distfn.a < x) & (x < distfn.b)]
239: 
240:         pdf, cdf, sf = distfn.pdf(x, *arg), distfn.cdf(x, *arg), distfn.sf(x, *arg)
241:         assert_allclose(deriv(distfn.cdf, x, *arg), pdf, rtol=1e-5)
242:         assert_allclose(deriv(distfn.logcdf, x, *arg), pdf/cdf, rtol=1e-5)
243: 
244:         assert_allclose(deriv(distfn.sf, x, *arg), -pdf, rtol=1e-5)
245:         assert_allclose(deriv(distfn.logsf, x, *arg), -pdf/sf, rtol=1e-5)
246: 
247:         assert_allclose(deriv(distfn.logpdf, x, *arg), 
248:                         deriv(distfn.pdf, x, *arg) / distfn.pdf(x, *arg),
249:                         rtol=1e-5)
250: 
251: 
252: def check_pickling(distfn, args):
253:     # check that a distribution instance pickles and unpickles
254:     # pay special attention to the random_state property
255: 
256:     # save the random_state (restore later)
257:     rndm = distfn.random_state
258: 
259:     distfn.random_state = 1234
260:     distfn.rvs(*args, size=8)
261:     s = pickle.dumps(distfn)
262:     r0 = distfn.rvs(*args, size=8)
263: 
264:     unpickled = pickle.loads(s)
265:     r1 = unpickled.rvs(*args, size=8)
266:     npt.assert_equal(r0, r1)
267: 
268:     # also smoke test some methods
269:     medians = [distfn.ppf(0.5, *args), unpickled.ppf(0.5, *args)]
270:     npt.assert_equal(medians[0], medians[1])
271:     npt.assert_equal(distfn.cdf(medians[0], *args),
272:                      unpickled.cdf(medians[1], *args))
273: 
274:     # restore the random_state
275:     distfn.random_state = rndm
276: 
277: 
278: def check_rvs_broadcast(distfunc, distname, allargs, shape, shape_only, otype):
279:     np.random.seed(123)
280:     with suppress_warnings() as sup:
281:         # frechet_l and frechet_r are deprecated, so all their
282:         # methods generate DeprecationWarnings.
283:         sup.filter(category=DeprecationWarning, message=".*frechet_")
284:         sample = distfunc.rvs(*allargs)
285:         assert_equal(sample.shape, shape, "%s: rvs failed to broadcast" % distname)
286:         if not shape_only:
287:             rvs = np.vectorize(lambda *allargs: distfunc.rvs(*allargs), otypes=otype)
288:             np.random.seed(123)
289:             expected = rvs(*allargs)
290:             assert_allclose(sample, expected, rtol=1e-15)
291: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import pickle' statement (line 3)
import pickle

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'pickle', pickle, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import numpy' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_626829 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy')

if (type(import_626829) is not StypyTypeError):

    if (import_626829 != 'pyd_module'):
        __import__(import_626829)
        sys_modules_626830 = sys.modules[import_626829]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', sys_modules_626830.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy', import_626829)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy.testing' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_626831 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_626831) is not StypyTypeError):

    if (import_626831 != 'pyd_module'):
        __import__(import_626831)
        sys_modules_626832 = sys.modules[import_626831]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'npt', sys_modules_626832.module_type_store, module_type_store)
    else:
        import numpy.testing as npt

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'npt', numpy.testing, module_type_store)

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_626831)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_allclose, assert_equal' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_626833 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_626833) is not StypyTypeError):

    if (import_626833 != 'pyd_module'):
        __import__(import_626833)
        sys_modules_626834 = sys.modules[import_626833]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_626834.module_type_store, module_type_store, ['assert_allclose', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_626834, sys_modules_626834.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_allclose', 'assert_equal'], [assert_allclose, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_626833)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_626835 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat')

if (type(import_626835) is not StypyTypeError):

    if (import_626835 != 'pyd_module'):
        __import__(import_626835)
        sys_modules_626836 = sys.modules[import_626835]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat', sys_modules_626836.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_626836, sys_modules_626836.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy._lib._numpy_compat', import_626835)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from pytest import assert_raises' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_626837 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest')

if (type(import_626837) is not StypyTypeError):

    if (import_626837 != 'pyd_module'):
        __import__(import_626837)
        sys_modules_626838 = sys.modules[import_626837]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', sys_modules_626838.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_626838, sys_modules_626838.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', import_626837)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import numpy.ma.testutils' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_626839 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.ma.testutils')

if (type(import_626839) is not StypyTypeError):

    if (import_626839 != 'pyd_module'):
        __import__(import_626839)
        sys_modules_626840 = sys.modules[import_626839]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'ma_npt', sys_modules_626840.module_type_store, module_type_store)
    else:
        import numpy.ma.testutils as ma_npt

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'ma_npt', numpy.ma.testutils, module_type_store)

else:
    # Assigning a type to the variable 'numpy.ma.testutils' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.ma.testutils', import_626839)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy._lib._util import _getargspec' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_626841 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._util')

if (type(import_626841) is not StypyTypeError):

    if (import_626841 != 'pyd_module'):
        __import__(import_626841)
        sys_modules_626842 = sys.modules[import_626841]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._util', sys_modules_626842.module_type_store, module_type_store, ['getargspec_no_self'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_626842, sys_modules_626842.module_type_store, module_type_store)
    else:
        from scipy._lib._util import getargspec_no_self as _getargspec

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._util', None, module_type_store, ['getargspec_no_self'], [_getargspec])

else:
    # Assigning a type to the variable 'scipy._lib._util' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._util', import_626841)

# Adding an alias
module_type_store.add_alias('_getargspec', 'getargspec_no_self')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from scipy import stats' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_626843 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy')

if (type(import_626843) is not StypyTypeError):

    if (import_626843 != 'pyd_module'):
        __import__(import_626843)
        sys_modules_626844 = sys.modules[import_626843]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy', sys_modules_626844.module_type_store, module_type_store, ['stats'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_626844, sys_modules_626844.module_type_store, module_type_store)
    else:
        from scipy import stats

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy', None, module_type_store, ['stats'], [stats])

else:
    # Assigning a type to the variable 'scipy' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'scipy', import_626843)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')


@norecursion
def check_named_results(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 17)
    False_626845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 44), 'False')
    defaults = [False_626845]
    # Create a new context for function 'check_named_results'
    module_type_store = module_type_store.open_function_context('check_named_results', 17, 0, False)
    
    # Passed parameters checking function
    check_named_results.stypy_localization = localization
    check_named_results.stypy_type_of_self = None
    check_named_results.stypy_type_store = module_type_store
    check_named_results.stypy_function_name = 'check_named_results'
    check_named_results.stypy_param_names_list = ['res', 'attributes', 'ma']
    check_named_results.stypy_varargs_param_name = None
    check_named_results.stypy_kwargs_param_name = None
    check_named_results.stypy_call_defaults = defaults
    check_named_results.stypy_call_varargs = varargs
    check_named_results.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_named_results', ['res', 'attributes', 'ma'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_named_results', localization, ['res', 'attributes', 'ma'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_named_results(...)' code ##################

    
    
    # Call to enumerate(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'attributes' (line 18)
    attributes_626847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 29), 'attributes', False)
    # Processing the call keyword arguments (line 18)
    kwargs_626848 = {}
    # Getting the type of 'enumerate' (line 18)
    enumerate_626846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 19), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 18)
    enumerate_call_result_626849 = invoke(stypy.reporting.localization.Localization(__file__, 18, 19), enumerate_626846, *[attributes_626847], **kwargs_626848)
    
    # Testing the type of a for loop iterable (line 18)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 18, 4), enumerate_call_result_626849)
    # Getting the type of the for loop variable (line 18)
    for_loop_var_626850 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 18, 4), enumerate_call_result_626849)
    # Assigning a type to the variable 'i' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 4), for_loop_var_626850))
    # Assigning a type to the variable 'attr' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'attr', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 4), for_loop_var_626850))
    # SSA begins for a for statement (line 18)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'ma' (line 19)
    ma_626851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'ma')
    # Testing the type of an if condition (line 19)
    if_condition_626852 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 8), ma_626851)
    # Assigning a type to the variable 'if_condition_626852' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'if_condition_626852', if_condition_626852)
    # SSA begins for if statement (line 19)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_equal(...): (line 20)
    # Processing the call arguments (line 20)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 20)
    i_626855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 36), 'i', False)
    # Getting the type of 'res' (line 20)
    res_626856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 32), 'res', False)
    # Obtaining the member '__getitem__' of a type (line 20)
    getitem___626857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 32), res_626856, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 20)
    subscript_call_result_626858 = invoke(stypy.reporting.localization.Localization(__file__, 20, 32), getitem___626857, i_626855)
    
    
    # Call to getattr(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'res' (line 20)
    res_626860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 48), 'res', False)
    # Getting the type of 'attr' (line 20)
    attr_626861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 53), 'attr', False)
    # Processing the call keyword arguments (line 20)
    kwargs_626862 = {}
    # Getting the type of 'getattr' (line 20)
    getattr_626859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 40), 'getattr', False)
    # Calling getattr(args, kwargs) (line 20)
    getattr_call_result_626863 = invoke(stypy.reporting.localization.Localization(__file__, 20, 40), getattr_626859, *[res_626860, attr_626861], **kwargs_626862)
    
    # Processing the call keyword arguments (line 20)
    kwargs_626864 = {}
    # Getting the type of 'ma_npt' (line 20)
    ma_npt_626853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'ma_npt', False)
    # Obtaining the member 'assert_equal' of a type (line 20)
    assert_equal_626854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), ma_npt_626853, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 20)
    assert_equal_call_result_626865 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), assert_equal_626854, *[subscript_call_result_626858, getattr_call_result_626863], **kwargs_626864)
    
    # SSA branch for the else part of an if statement (line 19)
    module_type_store.open_ssa_branch('else')
    
    # Call to assert_equal(...): (line 22)
    # Processing the call arguments (line 22)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 22)
    i_626868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 33), 'i', False)
    # Getting the type of 'res' (line 22)
    res_626869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 29), 'res', False)
    # Obtaining the member '__getitem__' of a type (line 22)
    getitem___626870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 29), res_626869, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 22)
    subscript_call_result_626871 = invoke(stypy.reporting.localization.Localization(__file__, 22, 29), getitem___626870, i_626868)
    
    
    # Call to getattr(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'res' (line 22)
    res_626873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 45), 'res', False)
    # Getting the type of 'attr' (line 22)
    attr_626874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 50), 'attr', False)
    # Processing the call keyword arguments (line 22)
    kwargs_626875 = {}
    # Getting the type of 'getattr' (line 22)
    getattr_626872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 37), 'getattr', False)
    # Calling getattr(args, kwargs) (line 22)
    getattr_call_result_626876 = invoke(stypy.reporting.localization.Localization(__file__, 22, 37), getattr_626872, *[res_626873, attr_626874], **kwargs_626875)
    
    # Processing the call keyword arguments (line 22)
    kwargs_626877 = {}
    # Getting the type of 'npt' (line 22)
    npt_626866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'npt', False)
    # Obtaining the member 'assert_equal' of a type (line 22)
    assert_equal_626867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 12), npt_626866, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 22)
    assert_equal_call_result_626878 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), assert_equal_626867, *[subscript_call_result_626871, getattr_call_result_626876], **kwargs_626877)
    
    # SSA join for if statement (line 19)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_named_results(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_named_results' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_626879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_626879)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_named_results'
    return stypy_return_type_626879

# Assigning a type to the variable 'check_named_results' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'check_named_results', check_named_results)

@norecursion
def check_normalization(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_normalization'
    module_type_store = module_type_store.open_function_context('check_normalization', 25, 0, False)
    
    # Passed parameters checking function
    check_normalization.stypy_localization = localization
    check_normalization.stypy_type_of_self = None
    check_normalization.stypy_type_store = module_type_store
    check_normalization.stypy_function_name = 'check_normalization'
    check_normalization.stypy_param_names_list = ['distfn', 'args', 'distname']
    check_normalization.stypy_varargs_param_name = None
    check_normalization.stypy_kwargs_param_name = None
    check_normalization.stypy_call_defaults = defaults
    check_normalization.stypy_call_varargs = varargs
    check_normalization.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_normalization', ['distfn', 'args', 'distname'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_normalization', localization, ['distfn', 'args', 'distname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_normalization(...)' code ##################

    
    # Assigning a Call to a Name (line 26):
    
    # Assigning a Call to a Name (line 26):
    
    # Call to moment(...): (line 26)
    # Processing the call arguments (line 26)
    int_626882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 32), 'int')
    # Getting the type of 'args' (line 26)
    args_626883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 36), 'args', False)
    # Processing the call keyword arguments (line 26)
    kwargs_626884 = {}
    # Getting the type of 'distfn' (line 26)
    distfn_626880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'distfn', False)
    # Obtaining the member 'moment' of a type (line 26)
    moment_626881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 18), distfn_626880, 'moment')
    # Calling moment(args, kwargs) (line 26)
    moment_call_result_626885 = invoke(stypy.reporting.localization.Localization(__file__, 26, 18), moment_626881, *[int_626882, args_626883], **kwargs_626884)
    
    # Assigning a type to the variable 'norm_moment' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'norm_moment', moment_call_result_626885)
    
    # Call to assert_allclose(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'norm_moment' (line 27)
    norm_moment_626888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 24), 'norm_moment', False)
    float_626889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 37), 'float')
    # Processing the call keyword arguments (line 27)
    kwargs_626890 = {}
    # Getting the type of 'npt' (line 27)
    npt_626886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'npt', False)
    # Obtaining the member 'assert_allclose' of a type (line 27)
    assert_allclose_626887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 4), npt_626886, 'assert_allclose')
    # Calling assert_allclose(args, kwargs) (line 27)
    assert_allclose_call_result_626891 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), assert_allclose_626887, *[norm_moment_626888, float_626889], **kwargs_626890)
    
    
    
    # Getting the type of 'distname' (line 31)
    distname_626892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 7), 'distname')
    str_626893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 19), 'str', 'ncf')
    # Applying the binary operator '==' (line 31)
    result_eq_626894 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 7), '==', distname_626892, str_626893)
    
    # Testing the type of an if condition (line 31)
    if_condition_626895 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 4), result_eq_626894)
    # Assigning a type to the variable 'if_condition_626895' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'if_condition_626895', if_condition_626895)
    # SSA begins for if statement (line 31)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Tuple (line 32):
    
    # Assigning a Num to a Name (line 32):
    float_626896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 21), 'float')
    # Assigning a type to the variable 'tuple_assignment_626819' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'tuple_assignment_626819', float_626896)
    
    # Assigning a Num to a Name (line 32):
    int_626897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 27), 'int')
    # Assigning a type to the variable 'tuple_assignment_626820' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'tuple_assignment_626820', int_626897)
    
    # Assigning a Name to a Name (line 32):
    # Getting the type of 'tuple_assignment_626819' (line 32)
    tuple_assignment_626819_626898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'tuple_assignment_626819')
    # Assigning a type to the variable 'atol' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'atol', tuple_assignment_626819_626898)
    
    # Assigning a Name to a Name (line 32):
    # Getting the type of 'tuple_assignment_626820' (line 32)
    tuple_assignment_626820_626899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'tuple_assignment_626820')
    # Assigning a type to the variable 'rtol' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 14), 'rtol', tuple_assignment_626820_626899)
    # SSA branch for the else part of an if statement (line 31)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Tuple (line 34):
    
    # Assigning a Num to a Name (line 34):
    float_626900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 21), 'float')
    # Assigning a type to the variable 'tuple_assignment_626821' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_assignment_626821', float_626900)
    
    # Assigning a Num to a Name (line 34):
    float_626901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 27), 'float')
    # Assigning a type to the variable 'tuple_assignment_626822' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_assignment_626822', float_626901)
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'tuple_assignment_626821' (line 34)
    tuple_assignment_626821_626902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_assignment_626821')
    # Assigning a type to the variable 'atol' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'atol', tuple_assignment_626821_626902)
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'tuple_assignment_626822' (line 34)
    tuple_assignment_626822_626903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_assignment_626822')
    # Assigning a type to the variable 'rtol' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 14), 'rtol', tuple_assignment_626822_626903)
    # SSA join for if statement (line 31)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 36):
    
    # Assigning a Call to a Name (line 36):
    
    # Call to expect(...): (line 36)
    # Processing the call arguments (line 36)

    @norecursion
    def _stypy_temp_lambda_531(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_531'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_531', 36, 41, True)
        # Passed parameters checking function
        _stypy_temp_lambda_531.stypy_localization = localization
        _stypy_temp_lambda_531.stypy_type_of_self = None
        _stypy_temp_lambda_531.stypy_type_store = module_type_store
        _stypy_temp_lambda_531.stypy_function_name = '_stypy_temp_lambda_531'
        _stypy_temp_lambda_531.stypy_param_names_list = ['x']
        _stypy_temp_lambda_531.stypy_varargs_param_name = None
        _stypy_temp_lambda_531.stypy_kwargs_param_name = None
        _stypy_temp_lambda_531.stypy_call_defaults = defaults
        _stypy_temp_lambda_531.stypy_call_varargs = varargs
        _stypy_temp_lambda_531.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_531', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_531', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        int_626906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 51), 'int')
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 41), 'stypy_return_type', int_626906)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_531' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_626907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 41), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_626907)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_531'
        return stypy_return_type_626907

    # Assigning a type to the variable '_stypy_temp_lambda_531' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 41), '_stypy_temp_lambda_531', _stypy_temp_lambda_531)
    # Getting the type of '_stypy_temp_lambda_531' (line 36)
    _stypy_temp_lambda_531_626908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 41), '_stypy_temp_lambda_531')
    # Processing the call keyword arguments (line 36)
    # Getting the type of 'args' (line 36)
    args_626909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 59), 'args', False)
    keyword_626910 = args_626909
    kwargs_626911 = {'args': keyword_626910}
    # Getting the type of 'distfn' (line 36)
    distfn_626904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'distfn', False)
    # Obtaining the member 'expect' of a type (line 36)
    expect_626905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 27), distfn_626904, 'expect')
    # Calling expect(args, kwargs) (line 36)
    expect_call_result_626912 = invoke(stypy.reporting.localization.Localization(__file__, 36, 27), expect_626905, *[_stypy_temp_lambda_531_626908], **kwargs_626911)
    
    # Assigning a type to the variable 'normalization_expect' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'normalization_expect', expect_call_result_626912)
    
    # Call to assert_allclose(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'normalization_expect' (line 37)
    normalization_expect_626915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 24), 'normalization_expect', False)
    float_626916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 46), 'float')
    # Processing the call keyword arguments (line 37)
    # Getting the type of 'atol' (line 37)
    atol_626917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 56), 'atol', False)
    keyword_626918 = atol_626917
    # Getting the type of 'rtol' (line 37)
    rtol_626919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 67), 'rtol', False)
    keyword_626920 = rtol_626919
    # Getting the type of 'distname' (line 38)
    distname_626921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 20), 'distname', False)
    keyword_626922 = distname_626921
    # Getting the type of 'True' (line 38)
    True_626923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 38), 'True', False)
    keyword_626924 = True_626923
    kwargs_626925 = {'rtol': keyword_626920, 'err_msg': keyword_626922, 'verbose': keyword_626924, 'atol': keyword_626918}
    # Getting the type of 'npt' (line 37)
    npt_626913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'npt', False)
    # Obtaining the member 'assert_allclose' of a type (line 37)
    assert_allclose_626914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 4), npt_626913, 'assert_allclose')
    # Calling assert_allclose(args, kwargs) (line 37)
    assert_allclose_call_result_626926 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), assert_allclose_626914, *[normalization_expect_626915, float_626916], **kwargs_626925)
    
    
    # Assigning a Call to a Name (line 40):
    
    # Assigning a Call to a Name (line 40):
    
    # Call to cdf(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'distfn' (line 40)
    distfn_626929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 35), 'distfn', False)
    # Obtaining the member 'b' of a type (line 40)
    b_626930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 35), distfn_626929, 'b')
    # Getting the type of 'args' (line 40)
    args_626931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 46), 'args', False)
    # Processing the call keyword arguments (line 40)
    kwargs_626932 = {}
    # Getting the type of 'distfn' (line 40)
    distfn_626927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'distfn', False)
    # Obtaining the member 'cdf' of a type (line 40)
    cdf_626928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 24), distfn_626927, 'cdf')
    # Calling cdf(args, kwargs) (line 40)
    cdf_call_result_626933 = invoke(stypy.reporting.localization.Localization(__file__, 40, 24), cdf_626928, *[b_626930, args_626931], **kwargs_626932)
    
    # Assigning a type to the variable 'normalization_cdf' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'normalization_cdf', cdf_call_result_626933)
    
    # Call to assert_allclose(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'normalization_cdf' (line 41)
    normalization_cdf_626936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), 'normalization_cdf', False)
    float_626937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 43), 'float')
    # Processing the call keyword arguments (line 41)
    kwargs_626938 = {}
    # Getting the type of 'npt' (line 41)
    npt_626934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'npt', False)
    # Obtaining the member 'assert_allclose' of a type (line 41)
    assert_allclose_626935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), npt_626934, 'assert_allclose')
    # Calling assert_allclose(args, kwargs) (line 41)
    assert_allclose_call_result_626939 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), assert_allclose_626935, *[normalization_cdf_626936, float_626937], **kwargs_626938)
    
    
    # ################# End of 'check_normalization(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_normalization' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_626940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_626940)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_normalization'
    return stypy_return_type_626940

# Assigning a type to the variable 'check_normalization' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'check_normalization', check_normalization)

@norecursion
def check_moment(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_moment'
    module_type_store = module_type_store.open_function_context('check_moment', 44, 0, False)
    
    # Passed parameters checking function
    check_moment.stypy_localization = localization
    check_moment.stypy_type_of_self = None
    check_moment.stypy_type_store = module_type_store
    check_moment.stypy_function_name = 'check_moment'
    check_moment.stypy_param_names_list = ['distfn', 'arg', 'm', 'v', 'msg']
    check_moment.stypy_varargs_param_name = None
    check_moment.stypy_kwargs_param_name = None
    check_moment.stypy_call_defaults = defaults
    check_moment.stypy_call_varargs = varargs
    check_moment.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_moment', ['distfn', 'arg', 'm', 'v', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_moment', localization, ['distfn', 'arg', 'm', 'v', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_moment(...)' code ##################

    
    # Assigning a Call to a Name (line 45):
    
    # Assigning a Call to a Name (line 45):
    
    # Call to moment(...): (line 45)
    # Processing the call arguments (line 45)
    int_626943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 23), 'int')
    # Getting the type of 'arg' (line 45)
    arg_626944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 27), 'arg', False)
    # Processing the call keyword arguments (line 45)
    kwargs_626945 = {}
    # Getting the type of 'distfn' (line 45)
    distfn_626941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 9), 'distfn', False)
    # Obtaining the member 'moment' of a type (line 45)
    moment_626942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 9), distfn_626941, 'moment')
    # Calling moment(args, kwargs) (line 45)
    moment_call_result_626946 = invoke(stypy.reporting.localization.Localization(__file__, 45, 9), moment_626942, *[int_626943, arg_626944], **kwargs_626945)
    
    # Assigning a type to the variable 'm1' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'm1', moment_call_result_626946)
    
    # Assigning a Call to a Name (line 46):
    
    # Assigning a Call to a Name (line 46):
    
    # Call to moment(...): (line 46)
    # Processing the call arguments (line 46)
    int_626949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 23), 'int')
    # Getting the type of 'arg' (line 46)
    arg_626950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 27), 'arg', False)
    # Processing the call keyword arguments (line 46)
    kwargs_626951 = {}
    # Getting the type of 'distfn' (line 46)
    distfn_626947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 9), 'distfn', False)
    # Obtaining the member 'moment' of a type (line 46)
    moment_626948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 9), distfn_626947, 'moment')
    # Calling moment(args, kwargs) (line 46)
    moment_call_result_626952 = invoke(stypy.reporting.localization.Localization(__file__, 46, 9), moment_626948, *[int_626949, arg_626950], **kwargs_626951)
    
    # Assigning a type to the variable 'm2' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'm2', moment_call_result_626952)
    
    
    
    # Call to isinf(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'm' (line 47)
    m_626955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 20), 'm', False)
    # Processing the call keyword arguments (line 47)
    kwargs_626956 = {}
    # Getting the type of 'np' (line 47)
    np_626953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'np', False)
    # Obtaining the member 'isinf' of a type (line 47)
    isinf_626954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), np_626953, 'isinf')
    # Calling isinf(args, kwargs) (line 47)
    isinf_call_result_626957 = invoke(stypy.reporting.localization.Localization(__file__, 47, 11), isinf_626954, *[m_626955], **kwargs_626956)
    
    # Applying the 'not' unary operator (line 47)
    result_not__626958 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 7), 'not', isinf_call_result_626957)
    
    # Testing the type of an if condition (line 47)
    if_condition_626959 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 4), result_not__626958)
    # Assigning a type to the variable 'if_condition_626959' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'if_condition_626959', if_condition_626959)
    # SSA begins for if statement (line 47)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_almost_equal(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'm1' (line 48)
    m1_626962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 32), 'm1', False)
    # Getting the type of 'm' (line 48)
    m_626963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 36), 'm', False)
    # Processing the call keyword arguments (line 48)
    int_626964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 47), 'int')
    keyword_626965 = int_626964
    # Getting the type of 'msg' (line 48)
    msg_626966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 59), 'msg', False)
    str_626967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 28), 'str', ' - 1st moment')
    # Applying the binary operator '+' (line 48)
    result_add_626968 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 59), '+', msg_626966, str_626967)
    
    keyword_626969 = result_add_626968
    kwargs_626970 = {'decimal': keyword_626965, 'err_msg': keyword_626969}
    # Getting the type of 'npt' (line 48)
    npt_626960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'npt', False)
    # Obtaining the member 'assert_almost_equal' of a type (line 48)
    assert_almost_equal_626961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), npt_626960, 'assert_almost_equal')
    # Calling assert_almost_equal(args, kwargs) (line 48)
    assert_almost_equal_call_result_626971 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), assert_almost_equal_626961, *[m1_626962, m_626963], **kwargs_626970)
    
    # SSA branch for the else part of an if statement (line 47)
    module_type_store.open_ssa_branch('else')
    
    # Call to assert_(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Call to isinf(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'm1' (line 51)
    m1_626976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 29), 'm1', False)
    # Processing the call keyword arguments (line 51)
    kwargs_626977 = {}
    # Getting the type of 'np' (line 51)
    np_626974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'np', False)
    # Obtaining the member 'isinf' of a type (line 51)
    isinf_626975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 20), np_626974, 'isinf')
    # Calling isinf(args, kwargs) (line 51)
    isinf_call_result_626978 = invoke(stypy.reporting.localization.Localization(__file__, 51, 20), isinf_626975, *[m1_626976], **kwargs_626977)
    
    # Getting the type of 'msg' (line 52)
    msg_626979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'msg', False)
    str_626980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 21), 'str', ' - 1st moment -infinite, m1=%s')
    
    # Call to str(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'm1' (line 52)
    m1_626982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 60), 'm1', False)
    # Processing the call keyword arguments (line 52)
    kwargs_626983 = {}
    # Getting the type of 'str' (line 52)
    str_626981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 56), 'str', False)
    # Calling str(args, kwargs) (line 52)
    str_call_result_626984 = invoke(stypy.reporting.localization.Localization(__file__, 52, 56), str_626981, *[m1_626982], **kwargs_626983)
    
    # Applying the binary operator '%' (line 52)
    result_mod_626985 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 21), '%', str_626980, str_call_result_626984)
    
    # Applying the binary operator '+' (line 52)
    result_add_626986 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 15), '+', msg_626979, result_mod_626985)
    
    # Processing the call keyword arguments (line 51)
    kwargs_626987 = {}
    # Getting the type of 'npt' (line 51)
    npt_626972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 51)
    assert__626973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), npt_626972, 'assert_')
    # Calling assert_(args, kwargs) (line 51)
    assert__call_result_626988 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), assert__626973, *[isinf_call_result_626978, result_add_626986], **kwargs_626987)
    
    # SSA join for if statement (line 47)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to isinf(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'v' (line 54)
    v_626991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'v', False)
    # Processing the call keyword arguments (line 54)
    kwargs_626992 = {}
    # Getting the type of 'np' (line 54)
    np_626989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'np', False)
    # Obtaining the member 'isinf' of a type (line 54)
    isinf_626990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 11), np_626989, 'isinf')
    # Calling isinf(args, kwargs) (line 54)
    isinf_call_result_626993 = invoke(stypy.reporting.localization.Localization(__file__, 54, 11), isinf_626990, *[v_626991], **kwargs_626992)
    
    # Applying the 'not' unary operator (line 54)
    result_not__626994 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 7), 'not', isinf_call_result_626993)
    
    # Testing the type of an if condition (line 54)
    if_condition_626995 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 4), result_not__626994)
    # Assigning a type to the variable 'if_condition_626995' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'if_condition_626995', if_condition_626995)
    # SSA begins for if statement (line 54)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_almost_equal(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'm2' (line 55)
    m2_626998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 32), 'm2', False)
    # Getting the type of 'm1' (line 55)
    m1_626999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 37), 'm1', False)
    # Getting the type of 'm1' (line 55)
    m1_627000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 42), 'm1', False)
    # Applying the binary operator '*' (line 55)
    result_mul_627001 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 37), '*', m1_626999, m1_627000)
    
    # Applying the binary operator '-' (line 55)
    result_sub_627002 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 32), '-', m2_626998, result_mul_627001)
    
    # Getting the type of 'v' (line 55)
    v_627003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 46), 'v', False)
    # Processing the call keyword arguments (line 55)
    int_627004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 57), 'int')
    keyword_627005 = int_627004
    # Getting the type of 'msg' (line 55)
    msg_627006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 69), 'msg', False)
    str_627007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 28), 'str', ' - 2ndt moment')
    # Applying the binary operator '+' (line 55)
    result_add_627008 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 69), '+', msg_627006, str_627007)
    
    keyword_627009 = result_add_627008
    kwargs_627010 = {'decimal': keyword_627005, 'err_msg': keyword_627009}
    # Getting the type of 'npt' (line 55)
    npt_626996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'npt', False)
    # Obtaining the member 'assert_almost_equal' of a type (line 55)
    assert_almost_equal_626997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), npt_626996, 'assert_almost_equal')
    # Calling assert_almost_equal(args, kwargs) (line 55)
    assert_almost_equal_call_result_627011 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), assert_almost_equal_626997, *[result_sub_627002, v_627003], **kwargs_627010)
    
    # SSA branch for the else part of an if statement (line 54)
    module_type_store.open_ssa_branch('else')
    
    # Call to assert_(...): (line 58)
    # Processing the call arguments (line 58)
    
    # Call to isinf(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'm2' (line 58)
    m2_627016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 29), 'm2', False)
    # Processing the call keyword arguments (line 58)
    kwargs_627017 = {}
    # Getting the type of 'np' (line 58)
    np_627014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'np', False)
    # Obtaining the member 'isinf' of a type (line 58)
    isinf_627015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 20), np_627014, 'isinf')
    # Calling isinf(args, kwargs) (line 58)
    isinf_call_result_627018 = invoke(stypy.reporting.localization.Localization(__file__, 58, 20), isinf_627015, *[m2_627016], **kwargs_627017)
    
    # Getting the type of 'msg' (line 59)
    msg_627019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'msg', False)
    str_627020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 21), 'str', ' - 2nd moment -infinite, m2=%s')
    
    # Call to str(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'm2' (line 59)
    m2_627022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 60), 'm2', False)
    # Processing the call keyword arguments (line 59)
    kwargs_627023 = {}
    # Getting the type of 'str' (line 59)
    str_627021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 56), 'str', False)
    # Calling str(args, kwargs) (line 59)
    str_call_result_627024 = invoke(stypy.reporting.localization.Localization(__file__, 59, 56), str_627021, *[m2_627022], **kwargs_627023)
    
    # Applying the binary operator '%' (line 59)
    result_mod_627025 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 21), '%', str_627020, str_call_result_627024)
    
    # Applying the binary operator '+' (line 59)
    result_add_627026 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 15), '+', msg_627019, result_mod_627025)
    
    # Processing the call keyword arguments (line 58)
    kwargs_627027 = {}
    # Getting the type of 'npt' (line 58)
    npt_627012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 58)
    assert__627013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), npt_627012, 'assert_')
    # Calling assert_(args, kwargs) (line 58)
    assert__call_result_627028 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), assert__627013, *[isinf_call_result_627018, result_add_627026], **kwargs_627027)
    
    # SSA join for if statement (line 54)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_moment(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_moment' in the type store
    # Getting the type of 'stypy_return_type' (line 44)
    stypy_return_type_627029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_627029)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_moment'
    return stypy_return_type_627029

# Assigning a type to the variable 'check_moment' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'check_moment', check_moment)

@norecursion
def check_mean_expect(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_mean_expect'
    module_type_store = module_type_store.open_function_context('check_mean_expect', 62, 0, False)
    
    # Passed parameters checking function
    check_mean_expect.stypy_localization = localization
    check_mean_expect.stypy_type_of_self = None
    check_mean_expect.stypy_type_store = module_type_store
    check_mean_expect.stypy_function_name = 'check_mean_expect'
    check_mean_expect.stypy_param_names_list = ['distfn', 'arg', 'm', 'msg']
    check_mean_expect.stypy_varargs_param_name = None
    check_mean_expect.stypy_kwargs_param_name = None
    check_mean_expect.stypy_call_defaults = defaults
    check_mean_expect.stypy_call_varargs = varargs
    check_mean_expect.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_mean_expect', ['distfn', 'arg', 'm', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_mean_expect', localization, ['distfn', 'arg', 'm', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_mean_expect(...)' code ##################

    
    
    # Call to isfinite(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'm' (line 63)
    m_627032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'm', False)
    # Processing the call keyword arguments (line 63)
    kwargs_627033 = {}
    # Getting the type of 'np' (line 63)
    np_627030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 7), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 63)
    isfinite_627031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 7), np_627030, 'isfinite')
    # Calling isfinite(args, kwargs) (line 63)
    isfinite_call_result_627034 = invoke(stypy.reporting.localization.Localization(__file__, 63, 7), isfinite_627031, *[m_627032], **kwargs_627033)
    
    # Testing the type of an if condition (line 63)
    if_condition_627035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 4), isfinite_call_result_627034)
    # Assigning a type to the variable 'if_condition_627035' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'if_condition_627035', if_condition_627035)
    # SSA begins for if statement (line 63)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 64):
    
    # Assigning a Call to a Name (line 64):
    
    # Call to expect(...): (line 64)
    # Processing the call arguments (line 64)

    @norecursion
    def _stypy_temp_lambda_532(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_532'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_532', 64, 27, True)
        # Passed parameters checking function
        _stypy_temp_lambda_532.stypy_localization = localization
        _stypy_temp_lambda_532.stypy_type_of_self = None
        _stypy_temp_lambda_532.stypy_type_store = module_type_store
        _stypy_temp_lambda_532.stypy_function_name = '_stypy_temp_lambda_532'
        _stypy_temp_lambda_532.stypy_param_names_list = ['x']
        _stypy_temp_lambda_532.stypy_varargs_param_name = None
        _stypy_temp_lambda_532.stypy_kwargs_param_name = None
        _stypy_temp_lambda_532.stypy_call_defaults = defaults
        _stypy_temp_lambda_532.stypy_call_varargs = varargs
        _stypy_temp_lambda_532.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_532', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_532', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'x' (line 64)
        x_627038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 37), 'x', False)
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 27), 'stypy_return_type', x_627038)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_532' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_627039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 27), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_627039)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_532'
        return stypy_return_type_627039

    # Assigning a type to the variable '_stypy_temp_lambda_532' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 27), '_stypy_temp_lambda_532', _stypy_temp_lambda_532)
    # Getting the type of '_stypy_temp_lambda_532' (line 64)
    _stypy_temp_lambda_532_627040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 27), '_stypy_temp_lambda_532')
    # Getting the type of 'arg' (line 64)
    arg_627041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 40), 'arg', False)
    # Processing the call keyword arguments (line 64)
    kwargs_627042 = {}
    # Getting the type of 'distfn' (line 64)
    distfn_627036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 13), 'distfn', False)
    # Obtaining the member 'expect' of a type (line 64)
    expect_627037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 13), distfn_627036, 'expect')
    # Calling expect(args, kwargs) (line 64)
    expect_call_result_627043 = invoke(stypy.reporting.localization.Localization(__file__, 64, 13), expect_627037, *[_stypy_temp_lambda_532_627040, arg_627041], **kwargs_627042)
    
    # Assigning a type to the variable 'm1' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'm1', expect_call_result_627043)
    
    # Call to assert_almost_equal(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'm1' (line 65)
    m1_627046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 32), 'm1', False)
    # Getting the type of 'm' (line 65)
    m_627047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 36), 'm', False)
    # Processing the call keyword arguments (line 65)
    int_627048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 47), 'int')
    keyword_627049 = int_627048
    # Getting the type of 'msg' (line 65)
    msg_627050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 58), 'msg', False)
    str_627051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 28), 'str', ' - 1st moment (expect)')
    # Applying the binary operator '+' (line 65)
    result_add_627052 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 58), '+', msg_627050, str_627051)
    
    keyword_627053 = result_add_627052
    kwargs_627054 = {'decimal': keyword_627049, 'err_msg': keyword_627053}
    # Getting the type of 'npt' (line 65)
    npt_627044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'npt', False)
    # Obtaining the member 'assert_almost_equal' of a type (line 65)
    assert_almost_equal_627045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), npt_627044, 'assert_almost_equal')
    # Calling assert_almost_equal(args, kwargs) (line 65)
    assert_almost_equal_call_result_627055 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), assert_almost_equal_627045, *[m1_627046, m_627047], **kwargs_627054)
    
    # SSA join for if statement (line 63)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_mean_expect(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_mean_expect' in the type store
    # Getting the type of 'stypy_return_type' (line 62)
    stypy_return_type_627056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_627056)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_mean_expect'
    return stypy_return_type_627056

# Assigning a type to the variable 'check_mean_expect' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'check_mean_expect', check_mean_expect)

@norecursion
def check_var_expect(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_var_expect'
    module_type_store = module_type_store.open_function_context('check_var_expect', 69, 0, False)
    
    # Passed parameters checking function
    check_var_expect.stypy_localization = localization
    check_var_expect.stypy_type_of_self = None
    check_var_expect.stypy_type_store = module_type_store
    check_var_expect.stypy_function_name = 'check_var_expect'
    check_var_expect.stypy_param_names_list = ['distfn', 'arg', 'm', 'v', 'msg']
    check_var_expect.stypy_varargs_param_name = None
    check_var_expect.stypy_kwargs_param_name = None
    check_var_expect.stypy_call_defaults = defaults
    check_var_expect.stypy_call_varargs = varargs
    check_var_expect.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_var_expect', ['distfn', 'arg', 'm', 'v', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_var_expect', localization, ['distfn', 'arg', 'm', 'v', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_var_expect(...)' code ##################

    
    
    # Call to isfinite(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'v' (line 70)
    v_627059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'v', False)
    # Processing the call keyword arguments (line 70)
    kwargs_627060 = {}
    # Getting the type of 'np' (line 70)
    np_627057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 7), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 70)
    isfinite_627058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 7), np_627057, 'isfinite')
    # Calling isfinite(args, kwargs) (line 70)
    isfinite_call_result_627061 = invoke(stypy.reporting.localization.Localization(__file__, 70, 7), isfinite_627058, *[v_627059], **kwargs_627060)
    
    # Testing the type of an if condition (line 70)
    if_condition_627062 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 4), isfinite_call_result_627061)
    # Assigning a type to the variable 'if_condition_627062' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'if_condition_627062', if_condition_627062)
    # SSA begins for if statement (line 70)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 71):
    
    # Assigning a Call to a Name (line 71):
    
    # Call to expect(...): (line 71)
    # Processing the call arguments (line 71)

    @norecursion
    def _stypy_temp_lambda_533(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_533'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_533', 71, 27, True)
        # Passed parameters checking function
        _stypy_temp_lambda_533.stypy_localization = localization
        _stypy_temp_lambda_533.stypy_type_of_self = None
        _stypy_temp_lambda_533.stypy_type_store = module_type_store
        _stypy_temp_lambda_533.stypy_function_name = '_stypy_temp_lambda_533'
        _stypy_temp_lambda_533.stypy_param_names_list = ['x']
        _stypy_temp_lambda_533.stypy_varargs_param_name = None
        _stypy_temp_lambda_533.stypy_kwargs_param_name = None
        _stypy_temp_lambda_533.stypy_call_defaults = defaults
        _stypy_temp_lambda_533.stypy_call_varargs = varargs
        _stypy_temp_lambda_533.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_533', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_533', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'x' (line 71)
        x_627065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 37), 'x', False)
        # Getting the type of 'x' (line 71)
        x_627066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 39), 'x', False)
        # Applying the binary operator '*' (line 71)
        result_mul_627067 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 37), '*', x_627065, x_627066)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 'stypy_return_type', result_mul_627067)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_533' in the type store
        # Getting the type of 'stypy_return_type' (line 71)
        stypy_return_type_627068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_627068)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_533'
        return stypy_return_type_627068

    # Assigning a type to the variable '_stypy_temp_lambda_533' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), '_stypy_temp_lambda_533', _stypy_temp_lambda_533)
    # Getting the type of '_stypy_temp_lambda_533' (line 71)
    _stypy_temp_lambda_533_627069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), '_stypy_temp_lambda_533')
    # Getting the type of 'arg' (line 71)
    arg_627070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 42), 'arg', False)
    # Processing the call keyword arguments (line 71)
    kwargs_627071 = {}
    # Getting the type of 'distfn' (line 71)
    distfn_627063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 13), 'distfn', False)
    # Obtaining the member 'expect' of a type (line 71)
    expect_627064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 13), distfn_627063, 'expect')
    # Calling expect(args, kwargs) (line 71)
    expect_call_result_627072 = invoke(stypy.reporting.localization.Localization(__file__, 71, 13), expect_627064, *[_stypy_temp_lambda_533_627069, arg_627070], **kwargs_627071)
    
    # Assigning a type to the variable 'm2' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'm2', expect_call_result_627072)
    
    # Call to assert_almost_equal(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'm2' (line 72)
    m2_627075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'm2', False)
    # Getting the type of 'v' (line 72)
    v_627076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 36), 'v', False)
    # Getting the type of 'm' (line 72)
    m_627077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 40), 'm', False)
    # Getting the type of 'm' (line 72)
    m_627078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 42), 'm', False)
    # Applying the binary operator '*' (line 72)
    result_mul_627079 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 40), '*', m_627077, m_627078)
    
    # Applying the binary operator '+' (line 72)
    result_add_627080 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 36), '+', v_627076, result_mul_627079)
    
    # Processing the call keyword arguments (line 72)
    int_627081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 53), 'int')
    keyword_627082 = int_627081
    # Getting the type of 'msg' (line 72)
    msg_627083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 64), 'msg', False)
    str_627084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 28), 'str', ' - 2st moment (expect)')
    # Applying the binary operator '+' (line 72)
    result_add_627085 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 64), '+', msg_627083, str_627084)
    
    keyword_627086 = result_add_627085
    kwargs_627087 = {'decimal': keyword_627082, 'err_msg': keyword_627086}
    # Getting the type of 'npt' (line 72)
    npt_627073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'npt', False)
    # Obtaining the member 'assert_almost_equal' of a type (line 72)
    assert_almost_equal_627074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), npt_627073, 'assert_almost_equal')
    # Calling assert_almost_equal(args, kwargs) (line 72)
    assert_almost_equal_call_result_627088 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), assert_almost_equal_627074, *[m2_627075, result_add_627080], **kwargs_627087)
    
    # SSA join for if statement (line 70)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_var_expect(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_var_expect' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_627089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_627089)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_var_expect'
    return stypy_return_type_627089

# Assigning a type to the variable 'check_var_expect' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'check_var_expect', check_var_expect)

@norecursion
def check_skew_expect(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_skew_expect'
    module_type_store = module_type_store.open_function_context('check_skew_expect', 76, 0, False)
    
    # Passed parameters checking function
    check_skew_expect.stypy_localization = localization
    check_skew_expect.stypy_type_of_self = None
    check_skew_expect.stypy_type_store = module_type_store
    check_skew_expect.stypy_function_name = 'check_skew_expect'
    check_skew_expect.stypy_param_names_list = ['distfn', 'arg', 'm', 'v', 's', 'msg']
    check_skew_expect.stypy_varargs_param_name = None
    check_skew_expect.stypy_kwargs_param_name = None
    check_skew_expect.stypy_call_defaults = defaults
    check_skew_expect.stypy_call_varargs = varargs
    check_skew_expect.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_skew_expect', ['distfn', 'arg', 'm', 'v', 's', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_skew_expect', localization, ['distfn', 'arg', 'm', 'v', 's', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_skew_expect(...)' code ##################

    
    
    # Call to isfinite(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 's' (line 77)
    s_627092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 's', False)
    # Processing the call keyword arguments (line 77)
    kwargs_627093 = {}
    # Getting the type of 'np' (line 77)
    np_627090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 7), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 77)
    isfinite_627091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 7), np_627090, 'isfinite')
    # Calling isfinite(args, kwargs) (line 77)
    isfinite_call_result_627094 = invoke(stypy.reporting.localization.Localization(__file__, 77, 7), isfinite_627091, *[s_627092], **kwargs_627093)
    
    # Testing the type of an if condition (line 77)
    if_condition_627095 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 4), isfinite_call_result_627094)
    # Assigning a type to the variable 'if_condition_627095' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'if_condition_627095', if_condition_627095)
    # SSA begins for if statement (line 77)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 78):
    
    # Assigning a Call to a Name (line 78):
    
    # Call to expect(...): (line 78)
    # Processing the call arguments (line 78)

    @norecursion
    def _stypy_temp_lambda_534(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_534'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_534', 78, 28, True)
        # Passed parameters checking function
        _stypy_temp_lambda_534.stypy_localization = localization
        _stypy_temp_lambda_534.stypy_type_of_self = None
        _stypy_temp_lambda_534.stypy_type_store = module_type_store
        _stypy_temp_lambda_534.stypy_function_name = '_stypy_temp_lambda_534'
        _stypy_temp_lambda_534.stypy_param_names_list = ['x']
        _stypy_temp_lambda_534.stypy_varargs_param_name = None
        _stypy_temp_lambda_534.stypy_kwargs_param_name = None
        _stypy_temp_lambda_534.stypy_call_defaults = defaults
        _stypy_temp_lambda_534.stypy_call_varargs = varargs
        _stypy_temp_lambda_534.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_534', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_534', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to power(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'x' (line 78)
        x_627100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 47), 'x', False)
        # Getting the type of 'm' (line 78)
        m_627101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 49), 'm', False)
        # Applying the binary operator '-' (line 78)
        result_sub_627102 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 47), '-', x_627100, m_627101)
        
        int_627103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 52), 'int')
        # Processing the call keyword arguments (line 78)
        kwargs_627104 = {}
        # Getting the type of 'np' (line 78)
        np_627098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 38), 'np', False)
        # Obtaining the member 'power' of a type (line 78)
        power_627099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 38), np_627098, 'power')
        # Calling power(args, kwargs) (line 78)
        power_call_result_627105 = invoke(stypy.reporting.localization.Localization(__file__, 78, 38), power_627099, *[result_sub_627102, int_627103], **kwargs_627104)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 28), 'stypy_return_type', power_call_result_627105)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_534' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_627106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 28), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_627106)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_534'
        return stypy_return_type_627106

    # Assigning a type to the variable '_stypy_temp_lambda_534' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 28), '_stypy_temp_lambda_534', _stypy_temp_lambda_534)
    # Getting the type of '_stypy_temp_lambda_534' (line 78)
    _stypy_temp_lambda_534_627107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 28), '_stypy_temp_lambda_534')
    # Getting the type of 'arg' (line 78)
    arg_627108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 56), 'arg', False)
    # Processing the call keyword arguments (line 78)
    kwargs_627109 = {}
    # Getting the type of 'distfn' (line 78)
    distfn_627096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 14), 'distfn', False)
    # Obtaining the member 'expect' of a type (line 78)
    expect_627097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 14), distfn_627096, 'expect')
    # Calling expect(args, kwargs) (line 78)
    expect_call_result_627110 = invoke(stypy.reporting.localization.Localization(__file__, 78, 14), expect_627097, *[_stypy_temp_lambda_534_627107, arg_627108], **kwargs_627109)
    
    # Assigning a type to the variable 'm3e' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'm3e', expect_call_result_627110)
    
    # Call to assert_almost_equal(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'm3e' (line 79)
    m3e_627113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 32), 'm3e', False)
    # Getting the type of 's' (line 79)
    s_627114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 37), 's', False)
    
    # Call to power(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'v' (line 79)
    v_627117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 50), 'v', False)
    float_627118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 53), 'float')
    # Processing the call keyword arguments (line 79)
    kwargs_627119 = {}
    # Getting the type of 'np' (line 79)
    np_627115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 41), 'np', False)
    # Obtaining the member 'power' of a type (line 79)
    power_627116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 41), np_627115, 'power')
    # Calling power(args, kwargs) (line 79)
    power_call_result_627120 = invoke(stypy.reporting.localization.Localization(__file__, 79, 41), power_627116, *[v_627117, float_627118], **kwargs_627119)
    
    # Applying the binary operator '*' (line 79)
    result_mul_627121 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 37), '*', s_627114, power_call_result_627120)
    
    # Processing the call keyword arguments (line 79)
    int_627122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 24), 'int')
    keyword_627123 = int_627122
    # Getting the type of 'msg' (line 80)
    msg_627124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 35), 'msg', False)
    str_627125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 41), 'str', ' - skew')
    # Applying the binary operator '+' (line 80)
    result_add_627126 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 35), '+', msg_627124, str_627125)
    
    keyword_627127 = result_add_627126
    kwargs_627128 = {'decimal': keyword_627123, 'err_msg': keyword_627127}
    # Getting the type of 'npt' (line 79)
    npt_627111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'npt', False)
    # Obtaining the member 'assert_almost_equal' of a type (line 79)
    assert_almost_equal_627112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), npt_627111, 'assert_almost_equal')
    # Calling assert_almost_equal(args, kwargs) (line 79)
    assert_almost_equal_call_result_627129 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), assert_almost_equal_627112, *[m3e_627113, result_mul_627121], **kwargs_627128)
    
    # SSA branch for the else part of an if statement (line 77)
    module_type_store.open_ssa_branch('else')
    
    # Call to assert_(...): (line 82)
    # Processing the call arguments (line 82)
    
    # Call to isnan(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 's' (line 82)
    s_627134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 's', False)
    # Processing the call keyword arguments (line 82)
    kwargs_627135 = {}
    # Getting the type of 'np' (line 82)
    np_627132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'np', False)
    # Obtaining the member 'isnan' of a type (line 82)
    isnan_627133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 20), np_627132, 'isnan')
    # Calling isnan(args, kwargs) (line 82)
    isnan_call_result_627136 = invoke(stypy.reporting.localization.Localization(__file__, 82, 20), isnan_627133, *[s_627134], **kwargs_627135)
    
    # Processing the call keyword arguments (line 82)
    kwargs_627137 = {}
    # Getting the type of 'npt' (line 82)
    npt_627130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 82)
    assert__627131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), npt_627130, 'assert_')
    # Calling assert_(args, kwargs) (line 82)
    assert__call_result_627138 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), assert__627131, *[isnan_call_result_627136], **kwargs_627137)
    
    # SSA join for if statement (line 77)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_skew_expect(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_skew_expect' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_627139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_627139)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_skew_expect'
    return stypy_return_type_627139

# Assigning a type to the variable 'check_skew_expect' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'check_skew_expect', check_skew_expect)

@norecursion
def check_kurt_expect(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_kurt_expect'
    module_type_store = module_type_store.open_function_context('check_kurt_expect', 85, 0, False)
    
    # Passed parameters checking function
    check_kurt_expect.stypy_localization = localization
    check_kurt_expect.stypy_type_of_self = None
    check_kurt_expect.stypy_type_store = module_type_store
    check_kurt_expect.stypy_function_name = 'check_kurt_expect'
    check_kurt_expect.stypy_param_names_list = ['distfn', 'arg', 'm', 'v', 'k', 'msg']
    check_kurt_expect.stypy_varargs_param_name = None
    check_kurt_expect.stypy_kwargs_param_name = None
    check_kurt_expect.stypy_call_defaults = defaults
    check_kurt_expect.stypy_call_varargs = varargs
    check_kurt_expect.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_kurt_expect', ['distfn', 'arg', 'm', 'v', 'k', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_kurt_expect', localization, ['distfn', 'arg', 'm', 'v', 'k', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_kurt_expect(...)' code ##################

    
    
    # Call to isfinite(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'k' (line 86)
    k_627142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'k', False)
    # Processing the call keyword arguments (line 86)
    kwargs_627143 = {}
    # Getting the type of 'np' (line 86)
    np_627140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 7), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 86)
    isfinite_627141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 7), np_627140, 'isfinite')
    # Calling isfinite(args, kwargs) (line 86)
    isfinite_call_result_627144 = invoke(stypy.reporting.localization.Localization(__file__, 86, 7), isfinite_627141, *[k_627142], **kwargs_627143)
    
    # Testing the type of an if condition (line 86)
    if_condition_627145 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 4), isfinite_call_result_627144)
    # Assigning a type to the variable 'if_condition_627145' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'if_condition_627145', if_condition_627145)
    # SSA begins for if statement (line 86)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 87):
    
    # Assigning a Call to a Name (line 87):
    
    # Call to expect(...): (line 87)
    # Processing the call arguments (line 87)

    @norecursion
    def _stypy_temp_lambda_535(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_535'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_535', 87, 28, True)
        # Passed parameters checking function
        _stypy_temp_lambda_535.stypy_localization = localization
        _stypy_temp_lambda_535.stypy_type_of_self = None
        _stypy_temp_lambda_535.stypy_type_store = module_type_store
        _stypy_temp_lambda_535.stypy_function_name = '_stypy_temp_lambda_535'
        _stypy_temp_lambda_535.stypy_param_names_list = ['x']
        _stypy_temp_lambda_535.stypy_varargs_param_name = None
        _stypy_temp_lambda_535.stypy_kwargs_param_name = None
        _stypy_temp_lambda_535.stypy_call_defaults = defaults
        _stypy_temp_lambda_535.stypy_call_varargs = varargs
        _stypy_temp_lambda_535.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_535', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_535', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to power(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'x' (line 87)
        x_627150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 47), 'x', False)
        # Getting the type of 'm' (line 87)
        m_627151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 49), 'm', False)
        # Applying the binary operator '-' (line 87)
        result_sub_627152 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 47), '-', x_627150, m_627151)
        
        int_627153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 52), 'int')
        # Processing the call keyword arguments (line 87)
        kwargs_627154 = {}
        # Getting the type of 'np' (line 87)
        np_627148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 38), 'np', False)
        # Obtaining the member 'power' of a type (line 87)
        power_627149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 38), np_627148, 'power')
        # Calling power(args, kwargs) (line 87)
        power_call_result_627155 = invoke(stypy.reporting.localization.Localization(__file__, 87, 38), power_627149, *[result_sub_627152, int_627153], **kwargs_627154)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 28), 'stypy_return_type', power_call_result_627155)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_535' in the type store
        # Getting the type of 'stypy_return_type' (line 87)
        stypy_return_type_627156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 28), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_627156)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_535'
        return stypy_return_type_627156

    # Assigning a type to the variable '_stypy_temp_lambda_535' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 28), '_stypy_temp_lambda_535', _stypy_temp_lambda_535)
    # Getting the type of '_stypy_temp_lambda_535' (line 87)
    _stypy_temp_lambda_535_627157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 28), '_stypy_temp_lambda_535')
    # Getting the type of 'arg' (line 87)
    arg_627158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 56), 'arg', False)
    # Processing the call keyword arguments (line 87)
    kwargs_627159 = {}
    # Getting the type of 'distfn' (line 87)
    distfn_627146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 14), 'distfn', False)
    # Obtaining the member 'expect' of a type (line 87)
    expect_627147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 14), distfn_627146, 'expect')
    # Calling expect(args, kwargs) (line 87)
    expect_call_result_627160 = invoke(stypy.reporting.localization.Localization(__file__, 87, 14), expect_627147, *[_stypy_temp_lambda_535_627157, arg_627158], **kwargs_627159)
    
    # Assigning a type to the variable 'm4e' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'm4e', expect_call_result_627160)
    
    # Call to assert_allclose(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'm4e' (line 88)
    m4e_627163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 28), 'm4e', False)
    # Getting the type of 'k' (line 88)
    k_627164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 34), 'k', False)
    float_627165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 38), 'float')
    # Applying the binary operator '+' (line 88)
    result_add_627166 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 34), '+', k_627164, float_627165)
    
    
    # Call to power(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'v' (line 88)
    v_627169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 53), 'v', False)
    int_627170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 56), 'int')
    # Processing the call keyword arguments (line 88)
    kwargs_627171 = {}
    # Getting the type of 'np' (line 88)
    np_627167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 44), 'np', False)
    # Obtaining the member 'power' of a type (line 88)
    power_627168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 44), np_627167, 'power')
    # Calling power(args, kwargs) (line 88)
    power_call_result_627172 = invoke(stypy.reporting.localization.Localization(__file__, 88, 44), power_627168, *[v_627169, int_627170], **kwargs_627171)
    
    # Applying the binary operator '*' (line 88)
    result_mul_627173 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 33), '*', result_add_627166, power_call_result_627172)
    
    # Processing the call keyword arguments (line 88)
    float_627174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 65), 'float')
    keyword_627175 = float_627174
    float_627176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 76), 'float')
    keyword_627177 = float_627176
    # Getting the type of 'msg' (line 89)
    msg_627178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'msg', False)
    str_627179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 30), 'str', ' - kurtosis')
    # Applying the binary operator '+' (line 89)
    result_add_627180 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 24), '+', msg_627178, str_627179)
    
    keyword_627181 = result_add_627180
    kwargs_627182 = {'rtol': keyword_627177, 'err_msg': keyword_627181, 'atol': keyword_627175}
    # Getting the type of 'npt' (line 88)
    npt_627161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'npt', False)
    # Obtaining the member 'assert_allclose' of a type (line 88)
    assert_allclose_627162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), npt_627161, 'assert_allclose')
    # Calling assert_allclose(args, kwargs) (line 88)
    assert_allclose_call_result_627183 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), assert_allclose_627162, *[m4e_627163, result_mul_627173], **kwargs_627182)
    
    # SSA branch for the else part of an if statement (line 86)
    module_type_store.open_ssa_branch('else')
    
    # Call to assert_(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Call to isnan(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'k' (line 91)
    k_627188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 29), 'k', False)
    # Processing the call keyword arguments (line 91)
    kwargs_627189 = {}
    # Getting the type of 'np' (line 91)
    np_627186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'np', False)
    # Obtaining the member 'isnan' of a type (line 91)
    isnan_627187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 20), np_627186, 'isnan')
    # Calling isnan(args, kwargs) (line 91)
    isnan_call_result_627190 = invoke(stypy.reporting.localization.Localization(__file__, 91, 20), isnan_627187, *[k_627188], **kwargs_627189)
    
    # Processing the call keyword arguments (line 91)
    kwargs_627191 = {}
    # Getting the type of 'npt' (line 91)
    npt_627184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 91)
    assert__627185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), npt_627184, 'assert_')
    # Calling assert_(args, kwargs) (line 91)
    assert__call_result_627192 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), assert__627185, *[isnan_call_result_627190], **kwargs_627191)
    
    # SSA join for if statement (line 86)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_kurt_expect(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_kurt_expect' in the type store
    # Getting the type of 'stypy_return_type' (line 85)
    stypy_return_type_627193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_627193)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_kurt_expect'
    return stypy_return_type_627193

# Assigning a type to the variable 'check_kurt_expect' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'check_kurt_expect', check_kurt_expect)

@norecursion
def check_entropy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_entropy'
    module_type_store = module_type_store.open_function_context('check_entropy', 94, 0, False)
    
    # Passed parameters checking function
    check_entropy.stypy_localization = localization
    check_entropy.stypy_type_of_self = None
    check_entropy.stypy_type_store = module_type_store
    check_entropy.stypy_function_name = 'check_entropy'
    check_entropy.stypy_param_names_list = ['distfn', 'arg', 'msg']
    check_entropy.stypy_varargs_param_name = None
    check_entropy.stypy_kwargs_param_name = None
    check_entropy.stypy_call_defaults = defaults
    check_entropy.stypy_call_varargs = varargs
    check_entropy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_entropy', ['distfn', 'arg', 'msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_entropy', localization, ['distfn', 'arg', 'msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_entropy(...)' code ##################

    
    # Assigning a Call to a Name (line 95):
    
    # Assigning a Call to a Name (line 95):
    
    # Call to entropy(...): (line 95)
    # Getting the type of 'arg' (line 95)
    arg_627196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 26), 'arg', False)
    # Processing the call keyword arguments (line 95)
    kwargs_627197 = {}
    # Getting the type of 'distfn' (line 95)
    distfn_627194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 10), 'distfn', False)
    # Obtaining the member 'entropy' of a type (line 95)
    entropy_627195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 10), distfn_627194, 'entropy')
    # Calling entropy(args, kwargs) (line 95)
    entropy_call_result_627198 = invoke(stypy.reporting.localization.Localization(__file__, 95, 10), entropy_627195, *[arg_627196], **kwargs_627197)
    
    # Assigning a type to the variable 'ent' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'ent', entropy_call_result_627198)
    
    # Call to assert_(...): (line 96)
    # Processing the call arguments (line 96)
    
    
    # Call to isnan(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'ent' (line 96)
    ent_627203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 29), 'ent', False)
    # Processing the call keyword arguments (line 96)
    kwargs_627204 = {}
    # Getting the type of 'np' (line 96)
    np_627201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 20), 'np', False)
    # Obtaining the member 'isnan' of a type (line 96)
    isnan_627202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 20), np_627201, 'isnan')
    # Calling isnan(args, kwargs) (line 96)
    isnan_call_result_627205 = invoke(stypy.reporting.localization.Localization(__file__, 96, 20), isnan_627202, *[ent_627203], **kwargs_627204)
    
    # Applying the 'not' unary operator (line 96)
    result_not__627206 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 16), 'not', isnan_call_result_627205)
    
    # Getting the type of 'msg' (line 96)
    msg_627207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 35), 'msg', False)
    str_627208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 41), 'str', 'test Entropy is nan')
    # Applying the binary operator '+' (line 96)
    result_add_627209 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 35), '+', msg_627207, str_627208)
    
    # Processing the call keyword arguments (line 96)
    kwargs_627210 = {}
    # Getting the type of 'npt' (line 96)
    npt_627199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 96)
    assert__627200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 4), npt_627199, 'assert_')
    # Calling assert_(args, kwargs) (line 96)
    assert__call_result_627211 = invoke(stypy.reporting.localization.Localization(__file__, 96, 4), assert__627200, *[result_not__627206, result_add_627209], **kwargs_627210)
    
    
    # ################# End of 'check_entropy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_entropy' in the type store
    # Getting the type of 'stypy_return_type' (line 94)
    stypy_return_type_627212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_627212)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_entropy'
    return stypy_return_type_627212

# Assigning a type to the variable 'check_entropy' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'check_entropy', check_entropy)

@norecursion
def check_private_entropy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_private_entropy'
    module_type_store = module_type_store.open_function_context('check_private_entropy', 99, 0, False)
    
    # Passed parameters checking function
    check_private_entropy.stypy_localization = localization
    check_private_entropy.stypy_type_of_self = None
    check_private_entropy.stypy_type_store = module_type_store
    check_private_entropy.stypy_function_name = 'check_private_entropy'
    check_private_entropy.stypy_param_names_list = ['distfn', 'args', 'superclass']
    check_private_entropy.stypy_varargs_param_name = None
    check_private_entropy.stypy_kwargs_param_name = None
    check_private_entropy.stypy_call_defaults = defaults
    check_private_entropy.stypy_call_varargs = varargs
    check_private_entropy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_private_entropy', ['distfn', 'args', 'superclass'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_private_entropy', localization, ['distfn', 'args', 'superclass'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_private_entropy(...)' code ##################

    
    # Call to assert_allclose(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Call to _entropy(...): (line 101)
    # Getting the type of 'args' (line 101)
    args_627217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 41), 'args', False)
    # Processing the call keyword arguments (line 101)
    kwargs_627218 = {}
    # Getting the type of 'distfn' (line 101)
    distfn_627215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'distfn', False)
    # Obtaining the member '_entropy' of a type (line 101)
    _entropy_627216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 24), distfn_627215, '_entropy')
    # Calling _entropy(args, kwargs) (line 101)
    _entropy_call_result_627219 = invoke(stypy.reporting.localization.Localization(__file__, 101, 24), _entropy_627216, *[args_627217], **kwargs_627218)
    
    
    # Call to _entropy(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'distfn' (line 102)
    distfn_627222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 44), 'distfn', False)
    # Getting the type of 'args' (line 102)
    args_627223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 53), 'args', False)
    # Processing the call keyword arguments (line 102)
    kwargs_627224 = {}
    # Getting the type of 'superclass' (line 102)
    superclass_627220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'superclass', False)
    # Obtaining the member '_entropy' of a type (line 102)
    _entropy_627221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 24), superclass_627220, '_entropy')
    # Calling _entropy(args, kwargs) (line 102)
    _entropy_call_result_627225 = invoke(stypy.reporting.localization.Localization(__file__, 102, 24), _entropy_627221, *[distfn_627222, args_627223], **kwargs_627224)
    
    # Processing the call keyword arguments (line 101)
    kwargs_627226 = {}
    # Getting the type of 'npt' (line 101)
    npt_627213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'npt', False)
    # Obtaining the member 'assert_allclose' of a type (line 101)
    assert_allclose_627214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 4), npt_627213, 'assert_allclose')
    # Calling assert_allclose(args, kwargs) (line 101)
    assert_allclose_call_result_627227 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), assert_allclose_627214, *[_entropy_call_result_627219, _entropy_call_result_627225], **kwargs_627226)
    
    
    # ################# End of 'check_private_entropy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_private_entropy' in the type store
    # Getting the type of 'stypy_return_type' (line 99)
    stypy_return_type_627228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_627228)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_private_entropy'
    return stypy_return_type_627228

# Assigning a type to the variable 'check_private_entropy' (line 99)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'check_private_entropy', check_private_entropy)

@norecursion
def check_edge_support(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_edge_support'
    module_type_store = module_type_store.open_function_context('check_edge_support', 105, 0, False)
    
    # Passed parameters checking function
    check_edge_support.stypy_localization = localization
    check_edge_support.stypy_type_of_self = None
    check_edge_support.stypy_type_store = module_type_store
    check_edge_support.stypy_function_name = 'check_edge_support'
    check_edge_support.stypy_param_names_list = ['distfn', 'args']
    check_edge_support.stypy_varargs_param_name = None
    check_edge_support.stypy_kwargs_param_name = None
    check_edge_support.stypy_call_defaults = defaults
    check_edge_support.stypy_call_varargs = varargs
    check_edge_support.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_edge_support', ['distfn', 'args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_edge_support', localization, ['distfn', 'args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_edge_support(...)' code ##################

    
    # Assigning a List to a Name (line 107):
    
    # Assigning a List to a Name (line 107):
    
    # Obtaining an instance of the builtin type 'list' (line 107)
    list_627229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 107)
    # Adding element type (line 107)
    # Getting the type of 'distfn' (line 107)
    distfn_627230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 9), 'distfn')
    # Obtaining the member 'a' of a type (line 107)
    a_627231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 9), distfn_627230, 'a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 8), list_627229, a_627231)
    # Adding element type (line 107)
    # Getting the type of 'distfn' (line 107)
    distfn_627232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'distfn')
    # Obtaining the member 'b' of a type (line 107)
    b_627233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 19), distfn_627232, 'b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 8), list_627229, b_627233)
    
    # Assigning a type to the variable 'x' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'x', list_627229)
    
    
    # Call to isinstance(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'distfn' (line 108)
    distfn_627235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 18), 'distfn', False)
    # Getting the type of 'stats' (line 108)
    stats_627236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 26), 'stats', False)
    # Obtaining the member 'rv_discrete' of a type (line 108)
    rv_discrete_627237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 26), stats_627236, 'rv_discrete')
    # Processing the call keyword arguments (line 108)
    kwargs_627238 = {}
    # Getting the type of 'isinstance' (line 108)
    isinstance_627234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 108)
    isinstance_call_result_627239 = invoke(stypy.reporting.localization.Localization(__file__, 108, 7), isinstance_627234, *[distfn_627235, rv_discrete_627237], **kwargs_627238)
    
    # Testing the type of an if condition (line 108)
    if_condition_627240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 4), isinstance_call_result_627239)
    # Assigning a type to the variable 'if_condition_627240' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'if_condition_627240', if_condition_627240)
    # SSA begins for if statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 109):
    
    # Assigning a List to a Name (line 109):
    
    # Obtaining an instance of the builtin type 'list' (line 109)
    list_627241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 109)
    # Adding element type (line 109)
    # Getting the type of 'distfn' (line 109)
    distfn_627242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 13), 'distfn')
    # Obtaining the member 'a' of a type (line 109)
    a_627243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 13), distfn_627242, 'a')
    int_627244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 24), 'int')
    # Applying the binary operator '-' (line 109)
    result_sub_627245 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 13), '-', a_627243, int_627244)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 12), list_627241, result_sub_627245)
    # Adding element type (line 109)
    # Getting the type of 'distfn' (line 109)
    distfn_627246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'distfn')
    # Obtaining the member 'b' of a type (line 109)
    b_627247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 27), distfn_627246, 'b')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 12), list_627241, b_627247)
    
    # Assigning a type to the variable 'x' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'x', list_627241)
    # SSA join for if statement (line 108)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_equal(...): (line 111)
    # Processing the call arguments (line 111)
    
    # Call to cdf(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'x' (line 111)
    x_627252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 32), 'x', False)
    # Getting the type of 'args' (line 111)
    args_627253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 36), 'args', False)
    # Processing the call keyword arguments (line 111)
    kwargs_627254 = {}
    # Getting the type of 'distfn' (line 111)
    distfn_627250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 21), 'distfn', False)
    # Obtaining the member 'cdf' of a type (line 111)
    cdf_627251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 21), distfn_627250, 'cdf')
    # Calling cdf(args, kwargs) (line 111)
    cdf_call_result_627255 = invoke(stypy.reporting.localization.Localization(__file__, 111, 21), cdf_627251, *[x_627252, args_627253], **kwargs_627254)
    
    
    # Obtaining an instance of the builtin type 'list' (line 111)
    list_627256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 111)
    # Adding element type (line 111)
    float_627257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 44), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 43), list_627256, float_627257)
    # Adding element type (line 111)
    float_627258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 49), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 43), list_627256, float_627258)
    
    # Processing the call keyword arguments (line 111)
    kwargs_627259 = {}
    # Getting the type of 'npt' (line 111)
    npt_627248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'npt', False)
    # Obtaining the member 'assert_equal' of a type (line 111)
    assert_equal_627249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 4), npt_627248, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 111)
    assert_equal_call_result_627260 = invoke(stypy.reporting.localization.Localization(__file__, 111, 4), assert_equal_627249, *[cdf_call_result_627255, list_627256], **kwargs_627259)
    
    
    # Call to assert_equal(...): (line 112)
    # Processing the call arguments (line 112)
    
    # Call to sf(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'x' (line 112)
    x_627265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'x', False)
    # Getting the type of 'args' (line 112)
    args_627266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 35), 'args', False)
    # Processing the call keyword arguments (line 112)
    kwargs_627267 = {}
    # Getting the type of 'distfn' (line 112)
    distfn_627263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 21), 'distfn', False)
    # Obtaining the member 'sf' of a type (line 112)
    sf_627264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 21), distfn_627263, 'sf')
    # Calling sf(args, kwargs) (line 112)
    sf_call_result_627268 = invoke(stypy.reporting.localization.Localization(__file__, 112, 21), sf_627264, *[x_627265, args_627266], **kwargs_627267)
    
    
    # Obtaining an instance of the builtin type 'list' (line 112)
    list_627269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 42), 'list')
    # Adding type elements to the builtin type 'list' instance (line 112)
    # Adding element type (line 112)
    float_627270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 43), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 42), list_627269, float_627270)
    # Adding element type (line 112)
    float_627271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 48), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 42), list_627269, float_627271)
    
    # Processing the call keyword arguments (line 112)
    kwargs_627272 = {}
    # Getting the type of 'npt' (line 112)
    npt_627261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'npt', False)
    # Obtaining the member 'assert_equal' of a type (line 112)
    assert_equal_627262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 4), npt_627261, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 112)
    assert_equal_call_result_627273 = invoke(stypy.reporting.localization.Localization(__file__, 112, 4), assert_equal_627262, *[sf_call_result_627268, list_627269], **kwargs_627272)
    
    
    
    # Getting the type of 'distfn' (line 114)
    distfn_627274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 7), 'distfn')
    # Obtaining the member 'name' of a type (line 114)
    name_627275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 7), distfn_627274, 'name')
    
    # Obtaining an instance of the builtin type 'tuple' (line 114)
    tuple_627276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 114)
    # Adding element type (line 114)
    str_627277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 27), 'str', 'skellam')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 27), tuple_627276, str_627277)
    # Adding element type (line 114)
    str_627278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 38), 'str', 'dlaplace')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 27), tuple_627276, str_627278)
    
    # Applying the binary operator 'notin' (line 114)
    result_contains_627279 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 7), 'notin', name_627275, tuple_627276)
    
    # Testing the type of an if condition (line 114)
    if_condition_627280 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 4), result_contains_627279)
    # Assigning a type to the variable 'if_condition_627280' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'if_condition_627280', if_condition_627280)
    # SSA begins for if statement (line 114)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_equal(...): (line 116)
    # Processing the call arguments (line 116)
    
    # Call to logcdf(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'x' (line 116)
    x_627285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 39), 'x', False)
    # Getting the type of 'args' (line 116)
    args_627286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 43), 'args', False)
    # Processing the call keyword arguments (line 116)
    kwargs_627287 = {}
    # Getting the type of 'distfn' (line 116)
    distfn_627283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 25), 'distfn', False)
    # Obtaining the member 'logcdf' of a type (line 116)
    logcdf_627284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 25), distfn_627283, 'logcdf')
    # Calling logcdf(args, kwargs) (line 116)
    logcdf_call_result_627288 = invoke(stypy.reporting.localization.Localization(__file__, 116, 25), logcdf_627284, *[x_627285, args_627286], **kwargs_627287)
    
    
    # Obtaining an instance of the builtin type 'list' (line 116)
    list_627289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 50), 'list')
    # Adding type elements to the builtin type 'list' instance (line 116)
    # Adding element type (line 116)
    
    # Getting the type of 'np' (line 116)
    np_627290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 52), 'np', False)
    # Obtaining the member 'inf' of a type (line 116)
    inf_627291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 52), np_627290, 'inf')
    # Applying the 'usub' unary operator (line 116)
    result___neg___627292 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 51), 'usub', inf_627291)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 50), list_627289, result___neg___627292)
    # Adding element type (line 116)
    float_627293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 60), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 50), list_627289, float_627293)
    
    # Processing the call keyword arguments (line 116)
    kwargs_627294 = {}
    # Getting the type of 'npt' (line 116)
    npt_627281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'npt', False)
    # Obtaining the member 'assert_equal' of a type (line 116)
    assert_equal_627282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), npt_627281, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 116)
    assert_equal_call_result_627295 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), assert_equal_627282, *[logcdf_call_result_627288, list_627289], **kwargs_627294)
    
    
    # Call to assert_equal(...): (line 117)
    # Processing the call arguments (line 117)
    
    # Call to logsf(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'x' (line 117)
    x_627300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 38), 'x', False)
    # Getting the type of 'args' (line 117)
    args_627301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 42), 'args', False)
    # Processing the call keyword arguments (line 117)
    kwargs_627302 = {}
    # Getting the type of 'distfn' (line 117)
    distfn_627298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'distfn', False)
    # Obtaining the member 'logsf' of a type (line 117)
    logsf_627299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 25), distfn_627298, 'logsf')
    # Calling logsf(args, kwargs) (line 117)
    logsf_call_result_627303 = invoke(stypy.reporting.localization.Localization(__file__, 117, 25), logsf_627299, *[x_627300, args_627301], **kwargs_627302)
    
    
    # Obtaining an instance of the builtin type 'list' (line 117)
    list_627304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 117)
    # Adding element type (line 117)
    float_627305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 50), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 49), list_627304, float_627305)
    # Adding element type (line 117)
    
    # Getting the type of 'np' (line 117)
    np_627306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 56), 'np', False)
    # Obtaining the member 'inf' of a type (line 117)
    inf_627307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 56), np_627306, 'inf')
    # Applying the 'usub' unary operator (line 117)
    result___neg___627308 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 55), 'usub', inf_627307)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 49), list_627304, result___neg___627308)
    
    # Processing the call keyword arguments (line 117)
    kwargs_627309 = {}
    # Getting the type of 'npt' (line 117)
    npt_627296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'npt', False)
    # Obtaining the member 'assert_equal' of a type (line 117)
    assert_equal_627297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), npt_627296, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 117)
    assert_equal_call_result_627310 = invoke(stypy.reporting.localization.Localization(__file__, 117, 8), assert_equal_627297, *[logsf_call_result_627303, list_627304], **kwargs_627309)
    
    # SSA join for if statement (line 114)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_equal(...): (line 119)
    # Processing the call arguments (line 119)
    
    # Call to ppf(...): (line 119)
    # Processing the call arguments (line 119)
    
    # Obtaining an instance of the builtin type 'list' (line 119)
    list_627315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 119)
    # Adding element type (line 119)
    float_627316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 33), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 32), list_627315, float_627316)
    # Adding element type (line 119)
    float_627317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 32), list_627315, float_627317)
    
    # Getting the type of 'args' (line 119)
    args_627318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 45), 'args', False)
    # Processing the call keyword arguments (line 119)
    kwargs_627319 = {}
    # Getting the type of 'distfn' (line 119)
    distfn_627313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 21), 'distfn', False)
    # Obtaining the member 'ppf' of a type (line 119)
    ppf_627314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 21), distfn_627313, 'ppf')
    # Calling ppf(args, kwargs) (line 119)
    ppf_call_result_627320 = invoke(stypy.reporting.localization.Localization(__file__, 119, 21), ppf_627314, *[list_627315, args_627318], **kwargs_627319)
    
    # Getting the type of 'x' (line 119)
    x_627321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 52), 'x', False)
    # Processing the call keyword arguments (line 119)
    kwargs_627322 = {}
    # Getting the type of 'npt' (line 119)
    npt_627311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'npt', False)
    # Obtaining the member 'assert_equal' of a type (line 119)
    assert_equal_627312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 4), npt_627311, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 119)
    assert_equal_call_result_627323 = invoke(stypy.reporting.localization.Localization(__file__, 119, 4), assert_equal_627312, *[ppf_call_result_627320, x_627321], **kwargs_627322)
    
    
    # Call to assert_equal(...): (line 120)
    # Processing the call arguments (line 120)
    
    # Call to isf(...): (line 120)
    # Processing the call arguments (line 120)
    
    # Obtaining an instance of the builtin type 'list' (line 120)
    list_627328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 120)
    # Adding element type (line 120)
    float_627329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 33), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 32), list_627328, float_627329)
    # Adding element type (line 120)
    float_627330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 38), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 32), list_627328, float_627330)
    
    # Getting the type of 'args' (line 120)
    args_627331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 45), 'args', False)
    # Processing the call keyword arguments (line 120)
    kwargs_627332 = {}
    # Getting the type of 'distfn' (line 120)
    distfn_627326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 21), 'distfn', False)
    # Obtaining the member 'isf' of a type (line 120)
    isf_627327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 21), distfn_627326, 'isf')
    # Calling isf(args, kwargs) (line 120)
    isf_call_result_627333 = invoke(stypy.reporting.localization.Localization(__file__, 120, 21), isf_627327, *[list_627328, args_627331], **kwargs_627332)
    
    
    # Obtaining the type of the subscript
    int_627334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 56), 'int')
    slice_627335 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 120, 52), None, None, int_627334)
    # Getting the type of 'x' (line 120)
    x_627336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 52), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 120)
    getitem___627337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 52), x_627336, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 120)
    subscript_call_result_627338 = invoke(stypy.reporting.localization.Localization(__file__, 120, 52), getitem___627337, slice_627335)
    
    # Processing the call keyword arguments (line 120)
    kwargs_627339 = {}
    # Getting the type of 'npt' (line 120)
    npt_627324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'npt', False)
    # Obtaining the member 'assert_equal' of a type (line 120)
    assert_equal_627325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 4), npt_627324, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 120)
    assert_equal_call_result_627340 = invoke(stypy.reporting.localization.Localization(__file__, 120, 4), assert_equal_627325, *[isf_call_result_627333, subscript_call_result_627338], **kwargs_627339)
    
    
    # Call to assert_(...): (line 123)
    # Processing the call arguments (line 123)
    
    # Call to all(...): (line 123)
    # Processing the call keyword arguments (line 123)
    kwargs_627356 = {}
    
    # Call to isnan(...): (line 123)
    # Processing the call arguments (line 123)
    
    # Call to isf(...): (line 123)
    # Processing the call arguments (line 123)
    
    # Obtaining an instance of the builtin type 'list' (line 123)
    list_627347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 123)
    # Adding element type (line 123)
    int_627348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 36), list_627347, int_627348)
    # Adding element type (line 123)
    int_627349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 36), list_627347, int_627349)
    
    # Getting the type of 'args' (line 123)
    args_627350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 46), 'args', False)
    # Processing the call keyword arguments (line 123)
    kwargs_627351 = {}
    # Getting the type of 'distfn' (line 123)
    distfn_627345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 25), 'distfn', False)
    # Obtaining the member 'isf' of a type (line 123)
    isf_627346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 25), distfn_627345, 'isf')
    # Calling isf(args, kwargs) (line 123)
    isf_call_result_627352 = invoke(stypy.reporting.localization.Localization(__file__, 123, 25), isf_627346, *[list_627347, args_627350], **kwargs_627351)
    
    # Processing the call keyword arguments (line 123)
    kwargs_627353 = {}
    # Getting the type of 'np' (line 123)
    np_627343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'np', False)
    # Obtaining the member 'isnan' of a type (line 123)
    isnan_627344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 16), np_627343, 'isnan')
    # Calling isnan(args, kwargs) (line 123)
    isnan_call_result_627354 = invoke(stypy.reporting.localization.Localization(__file__, 123, 16), isnan_627344, *[isf_call_result_627352], **kwargs_627353)
    
    # Obtaining the member 'all' of a type (line 123)
    all_627355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 16), isnan_call_result_627354, 'all')
    # Calling all(args, kwargs) (line 123)
    all_call_result_627357 = invoke(stypy.reporting.localization.Localization(__file__, 123, 16), all_627355, *[], **kwargs_627356)
    
    # Processing the call keyword arguments (line 123)
    kwargs_627358 = {}
    # Getting the type of 'npt' (line 123)
    npt_627341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 123)
    assert__627342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 4), npt_627341, 'assert_')
    # Calling assert_(args, kwargs) (line 123)
    assert__call_result_627359 = invoke(stypy.reporting.localization.Localization(__file__, 123, 4), assert__627342, *[all_call_result_627357], **kwargs_627358)
    
    
    # Call to assert_(...): (line 124)
    # Processing the call arguments (line 124)
    
    # Call to all(...): (line 124)
    # Processing the call keyword arguments (line 124)
    kwargs_627375 = {}
    
    # Call to isnan(...): (line 124)
    # Processing the call arguments (line 124)
    
    # Call to ppf(...): (line 124)
    # Processing the call arguments (line 124)
    
    # Obtaining an instance of the builtin type 'list' (line 124)
    list_627366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 124)
    # Adding element type (line 124)
    int_627367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 36), list_627366, int_627367)
    # Adding element type (line 124)
    int_627368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 36), list_627366, int_627368)
    
    # Getting the type of 'args' (line 124)
    args_627369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 46), 'args', False)
    # Processing the call keyword arguments (line 124)
    kwargs_627370 = {}
    # Getting the type of 'distfn' (line 124)
    distfn_627364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 'distfn', False)
    # Obtaining the member 'ppf' of a type (line 124)
    ppf_627365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 25), distfn_627364, 'ppf')
    # Calling ppf(args, kwargs) (line 124)
    ppf_call_result_627371 = invoke(stypy.reporting.localization.Localization(__file__, 124, 25), ppf_627365, *[list_627366, args_627369], **kwargs_627370)
    
    # Processing the call keyword arguments (line 124)
    kwargs_627372 = {}
    # Getting the type of 'np' (line 124)
    np_627362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'np', False)
    # Obtaining the member 'isnan' of a type (line 124)
    isnan_627363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 16), np_627362, 'isnan')
    # Calling isnan(args, kwargs) (line 124)
    isnan_call_result_627373 = invoke(stypy.reporting.localization.Localization(__file__, 124, 16), isnan_627363, *[ppf_call_result_627371], **kwargs_627372)
    
    # Obtaining the member 'all' of a type (line 124)
    all_627374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 16), isnan_call_result_627373, 'all')
    # Calling all(args, kwargs) (line 124)
    all_call_result_627376 = invoke(stypy.reporting.localization.Localization(__file__, 124, 16), all_627374, *[], **kwargs_627375)
    
    # Processing the call keyword arguments (line 124)
    kwargs_627377 = {}
    # Getting the type of 'npt' (line 124)
    npt_627360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 124)
    assert__627361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 4), npt_627360, 'assert_')
    # Calling assert_(args, kwargs) (line 124)
    assert__call_result_627378 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), assert__627361, *[all_call_result_627376], **kwargs_627377)
    
    
    # ################# End of 'check_edge_support(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_edge_support' in the type store
    # Getting the type of 'stypy_return_type' (line 105)
    stypy_return_type_627379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_627379)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_edge_support'
    return stypy_return_type_627379

# Assigning a type to the variable 'check_edge_support' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'check_edge_support', check_edge_support)

@norecursion
def check_named_args(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_named_args'
    module_type_store = module_type_store.open_function_context('check_named_args', 127, 0, False)
    
    # Passed parameters checking function
    check_named_args.stypy_localization = localization
    check_named_args.stypy_type_of_self = None
    check_named_args.stypy_type_store = module_type_store
    check_named_args.stypy_function_name = 'check_named_args'
    check_named_args.stypy_param_names_list = ['distfn', 'x', 'shape_args', 'defaults', 'meths']
    check_named_args.stypy_varargs_param_name = None
    check_named_args.stypy_kwargs_param_name = None
    check_named_args.stypy_call_defaults = defaults
    check_named_args.stypy_call_varargs = varargs
    check_named_args.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_named_args', ['distfn', 'x', 'shape_args', 'defaults', 'meths'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_named_args', localization, ['distfn', 'x', 'shape_args', 'defaults', 'meths'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_named_args(...)' code ##################

    
    # Assigning a Call to a Name (line 131):
    
    # Assigning a Call to a Name (line 131):
    
    # Call to _getargspec(...): (line 131)
    # Processing the call arguments (line 131)
    # Getting the type of 'distfn' (line 131)
    distfn_627381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 28), 'distfn', False)
    # Obtaining the member '_parse_args' of a type (line 131)
    _parse_args_627382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 28), distfn_627381, '_parse_args')
    # Processing the call keyword arguments (line 131)
    kwargs_627383 = {}
    # Getting the type of '_getargspec' (line 131)
    _getargspec_627380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), '_getargspec', False)
    # Calling _getargspec(args, kwargs) (line 131)
    _getargspec_call_result_627384 = invoke(stypy.reporting.localization.Localization(__file__, 131, 16), _getargspec_627380, *[_parse_args_627382], **kwargs_627383)
    
    # Assigning a type to the variable 'signature' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'signature', _getargspec_call_result_627384)
    
    # Call to assert_(...): (line 132)
    # Processing the call arguments (line 132)
    
    # Getting the type of 'signature' (line 132)
    signature_627387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'signature', False)
    # Obtaining the member 'varargs' of a type (line 132)
    varargs_627388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 16), signature_627387, 'varargs')
    # Getting the type of 'None' (line 132)
    None_627389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 37), 'None', False)
    # Applying the binary operator 'is' (line 132)
    result_is__627390 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 16), 'is', varargs_627388, None_627389)
    
    # Processing the call keyword arguments (line 132)
    kwargs_627391 = {}
    # Getting the type of 'npt' (line 132)
    npt_627385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 132)
    assert__627386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 4), npt_627385, 'assert_')
    # Calling assert_(args, kwargs) (line 132)
    assert__call_result_627392 = invoke(stypy.reporting.localization.Localization(__file__, 132, 4), assert__627386, *[result_is__627390], **kwargs_627391)
    
    
    # Call to assert_(...): (line 133)
    # Processing the call arguments (line 133)
    
    # Getting the type of 'signature' (line 133)
    signature_627395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'signature', False)
    # Obtaining the member 'keywords' of a type (line 133)
    keywords_627396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 16), signature_627395, 'keywords')
    # Getting the type of 'None' (line 133)
    None_627397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 38), 'None', False)
    # Applying the binary operator 'is' (line 133)
    result_is__627398 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 16), 'is', keywords_627396, None_627397)
    
    # Processing the call keyword arguments (line 133)
    kwargs_627399 = {}
    # Getting the type of 'npt' (line 133)
    npt_627393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 133)
    assert__627394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 4), npt_627393, 'assert_')
    # Calling assert_(args, kwargs) (line 133)
    assert__call_result_627400 = invoke(stypy.reporting.localization.Localization(__file__, 133, 4), assert__627394, *[result_is__627398], **kwargs_627399)
    
    
    # Call to assert_(...): (line 134)
    # Processing the call arguments (line 134)
    
    
    # Call to list(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'signature' (line 134)
    signature_627404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 21), 'signature', False)
    # Obtaining the member 'defaults' of a type (line 134)
    defaults_627405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 21), signature_627404, 'defaults')
    # Processing the call keyword arguments (line 134)
    kwargs_627406 = {}
    # Getting the type of 'list' (line 134)
    list_627403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), 'list', False)
    # Calling list(args, kwargs) (line 134)
    list_call_result_627407 = invoke(stypy.reporting.localization.Localization(__file__, 134, 16), list_627403, *[defaults_627405], **kwargs_627406)
    
    
    # Call to list(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'defaults' (line 134)
    defaults_627409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 49), 'defaults', False)
    # Processing the call keyword arguments (line 134)
    kwargs_627410 = {}
    # Getting the type of 'list' (line 134)
    list_627408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 44), 'list', False)
    # Calling list(args, kwargs) (line 134)
    list_call_result_627411 = invoke(stypy.reporting.localization.Localization(__file__, 134, 44), list_627408, *[defaults_627409], **kwargs_627410)
    
    # Applying the binary operator '==' (line 134)
    result_eq_627412 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 16), '==', list_call_result_627407, list_call_result_627411)
    
    # Processing the call keyword arguments (line 134)
    kwargs_627413 = {}
    # Getting the type of 'npt' (line 134)
    npt_627401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 134)
    assert__627402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 4), npt_627401, 'assert_')
    # Calling assert_(args, kwargs) (line 134)
    assert__call_result_627414 = invoke(stypy.reporting.localization.Localization(__file__, 134, 4), assert__627402, *[result_eq_627412], **kwargs_627413)
    
    
    # Assigning a Subscript to a Name (line 136):
    
    # Assigning a Subscript to a Name (line 136):
    
    # Obtaining the type of the subscript
    
    
    # Call to len(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'defaults' (line 136)
    defaults_627416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 42), 'defaults', False)
    # Processing the call keyword arguments (line 136)
    kwargs_627417 = {}
    # Getting the type of 'len' (line 136)
    len_627415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 38), 'len', False)
    # Calling len(args, kwargs) (line 136)
    len_call_result_627418 = invoke(stypy.reporting.localization.Localization(__file__, 136, 38), len_627415, *[defaults_627416], **kwargs_627417)
    
    # Applying the 'usub' unary operator (line 136)
    result___neg___627419 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 37), 'usub', len_call_result_627418)
    
    slice_627420 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 136, 21), None, result___neg___627419, None)
    # Getting the type of 'signature' (line 136)
    signature_627421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 21), 'signature')
    # Obtaining the member 'args' of a type (line 136)
    args_627422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 21), signature_627421, 'args')
    # Obtaining the member '__getitem__' of a type (line 136)
    getitem___627423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 21), args_627422, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 136)
    subscript_call_result_627424 = invoke(stypy.reporting.localization.Localization(__file__, 136, 21), getitem___627423, slice_627420)
    
    # Assigning a type to the variable 'shape_argnames' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'shape_argnames', subscript_call_result_627424)
    
    # Getting the type of 'distfn' (line 137)
    distfn_627425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 7), 'distfn')
    # Obtaining the member 'shapes' of a type (line 137)
    shapes_627426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 7), distfn_627425, 'shapes')
    # Testing the type of an if condition (line 137)
    if_condition_627427 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 4), shapes_627426)
    # Assigning a type to the variable 'if_condition_627427' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'if_condition_627427', if_condition_627427)
    # SSA begins for if statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 138):
    
    # Assigning a Call to a Name (line 138):
    
    # Call to split(...): (line 138)
    # Processing the call keyword arguments (line 138)
    kwargs_627436 = {}
    
    # Call to replace(...): (line 138)
    # Processing the call arguments (line 138)
    str_627431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 40), 'str', ',')
    str_627432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 45), 'str', ' ')
    # Processing the call keyword arguments (line 138)
    kwargs_627433 = {}
    # Getting the type of 'distfn' (line 138)
    distfn_627428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 18), 'distfn', False)
    # Obtaining the member 'shapes' of a type (line 138)
    shapes_627429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 18), distfn_627428, 'shapes')
    # Obtaining the member 'replace' of a type (line 138)
    replace_627430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 18), shapes_627429, 'replace')
    # Calling replace(args, kwargs) (line 138)
    replace_call_result_627434 = invoke(stypy.reporting.localization.Localization(__file__, 138, 18), replace_627430, *[str_627431, str_627432], **kwargs_627433)
    
    # Obtaining the member 'split' of a type (line 138)
    split_627435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 18), replace_call_result_627434, 'split')
    # Calling split(args, kwargs) (line 138)
    split_call_result_627437 = invoke(stypy.reporting.localization.Localization(__file__, 138, 18), split_627435, *[], **kwargs_627436)
    
    # Assigning a type to the variable 'shapes_' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'shapes_', split_call_result_627437)
    # SSA branch for the else part of an if statement (line 137)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 140):
    
    # Assigning a Str to a Name (line 140):
    str_627438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 18), 'str', '')
    # Assigning a type to the variable 'shapes_' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'shapes_', str_627438)
    # SSA join for if statement (line 137)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to assert_(...): (line 141)
    # Processing the call arguments (line 141)
    
    
    # Call to len(...): (line 141)
    # Processing the call arguments (line 141)
    # Getting the type of 'shapes_' (line 141)
    shapes__627442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'shapes_', False)
    # Processing the call keyword arguments (line 141)
    kwargs_627443 = {}
    # Getting the type of 'len' (line 141)
    len_627441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'len', False)
    # Calling len(args, kwargs) (line 141)
    len_call_result_627444 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), len_627441, *[shapes__627442], **kwargs_627443)
    
    # Getting the type of 'distfn' (line 141)
    distfn_627445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 32), 'distfn', False)
    # Obtaining the member 'numargs' of a type (line 141)
    numargs_627446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 32), distfn_627445, 'numargs')
    # Applying the binary operator '==' (line 141)
    result_eq_627447 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 16), '==', len_call_result_627444, numargs_627446)
    
    # Processing the call keyword arguments (line 141)
    kwargs_627448 = {}
    # Getting the type of 'npt' (line 141)
    npt_627439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 141)
    assert__627440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 4), npt_627439, 'assert_')
    # Calling assert_(args, kwargs) (line 141)
    assert__call_result_627449 = invoke(stypy.reporting.localization.Localization(__file__, 141, 4), assert__627440, *[result_eq_627447], **kwargs_627448)
    
    
    # Call to assert_(...): (line 142)
    # Processing the call arguments (line 142)
    
    
    # Call to len(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'shapes_' (line 142)
    shapes__627453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), 'shapes_', False)
    # Processing the call keyword arguments (line 142)
    kwargs_627454 = {}
    # Getting the type of 'len' (line 142)
    len_627452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'len', False)
    # Calling len(args, kwargs) (line 142)
    len_call_result_627455 = invoke(stypy.reporting.localization.Localization(__file__, 142, 16), len_627452, *[shapes__627453], **kwargs_627454)
    
    
    # Call to len(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'shape_argnames' (line 142)
    shape_argnames_627457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 36), 'shape_argnames', False)
    # Processing the call keyword arguments (line 142)
    kwargs_627458 = {}
    # Getting the type of 'len' (line 142)
    len_627456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 32), 'len', False)
    # Calling len(args, kwargs) (line 142)
    len_call_result_627459 = invoke(stypy.reporting.localization.Localization(__file__, 142, 32), len_627456, *[shape_argnames_627457], **kwargs_627458)
    
    # Applying the binary operator '==' (line 142)
    result_eq_627460 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 16), '==', len_call_result_627455, len_call_result_627459)
    
    # Processing the call keyword arguments (line 142)
    kwargs_627461 = {}
    # Getting the type of 'npt' (line 142)
    npt_627450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 142)
    assert__627451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 4), npt_627450, 'assert_')
    # Calling assert_(args, kwargs) (line 142)
    assert__call_result_627462 = invoke(stypy.reporting.localization.Localization(__file__, 142, 4), assert__627451, *[result_eq_627460], **kwargs_627461)
    
    
    # Assigning a Call to a Name (line 145):
    
    # Assigning a Call to a Name (line 145):
    
    # Call to list(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'shape_args' (line 145)
    shape_args_627464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 22), 'shape_args', False)
    # Processing the call keyword arguments (line 145)
    kwargs_627465 = {}
    # Getting the type of 'list' (line 145)
    list_627463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 17), 'list', False)
    # Calling list(args, kwargs) (line 145)
    list_call_result_627466 = invoke(stypy.reporting.localization.Localization(__file__, 145, 17), list_627463, *[shape_args_627464], **kwargs_627465)
    
    # Assigning a type to the variable 'shape_args' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'shape_args', list_call_result_627466)
    
    # Assigning a ListComp to a Name (line 147):
    
    # Assigning a ListComp to a Name (line 147):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'meths' (line 147)
    meths_627472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 45), 'meths')
    comprehension_627473 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 12), meths_627472)
    # Assigning a type to the variable 'meth' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'meth', comprehension_627473)
    
    # Call to meth(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'x' (line 147)
    x_627468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 17), 'x', False)
    # Getting the type of 'shape_args' (line 147)
    shape_args_627469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 21), 'shape_args', False)
    # Processing the call keyword arguments (line 147)
    kwargs_627470 = {}
    # Getting the type of 'meth' (line 147)
    meth_627467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'meth', False)
    # Calling meth(args, kwargs) (line 147)
    meth_call_result_627471 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), meth_627467, *[x_627468, shape_args_627469], **kwargs_627470)
    
    list_627474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 12), list_627474, meth_call_result_627471)
    # Assigning a type to the variable 'vals' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'vals', list_627474)
    
    # Call to assert_(...): (line 148)
    # Processing the call arguments (line 148)
    
    # Call to all(...): (line 148)
    # Processing the call arguments (line 148)
    
    # Call to isfinite(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'vals' (line 148)
    vals_627481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 35), 'vals', False)
    # Processing the call keyword arguments (line 148)
    kwargs_627482 = {}
    # Getting the type of 'np' (line 148)
    np_627479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 23), 'np', False)
    # Obtaining the member 'isfinite' of a type (line 148)
    isfinite_627480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 23), np_627479, 'isfinite')
    # Calling isfinite(args, kwargs) (line 148)
    isfinite_call_result_627483 = invoke(stypy.reporting.localization.Localization(__file__, 148, 23), isfinite_627480, *[vals_627481], **kwargs_627482)
    
    # Processing the call keyword arguments (line 148)
    kwargs_627484 = {}
    # Getting the type of 'np' (line 148)
    np_627477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'np', False)
    # Obtaining the member 'all' of a type (line 148)
    all_627478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 16), np_627477, 'all')
    # Calling all(args, kwargs) (line 148)
    all_call_result_627485 = invoke(stypy.reporting.localization.Localization(__file__, 148, 16), all_627478, *[isfinite_call_result_627483], **kwargs_627484)
    
    # Processing the call keyword arguments (line 148)
    kwargs_627486 = {}
    # Getting the type of 'npt' (line 148)
    npt_627475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 148)
    assert__627476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 4), npt_627475, 'assert_')
    # Calling assert_(args, kwargs) (line 148)
    assert__call_result_627487 = invoke(stypy.reporting.localization.Localization(__file__, 148, 4), assert__627476, *[all_call_result_627485], **kwargs_627486)
    
    
    # Assigning a Tuple to a Tuple (line 150):
    
    # Assigning a Subscript to a Name (line 150):
    
    # Obtaining the type of the subscript
    slice_627488 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 150, 18), None, None, None)
    # Getting the type of 'shape_argnames' (line 150)
    shape_argnames_627489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 18), 'shape_argnames')
    # Obtaining the member '__getitem__' of a type (line 150)
    getitem___627490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 18), shape_argnames_627489, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 150)
    subscript_call_result_627491 = invoke(stypy.reporting.localization.Localization(__file__, 150, 18), getitem___627490, slice_627488)
    
    # Assigning a type to the variable 'tuple_assignment_626823' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_assignment_626823', subscript_call_result_627491)
    
    # Assigning a Subscript to a Name (line 150):
    
    # Obtaining the type of the subscript
    slice_627492 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 150, 37), None, None, None)
    # Getting the type of 'shape_args' (line 150)
    shape_args_627493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 37), 'shape_args')
    # Obtaining the member '__getitem__' of a type (line 150)
    getitem___627494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 37), shape_args_627493, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 150)
    subscript_call_result_627495 = invoke(stypy.reporting.localization.Localization(__file__, 150, 37), getitem___627494, slice_627492)
    
    # Assigning a type to the variable 'tuple_assignment_626824' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_assignment_626824', subscript_call_result_627495)
    
    # Assigning a Dict to a Name (line 150):
    
    # Obtaining an instance of the builtin type 'dict' (line 150)
    dict_627496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 52), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 150)
    
    # Assigning a type to the variable 'tuple_assignment_626825' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_assignment_626825', dict_627496)
    
    # Assigning a Name to a Name (line 150):
    # Getting the type of 'tuple_assignment_626823' (line 150)
    tuple_assignment_626823_627497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_assignment_626823')
    # Assigning a type to the variable 'names' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'names', tuple_assignment_626823_627497)
    
    # Assigning a Name to a Name (line 150):
    # Getting the type of 'tuple_assignment_626824' (line 150)
    tuple_assignment_626824_627498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_assignment_626824')
    # Assigning a type to the variable 'a' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'a', tuple_assignment_626824_627498)
    
    # Assigning a Name to a Name (line 150):
    # Getting the type of 'tuple_assignment_626825' (line 150)
    tuple_assignment_626825_627499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'tuple_assignment_626825')
    # Assigning a type to the variable 'k' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 14), 'k', tuple_assignment_626825_627499)
    
    # Getting the type of 'names' (line 151)
    names_627500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 10), 'names')
    # Testing the type of an if condition (line 151)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 4), names_627500)
    # SSA begins for while statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Call to update(...): (line 152)
    # Processing the call arguments (line 152)
    
    # Obtaining an instance of the builtin type 'dict' (line 152)
    dict_627503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 17), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 152)
    # Adding element type (key, value) (line 152)
    
    # Call to pop(...): (line 152)
    # Processing the call keyword arguments (line 152)
    kwargs_627506 = {}
    # Getting the type of 'names' (line 152)
    names_627504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'names', False)
    # Obtaining the member 'pop' of a type (line 152)
    pop_627505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 18), names_627504, 'pop')
    # Calling pop(args, kwargs) (line 152)
    pop_call_result_627507 = invoke(stypy.reporting.localization.Localization(__file__, 152, 18), pop_627505, *[], **kwargs_627506)
    
    
    # Call to pop(...): (line 152)
    # Processing the call keyword arguments (line 152)
    kwargs_627510 = {}
    # Getting the type of 'a' (line 152)
    a_627508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 31), 'a', False)
    # Obtaining the member 'pop' of a type (line 152)
    pop_627509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 31), a_627508, 'pop')
    # Calling pop(args, kwargs) (line 152)
    pop_call_result_627511 = invoke(stypy.reporting.localization.Localization(__file__, 152, 31), pop_627509, *[], **kwargs_627510)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 17), dict_627503, (pop_call_result_627507, pop_call_result_627511))
    
    # Processing the call keyword arguments (line 152)
    kwargs_627512 = {}
    # Getting the type of 'k' (line 152)
    k_627501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'k', False)
    # Obtaining the member 'update' of a type (line 152)
    update_627502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), k_627501, 'update')
    # Calling update(args, kwargs) (line 152)
    update_call_result_627513 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), update_627502, *[dict_627503], **kwargs_627512)
    
    
    # Assigning a ListComp to a Name (line 153):
    
    # Assigning a ListComp to a Name (line 153):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'meths' (line 153)
    meths_627520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 42), 'meths')
    comprehension_627521 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 13), meths_627520)
    # Assigning a type to the variable 'meth' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 13), 'meth', comprehension_627521)
    
    # Call to meth(...): (line 153)
    # Processing the call arguments (line 153)
    # Getting the type of 'x' (line 153)
    x_627515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 18), 'x', False)
    # Getting the type of 'a' (line 153)
    a_627516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 22), 'a', False)
    # Processing the call keyword arguments (line 153)
    # Getting the type of 'k' (line 153)
    k_627517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 27), 'k', False)
    kwargs_627518 = {'k_627517': k_627517}
    # Getting the type of 'meth' (line 153)
    meth_627514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 13), 'meth', False)
    # Calling meth(args, kwargs) (line 153)
    meth_call_result_627519 = invoke(stypy.reporting.localization.Localization(__file__, 153, 13), meth_627514, *[x_627515, a_627516], **kwargs_627518)
    
    list_627522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 13), list_627522, meth_call_result_627519)
    # Assigning a type to the variable 'v' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'v', list_627522)
    
    # Call to assert_array_equal(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'vals' (line 154)
    vals_627525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 31), 'vals', False)
    # Getting the type of 'v' (line 154)
    v_627526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 37), 'v', False)
    # Processing the call keyword arguments (line 154)
    kwargs_627527 = {}
    # Getting the type of 'npt' (line 154)
    npt_627523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'npt', False)
    # Obtaining the member 'assert_array_equal' of a type (line 154)
    assert_array_equal_627524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), npt_627523, 'assert_array_equal')
    # Calling assert_array_equal(args, kwargs) (line 154)
    assert_array_equal_call_result_627528 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), assert_array_equal_627524, *[vals_627525, v_627526], **kwargs_627527)
    
    
    
    str_627529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 11), 'str', 'n')
    
    # Call to keys(...): (line 155)
    # Processing the call keyword arguments (line 155)
    kwargs_627532 = {}
    # Getting the type of 'k' (line 155)
    k_627530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'k', False)
    # Obtaining the member 'keys' of a type (line 155)
    keys_627531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 22), k_627530, 'keys')
    # Calling keys(args, kwargs) (line 155)
    keys_call_result_627533 = invoke(stypy.reporting.localization.Localization(__file__, 155, 22), keys_627531, *[], **kwargs_627532)
    
    # Applying the binary operator 'notin' (line 155)
    result_contains_627534 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 11), 'notin', str_627529, keys_call_result_627533)
    
    # Testing the type of an if condition (line 155)
    if_condition_627535 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 8), result_contains_627534)
    # Assigning a type to the variable 'if_condition_627535' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'if_condition_627535', if_condition_627535)
    # SSA begins for if statement (line 155)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_equal(...): (line 157)
    # Processing the call arguments (line 157)
    
    # Call to moment(...): (line 157)
    # Processing the call arguments (line 157)
    int_627540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 43), 'int')
    # Getting the type of 'a' (line 157)
    a_627541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 47), 'a', False)
    # Processing the call keyword arguments (line 157)
    # Getting the type of 'k' (line 157)
    k_627542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 52), 'k', False)
    kwargs_627543 = {'k_627542': k_627542}
    # Getting the type of 'distfn' (line 157)
    distfn_627538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 29), 'distfn', False)
    # Obtaining the member 'moment' of a type (line 157)
    moment_627539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 29), distfn_627538, 'moment')
    # Calling moment(args, kwargs) (line 157)
    moment_call_result_627544 = invoke(stypy.reporting.localization.Localization(__file__, 157, 29), moment_627539, *[int_627540, a_627541], **kwargs_627543)
    
    
    # Call to moment(...): (line 158)
    # Processing the call arguments (line 158)
    int_627547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 43), 'int')
    # Getting the type of 'shape_args' (line 158)
    shape_args_627548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 47), 'shape_args', False)
    # Processing the call keyword arguments (line 158)
    kwargs_627549 = {}
    # Getting the type of 'distfn' (line 158)
    distfn_627545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 29), 'distfn', False)
    # Obtaining the member 'moment' of a type (line 158)
    moment_627546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 29), distfn_627545, 'moment')
    # Calling moment(args, kwargs) (line 158)
    moment_call_result_627550 = invoke(stypy.reporting.localization.Localization(__file__, 158, 29), moment_627546, *[int_627547, shape_args_627548], **kwargs_627549)
    
    # Processing the call keyword arguments (line 157)
    kwargs_627551 = {}
    # Getting the type of 'npt' (line 157)
    npt_627536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'npt', False)
    # Obtaining the member 'assert_equal' of a type (line 157)
    assert_equal_627537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 12), npt_627536, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 157)
    assert_equal_call_result_627552 = invoke(stypy.reporting.localization.Localization(__file__, 157, 12), assert_equal_627537, *[moment_call_result_627544, moment_call_result_627550], **kwargs_627551)
    
    # SSA join for if statement (line 155)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 151)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to update(...): (line 161)
    # Processing the call arguments (line 161)
    
    # Obtaining an instance of the builtin type 'dict' (line 161)
    dict_627555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 161)
    # Adding element type (key, value) (line 161)
    str_627556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 14), 'str', 'kaboom')
    int_627557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 24), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 13), dict_627555, (str_627556, int_627557))
    
    # Processing the call keyword arguments (line 161)
    kwargs_627558 = {}
    # Getting the type of 'k' (line 161)
    k_627553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'k', False)
    # Obtaining the member 'update' of a type (line 161)
    update_627554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 4), k_627553, 'update')
    # Calling update(args, kwargs) (line 161)
    update_call_result_627559 = invoke(stypy.reporting.localization.Localization(__file__, 161, 4), update_627554, *[dict_627555], **kwargs_627558)
    
    
    # Call to assert_raises(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'TypeError' (line 162)
    TypeError_627561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 18), 'TypeError', False)
    # Getting the type of 'distfn' (line 162)
    distfn_627562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 29), 'distfn', False)
    # Obtaining the member 'cdf' of a type (line 162)
    cdf_627563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 29), distfn_627562, 'cdf')
    # Getting the type of 'x' (line 162)
    x_627564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 41), 'x', False)
    # Processing the call keyword arguments (line 162)
    # Getting the type of 'k' (line 162)
    k_627565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 46), 'k', False)
    kwargs_627566 = {'k_627565': k_627565}
    # Getting the type of 'assert_raises' (line 162)
    assert_raises_627560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 162)
    assert_raises_call_result_627567 = invoke(stypy.reporting.localization.Localization(__file__, 162, 4), assert_raises_627560, *[TypeError_627561, cdf_627563, x_627564], **kwargs_627566)
    
    
    # ################# End of 'check_named_args(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_named_args' in the type store
    # Getting the type of 'stypy_return_type' (line 127)
    stypy_return_type_627568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_627568)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_named_args'
    return stypy_return_type_627568

# Assigning a type to the variable 'check_named_args' (line 127)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'check_named_args', check_named_args)

@norecursion
def check_random_state_property(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_random_state_property'
    module_type_store = module_type_store.open_function_context('check_random_state_property', 165, 0, False)
    
    # Passed parameters checking function
    check_random_state_property.stypy_localization = localization
    check_random_state_property.stypy_type_of_self = None
    check_random_state_property.stypy_type_store = module_type_store
    check_random_state_property.stypy_function_name = 'check_random_state_property'
    check_random_state_property.stypy_param_names_list = ['distfn', 'args']
    check_random_state_property.stypy_varargs_param_name = None
    check_random_state_property.stypy_kwargs_param_name = None
    check_random_state_property.stypy_call_defaults = defaults
    check_random_state_property.stypy_call_varargs = varargs
    check_random_state_property.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_random_state_property', ['distfn', 'args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_random_state_property', localization, ['distfn', 'args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_random_state_property(...)' code ##################

    
    # Assigning a Attribute to a Name (line 170):
    
    # Assigning a Attribute to a Name (line 170):
    # Getting the type of 'distfn' (line 170)
    distfn_627569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'distfn')
    # Obtaining the member 'random_state' of a type (line 170)
    random_state_627570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 11), distfn_627569, 'random_state')
    # Assigning a type to the variable 'rndm' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'rndm', random_state_627570)
    
    # Call to seed(...): (line 173)
    # Processing the call arguments (line 173)
    int_627574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 19), 'int')
    # Processing the call keyword arguments (line 173)
    kwargs_627575 = {}
    # Getting the type of 'np' (line 173)
    np_627571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 173)
    random_627572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 4), np_627571, 'random')
    # Obtaining the member 'seed' of a type (line 173)
    seed_627573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 4), random_627572, 'seed')
    # Calling seed(args, kwargs) (line 173)
    seed_call_result_627576 = invoke(stypy.reporting.localization.Localization(__file__, 173, 4), seed_627573, *[int_627574], **kwargs_627575)
    
    
    # Assigning a Name to a Attribute (line 174):
    
    # Assigning a Name to a Attribute (line 174):
    # Getting the type of 'None' (line 174)
    None_627577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 26), 'None')
    # Getting the type of 'distfn' (line 174)
    distfn_627578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'distfn')
    # Setting the type of the member 'random_state' of a type (line 174)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 4), distfn_627578, 'random_state', None_627577)
    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to rvs(...): (line 175)
    # Getting the type of 'args' (line 175)
    args_627581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 21), 'args', False)
    # Processing the call keyword arguments (line 175)
    int_627582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 32), 'int')
    keyword_627583 = int_627582
    kwargs_627584 = {'size': keyword_627583}
    # Getting the type of 'distfn' (line 175)
    distfn_627579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 9), 'distfn', False)
    # Obtaining the member 'rvs' of a type (line 175)
    rvs_627580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 9), distfn_627579, 'rvs')
    # Calling rvs(args, kwargs) (line 175)
    rvs_call_result_627585 = invoke(stypy.reporting.localization.Localization(__file__, 175, 9), rvs_627580, *[args_627581], **kwargs_627584)
    
    # Assigning a type to the variable 'r0' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'r0', rvs_call_result_627585)
    
    # Assigning a Num to a Attribute (line 178):
    
    # Assigning a Num to a Attribute (line 178):
    int_627586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 26), 'int')
    # Getting the type of 'distfn' (line 178)
    distfn_627587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'distfn')
    # Setting the type of the member 'random_state' of a type (line 178)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 4), distfn_627587, 'random_state', int_627586)
    
    # Assigning a Call to a Name (line 179):
    
    # Assigning a Call to a Name (line 179):
    
    # Call to rvs(...): (line 179)
    # Getting the type of 'args' (line 179)
    args_627590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 21), 'args', False)
    # Processing the call keyword arguments (line 179)
    int_627591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 32), 'int')
    keyword_627592 = int_627591
    kwargs_627593 = {'size': keyword_627592}
    # Getting the type of 'distfn' (line 179)
    distfn_627588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 9), 'distfn', False)
    # Obtaining the member 'rvs' of a type (line 179)
    rvs_627589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 9), distfn_627588, 'rvs')
    # Calling rvs(args, kwargs) (line 179)
    rvs_call_result_627594 = invoke(stypy.reporting.localization.Localization(__file__, 179, 9), rvs_627589, *[args_627590], **kwargs_627593)
    
    # Assigning a type to the variable 'r1' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'r1', rvs_call_result_627594)
    
    # Call to assert_equal(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'r0' (line 180)
    r0_627597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 21), 'r0', False)
    # Getting the type of 'r1' (line 180)
    r1_627598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 25), 'r1', False)
    # Processing the call keyword arguments (line 180)
    kwargs_627599 = {}
    # Getting the type of 'npt' (line 180)
    npt_627595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'npt', False)
    # Obtaining the member 'assert_equal' of a type (line 180)
    assert_equal_627596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 4), npt_627595, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 180)
    assert_equal_call_result_627600 = invoke(stypy.reporting.localization.Localization(__file__, 180, 4), assert_equal_627596, *[r0_627597, r1_627598], **kwargs_627599)
    
    
    # Assigning a Call to a Attribute (line 182):
    
    # Assigning a Call to a Attribute (line 182):
    
    # Call to RandomState(...): (line 182)
    # Processing the call arguments (line 182)
    int_627604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 48), 'int')
    # Processing the call keyword arguments (line 182)
    kwargs_627605 = {}
    # Getting the type of 'np' (line 182)
    np_627601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 26), 'np', False)
    # Obtaining the member 'random' of a type (line 182)
    random_627602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 26), np_627601, 'random')
    # Obtaining the member 'RandomState' of a type (line 182)
    RandomState_627603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 26), random_627602, 'RandomState')
    # Calling RandomState(args, kwargs) (line 182)
    RandomState_call_result_627606 = invoke(stypy.reporting.localization.Localization(__file__, 182, 26), RandomState_627603, *[int_627604], **kwargs_627605)
    
    # Getting the type of 'distfn' (line 182)
    distfn_627607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'distfn')
    # Setting the type of the member 'random_state' of a type (line 182)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 4), distfn_627607, 'random_state', RandomState_call_result_627606)
    
    # Assigning a Call to a Name (line 183):
    
    # Assigning a Call to a Name (line 183):
    
    # Call to rvs(...): (line 183)
    # Getting the type of 'args' (line 183)
    args_627610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 21), 'args', False)
    # Processing the call keyword arguments (line 183)
    int_627611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 32), 'int')
    keyword_627612 = int_627611
    kwargs_627613 = {'size': keyword_627612}
    # Getting the type of 'distfn' (line 183)
    distfn_627608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 9), 'distfn', False)
    # Obtaining the member 'rvs' of a type (line 183)
    rvs_627609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 9), distfn_627608, 'rvs')
    # Calling rvs(args, kwargs) (line 183)
    rvs_call_result_627614 = invoke(stypy.reporting.localization.Localization(__file__, 183, 9), rvs_627609, *[args_627610], **kwargs_627613)
    
    # Assigning a type to the variable 'r2' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'r2', rvs_call_result_627614)
    
    # Call to assert_equal(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'r0' (line 184)
    r0_627617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), 'r0', False)
    # Getting the type of 'r2' (line 184)
    r2_627618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 25), 'r2', False)
    # Processing the call keyword arguments (line 184)
    kwargs_627619 = {}
    # Getting the type of 'npt' (line 184)
    npt_627615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'npt', False)
    # Obtaining the member 'assert_equal' of a type (line 184)
    assert_equal_627616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 4), npt_627615, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 184)
    assert_equal_call_result_627620 = invoke(stypy.reporting.localization.Localization(__file__, 184, 4), assert_equal_627616, *[r0_627617, r2_627618], **kwargs_627619)
    
    
    # Assigning a Num to a Attribute (line 187):
    
    # Assigning a Num to a Attribute (line 187):
    int_627621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 26), 'int')
    # Getting the type of 'distfn' (line 187)
    distfn_627622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'distfn')
    # Setting the type of the member 'random_state' of a type (line 187)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 4), distfn_627622, 'random_state', int_627621)
    
    # Assigning a Call to a Name (line 188):
    
    # Assigning a Call to a Name (line 188):
    
    # Call to get_state(...): (line 188)
    # Processing the call keyword arguments (line 188)
    kwargs_627626 = {}
    # Getting the type of 'distfn' (line 188)
    distfn_627623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 17), 'distfn', False)
    # Obtaining the member 'random_state' of a type (line 188)
    random_state_627624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 17), distfn_627623, 'random_state')
    # Obtaining the member 'get_state' of a type (line 188)
    get_state_627625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 17), random_state_627624, 'get_state')
    # Calling get_state(args, kwargs) (line 188)
    get_state_call_result_627627 = invoke(stypy.reporting.localization.Localization(__file__, 188, 17), get_state_627625, *[], **kwargs_627626)
    
    # Assigning a type to the variable 'orig_state' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'orig_state', get_state_call_result_627627)
    
    # Assigning a Call to a Name (line 190):
    
    # Assigning a Call to a Name (line 190):
    
    # Call to rvs(...): (line 190)
    # Getting the type of 'args' (line 190)
    args_627630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 21), 'args', False)
    # Processing the call keyword arguments (line 190)
    int_627631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 32), 'int')
    keyword_627632 = int_627631
    
    # Call to RandomState(...): (line 190)
    # Processing the call arguments (line 190)
    int_627636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 70), 'int')
    # Processing the call keyword arguments (line 190)
    kwargs_627637 = {}
    # Getting the type of 'np' (line 190)
    np_627633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 48), 'np', False)
    # Obtaining the member 'random' of a type (line 190)
    random_627634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 48), np_627633, 'random')
    # Obtaining the member 'RandomState' of a type (line 190)
    RandomState_627635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 48), random_627634, 'RandomState')
    # Calling RandomState(args, kwargs) (line 190)
    RandomState_call_result_627638 = invoke(stypy.reporting.localization.Localization(__file__, 190, 48), RandomState_627635, *[int_627636], **kwargs_627637)
    
    keyword_627639 = RandomState_call_result_627638
    kwargs_627640 = {'random_state': keyword_627639, 'size': keyword_627632}
    # Getting the type of 'distfn' (line 190)
    distfn_627628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 9), 'distfn', False)
    # Obtaining the member 'rvs' of a type (line 190)
    rvs_627629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 9), distfn_627628, 'rvs')
    # Calling rvs(args, kwargs) (line 190)
    rvs_call_result_627641 = invoke(stypy.reporting.localization.Localization(__file__, 190, 9), rvs_627629, *[args_627630], **kwargs_627640)
    
    # Assigning a type to the variable 'r3' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'r3', rvs_call_result_627641)
    
    # Call to assert_equal(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'r0' (line 191)
    r0_627644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 21), 'r0', False)
    # Getting the type of 'r3' (line 191)
    r3_627645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 25), 'r3', False)
    # Processing the call keyword arguments (line 191)
    kwargs_627646 = {}
    # Getting the type of 'npt' (line 191)
    npt_627642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'npt', False)
    # Obtaining the member 'assert_equal' of a type (line 191)
    assert_equal_627643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 4), npt_627642, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 191)
    assert_equal_call_result_627647 = invoke(stypy.reporting.localization.Localization(__file__, 191, 4), assert_equal_627643, *[r0_627644, r3_627645], **kwargs_627646)
    
    
    # Call to assert_equal(...): (line 194)
    # Processing the call arguments (line 194)
    
    # Call to get_state(...): (line 194)
    # Processing the call keyword arguments (line 194)
    kwargs_627653 = {}
    # Getting the type of 'distfn' (line 194)
    distfn_627650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 21), 'distfn', False)
    # Obtaining the member 'random_state' of a type (line 194)
    random_state_627651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 21), distfn_627650, 'random_state')
    # Obtaining the member 'get_state' of a type (line 194)
    get_state_627652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 21), random_state_627651, 'get_state')
    # Calling get_state(args, kwargs) (line 194)
    get_state_call_result_627654 = invoke(stypy.reporting.localization.Localization(__file__, 194, 21), get_state_627652, *[], **kwargs_627653)
    
    # Getting the type of 'orig_state' (line 194)
    orig_state_627655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 54), 'orig_state', False)
    # Processing the call keyword arguments (line 194)
    kwargs_627656 = {}
    # Getting the type of 'npt' (line 194)
    npt_627648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'npt', False)
    # Obtaining the member 'assert_equal' of a type (line 194)
    assert_equal_627649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 4), npt_627648, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 194)
    assert_equal_call_result_627657 = invoke(stypy.reporting.localization.Localization(__file__, 194, 4), assert_equal_627649, *[get_state_call_result_627654, orig_state_627655], **kwargs_627656)
    
    
    # Assigning a Name to a Attribute (line 197):
    
    # Assigning a Name to a Attribute (line 197):
    # Getting the type of 'rndm' (line 197)
    rndm_627658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 26), 'rndm')
    # Getting the type of 'distfn' (line 197)
    distfn_627659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'distfn')
    # Setting the type of the member 'random_state' of a type (line 197)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 4), distfn_627659, 'random_state', rndm_627658)
    
    # ################# End of 'check_random_state_property(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_random_state_property' in the type store
    # Getting the type of 'stypy_return_type' (line 165)
    stypy_return_type_627660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_627660)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_random_state_property'
    return stypy_return_type_627660

# Assigning a type to the variable 'check_random_state_property' (line 165)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'check_random_state_property', check_random_state_property)

@norecursion
def check_meth_dtype(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_meth_dtype'
    module_type_store = module_type_store.open_function_context('check_meth_dtype', 200, 0, False)
    
    # Passed parameters checking function
    check_meth_dtype.stypy_localization = localization
    check_meth_dtype.stypy_type_of_self = None
    check_meth_dtype.stypy_type_store = module_type_store
    check_meth_dtype.stypy_function_name = 'check_meth_dtype'
    check_meth_dtype.stypy_param_names_list = ['distfn', 'arg', 'meths']
    check_meth_dtype.stypy_varargs_param_name = None
    check_meth_dtype.stypy_kwargs_param_name = None
    check_meth_dtype.stypy_call_defaults = defaults
    check_meth_dtype.stypy_call_varargs = varargs
    check_meth_dtype.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_meth_dtype', ['distfn', 'arg', 'meths'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_meth_dtype', localization, ['distfn', 'arg', 'meths'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_meth_dtype(...)' code ##################

    
    # Assigning a List to a Name (line 201):
    
    # Assigning a List to a Name (line 201):
    
    # Obtaining an instance of the builtin type 'list' (line 201)
    list_627661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 9), 'list')
    # Adding type elements to the builtin type 'list' instance (line 201)
    # Adding element type (line 201)
    float_627662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 10), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 9), list_627661, float_627662)
    # Adding element type (line 201)
    float_627663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 16), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 9), list_627661, float_627663)
    # Adding element type (line 201)
    float_627664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 9), list_627661, float_627664)
    
    # Assigning a type to the variable 'q0' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'q0', list_627661)
    
    # Assigning a Call to a Name (line 202):
    
    # Assigning a Call to a Name (line 202):
    
    # Call to ppf(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'q0' (line 202)
    q0_627667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), 'q0', False)
    # Getting the type of 'arg' (line 202)
    arg_627668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 25), 'arg', False)
    # Processing the call keyword arguments (line 202)
    kwargs_627669 = {}
    # Getting the type of 'distfn' (line 202)
    distfn_627665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 9), 'distfn', False)
    # Obtaining the member 'ppf' of a type (line 202)
    ppf_627666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 9), distfn_627665, 'ppf')
    # Calling ppf(args, kwargs) (line 202)
    ppf_call_result_627670 = invoke(stypy.reporting.localization.Localization(__file__, 202, 9), ppf_627666, *[q0_627667, arg_627668], **kwargs_627669)
    
    # Assigning a type to the variable 'x0' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'x0', ppf_call_result_627670)
    
    # Assigning a ListComp to a Name (line 203):
    
    # Assigning a ListComp to a Name (line 203):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'tuple' (line 204)
    tuple_627676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 204)
    # Adding element type (line 204)
    # Getting the type of 'np' (line 204)
    np_627677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 25), 'np')
    # Obtaining the member 'int_' of a type (line 204)
    int__627678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 25), np_627677, 'int_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 25), tuple_627676, int__627678)
    # Adding element type (line 204)
    # Getting the type of 'np' (line 204)
    np_627679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 34), 'np')
    # Obtaining the member 'float16' of a type (line 204)
    float16_627680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 34), np_627679, 'float16')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 25), tuple_627676, float16_627680)
    # Adding element type (line 204)
    # Getting the type of 'np' (line 204)
    np_627681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 46), 'np')
    # Obtaining the member 'float32' of a type (line 204)
    float32_627682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 46), np_627681, 'float32')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 25), tuple_627676, float32_627682)
    # Adding element type (line 204)
    # Getting the type of 'np' (line 204)
    np_627683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 58), 'np')
    # Obtaining the member 'float64' of a type (line 204)
    float64_627684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 58), np_627683, 'float64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 25), tuple_627676, float64_627684)
    
    comprehension_627685 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 14), tuple_627676)
    # Assigning a type to the variable 'tp' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 14), 'tp', comprehension_627685)
    
    # Call to astype(...): (line 203)
    # Processing the call arguments (line 203)
    # Getting the type of 'tp' (line 203)
    tp_627673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 24), 'tp', False)
    # Processing the call keyword arguments (line 203)
    kwargs_627674 = {}
    # Getting the type of 'x0' (line 203)
    x0_627671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 14), 'x0', False)
    # Obtaining the member 'astype' of a type (line 203)
    astype_627672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 14), x0_627671, 'astype')
    # Calling astype(args, kwargs) (line 203)
    astype_call_result_627675 = invoke(stypy.reporting.localization.Localization(__file__, 203, 14), astype_627672, *[tp_627673], **kwargs_627674)
    
    list_627686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 14), list_627686, astype_call_result_627675)
    # Assigning a type to the variable 'x_cast' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'x_cast', list_627686)
    
    # Getting the type of 'x_cast' (line 206)
    x_cast_627687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 13), 'x_cast')
    # Testing the type of a for loop iterable (line 206)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 206, 4), x_cast_627687)
    # Getting the type of the for loop variable (line 206)
    for_loop_var_627688 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 206, 4), x_cast_627687)
    # Assigning a type to the variable 'x' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'x', for_loop_var_627688)
    # SSA begins for a for statement (line 206)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to _argcheck(...): (line 208)
    # Getting the type of 'arg' (line 208)
    arg_627691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 26), 'arg', False)
    # Processing the call keyword arguments (line 208)
    kwargs_627692 = {}
    # Getting the type of 'distfn' (line 208)
    distfn_627689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'distfn', False)
    # Obtaining the member '_argcheck' of a type (line 208)
    _argcheck_627690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), distfn_627689, '_argcheck')
    # Calling _argcheck(args, kwargs) (line 208)
    _argcheck_call_result_627693 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), _argcheck_627690, *[arg_627691], **kwargs_627692)
    
    
    # Assigning a Subscript to a Name (line 209):
    
    # Assigning a Subscript to a Name (line 209):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'distfn' (line 209)
    distfn_627694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'distfn')
    # Obtaining the member 'a' of a type (line 209)
    a_627695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 15), distfn_627694, 'a')
    # Getting the type of 'x' (line 209)
    x_627696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 26), 'x')
    # Applying the binary operator '<' (line 209)
    result_lt_627697 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 15), '<', a_627695, x_627696)
    
    
    # Getting the type of 'x' (line 209)
    x_627698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 32), 'x')
    # Getting the type of 'distfn' (line 209)
    distfn_627699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 36), 'distfn')
    # Obtaining the member 'b' of a type (line 209)
    b_627700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 36), distfn_627699, 'b')
    # Applying the binary operator '<' (line 209)
    result_lt_627701 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 32), '<', x_627698, b_627700)
    
    # Applying the binary operator '&' (line 209)
    result_and__627702 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 14), '&', result_lt_627697, result_lt_627701)
    
    # Getting the type of 'x' (line 209)
    x_627703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'x')
    # Obtaining the member '__getitem__' of a type (line 209)
    getitem___627704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 12), x_627703, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 209)
    subscript_call_result_627705 = invoke(stypy.reporting.localization.Localization(__file__, 209, 12), getitem___627704, result_and__627702)
    
    # Assigning a type to the variable 'x' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'x', subscript_call_result_627705)
    
    # Getting the type of 'meths' (line 210)
    meths_627706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 20), 'meths')
    # Testing the type of a for loop iterable (line 210)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 210, 8), meths_627706)
    # Getting the type of the for loop variable (line 210)
    for_loop_var_627707 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 210, 8), meths_627706)
    # Assigning a type to the variable 'meth' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'meth', for_loop_var_627707)
    # SSA begins for a for statement (line 210)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 211):
    
    # Assigning a Call to a Name (line 211):
    
    # Call to meth(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'x' (line 211)
    x_627709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 23), 'x', False)
    # Getting the type of 'arg' (line 211)
    arg_627710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 27), 'arg', False)
    # Processing the call keyword arguments (line 211)
    kwargs_627711 = {}
    # Getting the type of 'meth' (line 211)
    meth_627708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'meth', False)
    # Calling meth(args, kwargs) (line 211)
    meth_call_result_627712 = invoke(stypy.reporting.localization.Localization(__file__, 211, 18), meth_627708, *[x_627709, arg_627710], **kwargs_627711)
    
    # Assigning a type to the variable 'val' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'val', meth_call_result_627712)
    
    # Call to assert_(...): (line 212)
    # Processing the call arguments (line 212)
    
    # Getting the type of 'val' (line 212)
    val_627715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 24), 'val', False)
    # Obtaining the member 'dtype' of a type (line 212)
    dtype_627716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 24), val_627715, 'dtype')
    # Getting the type of 'np' (line 212)
    np_627717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 37), 'np', False)
    # Obtaining the member 'float_' of a type (line 212)
    float__627718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 37), np_627717, 'float_')
    # Applying the binary operator '==' (line 212)
    result_eq_627719 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 24), '==', dtype_627716, float__627718)
    
    # Processing the call keyword arguments (line 212)
    kwargs_627720 = {}
    # Getting the type of 'npt' (line 212)
    npt_627713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 212)
    assert__627714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), npt_627713, 'assert_')
    # Calling assert_(args, kwargs) (line 212)
    assert__call_result_627721 = invoke(stypy.reporting.localization.Localization(__file__, 212, 12), assert__627714, *[result_eq_627719], **kwargs_627720)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_meth_dtype(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_meth_dtype' in the type store
    # Getting the type of 'stypy_return_type' (line 200)
    stypy_return_type_627722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_627722)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_meth_dtype'
    return stypy_return_type_627722

# Assigning a type to the variable 'check_meth_dtype' (line 200)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 0), 'check_meth_dtype', check_meth_dtype)

@norecursion
def check_ppf_dtype(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_ppf_dtype'
    module_type_store = module_type_store.open_function_context('check_ppf_dtype', 215, 0, False)
    
    # Passed parameters checking function
    check_ppf_dtype.stypy_localization = localization
    check_ppf_dtype.stypy_type_of_self = None
    check_ppf_dtype.stypy_type_store = module_type_store
    check_ppf_dtype.stypy_function_name = 'check_ppf_dtype'
    check_ppf_dtype.stypy_param_names_list = ['distfn', 'arg']
    check_ppf_dtype.stypy_varargs_param_name = None
    check_ppf_dtype.stypy_kwargs_param_name = None
    check_ppf_dtype.stypy_call_defaults = defaults
    check_ppf_dtype.stypy_call_varargs = varargs
    check_ppf_dtype.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_ppf_dtype', ['distfn', 'arg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_ppf_dtype', localization, ['distfn', 'arg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_ppf_dtype(...)' code ##################

    
    # Assigning a Call to a Name (line 216):
    
    # Assigning a Call to a Name (line 216):
    
    # Call to asarray(...): (line 216)
    # Processing the call arguments (line 216)
    
    # Obtaining an instance of the builtin type 'list' (line 216)
    list_627725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 216)
    # Adding element type (line 216)
    float_627726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 20), list_627725, float_627726)
    # Adding element type (line 216)
    float_627727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 27), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 20), list_627725, float_627727)
    # Adding element type (line 216)
    float_627728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 32), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 20), list_627725, float_627728)
    
    # Processing the call keyword arguments (line 216)
    kwargs_627729 = {}
    # Getting the type of 'np' (line 216)
    np_627723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 9), 'np', False)
    # Obtaining the member 'asarray' of a type (line 216)
    asarray_627724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 9), np_627723, 'asarray')
    # Calling asarray(args, kwargs) (line 216)
    asarray_call_result_627730 = invoke(stypy.reporting.localization.Localization(__file__, 216, 9), asarray_627724, *[list_627725], **kwargs_627729)
    
    # Assigning a type to the variable 'q0' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'q0', asarray_call_result_627730)
    
    # Assigning a ListComp to a Name (line 217):
    
    # Assigning a ListComp to a Name (line 217):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'tuple' (line 217)
    tuple_627736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 217)
    # Adding element type (line 217)
    # Getting the type of 'np' (line 217)
    np_627737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 39), 'np')
    # Obtaining the member 'float16' of a type (line 217)
    float16_627738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 39), np_627737, 'float16')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 39), tuple_627736, float16_627738)
    # Adding element type (line 217)
    # Getting the type of 'np' (line 217)
    np_627739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 51), 'np')
    # Obtaining the member 'float32' of a type (line 217)
    float32_627740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 51), np_627739, 'float32')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 39), tuple_627736, float32_627740)
    # Adding element type (line 217)
    # Getting the type of 'np' (line 217)
    np_627741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 63), 'np')
    # Obtaining the member 'float64' of a type (line 217)
    float64_627742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 63), np_627741, 'float64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 39), tuple_627736, float64_627742)
    
    comprehension_627743 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 14), tuple_627736)
    # Assigning a type to the variable 'tp' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 14), 'tp', comprehension_627743)
    
    # Call to astype(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'tp' (line 217)
    tp_627733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 24), 'tp', False)
    # Processing the call keyword arguments (line 217)
    kwargs_627734 = {}
    # Getting the type of 'q0' (line 217)
    q0_627731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 14), 'q0', False)
    # Obtaining the member 'astype' of a type (line 217)
    astype_627732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 14), q0_627731, 'astype')
    # Calling astype(args, kwargs) (line 217)
    astype_call_result_627735 = invoke(stypy.reporting.localization.Localization(__file__, 217, 14), astype_627732, *[tp_627733], **kwargs_627734)
    
    list_627744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 14), list_627744, astype_call_result_627735)
    # Assigning a type to the variable 'q_cast' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'q_cast', list_627744)
    
    # Getting the type of 'q_cast' (line 218)
    q_cast_627745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 13), 'q_cast')
    # Testing the type of a for loop iterable (line 218)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 218, 4), q_cast_627745)
    # Getting the type of the for loop variable (line 218)
    for_loop_var_627746 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 218, 4), q_cast_627745)
    # Assigning a type to the variable 'q' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'q', for_loop_var_627746)
    # SSA begins for a for statement (line 218)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining an instance of the builtin type 'list' (line 219)
    list_627747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 219)
    # Adding element type (line 219)
    # Getting the type of 'distfn' (line 219)
    distfn_627748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'distfn')
    # Obtaining the member 'ppf' of a type (line 219)
    ppf_627749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 21), distfn_627748, 'ppf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 20), list_627747, ppf_627749)
    # Adding element type (line 219)
    # Getting the type of 'distfn' (line 219)
    distfn_627750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 33), 'distfn')
    # Obtaining the member 'isf' of a type (line 219)
    isf_627751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 33), distfn_627750, 'isf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 20), list_627747, isf_627751)
    
    # Testing the type of a for loop iterable (line 219)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 219, 8), list_627747)
    # Getting the type of the for loop variable (line 219)
    for_loop_var_627752 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 219, 8), list_627747)
    # Assigning a type to the variable 'meth' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'meth', for_loop_var_627752)
    # SSA begins for a for statement (line 219)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 220):
    
    # Assigning a Call to a Name (line 220):
    
    # Call to meth(...): (line 220)
    # Processing the call arguments (line 220)
    # Getting the type of 'q' (line 220)
    q_627754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 23), 'q', False)
    # Getting the type of 'arg' (line 220)
    arg_627755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 27), 'arg', False)
    # Processing the call keyword arguments (line 220)
    kwargs_627756 = {}
    # Getting the type of 'meth' (line 220)
    meth_627753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 18), 'meth', False)
    # Calling meth(args, kwargs) (line 220)
    meth_call_result_627757 = invoke(stypy.reporting.localization.Localization(__file__, 220, 18), meth_627753, *[q_627754, arg_627755], **kwargs_627756)
    
    # Assigning a type to the variable 'val' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'val', meth_call_result_627757)
    
    # Call to assert_(...): (line 221)
    # Processing the call arguments (line 221)
    
    # Getting the type of 'val' (line 221)
    val_627760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 24), 'val', False)
    # Obtaining the member 'dtype' of a type (line 221)
    dtype_627761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 24), val_627760, 'dtype')
    # Getting the type of 'np' (line 221)
    np_627762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 37), 'np', False)
    # Obtaining the member 'float_' of a type (line 221)
    float__627763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 37), np_627762, 'float_')
    # Applying the binary operator '==' (line 221)
    result_eq_627764 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 24), '==', dtype_627761, float__627763)
    
    # Processing the call keyword arguments (line 221)
    kwargs_627765 = {}
    # Getting the type of 'npt' (line 221)
    npt_627758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'npt', False)
    # Obtaining the member 'assert_' of a type (line 221)
    assert__627759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 12), npt_627758, 'assert_')
    # Calling assert_(args, kwargs) (line 221)
    assert__call_result_627766 = invoke(stypy.reporting.localization.Localization(__file__, 221, 12), assert__627759, *[result_eq_627764], **kwargs_627765)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_ppf_dtype(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_ppf_dtype' in the type store
    # Getting the type of 'stypy_return_type' (line 215)
    stypy_return_type_627767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_627767)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_ppf_dtype'
    return stypy_return_type_627767

# Assigning a type to the variable 'check_ppf_dtype' (line 215)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 0), 'check_ppf_dtype', check_ppf_dtype)

@norecursion
def check_cmplx_deriv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_cmplx_deriv'
    module_type_store = module_type_store.open_function_context('check_cmplx_deriv', 224, 0, False)
    
    # Passed parameters checking function
    check_cmplx_deriv.stypy_localization = localization
    check_cmplx_deriv.stypy_type_of_self = None
    check_cmplx_deriv.stypy_type_store = module_type_store
    check_cmplx_deriv.stypy_function_name = 'check_cmplx_deriv'
    check_cmplx_deriv.stypy_param_names_list = ['distfn', 'arg']
    check_cmplx_deriv.stypy_varargs_param_name = None
    check_cmplx_deriv.stypy_kwargs_param_name = None
    check_cmplx_deriv.stypy_call_defaults = defaults
    check_cmplx_deriv.stypy_call_varargs = varargs
    check_cmplx_deriv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_cmplx_deriv', ['distfn', 'arg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_cmplx_deriv', localization, ['distfn', 'arg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_cmplx_deriv(...)' code ##################


    @norecursion
    def deriv(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'deriv'
        module_type_store = module_type_store.open_function_context('deriv', 226, 4, False)
        
        # Passed parameters checking function
        deriv.stypy_localization = localization
        deriv.stypy_type_of_self = None
        deriv.stypy_type_store = module_type_store
        deriv.stypy_function_name = 'deriv'
        deriv.stypy_param_names_list = ['f', 'x']
        deriv.stypy_varargs_param_name = 'arg'
        deriv.stypy_kwargs_param_name = None
        deriv.stypy_call_defaults = defaults
        deriv.stypy_call_varargs = varargs
        deriv.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'deriv', ['f', 'x'], 'arg', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'deriv', localization, ['f', 'x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'deriv(...)' code ##################

        
        # Assigning a Call to a Name (line 227):
        
        # Assigning a Call to a Name (line 227):
        
        # Call to asarray(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'x' (line 227)
        x_627770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 23), 'x', False)
        # Processing the call keyword arguments (line 227)
        kwargs_627771 = {}
        # Getting the type of 'np' (line 227)
        np_627768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'np', False)
        # Obtaining the member 'asarray' of a type (line 227)
        asarray_627769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 12), np_627768, 'asarray')
        # Calling asarray(args, kwargs) (line 227)
        asarray_call_result_627772 = invoke(stypy.reporting.localization.Localization(__file__, 227, 12), asarray_627769, *[x_627770], **kwargs_627771)
        
        # Assigning a type to the variable 'x' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'x', asarray_call_result_627772)
        
        # Assigning a Num to a Name (line 228):
        
        # Assigning a Num to a Name (line 228):
        float_627773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 12), 'float')
        # Assigning a type to the variable 'h' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'h', float_627773)
        
        # Call to f(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'x' (line 229)
        x_627775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 18), 'x', False)
        # Getting the type of 'h' (line 229)
        h_627776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 22), 'h', False)
        complex_627777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 24), 'complex')
        # Applying the binary operator '*' (line 229)
        result_mul_627778 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 22), '*', h_627776, complex_627777)
        
        # Applying the binary operator '+' (line 229)
        result_add_627779 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 18), '+', x_627775, result_mul_627778)
        
        # Getting the type of 'arg' (line 229)
        arg_627780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 29), 'arg', False)
        # Processing the call keyword arguments (line 229)
        kwargs_627781 = {}
        # Getting the type of 'f' (line 229)
        f_627774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 16), 'f', False)
        # Calling f(args, kwargs) (line 229)
        f_call_result_627782 = invoke(stypy.reporting.localization.Localization(__file__, 229, 16), f_627774, *[result_add_627779, arg_627780], **kwargs_627781)
        
        # Getting the type of 'h' (line 229)
        h_627783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 34), 'h')
        # Applying the binary operator 'div' (line 229)
        result_div_627784 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 16), 'div', f_call_result_627782, h_627783)
        
        # Obtaining the member 'imag' of a type (line 229)
        imag_627785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 16), result_div_627784, 'imag')
        # Assigning a type to the variable 'stypy_return_type' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'stypy_return_type', imag_627785)
        
        # ################# End of 'deriv(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'deriv' in the type store
        # Getting the type of 'stypy_return_type' (line 226)
        stypy_return_type_627786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_627786)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'deriv'
        return stypy_return_type_627786

    # Assigning a type to the variable 'deriv' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'deriv', deriv)
    
    # Assigning a Call to a Name (line 231):
    
    # Assigning a Call to a Name (line 231):
    
    # Call to ppf(...): (line 231)
    # Processing the call arguments (line 231)
    
    # Obtaining an instance of the builtin type 'list' (line 231)
    list_627789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 231)
    # Adding element type (line 231)
    float_627790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 21), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 20), list_627789, float_627790)
    # Adding element type (line 231)
    float_627791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 27), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 20), list_627789, float_627791)
    # Adding element type (line 231)
    float_627792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 33), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 20), list_627789, float_627792)
    
    # Getting the type of 'arg' (line 231)
    arg_627793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 41), 'arg', False)
    # Processing the call keyword arguments (line 231)
    kwargs_627794 = {}
    # Getting the type of 'distfn' (line 231)
    distfn_627787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 9), 'distfn', False)
    # Obtaining the member 'ppf' of a type (line 231)
    ppf_627788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 9), distfn_627787, 'ppf')
    # Calling ppf(args, kwargs) (line 231)
    ppf_call_result_627795 = invoke(stypy.reporting.localization.Localization(__file__, 231, 9), ppf_627788, *[list_627789, arg_627793], **kwargs_627794)
    
    # Assigning a type to the variable 'x0' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'x0', ppf_call_result_627795)
    
    # Assigning a ListComp to a Name (line 232):
    
    # Assigning a ListComp to a Name (line 232):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'tuple' (line 233)
    tuple_627801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 233)
    # Adding element type (line 233)
    # Getting the type of 'np' (line 233)
    np_627802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 25), 'np')
    # Obtaining the member 'int_' of a type (line 233)
    int__627803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 25), np_627802, 'int_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 25), tuple_627801, int__627803)
    # Adding element type (line 233)
    # Getting the type of 'np' (line 233)
    np_627804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 34), 'np')
    # Obtaining the member 'float16' of a type (line 233)
    float16_627805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 34), np_627804, 'float16')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 25), tuple_627801, float16_627805)
    # Adding element type (line 233)
    # Getting the type of 'np' (line 233)
    np_627806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 46), 'np')
    # Obtaining the member 'float32' of a type (line 233)
    float32_627807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 46), np_627806, 'float32')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 25), tuple_627801, float32_627807)
    # Adding element type (line 233)
    # Getting the type of 'np' (line 233)
    np_627808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 58), 'np')
    # Obtaining the member 'float64' of a type (line 233)
    float64_627809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 58), np_627808, 'float64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 25), tuple_627801, float64_627809)
    
    comprehension_627810 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 14), tuple_627801)
    # Assigning a type to the variable 'tp' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 14), 'tp', comprehension_627810)
    
    # Call to astype(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'tp' (line 232)
    tp_627798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 24), 'tp', False)
    # Processing the call keyword arguments (line 232)
    kwargs_627799 = {}
    # Getting the type of 'x0' (line 232)
    x0_627796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 14), 'x0', False)
    # Obtaining the member 'astype' of a type (line 232)
    astype_627797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 14), x0_627796, 'astype')
    # Calling astype(args, kwargs) (line 232)
    astype_call_result_627800 = invoke(stypy.reporting.localization.Localization(__file__, 232, 14), astype_627797, *[tp_627798], **kwargs_627799)
    
    list_627811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 14), list_627811, astype_call_result_627800)
    # Assigning a type to the variable 'x_cast' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'x_cast', list_627811)
    
    # Getting the type of 'x_cast' (line 235)
    x_cast_627812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 13), 'x_cast')
    # Testing the type of a for loop iterable (line 235)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 235, 4), x_cast_627812)
    # Getting the type of the for loop variable (line 235)
    for_loop_var_627813 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 235, 4), x_cast_627812)
    # Assigning a type to the variable 'x' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'x', for_loop_var_627813)
    # SSA begins for a for statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to _argcheck(...): (line 237)
    # Getting the type of 'arg' (line 237)
    arg_627816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 26), 'arg', False)
    # Processing the call keyword arguments (line 237)
    kwargs_627817 = {}
    # Getting the type of 'distfn' (line 237)
    distfn_627814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'distfn', False)
    # Obtaining the member '_argcheck' of a type (line 237)
    _argcheck_627815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), distfn_627814, '_argcheck')
    # Calling _argcheck(args, kwargs) (line 237)
    _argcheck_call_result_627818 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), _argcheck_627815, *[arg_627816], **kwargs_627817)
    
    
    # Assigning a Subscript to a Name (line 238):
    
    # Assigning a Subscript to a Name (line 238):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'distfn' (line 238)
    distfn_627819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'distfn')
    # Obtaining the member 'a' of a type (line 238)
    a_627820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 15), distfn_627819, 'a')
    # Getting the type of 'x' (line 238)
    x_627821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 26), 'x')
    # Applying the binary operator '<' (line 238)
    result_lt_627822 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 15), '<', a_627820, x_627821)
    
    
    # Getting the type of 'x' (line 238)
    x_627823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 32), 'x')
    # Getting the type of 'distfn' (line 238)
    distfn_627824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 36), 'distfn')
    # Obtaining the member 'b' of a type (line 238)
    b_627825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 36), distfn_627824, 'b')
    # Applying the binary operator '<' (line 238)
    result_lt_627826 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 32), '<', x_627823, b_627825)
    
    # Applying the binary operator '&' (line 238)
    result_and__627827 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 14), '&', result_lt_627822, result_lt_627826)
    
    # Getting the type of 'x' (line 238)
    x_627828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'x')
    # Obtaining the member '__getitem__' of a type (line 238)
    getitem___627829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 12), x_627828, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 238)
    subscript_call_result_627830 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), getitem___627829, result_and__627827)
    
    # Assigning a type to the variable 'x' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'x', subscript_call_result_627830)
    
    # Assigning a Tuple to a Tuple (line 240):
    
    # Assigning a Call to a Name (line 240):
    
    # Call to pdf(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'x' (line 240)
    x_627833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 34), 'x', False)
    # Getting the type of 'arg' (line 240)
    arg_627834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 38), 'arg', False)
    # Processing the call keyword arguments (line 240)
    kwargs_627835 = {}
    # Getting the type of 'distfn' (line 240)
    distfn_627831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 23), 'distfn', False)
    # Obtaining the member 'pdf' of a type (line 240)
    pdf_627832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 23), distfn_627831, 'pdf')
    # Calling pdf(args, kwargs) (line 240)
    pdf_call_result_627836 = invoke(stypy.reporting.localization.Localization(__file__, 240, 23), pdf_627832, *[x_627833, arg_627834], **kwargs_627835)
    
    # Assigning a type to the variable 'tuple_assignment_626826' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'tuple_assignment_626826', pdf_call_result_627836)
    
    # Assigning a Call to a Name (line 240):
    
    # Call to cdf(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'x' (line 240)
    x_627839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 55), 'x', False)
    # Getting the type of 'arg' (line 240)
    arg_627840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 59), 'arg', False)
    # Processing the call keyword arguments (line 240)
    kwargs_627841 = {}
    # Getting the type of 'distfn' (line 240)
    distfn_627837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 44), 'distfn', False)
    # Obtaining the member 'cdf' of a type (line 240)
    cdf_627838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 44), distfn_627837, 'cdf')
    # Calling cdf(args, kwargs) (line 240)
    cdf_call_result_627842 = invoke(stypy.reporting.localization.Localization(__file__, 240, 44), cdf_627838, *[x_627839, arg_627840], **kwargs_627841)
    
    # Assigning a type to the variable 'tuple_assignment_626827' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'tuple_assignment_626827', cdf_call_result_627842)
    
    # Assigning a Call to a Name (line 240):
    
    # Call to sf(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'x' (line 240)
    x_627845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 75), 'x', False)
    # Getting the type of 'arg' (line 240)
    arg_627846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 79), 'arg', False)
    # Processing the call keyword arguments (line 240)
    kwargs_627847 = {}
    # Getting the type of 'distfn' (line 240)
    distfn_627843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 65), 'distfn', False)
    # Obtaining the member 'sf' of a type (line 240)
    sf_627844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 65), distfn_627843, 'sf')
    # Calling sf(args, kwargs) (line 240)
    sf_call_result_627848 = invoke(stypy.reporting.localization.Localization(__file__, 240, 65), sf_627844, *[x_627845, arg_627846], **kwargs_627847)
    
    # Assigning a type to the variable 'tuple_assignment_626828' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'tuple_assignment_626828', sf_call_result_627848)
    
    # Assigning a Name to a Name (line 240):
    # Getting the type of 'tuple_assignment_626826' (line 240)
    tuple_assignment_626826_627849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'tuple_assignment_626826')
    # Assigning a type to the variable 'pdf' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'pdf', tuple_assignment_626826_627849)
    
    # Assigning a Name to a Name (line 240):
    # Getting the type of 'tuple_assignment_626827' (line 240)
    tuple_assignment_626827_627850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'tuple_assignment_626827')
    # Assigning a type to the variable 'cdf' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 13), 'cdf', tuple_assignment_626827_627850)
    
    # Assigning a Name to a Name (line 240):
    # Getting the type of 'tuple_assignment_626828' (line 240)
    tuple_assignment_626828_627851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'tuple_assignment_626828')
    # Assigning a type to the variable 'sf' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 18), 'sf', tuple_assignment_626828_627851)
    
    # Call to assert_allclose(...): (line 241)
    # Processing the call arguments (line 241)
    
    # Call to deriv(...): (line 241)
    # Processing the call arguments (line 241)
    # Getting the type of 'distfn' (line 241)
    distfn_627854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 30), 'distfn', False)
    # Obtaining the member 'cdf' of a type (line 241)
    cdf_627855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 30), distfn_627854, 'cdf')
    # Getting the type of 'x' (line 241)
    x_627856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 42), 'x', False)
    # Getting the type of 'arg' (line 241)
    arg_627857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 46), 'arg', False)
    # Processing the call keyword arguments (line 241)
    kwargs_627858 = {}
    # Getting the type of 'deriv' (line 241)
    deriv_627853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 24), 'deriv', False)
    # Calling deriv(args, kwargs) (line 241)
    deriv_call_result_627859 = invoke(stypy.reporting.localization.Localization(__file__, 241, 24), deriv_627853, *[cdf_627855, x_627856, arg_627857], **kwargs_627858)
    
    # Getting the type of 'pdf' (line 241)
    pdf_627860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 52), 'pdf', False)
    # Processing the call keyword arguments (line 241)
    float_627861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 62), 'float')
    keyword_627862 = float_627861
    kwargs_627863 = {'rtol': keyword_627862}
    # Getting the type of 'assert_allclose' (line 241)
    assert_allclose_627852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 241)
    assert_allclose_call_result_627864 = invoke(stypy.reporting.localization.Localization(__file__, 241, 8), assert_allclose_627852, *[deriv_call_result_627859, pdf_627860], **kwargs_627863)
    
    
    # Call to assert_allclose(...): (line 242)
    # Processing the call arguments (line 242)
    
    # Call to deriv(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'distfn' (line 242)
    distfn_627867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 30), 'distfn', False)
    # Obtaining the member 'logcdf' of a type (line 242)
    logcdf_627868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 30), distfn_627867, 'logcdf')
    # Getting the type of 'x' (line 242)
    x_627869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 45), 'x', False)
    # Getting the type of 'arg' (line 242)
    arg_627870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 49), 'arg', False)
    # Processing the call keyword arguments (line 242)
    kwargs_627871 = {}
    # Getting the type of 'deriv' (line 242)
    deriv_627866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), 'deriv', False)
    # Calling deriv(args, kwargs) (line 242)
    deriv_call_result_627872 = invoke(stypy.reporting.localization.Localization(__file__, 242, 24), deriv_627866, *[logcdf_627868, x_627869, arg_627870], **kwargs_627871)
    
    # Getting the type of 'pdf' (line 242)
    pdf_627873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 55), 'pdf', False)
    # Getting the type of 'cdf' (line 242)
    cdf_627874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 59), 'cdf', False)
    # Applying the binary operator 'div' (line 242)
    result_div_627875 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 55), 'div', pdf_627873, cdf_627874)
    
    # Processing the call keyword arguments (line 242)
    float_627876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 69), 'float')
    keyword_627877 = float_627876
    kwargs_627878 = {'rtol': keyword_627877}
    # Getting the type of 'assert_allclose' (line 242)
    assert_allclose_627865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 242)
    assert_allclose_call_result_627879 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), assert_allclose_627865, *[deriv_call_result_627872, result_div_627875], **kwargs_627878)
    
    
    # Call to assert_allclose(...): (line 244)
    # Processing the call arguments (line 244)
    
    # Call to deriv(...): (line 244)
    # Processing the call arguments (line 244)
    # Getting the type of 'distfn' (line 244)
    distfn_627882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 30), 'distfn', False)
    # Obtaining the member 'sf' of a type (line 244)
    sf_627883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 30), distfn_627882, 'sf')
    # Getting the type of 'x' (line 244)
    x_627884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 41), 'x', False)
    # Getting the type of 'arg' (line 244)
    arg_627885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 45), 'arg', False)
    # Processing the call keyword arguments (line 244)
    kwargs_627886 = {}
    # Getting the type of 'deriv' (line 244)
    deriv_627881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'deriv', False)
    # Calling deriv(args, kwargs) (line 244)
    deriv_call_result_627887 = invoke(stypy.reporting.localization.Localization(__file__, 244, 24), deriv_627881, *[sf_627883, x_627884, arg_627885], **kwargs_627886)
    
    
    # Getting the type of 'pdf' (line 244)
    pdf_627888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 52), 'pdf', False)
    # Applying the 'usub' unary operator (line 244)
    result___neg___627889 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 51), 'usub', pdf_627888)
    
    # Processing the call keyword arguments (line 244)
    float_627890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 62), 'float')
    keyword_627891 = float_627890
    kwargs_627892 = {'rtol': keyword_627891}
    # Getting the type of 'assert_allclose' (line 244)
    assert_allclose_627880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 244)
    assert_allclose_call_result_627893 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), assert_allclose_627880, *[deriv_call_result_627887, result___neg___627889], **kwargs_627892)
    
    
    # Call to assert_allclose(...): (line 245)
    # Processing the call arguments (line 245)
    
    # Call to deriv(...): (line 245)
    # Processing the call arguments (line 245)
    # Getting the type of 'distfn' (line 245)
    distfn_627896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 30), 'distfn', False)
    # Obtaining the member 'logsf' of a type (line 245)
    logsf_627897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 30), distfn_627896, 'logsf')
    # Getting the type of 'x' (line 245)
    x_627898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 44), 'x', False)
    # Getting the type of 'arg' (line 245)
    arg_627899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 48), 'arg', False)
    # Processing the call keyword arguments (line 245)
    kwargs_627900 = {}
    # Getting the type of 'deriv' (line 245)
    deriv_627895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 24), 'deriv', False)
    # Calling deriv(args, kwargs) (line 245)
    deriv_call_result_627901 = invoke(stypy.reporting.localization.Localization(__file__, 245, 24), deriv_627895, *[logsf_627897, x_627898, arg_627899], **kwargs_627900)
    
    
    # Getting the type of 'pdf' (line 245)
    pdf_627902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 55), 'pdf', False)
    # Applying the 'usub' unary operator (line 245)
    result___neg___627903 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 54), 'usub', pdf_627902)
    
    # Getting the type of 'sf' (line 245)
    sf_627904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 59), 'sf', False)
    # Applying the binary operator 'div' (line 245)
    result_div_627905 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 54), 'div', result___neg___627903, sf_627904)
    
    # Processing the call keyword arguments (line 245)
    float_627906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 68), 'float')
    keyword_627907 = float_627906
    kwargs_627908 = {'rtol': keyword_627907}
    # Getting the type of 'assert_allclose' (line 245)
    assert_allclose_627894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 245)
    assert_allclose_call_result_627909 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), assert_allclose_627894, *[deriv_call_result_627901, result_div_627905], **kwargs_627908)
    
    
    # Call to assert_allclose(...): (line 247)
    # Processing the call arguments (line 247)
    
    # Call to deriv(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'distfn' (line 247)
    distfn_627912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 30), 'distfn', False)
    # Obtaining the member 'logpdf' of a type (line 247)
    logpdf_627913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 30), distfn_627912, 'logpdf')
    # Getting the type of 'x' (line 247)
    x_627914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 45), 'x', False)
    # Getting the type of 'arg' (line 247)
    arg_627915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 49), 'arg', False)
    # Processing the call keyword arguments (line 247)
    kwargs_627916 = {}
    # Getting the type of 'deriv' (line 247)
    deriv_627911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 24), 'deriv', False)
    # Calling deriv(args, kwargs) (line 247)
    deriv_call_result_627917 = invoke(stypy.reporting.localization.Localization(__file__, 247, 24), deriv_627911, *[logpdf_627913, x_627914, arg_627915], **kwargs_627916)
    
    
    # Call to deriv(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'distfn' (line 248)
    distfn_627919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 30), 'distfn', False)
    # Obtaining the member 'pdf' of a type (line 248)
    pdf_627920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 30), distfn_627919, 'pdf')
    # Getting the type of 'x' (line 248)
    x_627921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 42), 'x', False)
    # Getting the type of 'arg' (line 248)
    arg_627922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 46), 'arg', False)
    # Processing the call keyword arguments (line 248)
    kwargs_627923 = {}
    # Getting the type of 'deriv' (line 248)
    deriv_627918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 24), 'deriv', False)
    # Calling deriv(args, kwargs) (line 248)
    deriv_call_result_627924 = invoke(stypy.reporting.localization.Localization(__file__, 248, 24), deriv_627918, *[pdf_627920, x_627921, arg_627922], **kwargs_627923)
    
    
    # Call to pdf(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'x' (line 248)
    x_627927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 64), 'x', False)
    # Getting the type of 'arg' (line 248)
    arg_627928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 68), 'arg', False)
    # Processing the call keyword arguments (line 248)
    kwargs_627929 = {}
    # Getting the type of 'distfn' (line 248)
    distfn_627925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 53), 'distfn', False)
    # Obtaining the member 'pdf' of a type (line 248)
    pdf_627926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 53), distfn_627925, 'pdf')
    # Calling pdf(args, kwargs) (line 248)
    pdf_call_result_627930 = invoke(stypy.reporting.localization.Localization(__file__, 248, 53), pdf_627926, *[x_627927, arg_627928], **kwargs_627929)
    
    # Applying the binary operator 'div' (line 248)
    result_div_627931 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 24), 'div', deriv_call_result_627924, pdf_call_result_627930)
    
    # Processing the call keyword arguments (line 247)
    float_627932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 29), 'float')
    keyword_627933 = float_627932
    kwargs_627934 = {'rtol': keyword_627933}
    # Getting the type of 'assert_allclose' (line 247)
    assert_allclose_627910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'assert_allclose', False)
    # Calling assert_allclose(args, kwargs) (line 247)
    assert_allclose_call_result_627935 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), assert_allclose_627910, *[deriv_call_result_627917, result_div_627931], **kwargs_627934)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_cmplx_deriv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_cmplx_deriv' in the type store
    # Getting the type of 'stypy_return_type' (line 224)
    stypy_return_type_627936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_627936)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_cmplx_deriv'
    return stypy_return_type_627936

# Assigning a type to the variable 'check_cmplx_deriv' (line 224)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), 'check_cmplx_deriv', check_cmplx_deriv)

@norecursion
def check_pickling(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_pickling'
    module_type_store = module_type_store.open_function_context('check_pickling', 252, 0, False)
    
    # Passed parameters checking function
    check_pickling.stypy_localization = localization
    check_pickling.stypy_type_of_self = None
    check_pickling.stypy_type_store = module_type_store
    check_pickling.stypy_function_name = 'check_pickling'
    check_pickling.stypy_param_names_list = ['distfn', 'args']
    check_pickling.stypy_varargs_param_name = None
    check_pickling.stypy_kwargs_param_name = None
    check_pickling.stypy_call_defaults = defaults
    check_pickling.stypy_call_varargs = varargs
    check_pickling.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_pickling', ['distfn', 'args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_pickling', localization, ['distfn', 'args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_pickling(...)' code ##################

    
    # Assigning a Attribute to a Name (line 257):
    
    # Assigning a Attribute to a Name (line 257):
    # Getting the type of 'distfn' (line 257)
    distfn_627937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 11), 'distfn')
    # Obtaining the member 'random_state' of a type (line 257)
    random_state_627938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 11), distfn_627937, 'random_state')
    # Assigning a type to the variable 'rndm' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'rndm', random_state_627938)
    
    # Assigning a Num to a Attribute (line 259):
    
    # Assigning a Num to a Attribute (line 259):
    int_627939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 26), 'int')
    # Getting the type of 'distfn' (line 259)
    distfn_627940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'distfn')
    # Setting the type of the member 'random_state' of a type (line 259)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 4), distfn_627940, 'random_state', int_627939)
    
    # Call to rvs(...): (line 260)
    # Getting the type of 'args' (line 260)
    args_627943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'args', False)
    # Processing the call keyword arguments (line 260)
    int_627944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 27), 'int')
    keyword_627945 = int_627944
    kwargs_627946 = {'size': keyword_627945}
    # Getting the type of 'distfn' (line 260)
    distfn_627941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'distfn', False)
    # Obtaining the member 'rvs' of a type (line 260)
    rvs_627942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 4), distfn_627941, 'rvs')
    # Calling rvs(args, kwargs) (line 260)
    rvs_call_result_627947 = invoke(stypy.reporting.localization.Localization(__file__, 260, 4), rvs_627942, *[args_627943], **kwargs_627946)
    
    
    # Assigning a Call to a Name (line 261):
    
    # Assigning a Call to a Name (line 261):
    
    # Call to dumps(...): (line 261)
    # Processing the call arguments (line 261)
    # Getting the type of 'distfn' (line 261)
    distfn_627950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 21), 'distfn', False)
    # Processing the call keyword arguments (line 261)
    kwargs_627951 = {}
    # Getting the type of 'pickle' (line 261)
    pickle_627948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'pickle', False)
    # Obtaining the member 'dumps' of a type (line 261)
    dumps_627949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), pickle_627948, 'dumps')
    # Calling dumps(args, kwargs) (line 261)
    dumps_call_result_627952 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), dumps_627949, *[distfn_627950], **kwargs_627951)
    
    # Assigning a type to the variable 's' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 's', dumps_call_result_627952)
    
    # Assigning a Call to a Name (line 262):
    
    # Assigning a Call to a Name (line 262):
    
    # Call to rvs(...): (line 262)
    # Getting the type of 'args' (line 262)
    args_627955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 21), 'args', False)
    # Processing the call keyword arguments (line 262)
    int_627956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 32), 'int')
    keyword_627957 = int_627956
    kwargs_627958 = {'size': keyword_627957}
    # Getting the type of 'distfn' (line 262)
    distfn_627953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 9), 'distfn', False)
    # Obtaining the member 'rvs' of a type (line 262)
    rvs_627954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 9), distfn_627953, 'rvs')
    # Calling rvs(args, kwargs) (line 262)
    rvs_call_result_627959 = invoke(stypy.reporting.localization.Localization(__file__, 262, 9), rvs_627954, *[args_627955], **kwargs_627958)
    
    # Assigning a type to the variable 'r0' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'r0', rvs_call_result_627959)
    
    # Assigning a Call to a Name (line 264):
    
    # Assigning a Call to a Name (line 264):
    
    # Call to loads(...): (line 264)
    # Processing the call arguments (line 264)
    # Getting the type of 's' (line 264)
    s_627962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 29), 's', False)
    # Processing the call keyword arguments (line 264)
    kwargs_627963 = {}
    # Getting the type of 'pickle' (line 264)
    pickle_627960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 'pickle', False)
    # Obtaining the member 'loads' of a type (line 264)
    loads_627961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 16), pickle_627960, 'loads')
    # Calling loads(args, kwargs) (line 264)
    loads_call_result_627964 = invoke(stypy.reporting.localization.Localization(__file__, 264, 16), loads_627961, *[s_627962], **kwargs_627963)
    
    # Assigning a type to the variable 'unpickled' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'unpickled', loads_call_result_627964)
    
    # Assigning a Call to a Name (line 265):
    
    # Assigning a Call to a Name (line 265):
    
    # Call to rvs(...): (line 265)
    # Getting the type of 'args' (line 265)
    args_627967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 24), 'args', False)
    # Processing the call keyword arguments (line 265)
    int_627968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 35), 'int')
    keyword_627969 = int_627968
    kwargs_627970 = {'size': keyword_627969}
    # Getting the type of 'unpickled' (line 265)
    unpickled_627965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 9), 'unpickled', False)
    # Obtaining the member 'rvs' of a type (line 265)
    rvs_627966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 9), unpickled_627965, 'rvs')
    # Calling rvs(args, kwargs) (line 265)
    rvs_call_result_627971 = invoke(stypy.reporting.localization.Localization(__file__, 265, 9), rvs_627966, *[args_627967], **kwargs_627970)
    
    # Assigning a type to the variable 'r1' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'r1', rvs_call_result_627971)
    
    # Call to assert_equal(...): (line 266)
    # Processing the call arguments (line 266)
    # Getting the type of 'r0' (line 266)
    r0_627974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 21), 'r0', False)
    # Getting the type of 'r1' (line 266)
    r1_627975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 25), 'r1', False)
    # Processing the call keyword arguments (line 266)
    kwargs_627976 = {}
    # Getting the type of 'npt' (line 266)
    npt_627972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'npt', False)
    # Obtaining the member 'assert_equal' of a type (line 266)
    assert_equal_627973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 4), npt_627972, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 266)
    assert_equal_call_result_627977 = invoke(stypy.reporting.localization.Localization(__file__, 266, 4), assert_equal_627973, *[r0_627974, r1_627975], **kwargs_627976)
    
    
    # Assigning a List to a Name (line 269):
    
    # Assigning a List to a Name (line 269):
    
    # Obtaining an instance of the builtin type 'list' (line 269)
    list_627978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 269)
    # Adding element type (line 269)
    
    # Call to ppf(...): (line 269)
    # Processing the call arguments (line 269)
    float_627981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 26), 'float')
    # Getting the type of 'args' (line 269)
    args_627982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 32), 'args', False)
    # Processing the call keyword arguments (line 269)
    kwargs_627983 = {}
    # Getting the type of 'distfn' (line 269)
    distfn_627979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 15), 'distfn', False)
    # Obtaining the member 'ppf' of a type (line 269)
    ppf_627980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 15), distfn_627979, 'ppf')
    # Calling ppf(args, kwargs) (line 269)
    ppf_call_result_627984 = invoke(stypy.reporting.localization.Localization(__file__, 269, 15), ppf_627980, *[float_627981, args_627982], **kwargs_627983)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 14), list_627978, ppf_call_result_627984)
    # Adding element type (line 269)
    
    # Call to ppf(...): (line 269)
    # Processing the call arguments (line 269)
    float_627987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 53), 'float')
    # Getting the type of 'args' (line 269)
    args_627988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 59), 'args', False)
    # Processing the call keyword arguments (line 269)
    kwargs_627989 = {}
    # Getting the type of 'unpickled' (line 269)
    unpickled_627985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 39), 'unpickled', False)
    # Obtaining the member 'ppf' of a type (line 269)
    ppf_627986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 39), unpickled_627985, 'ppf')
    # Calling ppf(args, kwargs) (line 269)
    ppf_call_result_627990 = invoke(stypy.reporting.localization.Localization(__file__, 269, 39), ppf_627986, *[float_627987, args_627988], **kwargs_627989)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 269, 14), list_627978, ppf_call_result_627990)
    
    # Assigning a type to the variable 'medians' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'medians', list_627978)
    
    # Call to assert_equal(...): (line 270)
    # Processing the call arguments (line 270)
    
    # Obtaining the type of the subscript
    int_627993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 29), 'int')
    # Getting the type of 'medians' (line 270)
    medians_627994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 21), 'medians', False)
    # Obtaining the member '__getitem__' of a type (line 270)
    getitem___627995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 21), medians_627994, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 270)
    subscript_call_result_627996 = invoke(stypy.reporting.localization.Localization(__file__, 270, 21), getitem___627995, int_627993)
    
    
    # Obtaining the type of the subscript
    int_627997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 41), 'int')
    # Getting the type of 'medians' (line 270)
    medians_627998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 33), 'medians', False)
    # Obtaining the member '__getitem__' of a type (line 270)
    getitem___627999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 33), medians_627998, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 270)
    subscript_call_result_628000 = invoke(stypy.reporting.localization.Localization(__file__, 270, 33), getitem___627999, int_627997)
    
    # Processing the call keyword arguments (line 270)
    kwargs_628001 = {}
    # Getting the type of 'npt' (line 270)
    npt_627991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'npt', False)
    # Obtaining the member 'assert_equal' of a type (line 270)
    assert_equal_627992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 4), npt_627991, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 270)
    assert_equal_call_result_628002 = invoke(stypy.reporting.localization.Localization(__file__, 270, 4), assert_equal_627992, *[subscript_call_result_627996, subscript_call_result_628000], **kwargs_628001)
    
    
    # Call to assert_equal(...): (line 271)
    # Processing the call arguments (line 271)
    
    # Call to cdf(...): (line 271)
    # Processing the call arguments (line 271)
    
    # Obtaining the type of the subscript
    int_628007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 40), 'int')
    # Getting the type of 'medians' (line 271)
    medians_628008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 32), 'medians', False)
    # Obtaining the member '__getitem__' of a type (line 271)
    getitem___628009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 32), medians_628008, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 271)
    subscript_call_result_628010 = invoke(stypy.reporting.localization.Localization(__file__, 271, 32), getitem___628009, int_628007)
    
    # Getting the type of 'args' (line 271)
    args_628011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 45), 'args', False)
    # Processing the call keyword arguments (line 271)
    kwargs_628012 = {}
    # Getting the type of 'distfn' (line 271)
    distfn_628005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 21), 'distfn', False)
    # Obtaining the member 'cdf' of a type (line 271)
    cdf_628006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 21), distfn_628005, 'cdf')
    # Calling cdf(args, kwargs) (line 271)
    cdf_call_result_628013 = invoke(stypy.reporting.localization.Localization(__file__, 271, 21), cdf_628006, *[subscript_call_result_628010, args_628011], **kwargs_628012)
    
    
    # Call to cdf(...): (line 272)
    # Processing the call arguments (line 272)
    
    # Obtaining the type of the subscript
    int_628016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 43), 'int')
    # Getting the type of 'medians' (line 272)
    medians_628017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 35), 'medians', False)
    # Obtaining the member '__getitem__' of a type (line 272)
    getitem___628018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 35), medians_628017, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 272)
    subscript_call_result_628019 = invoke(stypy.reporting.localization.Localization(__file__, 272, 35), getitem___628018, int_628016)
    
    # Getting the type of 'args' (line 272)
    args_628020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 48), 'args', False)
    # Processing the call keyword arguments (line 272)
    kwargs_628021 = {}
    # Getting the type of 'unpickled' (line 272)
    unpickled_628014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 21), 'unpickled', False)
    # Obtaining the member 'cdf' of a type (line 272)
    cdf_628015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 21), unpickled_628014, 'cdf')
    # Calling cdf(args, kwargs) (line 272)
    cdf_call_result_628022 = invoke(stypy.reporting.localization.Localization(__file__, 272, 21), cdf_628015, *[subscript_call_result_628019, args_628020], **kwargs_628021)
    
    # Processing the call keyword arguments (line 271)
    kwargs_628023 = {}
    # Getting the type of 'npt' (line 271)
    npt_628003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'npt', False)
    # Obtaining the member 'assert_equal' of a type (line 271)
    assert_equal_628004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 4), npt_628003, 'assert_equal')
    # Calling assert_equal(args, kwargs) (line 271)
    assert_equal_call_result_628024 = invoke(stypy.reporting.localization.Localization(__file__, 271, 4), assert_equal_628004, *[cdf_call_result_628013, cdf_call_result_628022], **kwargs_628023)
    
    
    # Assigning a Name to a Attribute (line 275):
    
    # Assigning a Name to a Attribute (line 275):
    # Getting the type of 'rndm' (line 275)
    rndm_628025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 26), 'rndm')
    # Getting the type of 'distfn' (line 275)
    distfn_628026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'distfn')
    # Setting the type of the member 'random_state' of a type (line 275)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 4), distfn_628026, 'random_state', rndm_628025)
    
    # ################# End of 'check_pickling(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_pickling' in the type store
    # Getting the type of 'stypy_return_type' (line 252)
    stypy_return_type_628027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_628027)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_pickling'
    return stypy_return_type_628027

# Assigning a type to the variable 'check_pickling' (line 252)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 0), 'check_pickling', check_pickling)

@norecursion
def check_rvs_broadcast(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_rvs_broadcast'
    module_type_store = module_type_store.open_function_context('check_rvs_broadcast', 278, 0, False)
    
    # Passed parameters checking function
    check_rvs_broadcast.stypy_localization = localization
    check_rvs_broadcast.stypy_type_of_self = None
    check_rvs_broadcast.stypy_type_store = module_type_store
    check_rvs_broadcast.stypy_function_name = 'check_rvs_broadcast'
    check_rvs_broadcast.stypy_param_names_list = ['distfunc', 'distname', 'allargs', 'shape', 'shape_only', 'otype']
    check_rvs_broadcast.stypy_varargs_param_name = None
    check_rvs_broadcast.stypy_kwargs_param_name = None
    check_rvs_broadcast.stypy_call_defaults = defaults
    check_rvs_broadcast.stypy_call_varargs = varargs
    check_rvs_broadcast.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_rvs_broadcast', ['distfunc', 'distname', 'allargs', 'shape', 'shape_only', 'otype'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_rvs_broadcast', localization, ['distfunc', 'distname', 'allargs', 'shape', 'shape_only', 'otype'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_rvs_broadcast(...)' code ##################

    
    # Call to seed(...): (line 279)
    # Processing the call arguments (line 279)
    int_628031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 19), 'int')
    # Processing the call keyword arguments (line 279)
    kwargs_628032 = {}
    # Getting the type of 'np' (line 279)
    np_628028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'np', False)
    # Obtaining the member 'random' of a type (line 279)
    random_628029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 4), np_628028, 'random')
    # Obtaining the member 'seed' of a type (line 279)
    seed_628030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 4), random_628029, 'seed')
    # Calling seed(args, kwargs) (line 279)
    seed_call_result_628033 = invoke(stypy.reporting.localization.Localization(__file__, 279, 4), seed_628030, *[int_628031], **kwargs_628032)
    
    
    # Call to suppress_warnings(...): (line 280)
    # Processing the call keyword arguments (line 280)
    kwargs_628035 = {}
    # Getting the type of 'suppress_warnings' (line 280)
    suppress_warnings_628034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 9), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 280)
    suppress_warnings_call_result_628036 = invoke(stypy.reporting.localization.Localization(__file__, 280, 9), suppress_warnings_628034, *[], **kwargs_628035)
    
    with_628037 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 280, 9), suppress_warnings_call_result_628036, 'with parameter', '__enter__', '__exit__')

    if with_628037:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 280)
        enter___628038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 9), suppress_warnings_call_result_628036, '__enter__')
        with_enter_628039 = invoke(stypy.reporting.localization.Localization(__file__, 280, 9), enter___628038)
        # Assigning a type to the variable 'sup' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 9), 'sup', with_enter_628039)
        
        # Call to filter(...): (line 283)
        # Processing the call keyword arguments (line 283)
        # Getting the type of 'DeprecationWarning' (line 283)
        DeprecationWarning_628042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 28), 'DeprecationWarning', False)
        keyword_628043 = DeprecationWarning_628042
        str_628044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 56), 'str', '.*frechet_')
        keyword_628045 = str_628044
        kwargs_628046 = {'category': keyword_628043, 'message': keyword_628045}
        # Getting the type of 'sup' (line 283)
        sup_628040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 283)
        filter_628041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), sup_628040, 'filter')
        # Calling filter(args, kwargs) (line 283)
        filter_call_result_628047 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), filter_628041, *[], **kwargs_628046)
        
        
        # Assigning a Call to a Name (line 284):
        
        # Assigning a Call to a Name (line 284):
        
        # Call to rvs(...): (line 284)
        # Getting the type of 'allargs' (line 284)
        allargs_628050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 31), 'allargs', False)
        # Processing the call keyword arguments (line 284)
        kwargs_628051 = {}
        # Getting the type of 'distfunc' (line 284)
        distfunc_628048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 17), 'distfunc', False)
        # Obtaining the member 'rvs' of a type (line 284)
        rvs_628049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 17), distfunc_628048, 'rvs')
        # Calling rvs(args, kwargs) (line 284)
        rvs_call_result_628052 = invoke(stypy.reporting.localization.Localization(__file__, 284, 17), rvs_628049, *[allargs_628050], **kwargs_628051)
        
        # Assigning a type to the variable 'sample' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'sample', rvs_call_result_628052)
        
        # Call to assert_equal(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'sample' (line 285)
        sample_628054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 21), 'sample', False)
        # Obtaining the member 'shape' of a type (line 285)
        shape_628055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 21), sample_628054, 'shape')
        # Getting the type of 'shape' (line 285)
        shape_628056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 35), 'shape', False)
        str_628057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 42), 'str', '%s: rvs failed to broadcast')
        # Getting the type of 'distname' (line 285)
        distname_628058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 74), 'distname', False)
        # Applying the binary operator '%' (line 285)
        result_mod_628059 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 42), '%', str_628057, distname_628058)
        
        # Processing the call keyword arguments (line 285)
        kwargs_628060 = {}
        # Getting the type of 'assert_equal' (line 285)
        assert_equal_628053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 285)
        assert_equal_call_result_628061 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), assert_equal_628053, *[shape_628055, shape_628056, result_mod_628059], **kwargs_628060)
        
        
        
        # Getting the type of 'shape_only' (line 286)
        shape_only_628062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 15), 'shape_only')
        # Applying the 'not' unary operator (line 286)
        result_not__628063 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 11), 'not', shape_only_628062)
        
        # Testing the type of an if condition (line 286)
        if_condition_628064 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 8), result_not__628063)
        # Assigning a type to the variable 'if_condition_628064' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'if_condition_628064', if_condition_628064)
        # SSA begins for if statement (line 286)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 287):
        
        # Assigning a Call to a Name (line 287):
        
        # Call to vectorize(...): (line 287)
        # Processing the call arguments (line 287)

        @norecursion
        def _stypy_temp_lambda_536(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_536'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_536', 287, 31, True)
            # Passed parameters checking function
            _stypy_temp_lambda_536.stypy_localization = localization
            _stypy_temp_lambda_536.stypy_type_of_self = None
            _stypy_temp_lambda_536.stypy_type_store = module_type_store
            _stypy_temp_lambda_536.stypy_function_name = '_stypy_temp_lambda_536'
            _stypy_temp_lambda_536.stypy_param_names_list = []
            _stypy_temp_lambda_536.stypy_varargs_param_name = 'allargs'
            _stypy_temp_lambda_536.stypy_kwargs_param_name = None
            _stypy_temp_lambda_536.stypy_call_defaults = defaults
            _stypy_temp_lambda_536.stypy_call_varargs = varargs
            _stypy_temp_lambda_536.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_536', [], 'allargs', None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_536', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to rvs(...): (line 287)
            # Getting the type of 'allargs' (line 287)
            allargs_628069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 62), 'allargs', False)
            # Processing the call keyword arguments (line 287)
            kwargs_628070 = {}
            # Getting the type of 'distfunc' (line 287)
            distfunc_628067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 48), 'distfunc', False)
            # Obtaining the member 'rvs' of a type (line 287)
            rvs_628068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 48), distfunc_628067, 'rvs')
            # Calling rvs(args, kwargs) (line 287)
            rvs_call_result_628071 = invoke(stypy.reporting.localization.Localization(__file__, 287, 48), rvs_628068, *[allargs_628069], **kwargs_628070)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 287)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), 'stypy_return_type', rvs_call_result_628071)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_536' in the type store
            # Getting the type of 'stypy_return_type' (line 287)
            stypy_return_type_628072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_628072)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_536'
            return stypy_return_type_628072

        # Assigning a type to the variable '_stypy_temp_lambda_536' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), '_stypy_temp_lambda_536', _stypy_temp_lambda_536)
        # Getting the type of '_stypy_temp_lambda_536' (line 287)
        _stypy_temp_lambda_536_628073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), '_stypy_temp_lambda_536')
        # Processing the call keyword arguments (line 287)
        # Getting the type of 'otype' (line 287)
        otype_628074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 79), 'otype', False)
        keyword_628075 = otype_628074
        kwargs_628076 = {'otypes': keyword_628075}
        # Getting the type of 'np' (line 287)
        np_628065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 18), 'np', False)
        # Obtaining the member 'vectorize' of a type (line 287)
        vectorize_628066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 18), np_628065, 'vectorize')
        # Calling vectorize(args, kwargs) (line 287)
        vectorize_call_result_628077 = invoke(stypy.reporting.localization.Localization(__file__, 287, 18), vectorize_628066, *[_stypy_temp_lambda_536_628073], **kwargs_628076)
        
        # Assigning a type to the variable 'rvs' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'rvs', vectorize_call_result_628077)
        
        # Call to seed(...): (line 288)
        # Processing the call arguments (line 288)
        int_628081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 27), 'int')
        # Processing the call keyword arguments (line 288)
        kwargs_628082 = {}
        # Getting the type of 'np' (line 288)
        np_628078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'np', False)
        # Obtaining the member 'random' of a type (line 288)
        random_628079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 12), np_628078, 'random')
        # Obtaining the member 'seed' of a type (line 288)
        seed_628080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 12), random_628079, 'seed')
        # Calling seed(args, kwargs) (line 288)
        seed_call_result_628083 = invoke(stypy.reporting.localization.Localization(__file__, 288, 12), seed_628080, *[int_628081], **kwargs_628082)
        
        
        # Assigning a Call to a Name (line 289):
        
        # Assigning a Call to a Name (line 289):
        
        # Call to rvs(...): (line 289)
        # Getting the type of 'allargs' (line 289)
        allargs_628085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 28), 'allargs', False)
        # Processing the call keyword arguments (line 289)
        kwargs_628086 = {}
        # Getting the type of 'rvs' (line 289)
        rvs_628084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 23), 'rvs', False)
        # Calling rvs(args, kwargs) (line 289)
        rvs_call_result_628087 = invoke(stypy.reporting.localization.Localization(__file__, 289, 23), rvs_628084, *[allargs_628085], **kwargs_628086)
        
        # Assigning a type to the variable 'expected' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'expected', rvs_call_result_628087)
        
        # Call to assert_allclose(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'sample' (line 290)
        sample_628089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 28), 'sample', False)
        # Getting the type of 'expected' (line 290)
        expected_628090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 36), 'expected', False)
        # Processing the call keyword arguments (line 290)
        float_628091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 51), 'float')
        keyword_628092 = float_628091
        kwargs_628093 = {'rtol': keyword_628092}
        # Getting the type of 'assert_allclose' (line 290)
        assert_allclose_628088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 290)
        assert_allclose_call_result_628094 = invoke(stypy.reporting.localization.Localization(__file__, 290, 12), assert_allclose_628088, *[sample_628089, expected_628090], **kwargs_628093)
        
        # SSA join for if statement (line 286)
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 280)
        exit___628095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 9), suppress_warnings_call_result_628036, '__exit__')
        with_exit_628096 = invoke(stypy.reporting.localization.Localization(__file__, 280, 9), exit___628095, None, None, None)

    
    # ################# End of 'check_rvs_broadcast(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_rvs_broadcast' in the type store
    # Getting the type of 'stypy_return_type' (line 278)
    stypy_return_type_628097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_628097)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_rvs_broadcast'
    return stypy_return_type_628097

# Assigning a type to the variable 'check_rvs_broadcast' (line 278)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 0), 'check_rvs_broadcast', check_rvs_broadcast)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
