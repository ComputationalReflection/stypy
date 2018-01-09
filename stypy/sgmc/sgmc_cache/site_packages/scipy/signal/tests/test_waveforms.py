
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import numpy as np
4: from numpy.testing import (assert_almost_equal, assert_equal,
5:                            assert_, assert_allclose, assert_array_equal)
6: from pytest import raises as assert_raises
7: 
8: import scipy.signal.waveforms as waveforms
9: 
10: 
11: # These chirp_* functions are the instantaneous frequencies of the signals
12: # returned by chirp().
13: 
14: def chirp_linear(t, f0, f1, t1):
15:     f = f0 + (f1 - f0) * t / t1
16:     return f
17: 
18: 
19: def chirp_quadratic(t, f0, f1, t1, vertex_zero=True):
20:     if vertex_zero:
21:         f = f0 + (f1 - f0) * t**2 / t1**2
22:     else:
23:         f = f1 - (f1 - f0) * (t1 - t)**2 / t1**2
24:     return f
25: 
26: 
27: def chirp_geometric(t, f0, f1, t1):
28:     f = f0 * (f1/f0)**(t/t1)
29:     return f
30: 
31: 
32: def chirp_hyperbolic(t, f0, f1, t1):
33:     f = f0*f1*t1 / ((f0 - f1)*t + f1*t1)
34:     return f
35: 
36: 
37: def compute_frequency(t, theta):
38:     '''
39:     Compute theta'(t)/(2*pi), where theta'(t) is the derivative of theta(t).
40:     '''
41:     # Assume theta and t are 1D numpy arrays.
42:     # Assume that t is uniformly spaced.
43:     dt = t[1] - t[0]
44:     f = np.diff(theta)/(2*np.pi) / dt
45:     tf = 0.5*(t[1:] + t[:-1])
46:     return tf, f
47: 
48: 
49: class TestChirp(object):
50: 
51:     def test_linear_at_zero(self):
52:         w = waveforms.chirp(t=0, f0=1.0, f1=2.0, t1=1.0, method='linear')
53:         assert_almost_equal(w, 1.0)
54: 
55:     def test_linear_freq_01(self):
56:         method = 'linear'
57:         f0 = 1.0
58:         f1 = 2.0
59:         t1 = 1.0
60:         t = np.linspace(0, t1, 100)
61:         phase = waveforms._chirp_phase(t, f0, t1, f1, method)
62:         tf, f = compute_frequency(t, phase)
63:         abserr = np.max(np.abs(f - chirp_linear(tf, f0, f1, t1)))
64:         assert_(abserr < 1e-6)
65: 
66:     def test_linear_freq_02(self):
67:         method = 'linear'
68:         f0 = 200.0
69:         f1 = 100.0
70:         t1 = 10.0
71:         t = np.linspace(0, t1, 100)
72:         phase = waveforms._chirp_phase(t, f0, t1, f1, method)
73:         tf, f = compute_frequency(t, phase)
74:         abserr = np.max(np.abs(f - chirp_linear(tf, f0, f1, t1)))
75:         assert_(abserr < 1e-6)
76: 
77:     def test_quadratic_at_zero(self):
78:         w = waveforms.chirp(t=0, f0=1.0, f1=2.0, t1=1.0, method='quadratic')
79:         assert_almost_equal(w, 1.0)
80: 
81:     def test_quadratic_at_zero2(self):
82:         w = waveforms.chirp(t=0, f0=1.0, f1=2.0, t1=1.0, method='quadratic',
83:                             vertex_zero=False)
84:         assert_almost_equal(w, 1.0)
85: 
86:     def test_quadratic_freq_01(self):
87:         method = 'quadratic'
88:         f0 = 1.0
89:         f1 = 2.0
90:         t1 = 1.0
91:         t = np.linspace(0, t1, 2000)
92:         phase = waveforms._chirp_phase(t, f0, t1, f1, method)
93:         tf, f = compute_frequency(t, phase)
94:         abserr = np.max(np.abs(f - chirp_quadratic(tf, f0, f1, t1)))
95:         assert_(abserr < 1e-6)
96: 
97:     def test_quadratic_freq_02(self):
98:         method = 'quadratic'
99:         f0 = 20.0
100:         f1 = 10.0
101:         t1 = 10.0
102:         t = np.linspace(0, t1, 2000)
103:         phase = waveforms._chirp_phase(t, f0, t1, f1, method)
104:         tf, f = compute_frequency(t, phase)
105:         abserr = np.max(np.abs(f - chirp_quadratic(tf, f0, f1, t1)))
106:         assert_(abserr < 1e-6)
107: 
108:     def test_logarithmic_at_zero(self):
109:         w = waveforms.chirp(t=0, f0=1.0, f1=2.0, t1=1.0, method='logarithmic')
110:         assert_almost_equal(w, 1.0)
111: 
112:     def test_logarithmic_freq_01(self):
113:         method = 'logarithmic'
114:         f0 = 1.0
115:         f1 = 2.0
116:         t1 = 1.0
117:         t = np.linspace(0, t1, 10000)
118:         phase = waveforms._chirp_phase(t, f0, t1, f1, method)
119:         tf, f = compute_frequency(t, phase)
120:         abserr = np.max(np.abs(f - chirp_geometric(tf, f0, f1, t1)))
121:         assert_(abserr < 1e-6)
122: 
123:     def test_logarithmic_freq_02(self):
124:         method = 'logarithmic'
125:         f0 = 200.0
126:         f1 = 100.0
127:         t1 = 10.0
128:         t = np.linspace(0, t1, 10000)
129:         phase = waveforms._chirp_phase(t, f0, t1, f1, method)
130:         tf, f = compute_frequency(t, phase)
131:         abserr = np.max(np.abs(f - chirp_geometric(tf, f0, f1, t1)))
132:         assert_(abserr < 1e-6)
133: 
134:     def test_logarithmic_freq_03(self):
135:         method = 'logarithmic'
136:         f0 = 100.0
137:         f1 = 100.0
138:         t1 = 10.0
139:         t = np.linspace(0, t1, 10000)
140:         phase = waveforms._chirp_phase(t, f0, t1, f1, method)
141:         tf, f = compute_frequency(t, phase)
142:         abserr = np.max(np.abs(f - chirp_geometric(tf, f0, f1, t1)))
143:         assert_(abserr < 1e-6)
144: 
145:     def test_hyperbolic_at_zero(self):
146:         w = waveforms.chirp(t=0, f0=10.0, f1=1.0, t1=1.0, method='hyperbolic')
147:         assert_almost_equal(w, 1.0)
148: 
149:     def test_hyperbolic_freq_01(self):
150:         method = 'hyperbolic'
151:         t1 = 1.0
152:         t = np.linspace(0, t1, 10000)
153:         #           f0     f1
154:         cases = [[10.0, 1.0],
155:                  [1.0, 10.0],
156:                  [-10.0, -1.0],
157:                  [-1.0, -10.0]]
158:         for f0, f1 in cases:
159:             phase = waveforms._chirp_phase(t, f0, t1, f1, method)
160:             tf, f = compute_frequency(t, phase)
161:             expected = chirp_hyperbolic(tf, f0, f1, t1)
162:             assert_allclose(f, expected)
163: 
164:     def test_hyperbolic_zero_freq(self):
165:         # f0=0 or f1=0 must raise a ValueError.
166:         method = 'hyperbolic'
167:         t1 = 1.0
168:         t = np.linspace(0, t1, 5)
169:         assert_raises(ValueError, waveforms.chirp, t, 0, t1, 1, method)
170:         assert_raises(ValueError, waveforms.chirp, t, 1, t1, 0, method)
171: 
172:     def test_unknown_method(self):
173:         method = "foo"
174:         f0 = 10.0
175:         f1 = 20.0
176:         t1 = 1.0
177:         t = np.linspace(0, t1, 10)
178:         assert_raises(ValueError, waveforms.chirp, t, f0, t1, f1, method)
179: 
180:     def test_integer_t1(self):
181:         f0 = 10.0
182:         f1 = 20.0
183:         t = np.linspace(-1, 1, 11)
184:         t1 = 3.0
185:         float_result = waveforms.chirp(t, f0, t1, f1)
186:         t1 = 3
187:         int_result = waveforms.chirp(t, f0, t1, f1)
188:         err_msg = "Integer input 't1=3' gives wrong result"
189:         assert_equal(int_result, float_result, err_msg=err_msg)
190: 
191:     def test_integer_f0(self):
192:         f1 = 20.0
193:         t1 = 3.0
194:         t = np.linspace(-1, 1, 11)
195:         f0 = 10.0
196:         float_result = waveforms.chirp(t, f0, t1, f1)
197:         f0 = 10
198:         int_result = waveforms.chirp(t, f0, t1, f1)
199:         err_msg = "Integer input 'f0=10' gives wrong result"
200:         assert_equal(int_result, float_result, err_msg=err_msg)
201: 
202:     def test_integer_f1(self):
203:         f0 = 10.0
204:         t1 = 3.0
205:         t = np.linspace(-1, 1, 11)
206:         f1 = 20.0
207:         float_result = waveforms.chirp(t, f0, t1, f1)
208:         f1 = 20
209:         int_result = waveforms.chirp(t, f0, t1, f1)
210:         err_msg = "Integer input 'f1=20' gives wrong result"
211:         assert_equal(int_result, float_result, err_msg=err_msg)
212: 
213:     def test_integer_all(self):
214:         f0 = 10
215:         t1 = 3
216:         f1 = 20
217:         t = np.linspace(-1, 1, 11)
218:         float_result = waveforms.chirp(t, float(f0), float(t1), float(f1))
219:         int_result = waveforms.chirp(t, f0, t1, f1)
220:         err_msg = "Integer input 'f0=10, t1=3, f1=20' gives wrong result"
221:         assert_equal(int_result, float_result, err_msg=err_msg)
222: 
223: 
224: class TestSweepPoly(object):
225: 
226:     def test_sweep_poly_quad1(self):
227:         p = np.poly1d([1.0, 0.0, 1.0])
228:         t = np.linspace(0, 3.0, 10000)
229:         phase = waveforms._sweep_poly_phase(t, p)
230:         tf, f = compute_frequency(t, phase)
231:         expected = p(tf)
232:         abserr = np.max(np.abs(f - expected))
233:         assert_(abserr < 1e-6)
234: 
235:     def test_sweep_poly_const(self):
236:         p = np.poly1d(2.0)
237:         t = np.linspace(0, 3.0, 10000)
238:         phase = waveforms._sweep_poly_phase(t, p)
239:         tf, f = compute_frequency(t, phase)
240:         expected = p(tf)
241:         abserr = np.max(np.abs(f - expected))
242:         assert_(abserr < 1e-6)
243: 
244:     def test_sweep_poly_linear(self):
245:         p = np.poly1d([-1.0, 10.0])
246:         t = np.linspace(0, 3.0, 10000)
247:         phase = waveforms._sweep_poly_phase(t, p)
248:         tf, f = compute_frequency(t, phase)
249:         expected = p(tf)
250:         abserr = np.max(np.abs(f - expected))
251:         assert_(abserr < 1e-6)
252: 
253:     def test_sweep_poly_quad2(self):
254:         p = np.poly1d([1.0, 0.0, -2.0])
255:         t = np.linspace(0, 3.0, 10000)
256:         phase = waveforms._sweep_poly_phase(t, p)
257:         tf, f = compute_frequency(t, phase)
258:         expected = p(tf)
259:         abserr = np.max(np.abs(f - expected))
260:         assert_(abserr < 1e-6)
261: 
262:     def test_sweep_poly_cubic(self):
263:         p = np.poly1d([2.0, 1.0, 0.0, -2.0])
264:         t = np.linspace(0, 2.0, 10000)
265:         phase = waveforms._sweep_poly_phase(t, p)
266:         tf, f = compute_frequency(t, phase)
267:         expected = p(tf)
268:         abserr = np.max(np.abs(f - expected))
269:         assert_(abserr < 1e-6)
270: 
271:     def test_sweep_poly_cubic2(self):
272:         '''Use an array of coefficients instead of a poly1d.'''
273:         p = np.array([2.0, 1.0, 0.0, -2.0])
274:         t = np.linspace(0, 2.0, 10000)
275:         phase = waveforms._sweep_poly_phase(t, p)
276:         tf, f = compute_frequency(t, phase)
277:         expected = np.poly1d(p)(tf)
278:         abserr = np.max(np.abs(f - expected))
279:         assert_(abserr < 1e-6)
280: 
281:     def test_sweep_poly_cubic3(self):
282:         '''Use a list of coefficients instead of a poly1d.'''
283:         p = [2.0, 1.0, 0.0, -2.0]
284:         t = np.linspace(0, 2.0, 10000)
285:         phase = waveforms._sweep_poly_phase(t, p)
286:         tf, f = compute_frequency(t, phase)
287:         expected = np.poly1d(p)(tf)
288:         abserr = np.max(np.abs(f - expected))
289:         assert_(abserr < 1e-6)
290: 
291: 
292: class TestGaussPulse(object):
293: 
294:     def test_integer_fc(self):
295:         float_result = waveforms.gausspulse('cutoff', fc=1000.0)
296:         int_result = waveforms.gausspulse('cutoff', fc=1000)
297:         err_msg = "Integer input 'fc=1000' gives wrong result"
298:         assert_equal(int_result, float_result, err_msg=err_msg)
299: 
300:     def test_integer_bw(self):
301:         float_result = waveforms.gausspulse('cutoff', bw=1.0)
302:         int_result = waveforms.gausspulse('cutoff', bw=1)
303:         err_msg = "Integer input 'bw=1' gives wrong result"
304:         assert_equal(int_result, float_result, err_msg=err_msg)
305: 
306:     def test_integer_bwr(self):
307:         float_result = waveforms.gausspulse('cutoff', bwr=-6.0)
308:         int_result = waveforms.gausspulse('cutoff', bwr=-6)
309:         err_msg = "Integer input 'bwr=-6' gives wrong result"
310:         assert_equal(int_result, float_result, err_msg=err_msg)
311: 
312:     def test_integer_tpr(self):
313:         float_result = waveforms.gausspulse('cutoff', tpr=-60.0)
314:         int_result = waveforms.gausspulse('cutoff', tpr=-60)
315:         err_msg = "Integer input 'tpr=-60' gives wrong result"
316:         assert_equal(int_result, float_result, err_msg=err_msg)
317: 
318: 
319: class TestUnitImpulse(object):
320: 
321:     def test_no_index(self):
322:         assert_array_equal(waveforms.unit_impulse(7), [1, 0, 0, 0, 0, 0, 0])
323:         assert_array_equal(waveforms.unit_impulse((3, 3)),
324:                            [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
325: 
326:     def test_index(self):
327:         assert_array_equal(waveforms.unit_impulse(10, 3),
328:                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
329:         assert_array_equal(waveforms.unit_impulse((3, 3), (1, 1)),
330:                            [[0, 0, 0], [0, 1, 0], [0, 0, 0]])
331: 
332:         # Broadcasting
333:         imp = waveforms.unit_impulse((4, 4), 2)
334:         assert_array_equal(imp, np.array([[0, 0, 0, 0],
335:                                           [0, 0, 0, 0],
336:                                           [0, 0, 1, 0],
337:                                           [0, 0, 0, 0]]))
338: 
339:     def test_mid(self):
340:         assert_array_equal(waveforms.unit_impulse((3, 3), 'mid'),
341:                            [[0, 0, 0], [0, 1, 0], [0, 0, 0]])
342:         assert_array_equal(waveforms.unit_impulse(9, 'mid'),
343:                            [0, 0, 0, 0, 1, 0, 0, 0, 0])
344: 
345:     def test_dtype(self):
346:         imp = waveforms.unit_impulse(7)
347:         assert_(np.issubdtype(imp.dtype, np.floating))
348: 
349:         imp = waveforms.unit_impulse(5, 3, dtype=int)
350:         assert_(np.issubdtype(imp.dtype, np.integer))
351: 
352:         imp = waveforms.unit_impulse((5, 2), (3, 1), dtype=complex)
353:         assert_(np.issubdtype(imp.dtype, np.complexfloating))
354: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_351444 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_351444) is not StypyTypeError):

    if (import_351444 != 'pyd_module'):
        __import__(import_351444)
        sys_modules_351445 = sys.modules[import_351444]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_351445.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_351444)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.testing import assert_almost_equal, assert_equal, assert_, assert_allclose, assert_array_equal' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_351446 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing')

if (type(import_351446) is not StypyTypeError):

    if (import_351446 != 'pyd_module'):
        __import__(import_351446)
        sys_modules_351447 = sys.modules[import_351446]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', sys_modules_351447.module_type_store, module_type_store, ['assert_almost_equal', 'assert_equal', 'assert_', 'assert_allclose', 'assert_array_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_351447, sys_modules_351447.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_almost_equal, assert_equal, assert_, assert_allclose, assert_array_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', None, module_type_store, ['assert_almost_equal', 'assert_equal', 'assert_', 'assert_allclose', 'assert_array_equal'], [assert_almost_equal, assert_equal, assert_, assert_allclose, assert_array_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.testing', import_351446)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from pytest import assert_raises' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_351448 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest')

if (type(import_351448) is not StypyTypeError):

    if (import_351448 != 'pyd_module'):
        __import__(import_351448)
        sys_modules_351449 = sys.modules[import_351448]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', sys_modules_351449.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_351449, sys_modules_351449.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', import_351448)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import scipy.signal.waveforms' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/tests/')
import_351450 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal.waveforms')

if (type(import_351450) is not StypyTypeError):

    if (import_351450 != 'pyd_module'):
        __import__(import_351450)
        sys_modules_351451 = sys.modules[import_351450]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'waveforms', sys_modules_351451.module_type_store, module_type_store)
    else:
        import scipy.signal.waveforms as waveforms

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'waveforms', scipy.signal.waveforms, module_type_store)

else:
    # Assigning a type to the variable 'scipy.signal.waveforms' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'scipy.signal.waveforms', import_351450)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/tests/')


@norecursion
def chirp_linear(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chirp_linear'
    module_type_store = module_type_store.open_function_context('chirp_linear', 14, 0, False)
    
    # Passed parameters checking function
    chirp_linear.stypy_localization = localization
    chirp_linear.stypy_type_of_self = None
    chirp_linear.stypy_type_store = module_type_store
    chirp_linear.stypy_function_name = 'chirp_linear'
    chirp_linear.stypy_param_names_list = ['t', 'f0', 'f1', 't1']
    chirp_linear.stypy_varargs_param_name = None
    chirp_linear.stypy_kwargs_param_name = None
    chirp_linear.stypy_call_defaults = defaults
    chirp_linear.stypy_call_varargs = varargs
    chirp_linear.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chirp_linear', ['t', 'f0', 'f1', 't1'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chirp_linear', localization, ['t', 'f0', 'f1', 't1'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chirp_linear(...)' code ##################

    
    # Assigning a BinOp to a Name (line 15):
    
    # Assigning a BinOp to a Name (line 15):
    # Getting the type of 'f0' (line 15)
    f0_351452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'f0')
    # Getting the type of 'f1' (line 15)
    f1_351453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 14), 'f1')
    # Getting the type of 'f0' (line 15)
    f0_351454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 19), 'f0')
    # Applying the binary operator '-' (line 15)
    result_sub_351455 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 14), '-', f1_351453, f0_351454)
    
    # Getting the type of 't' (line 15)
    t_351456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 25), 't')
    # Applying the binary operator '*' (line 15)
    result_mul_351457 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 13), '*', result_sub_351455, t_351456)
    
    # Getting the type of 't1' (line 15)
    t1_351458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 29), 't1')
    # Applying the binary operator 'div' (line 15)
    result_div_351459 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 27), 'div', result_mul_351457, t1_351458)
    
    # Applying the binary operator '+' (line 15)
    result_add_351460 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 8), '+', f0_351452, result_div_351459)
    
    # Assigning a type to the variable 'f' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'f', result_add_351460)
    # Getting the type of 'f' (line 16)
    f_351461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'f')
    # Assigning a type to the variable 'stypy_return_type' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type', f_351461)
    
    # ################# End of 'chirp_linear(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chirp_linear' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_351462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_351462)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chirp_linear'
    return stypy_return_type_351462

# Assigning a type to the variable 'chirp_linear' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'chirp_linear', chirp_linear)

@norecursion
def chirp_quadratic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 19)
    True_351463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 47), 'True')
    defaults = [True_351463]
    # Create a new context for function 'chirp_quadratic'
    module_type_store = module_type_store.open_function_context('chirp_quadratic', 19, 0, False)
    
    # Passed parameters checking function
    chirp_quadratic.stypy_localization = localization
    chirp_quadratic.stypy_type_of_self = None
    chirp_quadratic.stypy_type_store = module_type_store
    chirp_quadratic.stypy_function_name = 'chirp_quadratic'
    chirp_quadratic.stypy_param_names_list = ['t', 'f0', 'f1', 't1', 'vertex_zero']
    chirp_quadratic.stypy_varargs_param_name = None
    chirp_quadratic.stypy_kwargs_param_name = None
    chirp_quadratic.stypy_call_defaults = defaults
    chirp_quadratic.stypy_call_varargs = varargs
    chirp_quadratic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chirp_quadratic', ['t', 'f0', 'f1', 't1', 'vertex_zero'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chirp_quadratic', localization, ['t', 'f0', 'f1', 't1', 'vertex_zero'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chirp_quadratic(...)' code ##################

    
    # Getting the type of 'vertex_zero' (line 20)
    vertex_zero_351464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 7), 'vertex_zero')
    # Testing the type of an if condition (line 20)
    if_condition_351465 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 4), vertex_zero_351464)
    # Assigning a type to the variable 'if_condition_351465' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'if_condition_351465', if_condition_351465)
    # SSA begins for if statement (line 20)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 21):
    
    # Assigning a BinOp to a Name (line 21):
    # Getting the type of 'f0' (line 21)
    f0_351466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'f0')
    # Getting the type of 'f1' (line 21)
    f1_351467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 18), 'f1')
    # Getting the type of 'f0' (line 21)
    f0_351468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 23), 'f0')
    # Applying the binary operator '-' (line 21)
    result_sub_351469 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 18), '-', f1_351467, f0_351468)
    
    # Getting the type of 't' (line 21)
    t_351470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 29), 't')
    int_351471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 32), 'int')
    # Applying the binary operator '**' (line 21)
    result_pow_351472 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 29), '**', t_351470, int_351471)
    
    # Applying the binary operator '*' (line 21)
    result_mul_351473 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 17), '*', result_sub_351469, result_pow_351472)
    
    # Getting the type of 't1' (line 21)
    t1_351474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 36), 't1')
    int_351475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 40), 'int')
    # Applying the binary operator '**' (line 21)
    result_pow_351476 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 36), '**', t1_351474, int_351475)
    
    # Applying the binary operator 'div' (line 21)
    result_div_351477 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 34), 'div', result_mul_351473, result_pow_351476)
    
    # Applying the binary operator '+' (line 21)
    result_add_351478 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 12), '+', f0_351466, result_div_351477)
    
    # Assigning a type to the variable 'f' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'f', result_add_351478)
    # SSA branch for the else part of an if statement (line 20)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 23):
    
    # Assigning a BinOp to a Name (line 23):
    # Getting the type of 'f1' (line 23)
    f1_351479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'f1')
    # Getting the type of 'f1' (line 23)
    f1_351480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 18), 'f1')
    # Getting the type of 'f0' (line 23)
    f0_351481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'f0')
    # Applying the binary operator '-' (line 23)
    result_sub_351482 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 18), '-', f1_351480, f0_351481)
    
    # Getting the type of 't1' (line 23)
    t1_351483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 30), 't1')
    # Getting the type of 't' (line 23)
    t_351484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 35), 't')
    # Applying the binary operator '-' (line 23)
    result_sub_351485 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 30), '-', t1_351483, t_351484)
    
    int_351486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 39), 'int')
    # Applying the binary operator '**' (line 23)
    result_pow_351487 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 29), '**', result_sub_351485, int_351486)
    
    # Applying the binary operator '*' (line 23)
    result_mul_351488 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 17), '*', result_sub_351482, result_pow_351487)
    
    # Getting the type of 't1' (line 23)
    t1_351489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 43), 't1')
    int_351490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 47), 'int')
    # Applying the binary operator '**' (line 23)
    result_pow_351491 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 43), '**', t1_351489, int_351490)
    
    # Applying the binary operator 'div' (line 23)
    result_div_351492 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 41), 'div', result_mul_351488, result_pow_351491)
    
    # Applying the binary operator '-' (line 23)
    result_sub_351493 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 12), '-', f1_351479, result_div_351492)
    
    # Assigning a type to the variable 'f' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'f', result_sub_351493)
    # SSA join for if statement (line 20)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'f' (line 24)
    f_351494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'f')
    # Assigning a type to the variable 'stypy_return_type' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type', f_351494)
    
    # ################# End of 'chirp_quadratic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chirp_quadratic' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_351495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_351495)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chirp_quadratic'
    return stypy_return_type_351495

# Assigning a type to the variable 'chirp_quadratic' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'chirp_quadratic', chirp_quadratic)

@norecursion
def chirp_geometric(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chirp_geometric'
    module_type_store = module_type_store.open_function_context('chirp_geometric', 27, 0, False)
    
    # Passed parameters checking function
    chirp_geometric.stypy_localization = localization
    chirp_geometric.stypy_type_of_self = None
    chirp_geometric.stypy_type_store = module_type_store
    chirp_geometric.stypy_function_name = 'chirp_geometric'
    chirp_geometric.stypy_param_names_list = ['t', 'f0', 'f1', 't1']
    chirp_geometric.stypy_varargs_param_name = None
    chirp_geometric.stypy_kwargs_param_name = None
    chirp_geometric.stypy_call_defaults = defaults
    chirp_geometric.stypy_call_varargs = varargs
    chirp_geometric.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chirp_geometric', ['t', 'f0', 'f1', 't1'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chirp_geometric', localization, ['t', 'f0', 'f1', 't1'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chirp_geometric(...)' code ##################

    
    # Assigning a BinOp to a Name (line 28):
    
    # Assigning a BinOp to a Name (line 28):
    # Getting the type of 'f0' (line 28)
    f0_351496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'f0')
    # Getting the type of 'f1' (line 28)
    f1_351497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 14), 'f1')
    # Getting the type of 'f0' (line 28)
    f0_351498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'f0')
    # Applying the binary operator 'div' (line 28)
    result_div_351499 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 14), 'div', f1_351497, f0_351498)
    
    # Getting the type of 't' (line 28)
    t_351500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 't')
    # Getting the type of 't1' (line 28)
    t1_351501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 25), 't1')
    # Applying the binary operator 'div' (line 28)
    result_div_351502 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 23), 'div', t_351500, t1_351501)
    
    # Applying the binary operator '**' (line 28)
    result_pow_351503 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 13), '**', result_div_351499, result_div_351502)
    
    # Applying the binary operator '*' (line 28)
    result_mul_351504 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 8), '*', f0_351496, result_pow_351503)
    
    # Assigning a type to the variable 'f' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'f', result_mul_351504)
    # Getting the type of 'f' (line 29)
    f_351505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'f')
    # Assigning a type to the variable 'stypy_return_type' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type', f_351505)
    
    # ################# End of 'chirp_geometric(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chirp_geometric' in the type store
    # Getting the type of 'stypy_return_type' (line 27)
    stypy_return_type_351506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_351506)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chirp_geometric'
    return stypy_return_type_351506

# Assigning a type to the variable 'chirp_geometric' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'chirp_geometric', chirp_geometric)

@norecursion
def chirp_hyperbolic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'chirp_hyperbolic'
    module_type_store = module_type_store.open_function_context('chirp_hyperbolic', 32, 0, False)
    
    # Passed parameters checking function
    chirp_hyperbolic.stypy_localization = localization
    chirp_hyperbolic.stypy_type_of_self = None
    chirp_hyperbolic.stypy_type_store = module_type_store
    chirp_hyperbolic.stypy_function_name = 'chirp_hyperbolic'
    chirp_hyperbolic.stypy_param_names_list = ['t', 'f0', 'f1', 't1']
    chirp_hyperbolic.stypy_varargs_param_name = None
    chirp_hyperbolic.stypy_kwargs_param_name = None
    chirp_hyperbolic.stypy_call_defaults = defaults
    chirp_hyperbolic.stypy_call_varargs = varargs
    chirp_hyperbolic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'chirp_hyperbolic', ['t', 'f0', 'f1', 't1'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'chirp_hyperbolic', localization, ['t', 'f0', 'f1', 't1'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'chirp_hyperbolic(...)' code ##################

    
    # Assigning a BinOp to a Name (line 33):
    
    # Assigning a BinOp to a Name (line 33):
    # Getting the type of 'f0' (line 33)
    f0_351507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'f0')
    # Getting the type of 'f1' (line 33)
    f1_351508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'f1')
    # Applying the binary operator '*' (line 33)
    result_mul_351509 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 8), '*', f0_351507, f1_351508)
    
    # Getting the type of 't1' (line 33)
    t1_351510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 14), 't1')
    # Applying the binary operator '*' (line 33)
    result_mul_351511 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 13), '*', result_mul_351509, t1_351510)
    
    # Getting the type of 'f0' (line 33)
    f0_351512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 21), 'f0')
    # Getting the type of 'f1' (line 33)
    f1_351513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 26), 'f1')
    # Applying the binary operator '-' (line 33)
    result_sub_351514 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 21), '-', f0_351512, f1_351513)
    
    # Getting the type of 't' (line 33)
    t_351515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 30), 't')
    # Applying the binary operator '*' (line 33)
    result_mul_351516 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 20), '*', result_sub_351514, t_351515)
    
    # Getting the type of 'f1' (line 33)
    f1_351517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 34), 'f1')
    # Getting the type of 't1' (line 33)
    t1_351518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 37), 't1')
    # Applying the binary operator '*' (line 33)
    result_mul_351519 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 34), '*', f1_351517, t1_351518)
    
    # Applying the binary operator '+' (line 33)
    result_add_351520 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 20), '+', result_mul_351516, result_mul_351519)
    
    # Applying the binary operator 'div' (line 33)
    result_div_351521 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 17), 'div', result_mul_351511, result_add_351520)
    
    # Assigning a type to the variable 'f' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'f', result_div_351521)
    # Getting the type of 'f' (line 34)
    f_351522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'f')
    # Assigning a type to the variable 'stypy_return_type' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type', f_351522)
    
    # ################# End of 'chirp_hyperbolic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'chirp_hyperbolic' in the type store
    # Getting the type of 'stypy_return_type' (line 32)
    stypy_return_type_351523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_351523)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'chirp_hyperbolic'
    return stypy_return_type_351523

# Assigning a type to the variable 'chirp_hyperbolic' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'chirp_hyperbolic', chirp_hyperbolic)

@norecursion
def compute_frequency(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'compute_frequency'
    module_type_store = module_type_store.open_function_context('compute_frequency', 37, 0, False)
    
    # Passed parameters checking function
    compute_frequency.stypy_localization = localization
    compute_frequency.stypy_type_of_self = None
    compute_frequency.stypy_type_store = module_type_store
    compute_frequency.stypy_function_name = 'compute_frequency'
    compute_frequency.stypy_param_names_list = ['t', 'theta']
    compute_frequency.stypy_varargs_param_name = None
    compute_frequency.stypy_kwargs_param_name = None
    compute_frequency.stypy_call_defaults = defaults
    compute_frequency.stypy_call_varargs = varargs
    compute_frequency.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'compute_frequency', ['t', 'theta'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'compute_frequency', localization, ['t', 'theta'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'compute_frequency(...)' code ##################

    str_351524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, (-1)), 'str', "\n    Compute theta'(t)/(2*pi), where theta'(t) is the derivative of theta(t).\n    ")
    
    # Assigning a BinOp to a Name (line 43):
    
    # Assigning a BinOp to a Name (line 43):
    
    # Obtaining the type of the subscript
    int_351525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 11), 'int')
    # Getting the type of 't' (line 43)
    t_351526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 9), 't')
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___351527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 9), t_351526, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_351528 = invoke(stypy.reporting.localization.Localization(__file__, 43, 9), getitem___351527, int_351525)
    
    
    # Obtaining the type of the subscript
    int_351529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 18), 'int')
    # Getting the type of 't' (line 43)
    t_351530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 't')
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___351531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 16), t_351530, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_351532 = invoke(stypy.reporting.localization.Localization(__file__, 43, 16), getitem___351531, int_351529)
    
    # Applying the binary operator '-' (line 43)
    result_sub_351533 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 9), '-', subscript_call_result_351528, subscript_call_result_351532)
    
    # Assigning a type to the variable 'dt' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'dt', result_sub_351533)
    
    # Assigning a BinOp to a Name (line 44):
    
    # Assigning a BinOp to a Name (line 44):
    
    # Call to diff(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'theta' (line 44)
    theta_351536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'theta', False)
    # Processing the call keyword arguments (line 44)
    kwargs_351537 = {}
    # Getting the type of 'np' (line 44)
    np_351534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'np', False)
    # Obtaining the member 'diff' of a type (line 44)
    diff_351535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), np_351534, 'diff')
    # Calling diff(args, kwargs) (line 44)
    diff_call_result_351538 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), diff_351535, *[theta_351536], **kwargs_351537)
    
    int_351539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 24), 'int')
    # Getting the type of 'np' (line 44)
    np_351540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 26), 'np')
    # Obtaining the member 'pi' of a type (line 44)
    pi_351541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 26), np_351540, 'pi')
    # Applying the binary operator '*' (line 44)
    result_mul_351542 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 24), '*', int_351539, pi_351541)
    
    # Applying the binary operator 'div' (line 44)
    result_div_351543 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 8), 'div', diff_call_result_351538, result_mul_351542)
    
    # Getting the type of 'dt' (line 44)
    dt_351544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'dt')
    # Applying the binary operator 'div' (line 44)
    result_div_351545 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 33), 'div', result_div_351543, dt_351544)
    
    # Assigning a type to the variable 'f' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'f', result_div_351545)
    
    # Assigning a BinOp to a Name (line 45):
    
    # Assigning a BinOp to a Name (line 45):
    float_351546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 9), 'float')
    
    # Obtaining the type of the subscript
    int_351547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 16), 'int')
    slice_351548 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 45, 14), int_351547, None, None)
    # Getting the type of 't' (line 45)
    t_351549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 14), 't')
    # Obtaining the member '__getitem__' of a type (line 45)
    getitem___351550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 14), t_351549, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 45)
    subscript_call_result_351551 = invoke(stypy.reporting.localization.Localization(__file__, 45, 14), getitem___351550, slice_351548)
    
    
    # Obtaining the type of the subscript
    int_351552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 25), 'int')
    slice_351553 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 45, 22), None, int_351552, None)
    # Getting the type of 't' (line 45)
    t_351554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 22), 't')
    # Obtaining the member '__getitem__' of a type (line 45)
    getitem___351555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 22), t_351554, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 45)
    subscript_call_result_351556 = invoke(stypy.reporting.localization.Localization(__file__, 45, 22), getitem___351555, slice_351553)
    
    # Applying the binary operator '+' (line 45)
    result_add_351557 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 14), '+', subscript_call_result_351551, subscript_call_result_351556)
    
    # Applying the binary operator '*' (line 45)
    result_mul_351558 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 9), '*', float_351546, result_add_351557)
    
    # Assigning a type to the variable 'tf' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'tf', result_mul_351558)
    
    # Obtaining an instance of the builtin type 'tuple' (line 46)
    tuple_351559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 46)
    # Adding element type (line 46)
    # Getting the type of 'tf' (line 46)
    tf_351560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'tf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 11), tuple_351559, tf_351560)
    # Adding element type (line 46)
    # Getting the type of 'f' (line 46)
    f_351561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 'f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 11), tuple_351559, f_351561)
    
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type', tuple_351559)
    
    # ################# End of 'compute_frequency(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'compute_frequency' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_351562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_351562)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'compute_frequency'
    return stypy_return_type_351562

# Assigning a type to the variable 'compute_frequency' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'compute_frequency', compute_frequency)
# Declaration of the 'TestChirp' class

class TestChirp(object, ):

    @norecursion
    def test_linear_at_zero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_linear_at_zero'
        module_type_store = module_type_store.open_function_context('test_linear_at_zero', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_linear_at_zero.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_linear_at_zero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_linear_at_zero.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_linear_at_zero.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_linear_at_zero')
        TestChirp.test_linear_at_zero.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_linear_at_zero.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_linear_at_zero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_linear_at_zero.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_linear_at_zero.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_linear_at_zero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_linear_at_zero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_linear_at_zero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_linear_at_zero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_linear_at_zero(...)' code ##################

        
        # Assigning a Call to a Name (line 52):
        
        # Assigning a Call to a Name (line 52):
        
        # Call to chirp(...): (line 52)
        # Processing the call keyword arguments (line 52)
        int_351565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 30), 'int')
        keyword_351566 = int_351565
        float_351567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 36), 'float')
        keyword_351568 = float_351567
        float_351569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 44), 'float')
        keyword_351570 = float_351569
        float_351571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 52), 'float')
        keyword_351572 = float_351571
        str_351573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 64), 'str', 'linear')
        keyword_351574 = str_351573
        kwargs_351575 = {'f0': keyword_351568, 'f1': keyword_351570, 'method': keyword_351574, 't': keyword_351566, 't1': keyword_351572}
        # Getting the type of 'waveforms' (line 52)
        waveforms_351563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'waveforms', False)
        # Obtaining the member 'chirp' of a type (line 52)
        chirp_351564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), waveforms_351563, 'chirp')
        # Calling chirp(args, kwargs) (line 52)
        chirp_call_result_351576 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), chirp_351564, *[], **kwargs_351575)
        
        # Assigning a type to the variable 'w' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'w', chirp_call_result_351576)
        
        # Call to assert_almost_equal(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'w' (line 53)
        w_351578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 28), 'w', False)
        float_351579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 31), 'float')
        # Processing the call keyword arguments (line 53)
        kwargs_351580 = {}
        # Getting the type of 'assert_almost_equal' (line 53)
        assert_almost_equal_351577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 53)
        assert_almost_equal_call_result_351581 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), assert_almost_equal_351577, *[w_351578, float_351579], **kwargs_351580)
        
        
        # ################# End of 'test_linear_at_zero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_linear_at_zero' in the type store
        # Getting the type of 'stypy_return_type' (line 51)
        stypy_return_type_351582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_351582)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_linear_at_zero'
        return stypy_return_type_351582


    @norecursion
    def test_linear_freq_01(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_linear_freq_01'
        module_type_store = module_type_store.open_function_context('test_linear_freq_01', 55, 4, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_linear_freq_01.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_linear_freq_01.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_linear_freq_01.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_linear_freq_01.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_linear_freq_01')
        TestChirp.test_linear_freq_01.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_linear_freq_01.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_linear_freq_01.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_linear_freq_01.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_linear_freq_01.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_linear_freq_01.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_linear_freq_01.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_linear_freq_01', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_linear_freq_01', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_linear_freq_01(...)' code ##################

        
        # Assigning a Str to a Name (line 56):
        
        # Assigning a Str to a Name (line 56):
        str_351583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 17), 'str', 'linear')
        # Assigning a type to the variable 'method' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'method', str_351583)
        
        # Assigning a Num to a Name (line 57):
        
        # Assigning a Num to a Name (line 57):
        float_351584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 13), 'float')
        # Assigning a type to the variable 'f0' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'f0', float_351584)
        
        # Assigning a Num to a Name (line 58):
        
        # Assigning a Num to a Name (line 58):
        float_351585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 13), 'float')
        # Assigning a type to the variable 'f1' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'f1', float_351585)
        
        # Assigning a Num to a Name (line 59):
        
        # Assigning a Num to a Name (line 59):
        float_351586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 13), 'float')
        # Assigning a type to the variable 't1' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 't1', float_351586)
        
        # Assigning a Call to a Name (line 60):
        
        # Assigning a Call to a Name (line 60):
        
        # Call to linspace(...): (line 60)
        # Processing the call arguments (line 60)
        int_351589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 24), 'int')
        # Getting the type of 't1' (line 60)
        t1_351590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 27), 't1', False)
        int_351591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 31), 'int')
        # Processing the call keyword arguments (line 60)
        kwargs_351592 = {}
        # Getting the type of 'np' (line 60)
        np_351587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 60)
        linspace_351588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 12), np_351587, 'linspace')
        # Calling linspace(args, kwargs) (line 60)
        linspace_call_result_351593 = invoke(stypy.reporting.localization.Localization(__file__, 60, 12), linspace_351588, *[int_351589, t1_351590, int_351591], **kwargs_351592)
        
        # Assigning a type to the variable 't' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 't', linspace_call_result_351593)
        
        # Assigning a Call to a Name (line 61):
        
        # Assigning a Call to a Name (line 61):
        
        # Call to _chirp_phase(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 't' (line 61)
        t_351596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 39), 't', False)
        # Getting the type of 'f0' (line 61)
        f0_351597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 42), 'f0', False)
        # Getting the type of 't1' (line 61)
        t1_351598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 46), 't1', False)
        # Getting the type of 'f1' (line 61)
        f1_351599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 50), 'f1', False)
        # Getting the type of 'method' (line 61)
        method_351600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 54), 'method', False)
        # Processing the call keyword arguments (line 61)
        kwargs_351601 = {}
        # Getting the type of 'waveforms' (line 61)
        waveforms_351594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'waveforms', False)
        # Obtaining the member '_chirp_phase' of a type (line 61)
        _chirp_phase_351595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 16), waveforms_351594, '_chirp_phase')
        # Calling _chirp_phase(args, kwargs) (line 61)
        _chirp_phase_call_result_351602 = invoke(stypy.reporting.localization.Localization(__file__, 61, 16), _chirp_phase_351595, *[t_351596, f0_351597, t1_351598, f1_351599, method_351600], **kwargs_351601)
        
        # Assigning a type to the variable 'phase' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'phase', _chirp_phase_call_result_351602)
        
        # Assigning a Call to a Tuple (line 62):
        
        # Assigning a Subscript to a Name (line 62):
        
        # Obtaining the type of the subscript
        int_351603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 8), 'int')
        
        # Call to compute_frequency(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 't' (line 62)
        t_351605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 34), 't', False)
        # Getting the type of 'phase' (line 62)
        phase_351606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 37), 'phase', False)
        # Processing the call keyword arguments (line 62)
        kwargs_351607 = {}
        # Getting the type of 'compute_frequency' (line 62)
        compute_frequency_351604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 62)
        compute_frequency_call_result_351608 = invoke(stypy.reporting.localization.Localization(__file__, 62, 16), compute_frequency_351604, *[t_351605, phase_351606], **kwargs_351607)
        
        # Obtaining the member '__getitem__' of a type (line 62)
        getitem___351609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), compute_frequency_call_result_351608, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 62)
        subscript_call_result_351610 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), getitem___351609, int_351603)
        
        # Assigning a type to the variable 'tuple_var_assignment_351414' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_var_assignment_351414', subscript_call_result_351610)
        
        # Assigning a Subscript to a Name (line 62):
        
        # Obtaining the type of the subscript
        int_351611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 8), 'int')
        
        # Call to compute_frequency(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 't' (line 62)
        t_351613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 34), 't', False)
        # Getting the type of 'phase' (line 62)
        phase_351614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 37), 'phase', False)
        # Processing the call keyword arguments (line 62)
        kwargs_351615 = {}
        # Getting the type of 'compute_frequency' (line 62)
        compute_frequency_351612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 62)
        compute_frequency_call_result_351616 = invoke(stypy.reporting.localization.Localization(__file__, 62, 16), compute_frequency_351612, *[t_351613, phase_351614], **kwargs_351615)
        
        # Obtaining the member '__getitem__' of a type (line 62)
        getitem___351617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), compute_frequency_call_result_351616, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 62)
        subscript_call_result_351618 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), getitem___351617, int_351611)
        
        # Assigning a type to the variable 'tuple_var_assignment_351415' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_var_assignment_351415', subscript_call_result_351618)
        
        # Assigning a Name to a Name (line 62):
        # Getting the type of 'tuple_var_assignment_351414' (line 62)
        tuple_var_assignment_351414_351619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_var_assignment_351414')
        # Assigning a type to the variable 'tf' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tf', tuple_var_assignment_351414_351619)
        
        # Assigning a Name to a Name (line 62):
        # Getting the type of 'tuple_var_assignment_351415' (line 62)
        tuple_var_assignment_351415_351620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_var_assignment_351415')
        # Assigning a type to the variable 'f' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'f', tuple_var_assignment_351415_351620)
        
        # Assigning a Call to a Name (line 63):
        
        # Assigning a Call to a Name (line 63):
        
        # Call to max(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Call to abs(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'f' (line 63)
        f_351625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 31), 'f', False)
        
        # Call to chirp_linear(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'tf' (line 63)
        tf_351627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 48), 'tf', False)
        # Getting the type of 'f0' (line 63)
        f0_351628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 52), 'f0', False)
        # Getting the type of 'f1' (line 63)
        f1_351629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 56), 'f1', False)
        # Getting the type of 't1' (line 63)
        t1_351630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 60), 't1', False)
        # Processing the call keyword arguments (line 63)
        kwargs_351631 = {}
        # Getting the type of 'chirp_linear' (line 63)
        chirp_linear_351626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 35), 'chirp_linear', False)
        # Calling chirp_linear(args, kwargs) (line 63)
        chirp_linear_call_result_351632 = invoke(stypy.reporting.localization.Localization(__file__, 63, 35), chirp_linear_351626, *[tf_351627, f0_351628, f1_351629, t1_351630], **kwargs_351631)
        
        # Applying the binary operator '-' (line 63)
        result_sub_351633 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 31), '-', f_351625, chirp_linear_call_result_351632)
        
        # Processing the call keyword arguments (line 63)
        kwargs_351634 = {}
        # Getting the type of 'np' (line 63)
        np_351623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'np', False)
        # Obtaining the member 'abs' of a type (line 63)
        abs_351624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 24), np_351623, 'abs')
        # Calling abs(args, kwargs) (line 63)
        abs_call_result_351635 = invoke(stypy.reporting.localization.Localization(__file__, 63, 24), abs_351624, *[result_sub_351633], **kwargs_351634)
        
        # Processing the call keyword arguments (line 63)
        kwargs_351636 = {}
        # Getting the type of 'np' (line 63)
        np_351621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'np', False)
        # Obtaining the member 'max' of a type (line 63)
        max_351622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 17), np_351621, 'max')
        # Calling max(args, kwargs) (line 63)
        max_call_result_351637 = invoke(stypy.reporting.localization.Localization(__file__, 63, 17), max_351622, *[abs_call_result_351635], **kwargs_351636)
        
        # Assigning a type to the variable 'abserr' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'abserr', max_call_result_351637)
        
        # Call to assert_(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Getting the type of 'abserr' (line 64)
        abserr_351639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'abserr', False)
        float_351640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 25), 'float')
        # Applying the binary operator '<' (line 64)
        result_lt_351641 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 16), '<', abserr_351639, float_351640)
        
        # Processing the call keyword arguments (line 64)
        kwargs_351642 = {}
        # Getting the type of 'assert_' (line 64)
        assert__351638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 64)
        assert__call_result_351643 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), assert__351638, *[result_lt_351641], **kwargs_351642)
        
        
        # ################# End of 'test_linear_freq_01(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_linear_freq_01' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_351644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_351644)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_linear_freq_01'
        return stypy_return_type_351644


    @norecursion
    def test_linear_freq_02(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_linear_freq_02'
        module_type_store = module_type_store.open_function_context('test_linear_freq_02', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_linear_freq_02.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_linear_freq_02.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_linear_freq_02.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_linear_freq_02.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_linear_freq_02')
        TestChirp.test_linear_freq_02.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_linear_freq_02.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_linear_freq_02.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_linear_freq_02.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_linear_freq_02.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_linear_freq_02.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_linear_freq_02.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_linear_freq_02', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_linear_freq_02', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_linear_freq_02(...)' code ##################

        
        # Assigning a Str to a Name (line 67):
        
        # Assigning a Str to a Name (line 67):
        str_351645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 17), 'str', 'linear')
        # Assigning a type to the variable 'method' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'method', str_351645)
        
        # Assigning a Num to a Name (line 68):
        
        # Assigning a Num to a Name (line 68):
        float_351646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 13), 'float')
        # Assigning a type to the variable 'f0' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'f0', float_351646)
        
        # Assigning a Num to a Name (line 69):
        
        # Assigning a Num to a Name (line 69):
        float_351647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 13), 'float')
        # Assigning a type to the variable 'f1' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'f1', float_351647)
        
        # Assigning a Num to a Name (line 70):
        
        # Assigning a Num to a Name (line 70):
        float_351648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 13), 'float')
        # Assigning a type to the variable 't1' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 't1', float_351648)
        
        # Assigning a Call to a Name (line 71):
        
        # Assigning a Call to a Name (line 71):
        
        # Call to linspace(...): (line 71)
        # Processing the call arguments (line 71)
        int_351651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 24), 'int')
        # Getting the type of 't1' (line 71)
        t1_351652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 't1', False)
        int_351653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 31), 'int')
        # Processing the call keyword arguments (line 71)
        kwargs_351654 = {}
        # Getting the type of 'np' (line 71)
        np_351649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 71)
        linspace_351650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), np_351649, 'linspace')
        # Calling linspace(args, kwargs) (line 71)
        linspace_call_result_351655 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), linspace_351650, *[int_351651, t1_351652, int_351653], **kwargs_351654)
        
        # Assigning a type to the variable 't' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 't', linspace_call_result_351655)
        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to _chirp_phase(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 't' (line 72)
        t_351658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 39), 't', False)
        # Getting the type of 'f0' (line 72)
        f0_351659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 42), 'f0', False)
        # Getting the type of 't1' (line 72)
        t1_351660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 46), 't1', False)
        # Getting the type of 'f1' (line 72)
        f1_351661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 50), 'f1', False)
        # Getting the type of 'method' (line 72)
        method_351662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 54), 'method', False)
        # Processing the call keyword arguments (line 72)
        kwargs_351663 = {}
        # Getting the type of 'waveforms' (line 72)
        waveforms_351656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'waveforms', False)
        # Obtaining the member '_chirp_phase' of a type (line 72)
        _chirp_phase_351657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 16), waveforms_351656, '_chirp_phase')
        # Calling _chirp_phase(args, kwargs) (line 72)
        _chirp_phase_call_result_351664 = invoke(stypy.reporting.localization.Localization(__file__, 72, 16), _chirp_phase_351657, *[t_351658, f0_351659, t1_351660, f1_351661, method_351662], **kwargs_351663)
        
        # Assigning a type to the variable 'phase' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'phase', _chirp_phase_call_result_351664)
        
        # Assigning a Call to a Tuple (line 73):
        
        # Assigning a Subscript to a Name (line 73):
        
        # Obtaining the type of the subscript
        int_351665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 8), 'int')
        
        # Call to compute_frequency(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 't' (line 73)
        t_351667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 34), 't', False)
        # Getting the type of 'phase' (line 73)
        phase_351668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 37), 'phase', False)
        # Processing the call keyword arguments (line 73)
        kwargs_351669 = {}
        # Getting the type of 'compute_frequency' (line 73)
        compute_frequency_351666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 73)
        compute_frequency_call_result_351670 = invoke(stypy.reporting.localization.Localization(__file__, 73, 16), compute_frequency_351666, *[t_351667, phase_351668], **kwargs_351669)
        
        # Obtaining the member '__getitem__' of a type (line 73)
        getitem___351671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), compute_frequency_call_result_351670, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
        subscript_call_result_351672 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), getitem___351671, int_351665)
        
        # Assigning a type to the variable 'tuple_var_assignment_351416' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'tuple_var_assignment_351416', subscript_call_result_351672)
        
        # Assigning a Subscript to a Name (line 73):
        
        # Obtaining the type of the subscript
        int_351673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 8), 'int')
        
        # Call to compute_frequency(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 't' (line 73)
        t_351675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 34), 't', False)
        # Getting the type of 'phase' (line 73)
        phase_351676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 37), 'phase', False)
        # Processing the call keyword arguments (line 73)
        kwargs_351677 = {}
        # Getting the type of 'compute_frequency' (line 73)
        compute_frequency_351674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 73)
        compute_frequency_call_result_351678 = invoke(stypy.reporting.localization.Localization(__file__, 73, 16), compute_frequency_351674, *[t_351675, phase_351676], **kwargs_351677)
        
        # Obtaining the member '__getitem__' of a type (line 73)
        getitem___351679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), compute_frequency_call_result_351678, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
        subscript_call_result_351680 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), getitem___351679, int_351673)
        
        # Assigning a type to the variable 'tuple_var_assignment_351417' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'tuple_var_assignment_351417', subscript_call_result_351680)
        
        # Assigning a Name to a Name (line 73):
        # Getting the type of 'tuple_var_assignment_351416' (line 73)
        tuple_var_assignment_351416_351681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'tuple_var_assignment_351416')
        # Assigning a type to the variable 'tf' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'tf', tuple_var_assignment_351416_351681)
        
        # Assigning a Name to a Name (line 73):
        # Getting the type of 'tuple_var_assignment_351417' (line 73)
        tuple_var_assignment_351417_351682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'tuple_var_assignment_351417')
        # Assigning a type to the variable 'f' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'f', tuple_var_assignment_351417_351682)
        
        # Assigning a Call to a Name (line 74):
        
        # Assigning a Call to a Name (line 74):
        
        # Call to max(...): (line 74)
        # Processing the call arguments (line 74)
        
        # Call to abs(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'f' (line 74)
        f_351687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 31), 'f', False)
        
        # Call to chirp_linear(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'tf' (line 74)
        tf_351689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 48), 'tf', False)
        # Getting the type of 'f0' (line 74)
        f0_351690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 52), 'f0', False)
        # Getting the type of 'f1' (line 74)
        f1_351691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 56), 'f1', False)
        # Getting the type of 't1' (line 74)
        t1_351692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 60), 't1', False)
        # Processing the call keyword arguments (line 74)
        kwargs_351693 = {}
        # Getting the type of 'chirp_linear' (line 74)
        chirp_linear_351688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 35), 'chirp_linear', False)
        # Calling chirp_linear(args, kwargs) (line 74)
        chirp_linear_call_result_351694 = invoke(stypy.reporting.localization.Localization(__file__, 74, 35), chirp_linear_351688, *[tf_351689, f0_351690, f1_351691, t1_351692], **kwargs_351693)
        
        # Applying the binary operator '-' (line 74)
        result_sub_351695 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 31), '-', f_351687, chirp_linear_call_result_351694)
        
        # Processing the call keyword arguments (line 74)
        kwargs_351696 = {}
        # Getting the type of 'np' (line 74)
        np_351685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 24), 'np', False)
        # Obtaining the member 'abs' of a type (line 74)
        abs_351686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 24), np_351685, 'abs')
        # Calling abs(args, kwargs) (line 74)
        abs_call_result_351697 = invoke(stypy.reporting.localization.Localization(__file__, 74, 24), abs_351686, *[result_sub_351695], **kwargs_351696)
        
        # Processing the call keyword arguments (line 74)
        kwargs_351698 = {}
        # Getting the type of 'np' (line 74)
        np_351683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'np', False)
        # Obtaining the member 'max' of a type (line 74)
        max_351684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 17), np_351683, 'max')
        # Calling max(args, kwargs) (line 74)
        max_call_result_351699 = invoke(stypy.reporting.localization.Localization(__file__, 74, 17), max_351684, *[abs_call_result_351697], **kwargs_351698)
        
        # Assigning a type to the variable 'abserr' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'abserr', max_call_result_351699)
        
        # Call to assert_(...): (line 75)
        # Processing the call arguments (line 75)
        
        # Getting the type of 'abserr' (line 75)
        abserr_351701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'abserr', False)
        float_351702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'float')
        # Applying the binary operator '<' (line 75)
        result_lt_351703 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 16), '<', abserr_351701, float_351702)
        
        # Processing the call keyword arguments (line 75)
        kwargs_351704 = {}
        # Getting the type of 'assert_' (line 75)
        assert__351700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 75)
        assert__call_result_351705 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), assert__351700, *[result_lt_351703], **kwargs_351704)
        
        
        # ################# End of 'test_linear_freq_02(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_linear_freq_02' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_351706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_351706)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_linear_freq_02'
        return stypy_return_type_351706


    @norecursion
    def test_quadratic_at_zero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_quadratic_at_zero'
        module_type_store = module_type_store.open_function_context('test_quadratic_at_zero', 77, 4, False)
        # Assigning a type to the variable 'self' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_quadratic_at_zero.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_quadratic_at_zero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_quadratic_at_zero.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_quadratic_at_zero.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_quadratic_at_zero')
        TestChirp.test_quadratic_at_zero.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_quadratic_at_zero.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_quadratic_at_zero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_quadratic_at_zero.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_quadratic_at_zero.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_quadratic_at_zero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_quadratic_at_zero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_quadratic_at_zero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_quadratic_at_zero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_quadratic_at_zero(...)' code ##################

        
        # Assigning a Call to a Name (line 78):
        
        # Assigning a Call to a Name (line 78):
        
        # Call to chirp(...): (line 78)
        # Processing the call keyword arguments (line 78)
        int_351709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 30), 'int')
        keyword_351710 = int_351709
        float_351711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 36), 'float')
        keyword_351712 = float_351711
        float_351713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 44), 'float')
        keyword_351714 = float_351713
        float_351715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 52), 'float')
        keyword_351716 = float_351715
        str_351717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 64), 'str', 'quadratic')
        keyword_351718 = str_351717
        kwargs_351719 = {'f0': keyword_351712, 'f1': keyword_351714, 'method': keyword_351718, 't': keyword_351710, 't1': keyword_351716}
        # Getting the type of 'waveforms' (line 78)
        waveforms_351707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'waveforms', False)
        # Obtaining the member 'chirp' of a type (line 78)
        chirp_351708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), waveforms_351707, 'chirp')
        # Calling chirp(args, kwargs) (line 78)
        chirp_call_result_351720 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), chirp_351708, *[], **kwargs_351719)
        
        # Assigning a type to the variable 'w' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'w', chirp_call_result_351720)
        
        # Call to assert_almost_equal(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'w' (line 79)
        w_351722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 28), 'w', False)
        float_351723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 31), 'float')
        # Processing the call keyword arguments (line 79)
        kwargs_351724 = {}
        # Getting the type of 'assert_almost_equal' (line 79)
        assert_almost_equal_351721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 79)
        assert_almost_equal_call_result_351725 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), assert_almost_equal_351721, *[w_351722, float_351723], **kwargs_351724)
        
        
        # ################# End of 'test_quadratic_at_zero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_quadratic_at_zero' in the type store
        # Getting the type of 'stypy_return_type' (line 77)
        stypy_return_type_351726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_351726)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_quadratic_at_zero'
        return stypy_return_type_351726


    @norecursion
    def test_quadratic_at_zero2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_quadratic_at_zero2'
        module_type_store = module_type_store.open_function_context('test_quadratic_at_zero2', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_quadratic_at_zero2.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_quadratic_at_zero2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_quadratic_at_zero2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_quadratic_at_zero2.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_quadratic_at_zero2')
        TestChirp.test_quadratic_at_zero2.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_quadratic_at_zero2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_quadratic_at_zero2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_quadratic_at_zero2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_quadratic_at_zero2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_quadratic_at_zero2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_quadratic_at_zero2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_quadratic_at_zero2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_quadratic_at_zero2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_quadratic_at_zero2(...)' code ##################

        
        # Assigning a Call to a Name (line 82):
        
        # Assigning a Call to a Name (line 82):
        
        # Call to chirp(...): (line 82)
        # Processing the call keyword arguments (line 82)
        int_351729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 30), 'int')
        keyword_351730 = int_351729
        float_351731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 36), 'float')
        keyword_351732 = float_351731
        float_351733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 44), 'float')
        keyword_351734 = float_351733
        float_351735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 52), 'float')
        keyword_351736 = float_351735
        str_351737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 64), 'str', 'quadratic')
        keyword_351738 = str_351737
        # Getting the type of 'False' (line 83)
        False_351739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 40), 'False', False)
        keyword_351740 = False_351739
        kwargs_351741 = {'f0': keyword_351732, 'f1': keyword_351734, 'vertex_zero': keyword_351740, 't1': keyword_351736, 't': keyword_351730, 'method': keyword_351738}
        # Getting the type of 'waveforms' (line 82)
        waveforms_351727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'waveforms', False)
        # Obtaining the member 'chirp' of a type (line 82)
        chirp_351728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), waveforms_351727, 'chirp')
        # Calling chirp(args, kwargs) (line 82)
        chirp_call_result_351742 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), chirp_351728, *[], **kwargs_351741)
        
        # Assigning a type to the variable 'w' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'w', chirp_call_result_351742)
        
        # Call to assert_almost_equal(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'w' (line 84)
        w_351744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 28), 'w', False)
        float_351745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 31), 'float')
        # Processing the call keyword arguments (line 84)
        kwargs_351746 = {}
        # Getting the type of 'assert_almost_equal' (line 84)
        assert_almost_equal_351743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 84)
        assert_almost_equal_call_result_351747 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), assert_almost_equal_351743, *[w_351744, float_351745], **kwargs_351746)
        
        
        # ################# End of 'test_quadratic_at_zero2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_quadratic_at_zero2' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_351748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_351748)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_quadratic_at_zero2'
        return stypy_return_type_351748


    @norecursion
    def test_quadratic_freq_01(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_quadratic_freq_01'
        module_type_store = module_type_store.open_function_context('test_quadratic_freq_01', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_quadratic_freq_01.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_quadratic_freq_01.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_quadratic_freq_01.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_quadratic_freq_01.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_quadratic_freq_01')
        TestChirp.test_quadratic_freq_01.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_quadratic_freq_01.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_quadratic_freq_01.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_quadratic_freq_01.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_quadratic_freq_01.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_quadratic_freq_01.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_quadratic_freq_01.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_quadratic_freq_01', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_quadratic_freq_01', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_quadratic_freq_01(...)' code ##################

        
        # Assigning a Str to a Name (line 87):
        
        # Assigning a Str to a Name (line 87):
        str_351749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 17), 'str', 'quadratic')
        # Assigning a type to the variable 'method' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'method', str_351749)
        
        # Assigning a Num to a Name (line 88):
        
        # Assigning a Num to a Name (line 88):
        float_351750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 13), 'float')
        # Assigning a type to the variable 'f0' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'f0', float_351750)
        
        # Assigning a Num to a Name (line 89):
        
        # Assigning a Num to a Name (line 89):
        float_351751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 13), 'float')
        # Assigning a type to the variable 'f1' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'f1', float_351751)
        
        # Assigning a Num to a Name (line 90):
        
        # Assigning a Num to a Name (line 90):
        float_351752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 13), 'float')
        # Assigning a type to the variable 't1' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 't1', float_351752)
        
        # Assigning a Call to a Name (line 91):
        
        # Assigning a Call to a Name (line 91):
        
        # Call to linspace(...): (line 91)
        # Processing the call arguments (line 91)
        int_351755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 24), 'int')
        # Getting the type of 't1' (line 91)
        t1_351756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 't1', False)
        int_351757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 31), 'int')
        # Processing the call keyword arguments (line 91)
        kwargs_351758 = {}
        # Getting the type of 'np' (line 91)
        np_351753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 91)
        linspace_351754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), np_351753, 'linspace')
        # Calling linspace(args, kwargs) (line 91)
        linspace_call_result_351759 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), linspace_351754, *[int_351755, t1_351756, int_351757], **kwargs_351758)
        
        # Assigning a type to the variable 't' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 't', linspace_call_result_351759)
        
        # Assigning a Call to a Name (line 92):
        
        # Assigning a Call to a Name (line 92):
        
        # Call to _chirp_phase(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 't' (line 92)
        t_351762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 39), 't', False)
        # Getting the type of 'f0' (line 92)
        f0_351763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 42), 'f0', False)
        # Getting the type of 't1' (line 92)
        t1_351764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 46), 't1', False)
        # Getting the type of 'f1' (line 92)
        f1_351765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 50), 'f1', False)
        # Getting the type of 'method' (line 92)
        method_351766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 54), 'method', False)
        # Processing the call keyword arguments (line 92)
        kwargs_351767 = {}
        # Getting the type of 'waveforms' (line 92)
        waveforms_351760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'waveforms', False)
        # Obtaining the member '_chirp_phase' of a type (line 92)
        _chirp_phase_351761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 16), waveforms_351760, '_chirp_phase')
        # Calling _chirp_phase(args, kwargs) (line 92)
        _chirp_phase_call_result_351768 = invoke(stypy.reporting.localization.Localization(__file__, 92, 16), _chirp_phase_351761, *[t_351762, f0_351763, t1_351764, f1_351765, method_351766], **kwargs_351767)
        
        # Assigning a type to the variable 'phase' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'phase', _chirp_phase_call_result_351768)
        
        # Assigning a Call to a Tuple (line 93):
        
        # Assigning a Subscript to a Name (line 93):
        
        # Obtaining the type of the subscript
        int_351769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'int')
        
        # Call to compute_frequency(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 't' (line 93)
        t_351771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 34), 't', False)
        # Getting the type of 'phase' (line 93)
        phase_351772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 37), 'phase', False)
        # Processing the call keyword arguments (line 93)
        kwargs_351773 = {}
        # Getting the type of 'compute_frequency' (line 93)
        compute_frequency_351770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 93)
        compute_frequency_call_result_351774 = invoke(stypy.reporting.localization.Localization(__file__, 93, 16), compute_frequency_351770, *[t_351771, phase_351772], **kwargs_351773)
        
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___351775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), compute_frequency_call_result_351774, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_351776 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), getitem___351775, int_351769)
        
        # Assigning a type to the variable 'tuple_var_assignment_351418' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_var_assignment_351418', subscript_call_result_351776)
        
        # Assigning a Subscript to a Name (line 93):
        
        # Obtaining the type of the subscript
        int_351777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'int')
        
        # Call to compute_frequency(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 't' (line 93)
        t_351779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 34), 't', False)
        # Getting the type of 'phase' (line 93)
        phase_351780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 37), 'phase', False)
        # Processing the call keyword arguments (line 93)
        kwargs_351781 = {}
        # Getting the type of 'compute_frequency' (line 93)
        compute_frequency_351778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 93)
        compute_frequency_call_result_351782 = invoke(stypy.reporting.localization.Localization(__file__, 93, 16), compute_frequency_351778, *[t_351779, phase_351780], **kwargs_351781)
        
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___351783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), compute_frequency_call_result_351782, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_351784 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), getitem___351783, int_351777)
        
        # Assigning a type to the variable 'tuple_var_assignment_351419' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_var_assignment_351419', subscript_call_result_351784)
        
        # Assigning a Name to a Name (line 93):
        # Getting the type of 'tuple_var_assignment_351418' (line 93)
        tuple_var_assignment_351418_351785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_var_assignment_351418')
        # Assigning a type to the variable 'tf' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tf', tuple_var_assignment_351418_351785)
        
        # Assigning a Name to a Name (line 93):
        # Getting the type of 'tuple_var_assignment_351419' (line 93)
        tuple_var_assignment_351419_351786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'tuple_var_assignment_351419')
        # Assigning a type to the variable 'f' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'f', tuple_var_assignment_351419_351786)
        
        # Assigning a Call to a Name (line 94):
        
        # Assigning a Call to a Name (line 94):
        
        # Call to max(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Call to abs(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'f' (line 94)
        f_351791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 31), 'f', False)
        
        # Call to chirp_quadratic(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'tf' (line 94)
        tf_351793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 51), 'tf', False)
        # Getting the type of 'f0' (line 94)
        f0_351794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 55), 'f0', False)
        # Getting the type of 'f1' (line 94)
        f1_351795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 59), 'f1', False)
        # Getting the type of 't1' (line 94)
        t1_351796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 63), 't1', False)
        # Processing the call keyword arguments (line 94)
        kwargs_351797 = {}
        # Getting the type of 'chirp_quadratic' (line 94)
        chirp_quadratic_351792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 35), 'chirp_quadratic', False)
        # Calling chirp_quadratic(args, kwargs) (line 94)
        chirp_quadratic_call_result_351798 = invoke(stypy.reporting.localization.Localization(__file__, 94, 35), chirp_quadratic_351792, *[tf_351793, f0_351794, f1_351795, t1_351796], **kwargs_351797)
        
        # Applying the binary operator '-' (line 94)
        result_sub_351799 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 31), '-', f_351791, chirp_quadratic_call_result_351798)
        
        # Processing the call keyword arguments (line 94)
        kwargs_351800 = {}
        # Getting the type of 'np' (line 94)
        np_351789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 24), 'np', False)
        # Obtaining the member 'abs' of a type (line 94)
        abs_351790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 24), np_351789, 'abs')
        # Calling abs(args, kwargs) (line 94)
        abs_call_result_351801 = invoke(stypy.reporting.localization.Localization(__file__, 94, 24), abs_351790, *[result_sub_351799], **kwargs_351800)
        
        # Processing the call keyword arguments (line 94)
        kwargs_351802 = {}
        # Getting the type of 'np' (line 94)
        np_351787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 17), 'np', False)
        # Obtaining the member 'max' of a type (line 94)
        max_351788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 17), np_351787, 'max')
        # Calling max(args, kwargs) (line 94)
        max_call_result_351803 = invoke(stypy.reporting.localization.Localization(__file__, 94, 17), max_351788, *[abs_call_result_351801], **kwargs_351802)
        
        # Assigning a type to the variable 'abserr' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'abserr', max_call_result_351803)
        
        # Call to assert_(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Getting the type of 'abserr' (line 95)
        abserr_351805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'abserr', False)
        float_351806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 25), 'float')
        # Applying the binary operator '<' (line 95)
        result_lt_351807 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 16), '<', abserr_351805, float_351806)
        
        # Processing the call keyword arguments (line 95)
        kwargs_351808 = {}
        # Getting the type of 'assert_' (line 95)
        assert__351804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 95)
        assert__call_result_351809 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), assert__351804, *[result_lt_351807], **kwargs_351808)
        
        
        # ################# End of 'test_quadratic_freq_01(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_quadratic_freq_01' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_351810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_351810)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_quadratic_freq_01'
        return stypy_return_type_351810


    @norecursion
    def test_quadratic_freq_02(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_quadratic_freq_02'
        module_type_store = module_type_store.open_function_context('test_quadratic_freq_02', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_quadratic_freq_02.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_quadratic_freq_02.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_quadratic_freq_02.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_quadratic_freq_02.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_quadratic_freq_02')
        TestChirp.test_quadratic_freq_02.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_quadratic_freq_02.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_quadratic_freq_02.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_quadratic_freq_02.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_quadratic_freq_02.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_quadratic_freq_02.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_quadratic_freq_02.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_quadratic_freq_02', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_quadratic_freq_02', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_quadratic_freq_02(...)' code ##################

        
        # Assigning a Str to a Name (line 98):
        
        # Assigning a Str to a Name (line 98):
        str_351811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 17), 'str', 'quadratic')
        # Assigning a type to the variable 'method' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'method', str_351811)
        
        # Assigning a Num to a Name (line 99):
        
        # Assigning a Num to a Name (line 99):
        float_351812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 13), 'float')
        # Assigning a type to the variable 'f0' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'f0', float_351812)
        
        # Assigning a Num to a Name (line 100):
        
        # Assigning a Num to a Name (line 100):
        float_351813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 13), 'float')
        # Assigning a type to the variable 'f1' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'f1', float_351813)
        
        # Assigning a Num to a Name (line 101):
        
        # Assigning a Num to a Name (line 101):
        float_351814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 13), 'float')
        # Assigning a type to the variable 't1' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 't1', float_351814)
        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to linspace(...): (line 102)
        # Processing the call arguments (line 102)
        int_351817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 24), 'int')
        # Getting the type of 't1' (line 102)
        t1_351818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 't1', False)
        int_351819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 31), 'int')
        # Processing the call keyword arguments (line 102)
        kwargs_351820 = {}
        # Getting the type of 'np' (line 102)
        np_351815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 102)
        linspace_351816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), np_351815, 'linspace')
        # Calling linspace(args, kwargs) (line 102)
        linspace_call_result_351821 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), linspace_351816, *[int_351817, t1_351818, int_351819], **kwargs_351820)
        
        # Assigning a type to the variable 't' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 't', linspace_call_result_351821)
        
        # Assigning a Call to a Name (line 103):
        
        # Assigning a Call to a Name (line 103):
        
        # Call to _chirp_phase(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 't' (line 103)
        t_351824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 39), 't', False)
        # Getting the type of 'f0' (line 103)
        f0_351825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 42), 'f0', False)
        # Getting the type of 't1' (line 103)
        t1_351826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 46), 't1', False)
        # Getting the type of 'f1' (line 103)
        f1_351827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 50), 'f1', False)
        # Getting the type of 'method' (line 103)
        method_351828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 54), 'method', False)
        # Processing the call keyword arguments (line 103)
        kwargs_351829 = {}
        # Getting the type of 'waveforms' (line 103)
        waveforms_351822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'waveforms', False)
        # Obtaining the member '_chirp_phase' of a type (line 103)
        _chirp_phase_351823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 16), waveforms_351822, '_chirp_phase')
        # Calling _chirp_phase(args, kwargs) (line 103)
        _chirp_phase_call_result_351830 = invoke(stypy.reporting.localization.Localization(__file__, 103, 16), _chirp_phase_351823, *[t_351824, f0_351825, t1_351826, f1_351827, method_351828], **kwargs_351829)
        
        # Assigning a type to the variable 'phase' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'phase', _chirp_phase_call_result_351830)
        
        # Assigning a Call to a Tuple (line 104):
        
        # Assigning a Subscript to a Name (line 104):
        
        # Obtaining the type of the subscript
        int_351831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 8), 'int')
        
        # Call to compute_frequency(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 't' (line 104)
        t_351833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 't', False)
        # Getting the type of 'phase' (line 104)
        phase_351834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 37), 'phase', False)
        # Processing the call keyword arguments (line 104)
        kwargs_351835 = {}
        # Getting the type of 'compute_frequency' (line 104)
        compute_frequency_351832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 104)
        compute_frequency_call_result_351836 = invoke(stypy.reporting.localization.Localization(__file__, 104, 16), compute_frequency_351832, *[t_351833, phase_351834], **kwargs_351835)
        
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___351837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), compute_frequency_call_result_351836, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_351838 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), getitem___351837, int_351831)
        
        # Assigning a type to the variable 'tuple_var_assignment_351420' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'tuple_var_assignment_351420', subscript_call_result_351838)
        
        # Assigning a Subscript to a Name (line 104):
        
        # Obtaining the type of the subscript
        int_351839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 8), 'int')
        
        # Call to compute_frequency(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 't' (line 104)
        t_351841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 't', False)
        # Getting the type of 'phase' (line 104)
        phase_351842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 37), 'phase', False)
        # Processing the call keyword arguments (line 104)
        kwargs_351843 = {}
        # Getting the type of 'compute_frequency' (line 104)
        compute_frequency_351840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 104)
        compute_frequency_call_result_351844 = invoke(stypy.reporting.localization.Localization(__file__, 104, 16), compute_frequency_351840, *[t_351841, phase_351842], **kwargs_351843)
        
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___351845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), compute_frequency_call_result_351844, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_351846 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), getitem___351845, int_351839)
        
        # Assigning a type to the variable 'tuple_var_assignment_351421' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'tuple_var_assignment_351421', subscript_call_result_351846)
        
        # Assigning a Name to a Name (line 104):
        # Getting the type of 'tuple_var_assignment_351420' (line 104)
        tuple_var_assignment_351420_351847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'tuple_var_assignment_351420')
        # Assigning a type to the variable 'tf' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'tf', tuple_var_assignment_351420_351847)
        
        # Assigning a Name to a Name (line 104):
        # Getting the type of 'tuple_var_assignment_351421' (line 104)
        tuple_var_assignment_351421_351848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'tuple_var_assignment_351421')
        # Assigning a type to the variable 'f' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'f', tuple_var_assignment_351421_351848)
        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to max(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Call to abs(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'f' (line 105)
        f_351853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 31), 'f', False)
        
        # Call to chirp_quadratic(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'tf' (line 105)
        tf_351855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 51), 'tf', False)
        # Getting the type of 'f0' (line 105)
        f0_351856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 55), 'f0', False)
        # Getting the type of 'f1' (line 105)
        f1_351857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 59), 'f1', False)
        # Getting the type of 't1' (line 105)
        t1_351858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 63), 't1', False)
        # Processing the call keyword arguments (line 105)
        kwargs_351859 = {}
        # Getting the type of 'chirp_quadratic' (line 105)
        chirp_quadratic_351854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 35), 'chirp_quadratic', False)
        # Calling chirp_quadratic(args, kwargs) (line 105)
        chirp_quadratic_call_result_351860 = invoke(stypy.reporting.localization.Localization(__file__, 105, 35), chirp_quadratic_351854, *[tf_351855, f0_351856, f1_351857, t1_351858], **kwargs_351859)
        
        # Applying the binary operator '-' (line 105)
        result_sub_351861 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 31), '-', f_351853, chirp_quadratic_call_result_351860)
        
        # Processing the call keyword arguments (line 105)
        kwargs_351862 = {}
        # Getting the type of 'np' (line 105)
        np_351851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'np', False)
        # Obtaining the member 'abs' of a type (line 105)
        abs_351852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 24), np_351851, 'abs')
        # Calling abs(args, kwargs) (line 105)
        abs_call_result_351863 = invoke(stypy.reporting.localization.Localization(__file__, 105, 24), abs_351852, *[result_sub_351861], **kwargs_351862)
        
        # Processing the call keyword arguments (line 105)
        kwargs_351864 = {}
        # Getting the type of 'np' (line 105)
        np_351849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 17), 'np', False)
        # Obtaining the member 'max' of a type (line 105)
        max_351850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 17), np_351849, 'max')
        # Calling max(args, kwargs) (line 105)
        max_call_result_351865 = invoke(stypy.reporting.localization.Localization(__file__, 105, 17), max_351850, *[abs_call_result_351863], **kwargs_351864)
        
        # Assigning a type to the variable 'abserr' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'abserr', max_call_result_351865)
        
        # Call to assert_(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Getting the type of 'abserr' (line 106)
        abserr_351867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'abserr', False)
        float_351868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 25), 'float')
        # Applying the binary operator '<' (line 106)
        result_lt_351869 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 16), '<', abserr_351867, float_351868)
        
        # Processing the call keyword arguments (line 106)
        kwargs_351870 = {}
        # Getting the type of 'assert_' (line 106)
        assert__351866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 106)
        assert__call_result_351871 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), assert__351866, *[result_lt_351869], **kwargs_351870)
        
        
        # ################# End of 'test_quadratic_freq_02(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_quadratic_freq_02' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_351872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_351872)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_quadratic_freq_02'
        return stypy_return_type_351872


    @norecursion
    def test_logarithmic_at_zero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_logarithmic_at_zero'
        module_type_store = module_type_store.open_function_context('test_logarithmic_at_zero', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_logarithmic_at_zero.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_logarithmic_at_zero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_logarithmic_at_zero.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_logarithmic_at_zero.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_logarithmic_at_zero')
        TestChirp.test_logarithmic_at_zero.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_logarithmic_at_zero.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_logarithmic_at_zero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_logarithmic_at_zero.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_logarithmic_at_zero.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_logarithmic_at_zero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_logarithmic_at_zero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_logarithmic_at_zero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_logarithmic_at_zero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_logarithmic_at_zero(...)' code ##################

        
        # Assigning a Call to a Name (line 109):
        
        # Assigning a Call to a Name (line 109):
        
        # Call to chirp(...): (line 109)
        # Processing the call keyword arguments (line 109)
        int_351875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 30), 'int')
        keyword_351876 = int_351875
        float_351877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 36), 'float')
        keyword_351878 = float_351877
        float_351879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 44), 'float')
        keyword_351880 = float_351879
        float_351881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 52), 'float')
        keyword_351882 = float_351881
        str_351883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 64), 'str', 'logarithmic')
        keyword_351884 = str_351883
        kwargs_351885 = {'f0': keyword_351878, 'f1': keyword_351880, 'method': keyword_351884, 't': keyword_351876, 't1': keyword_351882}
        # Getting the type of 'waveforms' (line 109)
        waveforms_351873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'waveforms', False)
        # Obtaining the member 'chirp' of a type (line 109)
        chirp_351874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 12), waveforms_351873, 'chirp')
        # Calling chirp(args, kwargs) (line 109)
        chirp_call_result_351886 = invoke(stypy.reporting.localization.Localization(__file__, 109, 12), chirp_351874, *[], **kwargs_351885)
        
        # Assigning a type to the variable 'w' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'w', chirp_call_result_351886)
        
        # Call to assert_almost_equal(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'w' (line 110)
        w_351888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 28), 'w', False)
        float_351889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 31), 'float')
        # Processing the call keyword arguments (line 110)
        kwargs_351890 = {}
        # Getting the type of 'assert_almost_equal' (line 110)
        assert_almost_equal_351887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 110)
        assert_almost_equal_call_result_351891 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), assert_almost_equal_351887, *[w_351888, float_351889], **kwargs_351890)
        
        
        # ################# End of 'test_logarithmic_at_zero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_logarithmic_at_zero' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_351892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_351892)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_logarithmic_at_zero'
        return stypy_return_type_351892


    @norecursion
    def test_logarithmic_freq_01(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_logarithmic_freq_01'
        module_type_store = module_type_store.open_function_context('test_logarithmic_freq_01', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_logarithmic_freq_01.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_logarithmic_freq_01.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_logarithmic_freq_01.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_logarithmic_freq_01.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_logarithmic_freq_01')
        TestChirp.test_logarithmic_freq_01.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_logarithmic_freq_01.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_logarithmic_freq_01.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_logarithmic_freq_01.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_logarithmic_freq_01.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_logarithmic_freq_01.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_logarithmic_freq_01.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_logarithmic_freq_01', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_logarithmic_freq_01', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_logarithmic_freq_01(...)' code ##################

        
        # Assigning a Str to a Name (line 113):
        
        # Assigning a Str to a Name (line 113):
        str_351893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 17), 'str', 'logarithmic')
        # Assigning a type to the variable 'method' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'method', str_351893)
        
        # Assigning a Num to a Name (line 114):
        
        # Assigning a Num to a Name (line 114):
        float_351894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 13), 'float')
        # Assigning a type to the variable 'f0' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'f0', float_351894)
        
        # Assigning a Num to a Name (line 115):
        
        # Assigning a Num to a Name (line 115):
        float_351895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 13), 'float')
        # Assigning a type to the variable 'f1' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'f1', float_351895)
        
        # Assigning a Num to a Name (line 116):
        
        # Assigning a Num to a Name (line 116):
        float_351896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 13), 'float')
        # Assigning a type to the variable 't1' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 't1', float_351896)
        
        # Assigning a Call to a Name (line 117):
        
        # Assigning a Call to a Name (line 117):
        
        # Call to linspace(...): (line 117)
        # Processing the call arguments (line 117)
        int_351899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 24), 'int')
        # Getting the type of 't1' (line 117)
        t1_351900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 't1', False)
        int_351901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 31), 'int')
        # Processing the call keyword arguments (line 117)
        kwargs_351902 = {}
        # Getting the type of 'np' (line 117)
        np_351897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 117)
        linspace_351898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), np_351897, 'linspace')
        # Calling linspace(args, kwargs) (line 117)
        linspace_call_result_351903 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), linspace_351898, *[int_351899, t1_351900, int_351901], **kwargs_351902)
        
        # Assigning a type to the variable 't' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 't', linspace_call_result_351903)
        
        # Assigning a Call to a Name (line 118):
        
        # Assigning a Call to a Name (line 118):
        
        # Call to _chirp_phase(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 't' (line 118)
        t_351906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 39), 't', False)
        # Getting the type of 'f0' (line 118)
        f0_351907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 42), 'f0', False)
        # Getting the type of 't1' (line 118)
        t1_351908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 46), 't1', False)
        # Getting the type of 'f1' (line 118)
        f1_351909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 50), 'f1', False)
        # Getting the type of 'method' (line 118)
        method_351910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 54), 'method', False)
        # Processing the call keyword arguments (line 118)
        kwargs_351911 = {}
        # Getting the type of 'waveforms' (line 118)
        waveforms_351904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'waveforms', False)
        # Obtaining the member '_chirp_phase' of a type (line 118)
        _chirp_phase_351905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 16), waveforms_351904, '_chirp_phase')
        # Calling _chirp_phase(args, kwargs) (line 118)
        _chirp_phase_call_result_351912 = invoke(stypy.reporting.localization.Localization(__file__, 118, 16), _chirp_phase_351905, *[t_351906, f0_351907, t1_351908, f1_351909, method_351910], **kwargs_351911)
        
        # Assigning a type to the variable 'phase' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'phase', _chirp_phase_call_result_351912)
        
        # Assigning a Call to a Tuple (line 119):
        
        # Assigning a Subscript to a Name (line 119):
        
        # Obtaining the type of the subscript
        int_351913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 8), 'int')
        
        # Call to compute_frequency(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 't' (line 119)
        t_351915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 34), 't', False)
        # Getting the type of 'phase' (line 119)
        phase_351916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 37), 'phase', False)
        # Processing the call keyword arguments (line 119)
        kwargs_351917 = {}
        # Getting the type of 'compute_frequency' (line 119)
        compute_frequency_351914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 119)
        compute_frequency_call_result_351918 = invoke(stypy.reporting.localization.Localization(__file__, 119, 16), compute_frequency_351914, *[t_351915, phase_351916], **kwargs_351917)
        
        # Obtaining the member '__getitem__' of a type (line 119)
        getitem___351919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), compute_frequency_call_result_351918, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 119)
        subscript_call_result_351920 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), getitem___351919, int_351913)
        
        # Assigning a type to the variable 'tuple_var_assignment_351422' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'tuple_var_assignment_351422', subscript_call_result_351920)
        
        # Assigning a Subscript to a Name (line 119):
        
        # Obtaining the type of the subscript
        int_351921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 8), 'int')
        
        # Call to compute_frequency(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 't' (line 119)
        t_351923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 34), 't', False)
        # Getting the type of 'phase' (line 119)
        phase_351924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 37), 'phase', False)
        # Processing the call keyword arguments (line 119)
        kwargs_351925 = {}
        # Getting the type of 'compute_frequency' (line 119)
        compute_frequency_351922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 119)
        compute_frequency_call_result_351926 = invoke(stypy.reporting.localization.Localization(__file__, 119, 16), compute_frequency_351922, *[t_351923, phase_351924], **kwargs_351925)
        
        # Obtaining the member '__getitem__' of a type (line 119)
        getitem___351927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), compute_frequency_call_result_351926, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 119)
        subscript_call_result_351928 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), getitem___351927, int_351921)
        
        # Assigning a type to the variable 'tuple_var_assignment_351423' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'tuple_var_assignment_351423', subscript_call_result_351928)
        
        # Assigning a Name to a Name (line 119):
        # Getting the type of 'tuple_var_assignment_351422' (line 119)
        tuple_var_assignment_351422_351929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'tuple_var_assignment_351422')
        # Assigning a type to the variable 'tf' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'tf', tuple_var_assignment_351422_351929)
        
        # Assigning a Name to a Name (line 119):
        # Getting the type of 'tuple_var_assignment_351423' (line 119)
        tuple_var_assignment_351423_351930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'tuple_var_assignment_351423')
        # Assigning a type to the variable 'f' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'f', tuple_var_assignment_351423_351930)
        
        # Assigning a Call to a Name (line 120):
        
        # Assigning a Call to a Name (line 120):
        
        # Call to max(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Call to abs(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'f' (line 120)
        f_351935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 31), 'f', False)
        
        # Call to chirp_geometric(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'tf' (line 120)
        tf_351937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 51), 'tf', False)
        # Getting the type of 'f0' (line 120)
        f0_351938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 55), 'f0', False)
        # Getting the type of 'f1' (line 120)
        f1_351939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 59), 'f1', False)
        # Getting the type of 't1' (line 120)
        t1_351940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 63), 't1', False)
        # Processing the call keyword arguments (line 120)
        kwargs_351941 = {}
        # Getting the type of 'chirp_geometric' (line 120)
        chirp_geometric_351936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 35), 'chirp_geometric', False)
        # Calling chirp_geometric(args, kwargs) (line 120)
        chirp_geometric_call_result_351942 = invoke(stypy.reporting.localization.Localization(__file__, 120, 35), chirp_geometric_351936, *[tf_351937, f0_351938, f1_351939, t1_351940], **kwargs_351941)
        
        # Applying the binary operator '-' (line 120)
        result_sub_351943 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 31), '-', f_351935, chirp_geometric_call_result_351942)
        
        # Processing the call keyword arguments (line 120)
        kwargs_351944 = {}
        # Getting the type of 'np' (line 120)
        np_351933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 24), 'np', False)
        # Obtaining the member 'abs' of a type (line 120)
        abs_351934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 24), np_351933, 'abs')
        # Calling abs(args, kwargs) (line 120)
        abs_call_result_351945 = invoke(stypy.reporting.localization.Localization(__file__, 120, 24), abs_351934, *[result_sub_351943], **kwargs_351944)
        
        # Processing the call keyword arguments (line 120)
        kwargs_351946 = {}
        # Getting the type of 'np' (line 120)
        np_351931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 17), 'np', False)
        # Obtaining the member 'max' of a type (line 120)
        max_351932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 17), np_351931, 'max')
        # Calling max(args, kwargs) (line 120)
        max_call_result_351947 = invoke(stypy.reporting.localization.Localization(__file__, 120, 17), max_351932, *[abs_call_result_351945], **kwargs_351946)
        
        # Assigning a type to the variable 'abserr' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'abserr', max_call_result_351947)
        
        # Call to assert_(...): (line 121)
        # Processing the call arguments (line 121)
        
        # Getting the type of 'abserr' (line 121)
        abserr_351949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'abserr', False)
        float_351950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 25), 'float')
        # Applying the binary operator '<' (line 121)
        result_lt_351951 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 16), '<', abserr_351949, float_351950)
        
        # Processing the call keyword arguments (line 121)
        kwargs_351952 = {}
        # Getting the type of 'assert_' (line 121)
        assert__351948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 121)
        assert__call_result_351953 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), assert__351948, *[result_lt_351951], **kwargs_351952)
        
        
        # ################# End of 'test_logarithmic_freq_01(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_logarithmic_freq_01' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_351954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_351954)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_logarithmic_freq_01'
        return stypy_return_type_351954


    @norecursion
    def test_logarithmic_freq_02(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_logarithmic_freq_02'
        module_type_store = module_type_store.open_function_context('test_logarithmic_freq_02', 123, 4, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_logarithmic_freq_02.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_logarithmic_freq_02.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_logarithmic_freq_02.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_logarithmic_freq_02.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_logarithmic_freq_02')
        TestChirp.test_logarithmic_freq_02.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_logarithmic_freq_02.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_logarithmic_freq_02.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_logarithmic_freq_02.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_logarithmic_freq_02.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_logarithmic_freq_02.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_logarithmic_freq_02.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_logarithmic_freq_02', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_logarithmic_freq_02', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_logarithmic_freq_02(...)' code ##################

        
        # Assigning a Str to a Name (line 124):
        
        # Assigning a Str to a Name (line 124):
        str_351955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 17), 'str', 'logarithmic')
        # Assigning a type to the variable 'method' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'method', str_351955)
        
        # Assigning a Num to a Name (line 125):
        
        # Assigning a Num to a Name (line 125):
        float_351956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 13), 'float')
        # Assigning a type to the variable 'f0' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'f0', float_351956)
        
        # Assigning a Num to a Name (line 126):
        
        # Assigning a Num to a Name (line 126):
        float_351957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 13), 'float')
        # Assigning a type to the variable 'f1' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'f1', float_351957)
        
        # Assigning a Num to a Name (line 127):
        
        # Assigning a Num to a Name (line 127):
        float_351958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 13), 'float')
        # Assigning a type to the variable 't1' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 't1', float_351958)
        
        # Assigning a Call to a Name (line 128):
        
        # Assigning a Call to a Name (line 128):
        
        # Call to linspace(...): (line 128)
        # Processing the call arguments (line 128)
        int_351961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 24), 'int')
        # Getting the type of 't1' (line 128)
        t1_351962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 27), 't1', False)
        int_351963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 31), 'int')
        # Processing the call keyword arguments (line 128)
        kwargs_351964 = {}
        # Getting the type of 'np' (line 128)
        np_351959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 128)
        linspace_351960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), np_351959, 'linspace')
        # Calling linspace(args, kwargs) (line 128)
        linspace_call_result_351965 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), linspace_351960, *[int_351961, t1_351962, int_351963], **kwargs_351964)
        
        # Assigning a type to the variable 't' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 't', linspace_call_result_351965)
        
        # Assigning a Call to a Name (line 129):
        
        # Assigning a Call to a Name (line 129):
        
        # Call to _chirp_phase(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 't' (line 129)
        t_351968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 39), 't', False)
        # Getting the type of 'f0' (line 129)
        f0_351969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 42), 'f0', False)
        # Getting the type of 't1' (line 129)
        t1_351970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 46), 't1', False)
        # Getting the type of 'f1' (line 129)
        f1_351971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 50), 'f1', False)
        # Getting the type of 'method' (line 129)
        method_351972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 54), 'method', False)
        # Processing the call keyword arguments (line 129)
        kwargs_351973 = {}
        # Getting the type of 'waveforms' (line 129)
        waveforms_351966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'waveforms', False)
        # Obtaining the member '_chirp_phase' of a type (line 129)
        _chirp_phase_351967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 16), waveforms_351966, '_chirp_phase')
        # Calling _chirp_phase(args, kwargs) (line 129)
        _chirp_phase_call_result_351974 = invoke(stypy.reporting.localization.Localization(__file__, 129, 16), _chirp_phase_351967, *[t_351968, f0_351969, t1_351970, f1_351971, method_351972], **kwargs_351973)
        
        # Assigning a type to the variable 'phase' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'phase', _chirp_phase_call_result_351974)
        
        # Assigning a Call to a Tuple (line 130):
        
        # Assigning a Subscript to a Name (line 130):
        
        # Obtaining the type of the subscript
        int_351975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 8), 'int')
        
        # Call to compute_frequency(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 't' (line 130)
        t_351977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 34), 't', False)
        # Getting the type of 'phase' (line 130)
        phase_351978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 37), 'phase', False)
        # Processing the call keyword arguments (line 130)
        kwargs_351979 = {}
        # Getting the type of 'compute_frequency' (line 130)
        compute_frequency_351976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 130)
        compute_frequency_call_result_351980 = invoke(stypy.reporting.localization.Localization(__file__, 130, 16), compute_frequency_351976, *[t_351977, phase_351978], **kwargs_351979)
        
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___351981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), compute_frequency_call_result_351980, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_351982 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), getitem___351981, int_351975)
        
        # Assigning a type to the variable 'tuple_var_assignment_351424' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'tuple_var_assignment_351424', subscript_call_result_351982)
        
        # Assigning a Subscript to a Name (line 130):
        
        # Obtaining the type of the subscript
        int_351983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 8), 'int')
        
        # Call to compute_frequency(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 't' (line 130)
        t_351985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 34), 't', False)
        # Getting the type of 'phase' (line 130)
        phase_351986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 37), 'phase', False)
        # Processing the call keyword arguments (line 130)
        kwargs_351987 = {}
        # Getting the type of 'compute_frequency' (line 130)
        compute_frequency_351984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 130)
        compute_frequency_call_result_351988 = invoke(stypy.reporting.localization.Localization(__file__, 130, 16), compute_frequency_351984, *[t_351985, phase_351986], **kwargs_351987)
        
        # Obtaining the member '__getitem__' of a type (line 130)
        getitem___351989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), compute_frequency_call_result_351988, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 130)
        subscript_call_result_351990 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), getitem___351989, int_351983)
        
        # Assigning a type to the variable 'tuple_var_assignment_351425' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'tuple_var_assignment_351425', subscript_call_result_351990)
        
        # Assigning a Name to a Name (line 130):
        # Getting the type of 'tuple_var_assignment_351424' (line 130)
        tuple_var_assignment_351424_351991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'tuple_var_assignment_351424')
        # Assigning a type to the variable 'tf' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'tf', tuple_var_assignment_351424_351991)
        
        # Assigning a Name to a Name (line 130):
        # Getting the type of 'tuple_var_assignment_351425' (line 130)
        tuple_var_assignment_351425_351992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'tuple_var_assignment_351425')
        # Assigning a type to the variable 'f' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'f', tuple_var_assignment_351425_351992)
        
        # Assigning a Call to a Name (line 131):
        
        # Assigning a Call to a Name (line 131):
        
        # Call to max(...): (line 131)
        # Processing the call arguments (line 131)
        
        # Call to abs(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'f' (line 131)
        f_351997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 31), 'f', False)
        
        # Call to chirp_geometric(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'tf' (line 131)
        tf_351999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 51), 'tf', False)
        # Getting the type of 'f0' (line 131)
        f0_352000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 55), 'f0', False)
        # Getting the type of 'f1' (line 131)
        f1_352001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 59), 'f1', False)
        # Getting the type of 't1' (line 131)
        t1_352002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 63), 't1', False)
        # Processing the call keyword arguments (line 131)
        kwargs_352003 = {}
        # Getting the type of 'chirp_geometric' (line 131)
        chirp_geometric_351998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 35), 'chirp_geometric', False)
        # Calling chirp_geometric(args, kwargs) (line 131)
        chirp_geometric_call_result_352004 = invoke(stypy.reporting.localization.Localization(__file__, 131, 35), chirp_geometric_351998, *[tf_351999, f0_352000, f1_352001, t1_352002], **kwargs_352003)
        
        # Applying the binary operator '-' (line 131)
        result_sub_352005 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 31), '-', f_351997, chirp_geometric_call_result_352004)
        
        # Processing the call keyword arguments (line 131)
        kwargs_352006 = {}
        # Getting the type of 'np' (line 131)
        np_351995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 24), 'np', False)
        # Obtaining the member 'abs' of a type (line 131)
        abs_351996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 24), np_351995, 'abs')
        # Calling abs(args, kwargs) (line 131)
        abs_call_result_352007 = invoke(stypy.reporting.localization.Localization(__file__, 131, 24), abs_351996, *[result_sub_352005], **kwargs_352006)
        
        # Processing the call keyword arguments (line 131)
        kwargs_352008 = {}
        # Getting the type of 'np' (line 131)
        np_351993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 17), 'np', False)
        # Obtaining the member 'max' of a type (line 131)
        max_351994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 17), np_351993, 'max')
        # Calling max(args, kwargs) (line 131)
        max_call_result_352009 = invoke(stypy.reporting.localization.Localization(__file__, 131, 17), max_351994, *[abs_call_result_352007], **kwargs_352008)
        
        # Assigning a type to the variable 'abserr' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'abserr', max_call_result_352009)
        
        # Call to assert_(...): (line 132)
        # Processing the call arguments (line 132)
        
        # Getting the type of 'abserr' (line 132)
        abserr_352011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'abserr', False)
        float_352012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 25), 'float')
        # Applying the binary operator '<' (line 132)
        result_lt_352013 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 16), '<', abserr_352011, float_352012)
        
        # Processing the call keyword arguments (line 132)
        kwargs_352014 = {}
        # Getting the type of 'assert_' (line 132)
        assert__352010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 132)
        assert__call_result_352015 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), assert__352010, *[result_lt_352013], **kwargs_352014)
        
        
        # ################# End of 'test_logarithmic_freq_02(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_logarithmic_freq_02' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_352016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352016)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_logarithmic_freq_02'
        return stypy_return_type_352016


    @norecursion
    def test_logarithmic_freq_03(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_logarithmic_freq_03'
        module_type_store = module_type_store.open_function_context('test_logarithmic_freq_03', 134, 4, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_logarithmic_freq_03.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_logarithmic_freq_03.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_logarithmic_freq_03.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_logarithmic_freq_03.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_logarithmic_freq_03')
        TestChirp.test_logarithmic_freq_03.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_logarithmic_freq_03.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_logarithmic_freq_03.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_logarithmic_freq_03.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_logarithmic_freq_03.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_logarithmic_freq_03.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_logarithmic_freq_03.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_logarithmic_freq_03', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_logarithmic_freq_03', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_logarithmic_freq_03(...)' code ##################

        
        # Assigning a Str to a Name (line 135):
        
        # Assigning a Str to a Name (line 135):
        str_352017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 17), 'str', 'logarithmic')
        # Assigning a type to the variable 'method' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'method', str_352017)
        
        # Assigning a Num to a Name (line 136):
        
        # Assigning a Num to a Name (line 136):
        float_352018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 13), 'float')
        # Assigning a type to the variable 'f0' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'f0', float_352018)
        
        # Assigning a Num to a Name (line 137):
        
        # Assigning a Num to a Name (line 137):
        float_352019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 13), 'float')
        # Assigning a type to the variable 'f1' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'f1', float_352019)
        
        # Assigning a Num to a Name (line 138):
        
        # Assigning a Num to a Name (line 138):
        float_352020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 13), 'float')
        # Assigning a type to the variable 't1' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 't1', float_352020)
        
        # Assigning a Call to a Name (line 139):
        
        # Assigning a Call to a Name (line 139):
        
        # Call to linspace(...): (line 139)
        # Processing the call arguments (line 139)
        int_352023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 24), 'int')
        # Getting the type of 't1' (line 139)
        t1_352024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 27), 't1', False)
        int_352025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 31), 'int')
        # Processing the call keyword arguments (line 139)
        kwargs_352026 = {}
        # Getting the type of 'np' (line 139)
        np_352021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 139)
        linspace_352022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 12), np_352021, 'linspace')
        # Calling linspace(args, kwargs) (line 139)
        linspace_call_result_352027 = invoke(stypy.reporting.localization.Localization(__file__, 139, 12), linspace_352022, *[int_352023, t1_352024, int_352025], **kwargs_352026)
        
        # Assigning a type to the variable 't' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 't', linspace_call_result_352027)
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to _chirp_phase(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 't' (line 140)
        t_352030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 39), 't', False)
        # Getting the type of 'f0' (line 140)
        f0_352031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 42), 'f0', False)
        # Getting the type of 't1' (line 140)
        t1_352032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 46), 't1', False)
        # Getting the type of 'f1' (line 140)
        f1_352033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 50), 'f1', False)
        # Getting the type of 'method' (line 140)
        method_352034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 54), 'method', False)
        # Processing the call keyword arguments (line 140)
        kwargs_352035 = {}
        # Getting the type of 'waveforms' (line 140)
        waveforms_352028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'waveforms', False)
        # Obtaining the member '_chirp_phase' of a type (line 140)
        _chirp_phase_352029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 16), waveforms_352028, '_chirp_phase')
        # Calling _chirp_phase(args, kwargs) (line 140)
        _chirp_phase_call_result_352036 = invoke(stypy.reporting.localization.Localization(__file__, 140, 16), _chirp_phase_352029, *[t_352030, f0_352031, t1_352032, f1_352033, method_352034], **kwargs_352035)
        
        # Assigning a type to the variable 'phase' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'phase', _chirp_phase_call_result_352036)
        
        # Assigning a Call to a Tuple (line 141):
        
        # Assigning a Subscript to a Name (line 141):
        
        # Obtaining the type of the subscript
        int_352037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 8), 'int')
        
        # Call to compute_frequency(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 't' (line 141)
        t_352039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 34), 't', False)
        # Getting the type of 'phase' (line 141)
        phase_352040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 37), 'phase', False)
        # Processing the call keyword arguments (line 141)
        kwargs_352041 = {}
        # Getting the type of 'compute_frequency' (line 141)
        compute_frequency_352038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 141)
        compute_frequency_call_result_352042 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), compute_frequency_352038, *[t_352039, phase_352040], **kwargs_352041)
        
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___352043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), compute_frequency_call_result_352042, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_352044 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), getitem___352043, int_352037)
        
        # Assigning a type to the variable 'tuple_var_assignment_351426' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_351426', subscript_call_result_352044)
        
        # Assigning a Subscript to a Name (line 141):
        
        # Obtaining the type of the subscript
        int_352045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 8), 'int')
        
        # Call to compute_frequency(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 't' (line 141)
        t_352047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 34), 't', False)
        # Getting the type of 'phase' (line 141)
        phase_352048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 37), 'phase', False)
        # Processing the call keyword arguments (line 141)
        kwargs_352049 = {}
        # Getting the type of 'compute_frequency' (line 141)
        compute_frequency_352046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 141)
        compute_frequency_call_result_352050 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), compute_frequency_352046, *[t_352047, phase_352048], **kwargs_352049)
        
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___352051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), compute_frequency_call_result_352050, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_352052 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), getitem___352051, int_352045)
        
        # Assigning a type to the variable 'tuple_var_assignment_351427' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_351427', subscript_call_result_352052)
        
        # Assigning a Name to a Name (line 141):
        # Getting the type of 'tuple_var_assignment_351426' (line 141)
        tuple_var_assignment_351426_352053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_351426')
        # Assigning a type to the variable 'tf' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tf', tuple_var_assignment_351426_352053)
        
        # Assigning a Name to a Name (line 141):
        # Getting the type of 'tuple_var_assignment_351427' (line 141)
        tuple_var_assignment_351427_352054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'tuple_var_assignment_351427')
        # Assigning a type to the variable 'f' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'f', tuple_var_assignment_351427_352054)
        
        # Assigning a Call to a Name (line 142):
        
        # Assigning a Call to a Name (line 142):
        
        # Call to max(...): (line 142)
        # Processing the call arguments (line 142)
        
        # Call to abs(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'f' (line 142)
        f_352059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 31), 'f', False)
        
        # Call to chirp_geometric(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'tf' (line 142)
        tf_352061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 51), 'tf', False)
        # Getting the type of 'f0' (line 142)
        f0_352062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 55), 'f0', False)
        # Getting the type of 'f1' (line 142)
        f1_352063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 59), 'f1', False)
        # Getting the type of 't1' (line 142)
        t1_352064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 63), 't1', False)
        # Processing the call keyword arguments (line 142)
        kwargs_352065 = {}
        # Getting the type of 'chirp_geometric' (line 142)
        chirp_geometric_352060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 35), 'chirp_geometric', False)
        # Calling chirp_geometric(args, kwargs) (line 142)
        chirp_geometric_call_result_352066 = invoke(stypy.reporting.localization.Localization(__file__, 142, 35), chirp_geometric_352060, *[tf_352061, f0_352062, f1_352063, t1_352064], **kwargs_352065)
        
        # Applying the binary operator '-' (line 142)
        result_sub_352067 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 31), '-', f_352059, chirp_geometric_call_result_352066)
        
        # Processing the call keyword arguments (line 142)
        kwargs_352068 = {}
        # Getting the type of 'np' (line 142)
        np_352057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 24), 'np', False)
        # Obtaining the member 'abs' of a type (line 142)
        abs_352058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 24), np_352057, 'abs')
        # Calling abs(args, kwargs) (line 142)
        abs_call_result_352069 = invoke(stypy.reporting.localization.Localization(__file__, 142, 24), abs_352058, *[result_sub_352067], **kwargs_352068)
        
        # Processing the call keyword arguments (line 142)
        kwargs_352070 = {}
        # Getting the type of 'np' (line 142)
        np_352055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 17), 'np', False)
        # Obtaining the member 'max' of a type (line 142)
        max_352056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 17), np_352055, 'max')
        # Calling max(args, kwargs) (line 142)
        max_call_result_352071 = invoke(stypy.reporting.localization.Localization(__file__, 142, 17), max_352056, *[abs_call_result_352069], **kwargs_352070)
        
        # Assigning a type to the variable 'abserr' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'abserr', max_call_result_352071)
        
        # Call to assert_(...): (line 143)
        # Processing the call arguments (line 143)
        
        # Getting the type of 'abserr' (line 143)
        abserr_352073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'abserr', False)
        float_352074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 25), 'float')
        # Applying the binary operator '<' (line 143)
        result_lt_352075 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 16), '<', abserr_352073, float_352074)
        
        # Processing the call keyword arguments (line 143)
        kwargs_352076 = {}
        # Getting the type of 'assert_' (line 143)
        assert__352072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 143)
        assert__call_result_352077 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), assert__352072, *[result_lt_352075], **kwargs_352076)
        
        
        # ################# End of 'test_logarithmic_freq_03(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_logarithmic_freq_03' in the type store
        # Getting the type of 'stypy_return_type' (line 134)
        stypy_return_type_352078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352078)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_logarithmic_freq_03'
        return stypy_return_type_352078


    @norecursion
    def test_hyperbolic_at_zero(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_hyperbolic_at_zero'
        module_type_store = module_type_store.open_function_context('test_hyperbolic_at_zero', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_hyperbolic_at_zero.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_hyperbolic_at_zero.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_hyperbolic_at_zero.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_hyperbolic_at_zero.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_hyperbolic_at_zero')
        TestChirp.test_hyperbolic_at_zero.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_hyperbolic_at_zero.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_hyperbolic_at_zero.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_hyperbolic_at_zero.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_hyperbolic_at_zero.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_hyperbolic_at_zero.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_hyperbolic_at_zero.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_hyperbolic_at_zero', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_hyperbolic_at_zero', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_hyperbolic_at_zero(...)' code ##################

        
        # Assigning a Call to a Name (line 146):
        
        # Assigning a Call to a Name (line 146):
        
        # Call to chirp(...): (line 146)
        # Processing the call keyword arguments (line 146)
        int_352081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 30), 'int')
        keyword_352082 = int_352081
        float_352083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 36), 'float')
        keyword_352084 = float_352083
        float_352085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 45), 'float')
        keyword_352086 = float_352085
        float_352087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 53), 'float')
        keyword_352088 = float_352087
        str_352089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 65), 'str', 'hyperbolic')
        keyword_352090 = str_352089
        kwargs_352091 = {'f0': keyword_352084, 'f1': keyword_352086, 'method': keyword_352090, 't': keyword_352082, 't1': keyword_352088}
        # Getting the type of 'waveforms' (line 146)
        waveforms_352079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'waveforms', False)
        # Obtaining the member 'chirp' of a type (line 146)
        chirp_352080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 12), waveforms_352079, 'chirp')
        # Calling chirp(args, kwargs) (line 146)
        chirp_call_result_352092 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), chirp_352080, *[], **kwargs_352091)
        
        # Assigning a type to the variable 'w' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'w', chirp_call_result_352092)
        
        # Call to assert_almost_equal(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'w' (line 147)
        w_352094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 28), 'w', False)
        float_352095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 31), 'float')
        # Processing the call keyword arguments (line 147)
        kwargs_352096 = {}
        # Getting the type of 'assert_almost_equal' (line 147)
        assert_almost_equal_352093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 147)
        assert_almost_equal_call_result_352097 = invoke(stypy.reporting.localization.Localization(__file__, 147, 8), assert_almost_equal_352093, *[w_352094, float_352095], **kwargs_352096)
        
        
        # ################# End of 'test_hyperbolic_at_zero(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_hyperbolic_at_zero' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_352098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352098)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_hyperbolic_at_zero'
        return stypy_return_type_352098


    @norecursion
    def test_hyperbolic_freq_01(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_hyperbolic_freq_01'
        module_type_store = module_type_store.open_function_context('test_hyperbolic_freq_01', 149, 4, False)
        # Assigning a type to the variable 'self' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_hyperbolic_freq_01.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_hyperbolic_freq_01.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_hyperbolic_freq_01.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_hyperbolic_freq_01.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_hyperbolic_freq_01')
        TestChirp.test_hyperbolic_freq_01.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_hyperbolic_freq_01.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_hyperbolic_freq_01.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_hyperbolic_freq_01.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_hyperbolic_freq_01.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_hyperbolic_freq_01.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_hyperbolic_freq_01.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_hyperbolic_freq_01', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_hyperbolic_freq_01', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_hyperbolic_freq_01(...)' code ##################

        
        # Assigning a Str to a Name (line 150):
        
        # Assigning a Str to a Name (line 150):
        str_352099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 17), 'str', 'hyperbolic')
        # Assigning a type to the variable 'method' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'method', str_352099)
        
        # Assigning a Num to a Name (line 151):
        
        # Assigning a Num to a Name (line 151):
        float_352100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 13), 'float')
        # Assigning a type to the variable 't1' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 't1', float_352100)
        
        # Assigning a Call to a Name (line 152):
        
        # Assigning a Call to a Name (line 152):
        
        # Call to linspace(...): (line 152)
        # Processing the call arguments (line 152)
        int_352103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 24), 'int')
        # Getting the type of 't1' (line 152)
        t1_352104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 27), 't1', False)
        int_352105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 31), 'int')
        # Processing the call keyword arguments (line 152)
        kwargs_352106 = {}
        # Getting the type of 'np' (line 152)
        np_352101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 152)
        linspace_352102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 12), np_352101, 'linspace')
        # Calling linspace(args, kwargs) (line 152)
        linspace_call_result_352107 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), linspace_352102, *[int_352103, t1_352104, int_352105], **kwargs_352106)
        
        # Assigning a type to the variable 't' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 't', linspace_call_result_352107)
        
        # Assigning a List to a Name (line 154):
        
        # Assigning a List to a Name (line 154):
        
        # Obtaining an instance of the builtin type 'list' (line 154)
        list_352108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 154)
        # Adding element type (line 154)
        
        # Obtaining an instance of the builtin type 'list' (line 154)
        list_352109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 154)
        # Adding element type (line 154)
        float_352110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 17), list_352109, float_352110)
        # Adding element type (line 154)
        float_352111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 17), list_352109, float_352111)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 16), list_352108, list_352109)
        # Adding element type (line 154)
        
        # Obtaining an instance of the builtin type 'list' (line 155)
        list_352112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 155)
        # Adding element type (line 155)
        float_352113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 17), list_352112, float_352113)
        # Adding element type (line 155)
        float_352114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 17), list_352112, float_352114)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 16), list_352108, list_352112)
        # Adding element type (line 154)
        
        # Obtaining an instance of the builtin type 'list' (line 156)
        list_352115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 156)
        # Adding element type (line 156)
        float_352116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 17), list_352115, float_352116)
        # Adding element type (line 156)
        float_352117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 17), list_352115, float_352117)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 16), list_352108, list_352115)
        # Adding element type (line 154)
        
        # Obtaining an instance of the builtin type 'list' (line 157)
        list_352118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 157)
        # Adding element type (line 157)
        float_352119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 17), list_352118, float_352119)
        # Adding element type (line 157)
        float_352120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 24), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 17), list_352118, float_352120)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 16), list_352108, list_352118)
        
        # Assigning a type to the variable 'cases' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'cases', list_352108)
        
        # Getting the type of 'cases' (line 158)
        cases_352121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 22), 'cases')
        # Testing the type of a for loop iterable (line 158)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 158, 8), cases_352121)
        # Getting the type of the for loop variable (line 158)
        for_loop_var_352122 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 158, 8), cases_352121)
        # Assigning a type to the variable 'f0' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'f0', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 8), for_loop_var_352122))
        # Assigning a type to the variable 'f1' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'f1', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 8), for_loop_var_352122))
        # SSA begins for a for statement (line 158)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 159):
        
        # Assigning a Call to a Name (line 159):
        
        # Call to _chirp_phase(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 't' (line 159)
        t_352125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 43), 't', False)
        # Getting the type of 'f0' (line 159)
        f0_352126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 46), 'f0', False)
        # Getting the type of 't1' (line 159)
        t1_352127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 50), 't1', False)
        # Getting the type of 'f1' (line 159)
        f1_352128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 54), 'f1', False)
        # Getting the type of 'method' (line 159)
        method_352129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 58), 'method', False)
        # Processing the call keyword arguments (line 159)
        kwargs_352130 = {}
        # Getting the type of 'waveforms' (line 159)
        waveforms_352123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'waveforms', False)
        # Obtaining the member '_chirp_phase' of a type (line 159)
        _chirp_phase_352124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 20), waveforms_352123, '_chirp_phase')
        # Calling _chirp_phase(args, kwargs) (line 159)
        _chirp_phase_call_result_352131 = invoke(stypy.reporting.localization.Localization(__file__, 159, 20), _chirp_phase_352124, *[t_352125, f0_352126, t1_352127, f1_352128, method_352129], **kwargs_352130)
        
        # Assigning a type to the variable 'phase' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'phase', _chirp_phase_call_result_352131)
        
        # Assigning a Call to a Tuple (line 160):
        
        # Assigning a Subscript to a Name (line 160):
        
        # Obtaining the type of the subscript
        int_352132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 12), 'int')
        
        # Call to compute_frequency(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 't' (line 160)
        t_352134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 38), 't', False)
        # Getting the type of 'phase' (line 160)
        phase_352135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 41), 'phase', False)
        # Processing the call keyword arguments (line 160)
        kwargs_352136 = {}
        # Getting the type of 'compute_frequency' (line 160)
        compute_frequency_352133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 160)
        compute_frequency_call_result_352137 = invoke(stypy.reporting.localization.Localization(__file__, 160, 20), compute_frequency_352133, *[t_352134, phase_352135], **kwargs_352136)
        
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___352138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), compute_frequency_call_result_352137, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_352139 = invoke(stypy.reporting.localization.Localization(__file__, 160, 12), getitem___352138, int_352132)
        
        # Assigning a type to the variable 'tuple_var_assignment_351428' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'tuple_var_assignment_351428', subscript_call_result_352139)
        
        # Assigning a Subscript to a Name (line 160):
        
        # Obtaining the type of the subscript
        int_352140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 12), 'int')
        
        # Call to compute_frequency(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 't' (line 160)
        t_352142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 38), 't', False)
        # Getting the type of 'phase' (line 160)
        phase_352143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 41), 'phase', False)
        # Processing the call keyword arguments (line 160)
        kwargs_352144 = {}
        # Getting the type of 'compute_frequency' (line 160)
        compute_frequency_352141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 160)
        compute_frequency_call_result_352145 = invoke(stypy.reporting.localization.Localization(__file__, 160, 20), compute_frequency_352141, *[t_352142, phase_352143], **kwargs_352144)
        
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___352146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), compute_frequency_call_result_352145, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_352147 = invoke(stypy.reporting.localization.Localization(__file__, 160, 12), getitem___352146, int_352140)
        
        # Assigning a type to the variable 'tuple_var_assignment_351429' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'tuple_var_assignment_351429', subscript_call_result_352147)
        
        # Assigning a Name to a Name (line 160):
        # Getting the type of 'tuple_var_assignment_351428' (line 160)
        tuple_var_assignment_351428_352148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'tuple_var_assignment_351428')
        # Assigning a type to the variable 'tf' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'tf', tuple_var_assignment_351428_352148)
        
        # Assigning a Name to a Name (line 160):
        # Getting the type of 'tuple_var_assignment_351429' (line 160)
        tuple_var_assignment_351429_352149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'tuple_var_assignment_351429')
        # Assigning a type to the variable 'f' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'f', tuple_var_assignment_351429_352149)
        
        # Assigning a Call to a Name (line 161):
        
        # Assigning a Call to a Name (line 161):
        
        # Call to chirp_hyperbolic(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'tf' (line 161)
        tf_352151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 40), 'tf', False)
        # Getting the type of 'f0' (line 161)
        f0_352152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 44), 'f0', False)
        # Getting the type of 'f1' (line 161)
        f1_352153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 48), 'f1', False)
        # Getting the type of 't1' (line 161)
        t1_352154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 52), 't1', False)
        # Processing the call keyword arguments (line 161)
        kwargs_352155 = {}
        # Getting the type of 'chirp_hyperbolic' (line 161)
        chirp_hyperbolic_352150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 23), 'chirp_hyperbolic', False)
        # Calling chirp_hyperbolic(args, kwargs) (line 161)
        chirp_hyperbolic_call_result_352156 = invoke(stypy.reporting.localization.Localization(__file__, 161, 23), chirp_hyperbolic_352150, *[tf_352151, f0_352152, f1_352153, t1_352154], **kwargs_352155)
        
        # Assigning a type to the variable 'expected' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'expected', chirp_hyperbolic_call_result_352156)
        
        # Call to assert_allclose(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'f' (line 162)
        f_352158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 28), 'f', False)
        # Getting the type of 'expected' (line 162)
        expected_352159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 31), 'expected', False)
        # Processing the call keyword arguments (line 162)
        kwargs_352160 = {}
        # Getting the type of 'assert_allclose' (line 162)
        assert_allclose_352157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 162)
        assert_allclose_call_result_352161 = invoke(stypy.reporting.localization.Localization(__file__, 162, 12), assert_allclose_352157, *[f_352158, expected_352159], **kwargs_352160)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_hyperbolic_freq_01(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_hyperbolic_freq_01' in the type store
        # Getting the type of 'stypy_return_type' (line 149)
        stypy_return_type_352162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352162)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_hyperbolic_freq_01'
        return stypy_return_type_352162


    @norecursion
    def test_hyperbolic_zero_freq(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_hyperbolic_zero_freq'
        module_type_store = module_type_store.open_function_context('test_hyperbolic_zero_freq', 164, 4, False)
        # Assigning a type to the variable 'self' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_hyperbolic_zero_freq.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_hyperbolic_zero_freq.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_hyperbolic_zero_freq.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_hyperbolic_zero_freq.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_hyperbolic_zero_freq')
        TestChirp.test_hyperbolic_zero_freq.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_hyperbolic_zero_freq.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_hyperbolic_zero_freq.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_hyperbolic_zero_freq.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_hyperbolic_zero_freq.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_hyperbolic_zero_freq.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_hyperbolic_zero_freq.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_hyperbolic_zero_freq', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_hyperbolic_zero_freq', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_hyperbolic_zero_freq(...)' code ##################

        
        # Assigning a Str to a Name (line 166):
        
        # Assigning a Str to a Name (line 166):
        str_352163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 17), 'str', 'hyperbolic')
        # Assigning a type to the variable 'method' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'method', str_352163)
        
        # Assigning a Num to a Name (line 167):
        
        # Assigning a Num to a Name (line 167):
        float_352164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 13), 'float')
        # Assigning a type to the variable 't1' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 't1', float_352164)
        
        # Assigning a Call to a Name (line 168):
        
        # Assigning a Call to a Name (line 168):
        
        # Call to linspace(...): (line 168)
        # Processing the call arguments (line 168)
        int_352167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 24), 'int')
        # Getting the type of 't1' (line 168)
        t1_352168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 't1', False)
        int_352169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 31), 'int')
        # Processing the call keyword arguments (line 168)
        kwargs_352170 = {}
        # Getting the type of 'np' (line 168)
        np_352165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 168)
        linspace_352166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 12), np_352165, 'linspace')
        # Calling linspace(args, kwargs) (line 168)
        linspace_call_result_352171 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), linspace_352166, *[int_352167, t1_352168, int_352169], **kwargs_352170)
        
        # Assigning a type to the variable 't' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 't', linspace_call_result_352171)
        
        # Call to assert_raises(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'ValueError' (line 169)
        ValueError_352173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 22), 'ValueError', False)
        # Getting the type of 'waveforms' (line 169)
        waveforms_352174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'waveforms', False)
        # Obtaining the member 'chirp' of a type (line 169)
        chirp_352175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 34), waveforms_352174, 'chirp')
        # Getting the type of 't' (line 169)
        t_352176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 51), 't', False)
        int_352177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 54), 'int')
        # Getting the type of 't1' (line 169)
        t1_352178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 57), 't1', False)
        int_352179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 61), 'int')
        # Getting the type of 'method' (line 169)
        method_352180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 64), 'method', False)
        # Processing the call keyword arguments (line 169)
        kwargs_352181 = {}
        # Getting the type of 'assert_raises' (line 169)
        assert_raises_352172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 169)
        assert_raises_call_result_352182 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), assert_raises_352172, *[ValueError_352173, chirp_352175, t_352176, int_352177, t1_352178, int_352179, method_352180], **kwargs_352181)
        
        
        # Call to assert_raises(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'ValueError' (line 170)
        ValueError_352184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 22), 'ValueError', False)
        # Getting the type of 'waveforms' (line 170)
        waveforms_352185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 34), 'waveforms', False)
        # Obtaining the member 'chirp' of a type (line 170)
        chirp_352186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 34), waveforms_352185, 'chirp')
        # Getting the type of 't' (line 170)
        t_352187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 51), 't', False)
        int_352188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 54), 'int')
        # Getting the type of 't1' (line 170)
        t1_352189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 57), 't1', False)
        int_352190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 61), 'int')
        # Getting the type of 'method' (line 170)
        method_352191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 64), 'method', False)
        # Processing the call keyword arguments (line 170)
        kwargs_352192 = {}
        # Getting the type of 'assert_raises' (line 170)
        assert_raises_352183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 170)
        assert_raises_call_result_352193 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), assert_raises_352183, *[ValueError_352184, chirp_352186, t_352187, int_352188, t1_352189, int_352190, method_352191], **kwargs_352192)
        
        
        # ################# End of 'test_hyperbolic_zero_freq(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_hyperbolic_zero_freq' in the type store
        # Getting the type of 'stypy_return_type' (line 164)
        stypy_return_type_352194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352194)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_hyperbolic_zero_freq'
        return stypy_return_type_352194


    @norecursion
    def test_unknown_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_unknown_method'
        module_type_store = module_type_store.open_function_context('test_unknown_method', 172, 4, False)
        # Assigning a type to the variable 'self' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_unknown_method.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_unknown_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_unknown_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_unknown_method.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_unknown_method')
        TestChirp.test_unknown_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_unknown_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_unknown_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_unknown_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_unknown_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_unknown_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_unknown_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_unknown_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_unknown_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_unknown_method(...)' code ##################

        
        # Assigning a Str to a Name (line 173):
        
        # Assigning a Str to a Name (line 173):
        str_352195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 17), 'str', 'foo')
        # Assigning a type to the variable 'method' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'method', str_352195)
        
        # Assigning a Num to a Name (line 174):
        
        # Assigning a Num to a Name (line 174):
        float_352196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 13), 'float')
        # Assigning a type to the variable 'f0' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'f0', float_352196)
        
        # Assigning a Num to a Name (line 175):
        
        # Assigning a Num to a Name (line 175):
        float_352197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 13), 'float')
        # Assigning a type to the variable 'f1' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'f1', float_352197)
        
        # Assigning a Num to a Name (line 176):
        
        # Assigning a Num to a Name (line 176):
        float_352198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 13), 'float')
        # Assigning a type to the variable 't1' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 't1', float_352198)
        
        # Assigning a Call to a Name (line 177):
        
        # Assigning a Call to a Name (line 177):
        
        # Call to linspace(...): (line 177)
        # Processing the call arguments (line 177)
        int_352201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 24), 'int')
        # Getting the type of 't1' (line 177)
        t1_352202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 27), 't1', False)
        int_352203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 31), 'int')
        # Processing the call keyword arguments (line 177)
        kwargs_352204 = {}
        # Getting the type of 'np' (line 177)
        np_352199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 177)
        linspace_352200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 12), np_352199, 'linspace')
        # Calling linspace(args, kwargs) (line 177)
        linspace_call_result_352205 = invoke(stypy.reporting.localization.Localization(__file__, 177, 12), linspace_352200, *[int_352201, t1_352202, int_352203], **kwargs_352204)
        
        # Assigning a type to the variable 't' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 't', linspace_call_result_352205)
        
        # Call to assert_raises(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'ValueError' (line 178)
        ValueError_352207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 22), 'ValueError', False)
        # Getting the type of 'waveforms' (line 178)
        waveforms_352208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 34), 'waveforms', False)
        # Obtaining the member 'chirp' of a type (line 178)
        chirp_352209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 34), waveforms_352208, 'chirp')
        # Getting the type of 't' (line 178)
        t_352210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 51), 't', False)
        # Getting the type of 'f0' (line 178)
        f0_352211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 54), 'f0', False)
        # Getting the type of 't1' (line 178)
        t1_352212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 58), 't1', False)
        # Getting the type of 'f1' (line 178)
        f1_352213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 62), 'f1', False)
        # Getting the type of 'method' (line 178)
        method_352214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 66), 'method', False)
        # Processing the call keyword arguments (line 178)
        kwargs_352215 = {}
        # Getting the type of 'assert_raises' (line 178)
        assert_raises_352206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 178)
        assert_raises_call_result_352216 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), assert_raises_352206, *[ValueError_352207, chirp_352209, t_352210, f0_352211, t1_352212, f1_352213, method_352214], **kwargs_352215)
        
        
        # ################# End of 'test_unknown_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_unknown_method' in the type store
        # Getting the type of 'stypy_return_type' (line 172)
        stypy_return_type_352217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352217)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_unknown_method'
        return stypy_return_type_352217


    @norecursion
    def test_integer_t1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_integer_t1'
        module_type_store = module_type_store.open_function_context('test_integer_t1', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_integer_t1.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_integer_t1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_integer_t1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_integer_t1.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_integer_t1')
        TestChirp.test_integer_t1.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_integer_t1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_integer_t1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_integer_t1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_integer_t1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_integer_t1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_integer_t1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_integer_t1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_integer_t1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_integer_t1(...)' code ##################

        
        # Assigning a Num to a Name (line 181):
        
        # Assigning a Num to a Name (line 181):
        float_352218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 13), 'float')
        # Assigning a type to the variable 'f0' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'f0', float_352218)
        
        # Assigning a Num to a Name (line 182):
        
        # Assigning a Num to a Name (line 182):
        float_352219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 13), 'float')
        # Assigning a type to the variable 'f1' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'f1', float_352219)
        
        # Assigning a Call to a Name (line 183):
        
        # Assigning a Call to a Name (line 183):
        
        # Call to linspace(...): (line 183)
        # Processing the call arguments (line 183)
        int_352222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 24), 'int')
        int_352223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 28), 'int')
        int_352224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 31), 'int')
        # Processing the call keyword arguments (line 183)
        kwargs_352225 = {}
        # Getting the type of 'np' (line 183)
        np_352220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 183)
        linspace_352221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), np_352220, 'linspace')
        # Calling linspace(args, kwargs) (line 183)
        linspace_call_result_352226 = invoke(stypy.reporting.localization.Localization(__file__, 183, 12), linspace_352221, *[int_352222, int_352223, int_352224], **kwargs_352225)
        
        # Assigning a type to the variable 't' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 't', linspace_call_result_352226)
        
        # Assigning a Num to a Name (line 184):
        
        # Assigning a Num to a Name (line 184):
        float_352227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 13), 'float')
        # Assigning a type to the variable 't1' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 't1', float_352227)
        
        # Assigning a Call to a Name (line 185):
        
        # Assigning a Call to a Name (line 185):
        
        # Call to chirp(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 't' (line 185)
        t_352230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 39), 't', False)
        # Getting the type of 'f0' (line 185)
        f0_352231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 42), 'f0', False)
        # Getting the type of 't1' (line 185)
        t1_352232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 46), 't1', False)
        # Getting the type of 'f1' (line 185)
        f1_352233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 50), 'f1', False)
        # Processing the call keyword arguments (line 185)
        kwargs_352234 = {}
        # Getting the type of 'waveforms' (line 185)
        waveforms_352228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 23), 'waveforms', False)
        # Obtaining the member 'chirp' of a type (line 185)
        chirp_352229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 23), waveforms_352228, 'chirp')
        # Calling chirp(args, kwargs) (line 185)
        chirp_call_result_352235 = invoke(stypy.reporting.localization.Localization(__file__, 185, 23), chirp_352229, *[t_352230, f0_352231, t1_352232, f1_352233], **kwargs_352234)
        
        # Assigning a type to the variable 'float_result' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'float_result', chirp_call_result_352235)
        
        # Assigning a Num to a Name (line 186):
        
        # Assigning a Num to a Name (line 186):
        int_352236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 13), 'int')
        # Assigning a type to the variable 't1' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 't1', int_352236)
        
        # Assigning a Call to a Name (line 187):
        
        # Assigning a Call to a Name (line 187):
        
        # Call to chirp(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 't' (line 187)
        t_352239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 37), 't', False)
        # Getting the type of 'f0' (line 187)
        f0_352240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 40), 'f0', False)
        # Getting the type of 't1' (line 187)
        t1_352241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 44), 't1', False)
        # Getting the type of 'f1' (line 187)
        f1_352242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 48), 'f1', False)
        # Processing the call keyword arguments (line 187)
        kwargs_352243 = {}
        # Getting the type of 'waveforms' (line 187)
        waveforms_352237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 21), 'waveforms', False)
        # Obtaining the member 'chirp' of a type (line 187)
        chirp_352238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 21), waveforms_352237, 'chirp')
        # Calling chirp(args, kwargs) (line 187)
        chirp_call_result_352244 = invoke(stypy.reporting.localization.Localization(__file__, 187, 21), chirp_352238, *[t_352239, f0_352240, t1_352241, f1_352242], **kwargs_352243)
        
        # Assigning a type to the variable 'int_result' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'int_result', chirp_call_result_352244)
        
        # Assigning a Str to a Name (line 188):
        
        # Assigning a Str to a Name (line 188):
        str_352245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 18), 'str', "Integer input 't1=3' gives wrong result")
        # Assigning a type to the variable 'err_msg' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'err_msg', str_352245)
        
        # Call to assert_equal(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'int_result' (line 189)
        int_result_352247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 21), 'int_result', False)
        # Getting the type of 'float_result' (line 189)
        float_result_352248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 33), 'float_result', False)
        # Processing the call keyword arguments (line 189)
        # Getting the type of 'err_msg' (line 189)
        err_msg_352249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 55), 'err_msg', False)
        keyword_352250 = err_msg_352249
        kwargs_352251 = {'err_msg': keyword_352250}
        # Getting the type of 'assert_equal' (line 189)
        assert_equal_352246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 189)
        assert_equal_call_result_352252 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), assert_equal_352246, *[int_result_352247, float_result_352248], **kwargs_352251)
        
        
        # ################# End of 'test_integer_t1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_integer_t1' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_352253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352253)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_integer_t1'
        return stypy_return_type_352253


    @norecursion
    def test_integer_f0(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_integer_f0'
        module_type_store = module_type_store.open_function_context('test_integer_f0', 191, 4, False)
        # Assigning a type to the variable 'self' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_integer_f0.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_integer_f0.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_integer_f0.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_integer_f0.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_integer_f0')
        TestChirp.test_integer_f0.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_integer_f0.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_integer_f0.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_integer_f0.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_integer_f0.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_integer_f0.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_integer_f0.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_integer_f0', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_integer_f0', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_integer_f0(...)' code ##################

        
        # Assigning a Num to a Name (line 192):
        
        # Assigning a Num to a Name (line 192):
        float_352254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 13), 'float')
        # Assigning a type to the variable 'f1' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'f1', float_352254)
        
        # Assigning a Num to a Name (line 193):
        
        # Assigning a Num to a Name (line 193):
        float_352255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 13), 'float')
        # Assigning a type to the variable 't1' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 't1', float_352255)
        
        # Assigning a Call to a Name (line 194):
        
        # Assigning a Call to a Name (line 194):
        
        # Call to linspace(...): (line 194)
        # Processing the call arguments (line 194)
        int_352258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 24), 'int')
        int_352259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 28), 'int')
        int_352260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 31), 'int')
        # Processing the call keyword arguments (line 194)
        kwargs_352261 = {}
        # Getting the type of 'np' (line 194)
        np_352256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 194)
        linspace_352257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), np_352256, 'linspace')
        # Calling linspace(args, kwargs) (line 194)
        linspace_call_result_352262 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), linspace_352257, *[int_352258, int_352259, int_352260], **kwargs_352261)
        
        # Assigning a type to the variable 't' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 't', linspace_call_result_352262)
        
        # Assigning a Num to a Name (line 195):
        
        # Assigning a Num to a Name (line 195):
        float_352263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 13), 'float')
        # Assigning a type to the variable 'f0' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'f0', float_352263)
        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Call to chirp(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 't' (line 196)
        t_352266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 39), 't', False)
        # Getting the type of 'f0' (line 196)
        f0_352267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 42), 'f0', False)
        # Getting the type of 't1' (line 196)
        t1_352268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 46), 't1', False)
        # Getting the type of 'f1' (line 196)
        f1_352269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 50), 'f1', False)
        # Processing the call keyword arguments (line 196)
        kwargs_352270 = {}
        # Getting the type of 'waveforms' (line 196)
        waveforms_352264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 23), 'waveforms', False)
        # Obtaining the member 'chirp' of a type (line 196)
        chirp_352265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 23), waveforms_352264, 'chirp')
        # Calling chirp(args, kwargs) (line 196)
        chirp_call_result_352271 = invoke(stypy.reporting.localization.Localization(__file__, 196, 23), chirp_352265, *[t_352266, f0_352267, t1_352268, f1_352269], **kwargs_352270)
        
        # Assigning a type to the variable 'float_result' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'float_result', chirp_call_result_352271)
        
        # Assigning a Num to a Name (line 197):
        
        # Assigning a Num to a Name (line 197):
        int_352272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 13), 'int')
        # Assigning a type to the variable 'f0' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'f0', int_352272)
        
        # Assigning a Call to a Name (line 198):
        
        # Assigning a Call to a Name (line 198):
        
        # Call to chirp(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 't' (line 198)
        t_352275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 37), 't', False)
        # Getting the type of 'f0' (line 198)
        f0_352276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 40), 'f0', False)
        # Getting the type of 't1' (line 198)
        t1_352277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 44), 't1', False)
        # Getting the type of 'f1' (line 198)
        f1_352278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 48), 'f1', False)
        # Processing the call keyword arguments (line 198)
        kwargs_352279 = {}
        # Getting the type of 'waveforms' (line 198)
        waveforms_352273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 21), 'waveforms', False)
        # Obtaining the member 'chirp' of a type (line 198)
        chirp_352274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 21), waveforms_352273, 'chirp')
        # Calling chirp(args, kwargs) (line 198)
        chirp_call_result_352280 = invoke(stypy.reporting.localization.Localization(__file__, 198, 21), chirp_352274, *[t_352275, f0_352276, t1_352277, f1_352278], **kwargs_352279)
        
        # Assigning a type to the variable 'int_result' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'int_result', chirp_call_result_352280)
        
        # Assigning a Str to a Name (line 199):
        
        # Assigning a Str to a Name (line 199):
        str_352281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 18), 'str', "Integer input 'f0=10' gives wrong result")
        # Assigning a type to the variable 'err_msg' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'err_msg', str_352281)
        
        # Call to assert_equal(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'int_result' (line 200)
        int_result_352283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 'int_result', False)
        # Getting the type of 'float_result' (line 200)
        float_result_352284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 33), 'float_result', False)
        # Processing the call keyword arguments (line 200)
        # Getting the type of 'err_msg' (line 200)
        err_msg_352285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 55), 'err_msg', False)
        keyword_352286 = err_msg_352285
        kwargs_352287 = {'err_msg': keyword_352286}
        # Getting the type of 'assert_equal' (line 200)
        assert_equal_352282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 200)
        assert_equal_call_result_352288 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), assert_equal_352282, *[int_result_352283, float_result_352284], **kwargs_352287)
        
        
        # ################# End of 'test_integer_f0(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_integer_f0' in the type store
        # Getting the type of 'stypy_return_type' (line 191)
        stypy_return_type_352289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352289)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_integer_f0'
        return stypy_return_type_352289


    @norecursion
    def test_integer_f1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_integer_f1'
        module_type_store = module_type_store.open_function_context('test_integer_f1', 202, 4, False)
        # Assigning a type to the variable 'self' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_integer_f1.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_integer_f1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_integer_f1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_integer_f1.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_integer_f1')
        TestChirp.test_integer_f1.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_integer_f1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_integer_f1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_integer_f1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_integer_f1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_integer_f1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_integer_f1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_integer_f1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_integer_f1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_integer_f1(...)' code ##################

        
        # Assigning a Num to a Name (line 203):
        
        # Assigning a Num to a Name (line 203):
        float_352290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 13), 'float')
        # Assigning a type to the variable 'f0' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'f0', float_352290)
        
        # Assigning a Num to a Name (line 204):
        
        # Assigning a Num to a Name (line 204):
        float_352291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 13), 'float')
        # Assigning a type to the variable 't1' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 't1', float_352291)
        
        # Assigning a Call to a Name (line 205):
        
        # Assigning a Call to a Name (line 205):
        
        # Call to linspace(...): (line 205)
        # Processing the call arguments (line 205)
        int_352294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 24), 'int')
        int_352295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 28), 'int')
        int_352296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 31), 'int')
        # Processing the call keyword arguments (line 205)
        kwargs_352297 = {}
        # Getting the type of 'np' (line 205)
        np_352292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 205)
        linspace_352293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 12), np_352292, 'linspace')
        # Calling linspace(args, kwargs) (line 205)
        linspace_call_result_352298 = invoke(stypy.reporting.localization.Localization(__file__, 205, 12), linspace_352293, *[int_352294, int_352295, int_352296], **kwargs_352297)
        
        # Assigning a type to the variable 't' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 't', linspace_call_result_352298)
        
        # Assigning a Num to a Name (line 206):
        
        # Assigning a Num to a Name (line 206):
        float_352299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 13), 'float')
        # Assigning a type to the variable 'f1' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'f1', float_352299)
        
        # Assigning a Call to a Name (line 207):
        
        # Assigning a Call to a Name (line 207):
        
        # Call to chirp(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 't' (line 207)
        t_352302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 39), 't', False)
        # Getting the type of 'f0' (line 207)
        f0_352303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 42), 'f0', False)
        # Getting the type of 't1' (line 207)
        t1_352304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 46), 't1', False)
        # Getting the type of 'f1' (line 207)
        f1_352305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 50), 'f1', False)
        # Processing the call keyword arguments (line 207)
        kwargs_352306 = {}
        # Getting the type of 'waveforms' (line 207)
        waveforms_352300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 23), 'waveforms', False)
        # Obtaining the member 'chirp' of a type (line 207)
        chirp_352301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 23), waveforms_352300, 'chirp')
        # Calling chirp(args, kwargs) (line 207)
        chirp_call_result_352307 = invoke(stypy.reporting.localization.Localization(__file__, 207, 23), chirp_352301, *[t_352302, f0_352303, t1_352304, f1_352305], **kwargs_352306)
        
        # Assigning a type to the variable 'float_result' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'float_result', chirp_call_result_352307)
        
        # Assigning a Num to a Name (line 208):
        
        # Assigning a Num to a Name (line 208):
        int_352308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 13), 'int')
        # Assigning a type to the variable 'f1' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'f1', int_352308)
        
        # Assigning a Call to a Name (line 209):
        
        # Assigning a Call to a Name (line 209):
        
        # Call to chirp(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 't' (line 209)
        t_352311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 37), 't', False)
        # Getting the type of 'f0' (line 209)
        f0_352312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 40), 'f0', False)
        # Getting the type of 't1' (line 209)
        t1_352313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 44), 't1', False)
        # Getting the type of 'f1' (line 209)
        f1_352314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 48), 'f1', False)
        # Processing the call keyword arguments (line 209)
        kwargs_352315 = {}
        # Getting the type of 'waveforms' (line 209)
        waveforms_352309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 21), 'waveforms', False)
        # Obtaining the member 'chirp' of a type (line 209)
        chirp_352310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 21), waveforms_352309, 'chirp')
        # Calling chirp(args, kwargs) (line 209)
        chirp_call_result_352316 = invoke(stypy.reporting.localization.Localization(__file__, 209, 21), chirp_352310, *[t_352311, f0_352312, t1_352313, f1_352314], **kwargs_352315)
        
        # Assigning a type to the variable 'int_result' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'int_result', chirp_call_result_352316)
        
        # Assigning a Str to a Name (line 210):
        
        # Assigning a Str to a Name (line 210):
        str_352317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 18), 'str', "Integer input 'f1=20' gives wrong result")
        # Assigning a type to the variable 'err_msg' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'err_msg', str_352317)
        
        # Call to assert_equal(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'int_result' (line 211)
        int_result_352319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 21), 'int_result', False)
        # Getting the type of 'float_result' (line 211)
        float_result_352320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 33), 'float_result', False)
        # Processing the call keyword arguments (line 211)
        # Getting the type of 'err_msg' (line 211)
        err_msg_352321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 55), 'err_msg', False)
        keyword_352322 = err_msg_352321
        kwargs_352323 = {'err_msg': keyword_352322}
        # Getting the type of 'assert_equal' (line 211)
        assert_equal_352318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 211)
        assert_equal_call_result_352324 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), assert_equal_352318, *[int_result_352319, float_result_352320], **kwargs_352323)
        
        
        # ################# End of 'test_integer_f1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_integer_f1' in the type store
        # Getting the type of 'stypy_return_type' (line 202)
        stypy_return_type_352325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352325)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_integer_f1'
        return stypy_return_type_352325


    @norecursion
    def test_integer_all(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_integer_all'
        module_type_store = module_type_store.open_function_context('test_integer_all', 213, 4, False)
        # Assigning a type to the variable 'self' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestChirp.test_integer_all.__dict__.__setitem__('stypy_localization', localization)
        TestChirp.test_integer_all.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestChirp.test_integer_all.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestChirp.test_integer_all.__dict__.__setitem__('stypy_function_name', 'TestChirp.test_integer_all')
        TestChirp.test_integer_all.__dict__.__setitem__('stypy_param_names_list', [])
        TestChirp.test_integer_all.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestChirp.test_integer_all.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestChirp.test_integer_all.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestChirp.test_integer_all.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestChirp.test_integer_all.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestChirp.test_integer_all.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.test_integer_all', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_integer_all', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_integer_all(...)' code ##################

        
        # Assigning a Num to a Name (line 214):
        
        # Assigning a Num to a Name (line 214):
        int_352326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 13), 'int')
        # Assigning a type to the variable 'f0' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'f0', int_352326)
        
        # Assigning a Num to a Name (line 215):
        
        # Assigning a Num to a Name (line 215):
        int_352327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 13), 'int')
        # Assigning a type to the variable 't1' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 't1', int_352327)
        
        # Assigning a Num to a Name (line 216):
        
        # Assigning a Num to a Name (line 216):
        int_352328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 13), 'int')
        # Assigning a type to the variable 'f1' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'f1', int_352328)
        
        # Assigning a Call to a Name (line 217):
        
        # Assigning a Call to a Name (line 217):
        
        # Call to linspace(...): (line 217)
        # Processing the call arguments (line 217)
        int_352331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 24), 'int')
        int_352332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 28), 'int')
        int_352333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 31), 'int')
        # Processing the call keyword arguments (line 217)
        kwargs_352334 = {}
        # Getting the type of 'np' (line 217)
        np_352329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 217)
        linspace_352330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 12), np_352329, 'linspace')
        # Calling linspace(args, kwargs) (line 217)
        linspace_call_result_352335 = invoke(stypy.reporting.localization.Localization(__file__, 217, 12), linspace_352330, *[int_352331, int_352332, int_352333], **kwargs_352334)
        
        # Assigning a type to the variable 't' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 't', linspace_call_result_352335)
        
        # Assigning a Call to a Name (line 218):
        
        # Assigning a Call to a Name (line 218):
        
        # Call to chirp(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 't' (line 218)
        t_352338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 39), 't', False)
        
        # Call to float(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'f0' (line 218)
        f0_352340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 48), 'f0', False)
        # Processing the call keyword arguments (line 218)
        kwargs_352341 = {}
        # Getting the type of 'float' (line 218)
        float_352339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 42), 'float', False)
        # Calling float(args, kwargs) (line 218)
        float_call_result_352342 = invoke(stypy.reporting.localization.Localization(__file__, 218, 42), float_352339, *[f0_352340], **kwargs_352341)
        
        
        # Call to float(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 't1' (line 218)
        t1_352344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 59), 't1', False)
        # Processing the call keyword arguments (line 218)
        kwargs_352345 = {}
        # Getting the type of 'float' (line 218)
        float_352343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 53), 'float', False)
        # Calling float(args, kwargs) (line 218)
        float_call_result_352346 = invoke(stypy.reporting.localization.Localization(__file__, 218, 53), float_352343, *[t1_352344], **kwargs_352345)
        
        
        # Call to float(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'f1' (line 218)
        f1_352348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 70), 'f1', False)
        # Processing the call keyword arguments (line 218)
        kwargs_352349 = {}
        # Getting the type of 'float' (line 218)
        float_352347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 64), 'float', False)
        # Calling float(args, kwargs) (line 218)
        float_call_result_352350 = invoke(stypy.reporting.localization.Localization(__file__, 218, 64), float_352347, *[f1_352348], **kwargs_352349)
        
        # Processing the call keyword arguments (line 218)
        kwargs_352351 = {}
        # Getting the type of 'waveforms' (line 218)
        waveforms_352336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 23), 'waveforms', False)
        # Obtaining the member 'chirp' of a type (line 218)
        chirp_352337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 23), waveforms_352336, 'chirp')
        # Calling chirp(args, kwargs) (line 218)
        chirp_call_result_352352 = invoke(stypy.reporting.localization.Localization(__file__, 218, 23), chirp_352337, *[t_352338, float_call_result_352342, float_call_result_352346, float_call_result_352350], **kwargs_352351)
        
        # Assigning a type to the variable 'float_result' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'float_result', chirp_call_result_352352)
        
        # Assigning a Call to a Name (line 219):
        
        # Assigning a Call to a Name (line 219):
        
        # Call to chirp(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 't' (line 219)
        t_352355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 37), 't', False)
        # Getting the type of 'f0' (line 219)
        f0_352356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 40), 'f0', False)
        # Getting the type of 't1' (line 219)
        t1_352357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 44), 't1', False)
        # Getting the type of 'f1' (line 219)
        f1_352358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 48), 'f1', False)
        # Processing the call keyword arguments (line 219)
        kwargs_352359 = {}
        # Getting the type of 'waveforms' (line 219)
        waveforms_352353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'waveforms', False)
        # Obtaining the member 'chirp' of a type (line 219)
        chirp_352354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 21), waveforms_352353, 'chirp')
        # Calling chirp(args, kwargs) (line 219)
        chirp_call_result_352360 = invoke(stypy.reporting.localization.Localization(__file__, 219, 21), chirp_352354, *[t_352355, f0_352356, t1_352357, f1_352358], **kwargs_352359)
        
        # Assigning a type to the variable 'int_result' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'int_result', chirp_call_result_352360)
        
        # Assigning a Str to a Name (line 220):
        
        # Assigning a Str to a Name (line 220):
        str_352361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 18), 'str', "Integer input 'f0=10, t1=3, f1=20' gives wrong result")
        # Assigning a type to the variable 'err_msg' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'err_msg', str_352361)
        
        # Call to assert_equal(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'int_result' (line 221)
        int_result_352363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 21), 'int_result', False)
        # Getting the type of 'float_result' (line 221)
        float_result_352364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 33), 'float_result', False)
        # Processing the call keyword arguments (line 221)
        # Getting the type of 'err_msg' (line 221)
        err_msg_352365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 55), 'err_msg', False)
        keyword_352366 = err_msg_352365
        kwargs_352367 = {'err_msg': keyword_352366}
        # Getting the type of 'assert_equal' (line 221)
        assert_equal_352362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 221)
        assert_equal_call_result_352368 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), assert_equal_352362, *[int_result_352363, float_result_352364], **kwargs_352367)
        
        
        # ################# End of 'test_integer_all(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_integer_all' in the type store
        # Getting the type of 'stypy_return_type' (line 213)
        stypy_return_type_352369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352369)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_integer_all'
        return stypy_return_type_352369


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 49, 0, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestChirp.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'TestChirp' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'TestChirp', TestChirp)
# Declaration of the 'TestSweepPoly' class

class TestSweepPoly(object, ):

    @norecursion
    def test_sweep_poly_quad1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sweep_poly_quad1'
        module_type_store = module_type_store.open_function_context('test_sweep_poly_quad1', 226, 4, False)
        # Assigning a type to the variable 'self' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSweepPoly.test_sweep_poly_quad1.__dict__.__setitem__('stypy_localization', localization)
        TestSweepPoly.test_sweep_poly_quad1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSweepPoly.test_sweep_poly_quad1.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSweepPoly.test_sweep_poly_quad1.__dict__.__setitem__('stypy_function_name', 'TestSweepPoly.test_sweep_poly_quad1')
        TestSweepPoly.test_sweep_poly_quad1.__dict__.__setitem__('stypy_param_names_list', [])
        TestSweepPoly.test_sweep_poly_quad1.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSweepPoly.test_sweep_poly_quad1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSweepPoly.test_sweep_poly_quad1.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSweepPoly.test_sweep_poly_quad1.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSweepPoly.test_sweep_poly_quad1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSweepPoly.test_sweep_poly_quad1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSweepPoly.test_sweep_poly_quad1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sweep_poly_quad1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sweep_poly_quad1(...)' code ##################

        
        # Assigning a Call to a Name (line 227):
        
        # Assigning a Call to a Name (line 227):
        
        # Call to poly1d(...): (line 227)
        # Processing the call arguments (line 227)
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_352372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        float_352373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 22), list_352372, float_352373)
        # Adding element type (line 227)
        float_352374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 22), list_352372, float_352374)
        # Adding element type (line 227)
        float_352375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 22), list_352372, float_352375)
        
        # Processing the call keyword arguments (line 227)
        kwargs_352376 = {}
        # Getting the type of 'np' (line 227)
        np_352370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'np', False)
        # Obtaining the member 'poly1d' of a type (line 227)
        poly1d_352371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 12), np_352370, 'poly1d')
        # Calling poly1d(args, kwargs) (line 227)
        poly1d_call_result_352377 = invoke(stypy.reporting.localization.Localization(__file__, 227, 12), poly1d_352371, *[list_352372], **kwargs_352376)
        
        # Assigning a type to the variable 'p' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'p', poly1d_call_result_352377)
        
        # Assigning a Call to a Name (line 228):
        
        # Assigning a Call to a Name (line 228):
        
        # Call to linspace(...): (line 228)
        # Processing the call arguments (line 228)
        int_352380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 24), 'int')
        float_352381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 27), 'float')
        int_352382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 32), 'int')
        # Processing the call keyword arguments (line 228)
        kwargs_352383 = {}
        # Getting the type of 'np' (line 228)
        np_352378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 228)
        linspace_352379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 12), np_352378, 'linspace')
        # Calling linspace(args, kwargs) (line 228)
        linspace_call_result_352384 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), linspace_352379, *[int_352380, float_352381, int_352382], **kwargs_352383)
        
        # Assigning a type to the variable 't' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 't', linspace_call_result_352384)
        
        # Assigning a Call to a Name (line 229):
        
        # Assigning a Call to a Name (line 229):
        
        # Call to _sweep_poly_phase(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 't' (line 229)
        t_352387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 44), 't', False)
        # Getting the type of 'p' (line 229)
        p_352388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 47), 'p', False)
        # Processing the call keyword arguments (line 229)
        kwargs_352389 = {}
        # Getting the type of 'waveforms' (line 229)
        waveforms_352385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 16), 'waveforms', False)
        # Obtaining the member '_sweep_poly_phase' of a type (line 229)
        _sweep_poly_phase_352386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 16), waveforms_352385, '_sweep_poly_phase')
        # Calling _sweep_poly_phase(args, kwargs) (line 229)
        _sweep_poly_phase_call_result_352390 = invoke(stypy.reporting.localization.Localization(__file__, 229, 16), _sweep_poly_phase_352386, *[t_352387, p_352388], **kwargs_352389)
        
        # Assigning a type to the variable 'phase' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'phase', _sweep_poly_phase_call_result_352390)
        
        # Assigning a Call to a Tuple (line 230):
        
        # Assigning a Subscript to a Name (line 230):
        
        # Obtaining the type of the subscript
        int_352391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 8), 'int')
        
        # Call to compute_frequency(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 't' (line 230)
        t_352393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 34), 't', False)
        # Getting the type of 'phase' (line 230)
        phase_352394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 37), 'phase', False)
        # Processing the call keyword arguments (line 230)
        kwargs_352395 = {}
        # Getting the type of 'compute_frequency' (line 230)
        compute_frequency_352392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 230)
        compute_frequency_call_result_352396 = invoke(stypy.reporting.localization.Localization(__file__, 230, 16), compute_frequency_352392, *[t_352393, phase_352394], **kwargs_352395)
        
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___352397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), compute_frequency_call_result_352396, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 230)
        subscript_call_result_352398 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), getitem___352397, int_352391)
        
        # Assigning a type to the variable 'tuple_var_assignment_351430' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'tuple_var_assignment_351430', subscript_call_result_352398)
        
        # Assigning a Subscript to a Name (line 230):
        
        # Obtaining the type of the subscript
        int_352399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 8), 'int')
        
        # Call to compute_frequency(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 't' (line 230)
        t_352401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 34), 't', False)
        # Getting the type of 'phase' (line 230)
        phase_352402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 37), 'phase', False)
        # Processing the call keyword arguments (line 230)
        kwargs_352403 = {}
        # Getting the type of 'compute_frequency' (line 230)
        compute_frequency_352400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 230)
        compute_frequency_call_result_352404 = invoke(stypy.reporting.localization.Localization(__file__, 230, 16), compute_frequency_352400, *[t_352401, phase_352402], **kwargs_352403)
        
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___352405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 8), compute_frequency_call_result_352404, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 230)
        subscript_call_result_352406 = invoke(stypy.reporting.localization.Localization(__file__, 230, 8), getitem___352405, int_352399)
        
        # Assigning a type to the variable 'tuple_var_assignment_351431' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'tuple_var_assignment_351431', subscript_call_result_352406)
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'tuple_var_assignment_351430' (line 230)
        tuple_var_assignment_351430_352407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'tuple_var_assignment_351430')
        # Assigning a type to the variable 'tf' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'tf', tuple_var_assignment_351430_352407)
        
        # Assigning a Name to a Name (line 230):
        # Getting the type of 'tuple_var_assignment_351431' (line 230)
        tuple_var_assignment_351431_352408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'tuple_var_assignment_351431')
        # Assigning a type to the variable 'f' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'f', tuple_var_assignment_351431_352408)
        
        # Assigning a Call to a Name (line 231):
        
        # Assigning a Call to a Name (line 231):
        
        # Call to p(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'tf' (line 231)
        tf_352410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 21), 'tf', False)
        # Processing the call keyword arguments (line 231)
        kwargs_352411 = {}
        # Getting the type of 'p' (line 231)
        p_352409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 19), 'p', False)
        # Calling p(args, kwargs) (line 231)
        p_call_result_352412 = invoke(stypy.reporting.localization.Localization(__file__, 231, 19), p_352409, *[tf_352410], **kwargs_352411)
        
        # Assigning a type to the variable 'expected' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'expected', p_call_result_352412)
        
        # Assigning a Call to a Name (line 232):
        
        # Assigning a Call to a Name (line 232):
        
        # Call to max(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Call to abs(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'f' (line 232)
        f_352417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 31), 'f', False)
        # Getting the type of 'expected' (line 232)
        expected_352418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 35), 'expected', False)
        # Applying the binary operator '-' (line 232)
        result_sub_352419 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 31), '-', f_352417, expected_352418)
        
        # Processing the call keyword arguments (line 232)
        kwargs_352420 = {}
        # Getting the type of 'np' (line 232)
        np_352415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 24), 'np', False)
        # Obtaining the member 'abs' of a type (line 232)
        abs_352416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 24), np_352415, 'abs')
        # Calling abs(args, kwargs) (line 232)
        abs_call_result_352421 = invoke(stypy.reporting.localization.Localization(__file__, 232, 24), abs_352416, *[result_sub_352419], **kwargs_352420)
        
        # Processing the call keyword arguments (line 232)
        kwargs_352422 = {}
        # Getting the type of 'np' (line 232)
        np_352413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 17), 'np', False)
        # Obtaining the member 'max' of a type (line 232)
        max_352414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 17), np_352413, 'max')
        # Calling max(args, kwargs) (line 232)
        max_call_result_352423 = invoke(stypy.reporting.localization.Localization(__file__, 232, 17), max_352414, *[abs_call_result_352421], **kwargs_352422)
        
        # Assigning a type to the variable 'abserr' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'abserr', max_call_result_352423)
        
        # Call to assert_(...): (line 233)
        # Processing the call arguments (line 233)
        
        # Getting the type of 'abserr' (line 233)
        abserr_352425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'abserr', False)
        float_352426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 25), 'float')
        # Applying the binary operator '<' (line 233)
        result_lt_352427 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 16), '<', abserr_352425, float_352426)
        
        # Processing the call keyword arguments (line 233)
        kwargs_352428 = {}
        # Getting the type of 'assert_' (line 233)
        assert__352424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 233)
        assert__call_result_352429 = invoke(stypy.reporting.localization.Localization(__file__, 233, 8), assert__352424, *[result_lt_352427], **kwargs_352428)
        
        
        # ################# End of 'test_sweep_poly_quad1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sweep_poly_quad1' in the type store
        # Getting the type of 'stypy_return_type' (line 226)
        stypy_return_type_352430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352430)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sweep_poly_quad1'
        return stypy_return_type_352430


    @norecursion
    def test_sweep_poly_const(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sweep_poly_const'
        module_type_store = module_type_store.open_function_context('test_sweep_poly_const', 235, 4, False)
        # Assigning a type to the variable 'self' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSweepPoly.test_sweep_poly_const.__dict__.__setitem__('stypy_localization', localization)
        TestSweepPoly.test_sweep_poly_const.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSweepPoly.test_sweep_poly_const.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSweepPoly.test_sweep_poly_const.__dict__.__setitem__('stypy_function_name', 'TestSweepPoly.test_sweep_poly_const')
        TestSweepPoly.test_sweep_poly_const.__dict__.__setitem__('stypy_param_names_list', [])
        TestSweepPoly.test_sweep_poly_const.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSweepPoly.test_sweep_poly_const.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSweepPoly.test_sweep_poly_const.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSweepPoly.test_sweep_poly_const.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSweepPoly.test_sweep_poly_const.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSweepPoly.test_sweep_poly_const.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSweepPoly.test_sweep_poly_const', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sweep_poly_const', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sweep_poly_const(...)' code ##################

        
        # Assigning a Call to a Name (line 236):
        
        # Assigning a Call to a Name (line 236):
        
        # Call to poly1d(...): (line 236)
        # Processing the call arguments (line 236)
        float_352433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 22), 'float')
        # Processing the call keyword arguments (line 236)
        kwargs_352434 = {}
        # Getting the type of 'np' (line 236)
        np_352431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'np', False)
        # Obtaining the member 'poly1d' of a type (line 236)
        poly1d_352432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 12), np_352431, 'poly1d')
        # Calling poly1d(args, kwargs) (line 236)
        poly1d_call_result_352435 = invoke(stypy.reporting.localization.Localization(__file__, 236, 12), poly1d_352432, *[float_352433], **kwargs_352434)
        
        # Assigning a type to the variable 'p' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'p', poly1d_call_result_352435)
        
        # Assigning a Call to a Name (line 237):
        
        # Assigning a Call to a Name (line 237):
        
        # Call to linspace(...): (line 237)
        # Processing the call arguments (line 237)
        int_352438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 24), 'int')
        float_352439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 27), 'float')
        int_352440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 32), 'int')
        # Processing the call keyword arguments (line 237)
        kwargs_352441 = {}
        # Getting the type of 'np' (line 237)
        np_352436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 237)
        linspace_352437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), np_352436, 'linspace')
        # Calling linspace(args, kwargs) (line 237)
        linspace_call_result_352442 = invoke(stypy.reporting.localization.Localization(__file__, 237, 12), linspace_352437, *[int_352438, float_352439, int_352440], **kwargs_352441)
        
        # Assigning a type to the variable 't' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 't', linspace_call_result_352442)
        
        # Assigning a Call to a Name (line 238):
        
        # Assigning a Call to a Name (line 238):
        
        # Call to _sweep_poly_phase(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 't' (line 238)
        t_352445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 44), 't', False)
        # Getting the type of 'p' (line 238)
        p_352446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 47), 'p', False)
        # Processing the call keyword arguments (line 238)
        kwargs_352447 = {}
        # Getting the type of 'waveforms' (line 238)
        waveforms_352443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'waveforms', False)
        # Obtaining the member '_sweep_poly_phase' of a type (line 238)
        _sweep_poly_phase_352444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 16), waveforms_352443, '_sweep_poly_phase')
        # Calling _sweep_poly_phase(args, kwargs) (line 238)
        _sweep_poly_phase_call_result_352448 = invoke(stypy.reporting.localization.Localization(__file__, 238, 16), _sweep_poly_phase_352444, *[t_352445, p_352446], **kwargs_352447)
        
        # Assigning a type to the variable 'phase' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'phase', _sweep_poly_phase_call_result_352448)
        
        # Assigning a Call to a Tuple (line 239):
        
        # Assigning a Subscript to a Name (line 239):
        
        # Obtaining the type of the subscript
        int_352449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 8), 'int')
        
        # Call to compute_frequency(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 't' (line 239)
        t_352451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 34), 't', False)
        # Getting the type of 'phase' (line 239)
        phase_352452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 37), 'phase', False)
        # Processing the call keyword arguments (line 239)
        kwargs_352453 = {}
        # Getting the type of 'compute_frequency' (line 239)
        compute_frequency_352450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 239)
        compute_frequency_call_result_352454 = invoke(stypy.reporting.localization.Localization(__file__, 239, 16), compute_frequency_352450, *[t_352451, phase_352452], **kwargs_352453)
        
        # Obtaining the member '__getitem__' of a type (line 239)
        getitem___352455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), compute_frequency_call_result_352454, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 239)
        subscript_call_result_352456 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), getitem___352455, int_352449)
        
        # Assigning a type to the variable 'tuple_var_assignment_351432' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'tuple_var_assignment_351432', subscript_call_result_352456)
        
        # Assigning a Subscript to a Name (line 239):
        
        # Obtaining the type of the subscript
        int_352457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 8), 'int')
        
        # Call to compute_frequency(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 't' (line 239)
        t_352459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 34), 't', False)
        # Getting the type of 'phase' (line 239)
        phase_352460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 37), 'phase', False)
        # Processing the call keyword arguments (line 239)
        kwargs_352461 = {}
        # Getting the type of 'compute_frequency' (line 239)
        compute_frequency_352458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 239)
        compute_frequency_call_result_352462 = invoke(stypy.reporting.localization.Localization(__file__, 239, 16), compute_frequency_352458, *[t_352459, phase_352460], **kwargs_352461)
        
        # Obtaining the member '__getitem__' of a type (line 239)
        getitem___352463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), compute_frequency_call_result_352462, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 239)
        subscript_call_result_352464 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), getitem___352463, int_352457)
        
        # Assigning a type to the variable 'tuple_var_assignment_351433' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'tuple_var_assignment_351433', subscript_call_result_352464)
        
        # Assigning a Name to a Name (line 239):
        # Getting the type of 'tuple_var_assignment_351432' (line 239)
        tuple_var_assignment_351432_352465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'tuple_var_assignment_351432')
        # Assigning a type to the variable 'tf' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'tf', tuple_var_assignment_351432_352465)
        
        # Assigning a Name to a Name (line 239):
        # Getting the type of 'tuple_var_assignment_351433' (line 239)
        tuple_var_assignment_351433_352466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'tuple_var_assignment_351433')
        # Assigning a type to the variable 'f' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'f', tuple_var_assignment_351433_352466)
        
        # Assigning a Call to a Name (line 240):
        
        # Assigning a Call to a Name (line 240):
        
        # Call to p(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'tf' (line 240)
        tf_352468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 21), 'tf', False)
        # Processing the call keyword arguments (line 240)
        kwargs_352469 = {}
        # Getting the type of 'p' (line 240)
        p_352467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 19), 'p', False)
        # Calling p(args, kwargs) (line 240)
        p_call_result_352470 = invoke(stypy.reporting.localization.Localization(__file__, 240, 19), p_352467, *[tf_352468], **kwargs_352469)
        
        # Assigning a type to the variable 'expected' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'expected', p_call_result_352470)
        
        # Assigning a Call to a Name (line 241):
        
        # Assigning a Call to a Name (line 241):
        
        # Call to max(...): (line 241)
        # Processing the call arguments (line 241)
        
        # Call to abs(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'f' (line 241)
        f_352475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 31), 'f', False)
        # Getting the type of 'expected' (line 241)
        expected_352476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 35), 'expected', False)
        # Applying the binary operator '-' (line 241)
        result_sub_352477 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 31), '-', f_352475, expected_352476)
        
        # Processing the call keyword arguments (line 241)
        kwargs_352478 = {}
        # Getting the type of 'np' (line 241)
        np_352473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 24), 'np', False)
        # Obtaining the member 'abs' of a type (line 241)
        abs_352474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 24), np_352473, 'abs')
        # Calling abs(args, kwargs) (line 241)
        abs_call_result_352479 = invoke(stypy.reporting.localization.Localization(__file__, 241, 24), abs_352474, *[result_sub_352477], **kwargs_352478)
        
        # Processing the call keyword arguments (line 241)
        kwargs_352480 = {}
        # Getting the type of 'np' (line 241)
        np_352471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 17), 'np', False)
        # Obtaining the member 'max' of a type (line 241)
        max_352472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 17), np_352471, 'max')
        # Calling max(args, kwargs) (line 241)
        max_call_result_352481 = invoke(stypy.reporting.localization.Localization(__file__, 241, 17), max_352472, *[abs_call_result_352479], **kwargs_352480)
        
        # Assigning a type to the variable 'abserr' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'abserr', max_call_result_352481)
        
        # Call to assert_(...): (line 242)
        # Processing the call arguments (line 242)
        
        # Getting the type of 'abserr' (line 242)
        abserr_352483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'abserr', False)
        float_352484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 25), 'float')
        # Applying the binary operator '<' (line 242)
        result_lt_352485 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 16), '<', abserr_352483, float_352484)
        
        # Processing the call keyword arguments (line 242)
        kwargs_352486 = {}
        # Getting the type of 'assert_' (line 242)
        assert__352482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 242)
        assert__call_result_352487 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), assert__352482, *[result_lt_352485], **kwargs_352486)
        
        
        # ################# End of 'test_sweep_poly_const(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sweep_poly_const' in the type store
        # Getting the type of 'stypy_return_type' (line 235)
        stypy_return_type_352488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352488)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sweep_poly_const'
        return stypy_return_type_352488


    @norecursion
    def test_sweep_poly_linear(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sweep_poly_linear'
        module_type_store = module_type_store.open_function_context('test_sweep_poly_linear', 244, 4, False)
        # Assigning a type to the variable 'self' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSweepPoly.test_sweep_poly_linear.__dict__.__setitem__('stypy_localization', localization)
        TestSweepPoly.test_sweep_poly_linear.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSweepPoly.test_sweep_poly_linear.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSweepPoly.test_sweep_poly_linear.__dict__.__setitem__('stypy_function_name', 'TestSweepPoly.test_sweep_poly_linear')
        TestSweepPoly.test_sweep_poly_linear.__dict__.__setitem__('stypy_param_names_list', [])
        TestSweepPoly.test_sweep_poly_linear.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSweepPoly.test_sweep_poly_linear.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSweepPoly.test_sweep_poly_linear.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSweepPoly.test_sweep_poly_linear.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSweepPoly.test_sweep_poly_linear.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSweepPoly.test_sweep_poly_linear.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSweepPoly.test_sweep_poly_linear', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sweep_poly_linear', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sweep_poly_linear(...)' code ##################

        
        # Assigning a Call to a Name (line 245):
        
        # Assigning a Call to a Name (line 245):
        
        # Call to poly1d(...): (line 245)
        # Processing the call arguments (line 245)
        
        # Obtaining an instance of the builtin type 'list' (line 245)
        list_352491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 245)
        # Adding element type (line 245)
        float_352492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 22), list_352491, float_352492)
        # Adding element type (line 245)
        float_352493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 29), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 22), list_352491, float_352493)
        
        # Processing the call keyword arguments (line 245)
        kwargs_352494 = {}
        # Getting the type of 'np' (line 245)
        np_352489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'np', False)
        # Obtaining the member 'poly1d' of a type (line 245)
        poly1d_352490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 12), np_352489, 'poly1d')
        # Calling poly1d(args, kwargs) (line 245)
        poly1d_call_result_352495 = invoke(stypy.reporting.localization.Localization(__file__, 245, 12), poly1d_352490, *[list_352491], **kwargs_352494)
        
        # Assigning a type to the variable 'p' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'p', poly1d_call_result_352495)
        
        # Assigning a Call to a Name (line 246):
        
        # Assigning a Call to a Name (line 246):
        
        # Call to linspace(...): (line 246)
        # Processing the call arguments (line 246)
        int_352498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 24), 'int')
        float_352499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 27), 'float')
        int_352500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 32), 'int')
        # Processing the call keyword arguments (line 246)
        kwargs_352501 = {}
        # Getting the type of 'np' (line 246)
        np_352496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 246)
        linspace_352497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 12), np_352496, 'linspace')
        # Calling linspace(args, kwargs) (line 246)
        linspace_call_result_352502 = invoke(stypy.reporting.localization.Localization(__file__, 246, 12), linspace_352497, *[int_352498, float_352499, int_352500], **kwargs_352501)
        
        # Assigning a type to the variable 't' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 't', linspace_call_result_352502)
        
        # Assigning a Call to a Name (line 247):
        
        # Assigning a Call to a Name (line 247):
        
        # Call to _sweep_poly_phase(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 't' (line 247)
        t_352505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 44), 't', False)
        # Getting the type of 'p' (line 247)
        p_352506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 47), 'p', False)
        # Processing the call keyword arguments (line 247)
        kwargs_352507 = {}
        # Getting the type of 'waveforms' (line 247)
        waveforms_352503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'waveforms', False)
        # Obtaining the member '_sweep_poly_phase' of a type (line 247)
        _sweep_poly_phase_352504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 16), waveforms_352503, '_sweep_poly_phase')
        # Calling _sweep_poly_phase(args, kwargs) (line 247)
        _sweep_poly_phase_call_result_352508 = invoke(stypy.reporting.localization.Localization(__file__, 247, 16), _sweep_poly_phase_352504, *[t_352505, p_352506], **kwargs_352507)
        
        # Assigning a type to the variable 'phase' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'phase', _sweep_poly_phase_call_result_352508)
        
        # Assigning a Call to a Tuple (line 248):
        
        # Assigning a Subscript to a Name (line 248):
        
        # Obtaining the type of the subscript
        int_352509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 8), 'int')
        
        # Call to compute_frequency(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 't' (line 248)
        t_352511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 34), 't', False)
        # Getting the type of 'phase' (line 248)
        phase_352512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 37), 'phase', False)
        # Processing the call keyword arguments (line 248)
        kwargs_352513 = {}
        # Getting the type of 'compute_frequency' (line 248)
        compute_frequency_352510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 248)
        compute_frequency_call_result_352514 = invoke(stypy.reporting.localization.Localization(__file__, 248, 16), compute_frequency_352510, *[t_352511, phase_352512], **kwargs_352513)
        
        # Obtaining the member '__getitem__' of a type (line 248)
        getitem___352515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), compute_frequency_call_result_352514, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 248)
        subscript_call_result_352516 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), getitem___352515, int_352509)
        
        # Assigning a type to the variable 'tuple_var_assignment_351434' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'tuple_var_assignment_351434', subscript_call_result_352516)
        
        # Assigning a Subscript to a Name (line 248):
        
        # Obtaining the type of the subscript
        int_352517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 8), 'int')
        
        # Call to compute_frequency(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 't' (line 248)
        t_352519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 34), 't', False)
        # Getting the type of 'phase' (line 248)
        phase_352520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 37), 'phase', False)
        # Processing the call keyword arguments (line 248)
        kwargs_352521 = {}
        # Getting the type of 'compute_frequency' (line 248)
        compute_frequency_352518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 248)
        compute_frequency_call_result_352522 = invoke(stypy.reporting.localization.Localization(__file__, 248, 16), compute_frequency_352518, *[t_352519, phase_352520], **kwargs_352521)
        
        # Obtaining the member '__getitem__' of a type (line 248)
        getitem___352523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), compute_frequency_call_result_352522, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 248)
        subscript_call_result_352524 = invoke(stypy.reporting.localization.Localization(__file__, 248, 8), getitem___352523, int_352517)
        
        # Assigning a type to the variable 'tuple_var_assignment_351435' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'tuple_var_assignment_351435', subscript_call_result_352524)
        
        # Assigning a Name to a Name (line 248):
        # Getting the type of 'tuple_var_assignment_351434' (line 248)
        tuple_var_assignment_351434_352525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'tuple_var_assignment_351434')
        # Assigning a type to the variable 'tf' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'tf', tuple_var_assignment_351434_352525)
        
        # Assigning a Name to a Name (line 248):
        # Getting the type of 'tuple_var_assignment_351435' (line 248)
        tuple_var_assignment_351435_352526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'tuple_var_assignment_351435')
        # Assigning a type to the variable 'f' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'f', tuple_var_assignment_351435_352526)
        
        # Assigning a Call to a Name (line 249):
        
        # Assigning a Call to a Name (line 249):
        
        # Call to p(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'tf' (line 249)
        tf_352528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 21), 'tf', False)
        # Processing the call keyword arguments (line 249)
        kwargs_352529 = {}
        # Getting the type of 'p' (line 249)
        p_352527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 19), 'p', False)
        # Calling p(args, kwargs) (line 249)
        p_call_result_352530 = invoke(stypy.reporting.localization.Localization(__file__, 249, 19), p_352527, *[tf_352528], **kwargs_352529)
        
        # Assigning a type to the variable 'expected' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'expected', p_call_result_352530)
        
        # Assigning a Call to a Name (line 250):
        
        # Assigning a Call to a Name (line 250):
        
        # Call to max(...): (line 250)
        # Processing the call arguments (line 250)
        
        # Call to abs(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'f' (line 250)
        f_352535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 31), 'f', False)
        # Getting the type of 'expected' (line 250)
        expected_352536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 35), 'expected', False)
        # Applying the binary operator '-' (line 250)
        result_sub_352537 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 31), '-', f_352535, expected_352536)
        
        # Processing the call keyword arguments (line 250)
        kwargs_352538 = {}
        # Getting the type of 'np' (line 250)
        np_352533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 24), 'np', False)
        # Obtaining the member 'abs' of a type (line 250)
        abs_352534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 24), np_352533, 'abs')
        # Calling abs(args, kwargs) (line 250)
        abs_call_result_352539 = invoke(stypy.reporting.localization.Localization(__file__, 250, 24), abs_352534, *[result_sub_352537], **kwargs_352538)
        
        # Processing the call keyword arguments (line 250)
        kwargs_352540 = {}
        # Getting the type of 'np' (line 250)
        np_352531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 17), 'np', False)
        # Obtaining the member 'max' of a type (line 250)
        max_352532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 17), np_352531, 'max')
        # Calling max(args, kwargs) (line 250)
        max_call_result_352541 = invoke(stypy.reporting.localization.Localization(__file__, 250, 17), max_352532, *[abs_call_result_352539], **kwargs_352540)
        
        # Assigning a type to the variable 'abserr' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'abserr', max_call_result_352541)
        
        # Call to assert_(...): (line 251)
        # Processing the call arguments (line 251)
        
        # Getting the type of 'abserr' (line 251)
        abserr_352543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 16), 'abserr', False)
        float_352544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 25), 'float')
        # Applying the binary operator '<' (line 251)
        result_lt_352545 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 16), '<', abserr_352543, float_352544)
        
        # Processing the call keyword arguments (line 251)
        kwargs_352546 = {}
        # Getting the type of 'assert_' (line 251)
        assert__352542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 251)
        assert__call_result_352547 = invoke(stypy.reporting.localization.Localization(__file__, 251, 8), assert__352542, *[result_lt_352545], **kwargs_352546)
        
        
        # ################# End of 'test_sweep_poly_linear(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sweep_poly_linear' in the type store
        # Getting the type of 'stypy_return_type' (line 244)
        stypy_return_type_352548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352548)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sweep_poly_linear'
        return stypy_return_type_352548


    @norecursion
    def test_sweep_poly_quad2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sweep_poly_quad2'
        module_type_store = module_type_store.open_function_context('test_sweep_poly_quad2', 253, 4, False)
        # Assigning a type to the variable 'self' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSweepPoly.test_sweep_poly_quad2.__dict__.__setitem__('stypy_localization', localization)
        TestSweepPoly.test_sweep_poly_quad2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSweepPoly.test_sweep_poly_quad2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSweepPoly.test_sweep_poly_quad2.__dict__.__setitem__('stypy_function_name', 'TestSweepPoly.test_sweep_poly_quad2')
        TestSweepPoly.test_sweep_poly_quad2.__dict__.__setitem__('stypy_param_names_list', [])
        TestSweepPoly.test_sweep_poly_quad2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSweepPoly.test_sweep_poly_quad2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSweepPoly.test_sweep_poly_quad2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSweepPoly.test_sweep_poly_quad2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSweepPoly.test_sweep_poly_quad2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSweepPoly.test_sweep_poly_quad2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSweepPoly.test_sweep_poly_quad2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sweep_poly_quad2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sweep_poly_quad2(...)' code ##################

        
        # Assigning a Call to a Name (line 254):
        
        # Assigning a Call to a Name (line 254):
        
        # Call to poly1d(...): (line 254)
        # Processing the call arguments (line 254)
        
        # Obtaining an instance of the builtin type 'list' (line 254)
        list_352551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 254)
        # Adding element type (line 254)
        float_352552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 22), list_352551, float_352552)
        # Adding element type (line 254)
        float_352553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 22), list_352551, float_352553)
        # Adding element type (line 254)
        float_352554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 22), list_352551, float_352554)
        
        # Processing the call keyword arguments (line 254)
        kwargs_352555 = {}
        # Getting the type of 'np' (line 254)
        np_352549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'np', False)
        # Obtaining the member 'poly1d' of a type (line 254)
        poly1d_352550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 12), np_352549, 'poly1d')
        # Calling poly1d(args, kwargs) (line 254)
        poly1d_call_result_352556 = invoke(stypy.reporting.localization.Localization(__file__, 254, 12), poly1d_352550, *[list_352551], **kwargs_352555)
        
        # Assigning a type to the variable 'p' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'p', poly1d_call_result_352556)
        
        # Assigning a Call to a Name (line 255):
        
        # Assigning a Call to a Name (line 255):
        
        # Call to linspace(...): (line 255)
        # Processing the call arguments (line 255)
        int_352559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 24), 'int')
        float_352560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 27), 'float')
        int_352561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 32), 'int')
        # Processing the call keyword arguments (line 255)
        kwargs_352562 = {}
        # Getting the type of 'np' (line 255)
        np_352557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 255)
        linspace_352558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), np_352557, 'linspace')
        # Calling linspace(args, kwargs) (line 255)
        linspace_call_result_352563 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), linspace_352558, *[int_352559, float_352560, int_352561], **kwargs_352562)
        
        # Assigning a type to the variable 't' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 't', linspace_call_result_352563)
        
        # Assigning a Call to a Name (line 256):
        
        # Assigning a Call to a Name (line 256):
        
        # Call to _sweep_poly_phase(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 't' (line 256)
        t_352566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 44), 't', False)
        # Getting the type of 'p' (line 256)
        p_352567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 47), 'p', False)
        # Processing the call keyword arguments (line 256)
        kwargs_352568 = {}
        # Getting the type of 'waveforms' (line 256)
        waveforms_352564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 16), 'waveforms', False)
        # Obtaining the member '_sweep_poly_phase' of a type (line 256)
        _sweep_poly_phase_352565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 16), waveforms_352564, '_sweep_poly_phase')
        # Calling _sweep_poly_phase(args, kwargs) (line 256)
        _sweep_poly_phase_call_result_352569 = invoke(stypy.reporting.localization.Localization(__file__, 256, 16), _sweep_poly_phase_352565, *[t_352566, p_352567], **kwargs_352568)
        
        # Assigning a type to the variable 'phase' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'phase', _sweep_poly_phase_call_result_352569)
        
        # Assigning a Call to a Tuple (line 257):
        
        # Assigning a Subscript to a Name (line 257):
        
        # Obtaining the type of the subscript
        int_352570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 8), 'int')
        
        # Call to compute_frequency(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 't' (line 257)
        t_352572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 34), 't', False)
        # Getting the type of 'phase' (line 257)
        phase_352573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 37), 'phase', False)
        # Processing the call keyword arguments (line 257)
        kwargs_352574 = {}
        # Getting the type of 'compute_frequency' (line 257)
        compute_frequency_352571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 257)
        compute_frequency_call_result_352575 = invoke(stypy.reporting.localization.Localization(__file__, 257, 16), compute_frequency_352571, *[t_352572, phase_352573], **kwargs_352574)
        
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___352576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), compute_frequency_call_result_352575, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_352577 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), getitem___352576, int_352570)
        
        # Assigning a type to the variable 'tuple_var_assignment_351436' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'tuple_var_assignment_351436', subscript_call_result_352577)
        
        # Assigning a Subscript to a Name (line 257):
        
        # Obtaining the type of the subscript
        int_352578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 8), 'int')
        
        # Call to compute_frequency(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 't' (line 257)
        t_352580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 34), 't', False)
        # Getting the type of 'phase' (line 257)
        phase_352581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 37), 'phase', False)
        # Processing the call keyword arguments (line 257)
        kwargs_352582 = {}
        # Getting the type of 'compute_frequency' (line 257)
        compute_frequency_352579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 257)
        compute_frequency_call_result_352583 = invoke(stypy.reporting.localization.Localization(__file__, 257, 16), compute_frequency_352579, *[t_352580, phase_352581], **kwargs_352582)
        
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___352584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), compute_frequency_call_result_352583, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_352585 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), getitem___352584, int_352578)
        
        # Assigning a type to the variable 'tuple_var_assignment_351437' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'tuple_var_assignment_351437', subscript_call_result_352585)
        
        # Assigning a Name to a Name (line 257):
        # Getting the type of 'tuple_var_assignment_351436' (line 257)
        tuple_var_assignment_351436_352586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'tuple_var_assignment_351436')
        # Assigning a type to the variable 'tf' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'tf', tuple_var_assignment_351436_352586)
        
        # Assigning a Name to a Name (line 257):
        # Getting the type of 'tuple_var_assignment_351437' (line 257)
        tuple_var_assignment_351437_352587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'tuple_var_assignment_351437')
        # Assigning a type to the variable 'f' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'f', tuple_var_assignment_351437_352587)
        
        # Assigning a Call to a Name (line 258):
        
        # Assigning a Call to a Name (line 258):
        
        # Call to p(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'tf' (line 258)
        tf_352589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 21), 'tf', False)
        # Processing the call keyword arguments (line 258)
        kwargs_352590 = {}
        # Getting the type of 'p' (line 258)
        p_352588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 19), 'p', False)
        # Calling p(args, kwargs) (line 258)
        p_call_result_352591 = invoke(stypy.reporting.localization.Localization(__file__, 258, 19), p_352588, *[tf_352589], **kwargs_352590)
        
        # Assigning a type to the variable 'expected' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'expected', p_call_result_352591)
        
        # Assigning a Call to a Name (line 259):
        
        # Assigning a Call to a Name (line 259):
        
        # Call to max(...): (line 259)
        # Processing the call arguments (line 259)
        
        # Call to abs(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'f' (line 259)
        f_352596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 31), 'f', False)
        # Getting the type of 'expected' (line 259)
        expected_352597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 35), 'expected', False)
        # Applying the binary operator '-' (line 259)
        result_sub_352598 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 31), '-', f_352596, expected_352597)
        
        # Processing the call keyword arguments (line 259)
        kwargs_352599 = {}
        # Getting the type of 'np' (line 259)
        np_352594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 24), 'np', False)
        # Obtaining the member 'abs' of a type (line 259)
        abs_352595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 24), np_352594, 'abs')
        # Calling abs(args, kwargs) (line 259)
        abs_call_result_352600 = invoke(stypy.reporting.localization.Localization(__file__, 259, 24), abs_352595, *[result_sub_352598], **kwargs_352599)
        
        # Processing the call keyword arguments (line 259)
        kwargs_352601 = {}
        # Getting the type of 'np' (line 259)
        np_352592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 17), 'np', False)
        # Obtaining the member 'max' of a type (line 259)
        max_352593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 17), np_352592, 'max')
        # Calling max(args, kwargs) (line 259)
        max_call_result_352602 = invoke(stypy.reporting.localization.Localization(__file__, 259, 17), max_352593, *[abs_call_result_352600], **kwargs_352601)
        
        # Assigning a type to the variable 'abserr' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'abserr', max_call_result_352602)
        
        # Call to assert_(...): (line 260)
        # Processing the call arguments (line 260)
        
        # Getting the type of 'abserr' (line 260)
        abserr_352604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 16), 'abserr', False)
        float_352605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 25), 'float')
        # Applying the binary operator '<' (line 260)
        result_lt_352606 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 16), '<', abserr_352604, float_352605)
        
        # Processing the call keyword arguments (line 260)
        kwargs_352607 = {}
        # Getting the type of 'assert_' (line 260)
        assert__352603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 260)
        assert__call_result_352608 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), assert__352603, *[result_lt_352606], **kwargs_352607)
        
        
        # ################# End of 'test_sweep_poly_quad2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sweep_poly_quad2' in the type store
        # Getting the type of 'stypy_return_type' (line 253)
        stypy_return_type_352609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352609)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sweep_poly_quad2'
        return stypy_return_type_352609


    @norecursion
    def test_sweep_poly_cubic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sweep_poly_cubic'
        module_type_store = module_type_store.open_function_context('test_sweep_poly_cubic', 262, 4, False)
        # Assigning a type to the variable 'self' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSweepPoly.test_sweep_poly_cubic.__dict__.__setitem__('stypy_localization', localization)
        TestSweepPoly.test_sweep_poly_cubic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSweepPoly.test_sweep_poly_cubic.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSweepPoly.test_sweep_poly_cubic.__dict__.__setitem__('stypy_function_name', 'TestSweepPoly.test_sweep_poly_cubic')
        TestSweepPoly.test_sweep_poly_cubic.__dict__.__setitem__('stypy_param_names_list', [])
        TestSweepPoly.test_sweep_poly_cubic.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSweepPoly.test_sweep_poly_cubic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSweepPoly.test_sweep_poly_cubic.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSweepPoly.test_sweep_poly_cubic.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSweepPoly.test_sweep_poly_cubic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSweepPoly.test_sweep_poly_cubic.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSweepPoly.test_sweep_poly_cubic', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sweep_poly_cubic', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sweep_poly_cubic(...)' code ##################

        
        # Assigning a Call to a Name (line 263):
        
        # Assigning a Call to a Name (line 263):
        
        # Call to poly1d(...): (line 263)
        # Processing the call arguments (line 263)
        
        # Obtaining an instance of the builtin type 'list' (line 263)
        list_352612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 263)
        # Adding element type (line 263)
        float_352613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 22), list_352612, float_352613)
        # Adding element type (line 263)
        float_352614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 22), list_352612, float_352614)
        # Adding element type (line 263)
        float_352615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 22), list_352612, float_352615)
        # Adding element type (line 263)
        float_352616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 22), list_352612, float_352616)
        
        # Processing the call keyword arguments (line 263)
        kwargs_352617 = {}
        # Getting the type of 'np' (line 263)
        np_352610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'np', False)
        # Obtaining the member 'poly1d' of a type (line 263)
        poly1d_352611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 12), np_352610, 'poly1d')
        # Calling poly1d(args, kwargs) (line 263)
        poly1d_call_result_352618 = invoke(stypy.reporting.localization.Localization(__file__, 263, 12), poly1d_352611, *[list_352612], **kwargs_352617)
        
        # Assigning a type to the variable 'p' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'p', poly1d_call_result_352618)
        
        # Assigning a Call to a Name (line 264):
        
        # Assigning a Call to a Name (line 264):
        
        # Call to linspace(...): (line 264)
        # Processing the call arguments (line 264)
        int_352621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 24), 'int')
        float_352622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 27), 'float')
        int_352623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 32), 'int')
        # Processing the call keyword arguments (line 264)
        kwargs_352624 = {}
        # Getting the type of 'np' (line 264)
        np_352619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 264)
        linspace_352620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 12), np_352619, 'linspace')
        # Calling linspace(args, kwargs) (line 264)
        linspace_call_result_352625 = invoke(stypy.reporting.localization.Localization(__file__, 264, 12), linspace_352620, *[int_352621, float_352622, int_352623], **kwargs_352624)
        
        # Assigning a type to the variable 't' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 't', linspace_call_result_352625)
        
        # Assigning a Call to a Name (line 265):
        
        # Assigning a Call to a Name (line 265):
        
        # Call to _sweep_poly_phase(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 't' (line 265)
        t_352628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 44), 't', False)
        # Getting the type of 'p' (line 265)
        p_352629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 47), 'p', False)
        # Processing the call keyword arguments (line 265)
        kwargs_352630 = {}
        # Getting the type of 'waveforms' (line 265)
        waveforms_352626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 16), 'waveforms', False)
        # Obtaining the member '_sweep_poly_phase' of a type (line 265)
        _sweep_poly_phase_352627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 16), waveforms_352626, '_sweep_poly_phase')
        # Calling _sweep_poly_phase(args, kwargs) (line 265)
        _sweep_poly_phase_call_result_352631 = invoke(stypy.reporting.localization.Localization(__file__, 265, 16), _sweep_poly_phase_352627, *[t_352628, p_352629], **kwargs_352630)
        
        # Assigning a type to the variable 'phase' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'phase', _sweep_poly_phase_call_result_352631)
        
        # Assigning a Call to a Tuple (line 266):
        
        # Assigning a Subscript to a Name (line 266):
        
        # Obtaining the type of the subscript
        int_352632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 8), 'int')
        
        # Call to compute_frequency(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 't' (line 266)
        t_352634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 34), 't', False)
        # Getting the type of 'phase' (line 266)
        phase_352635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 37), 'phase', False)
        # Processing the call keyword arguments (line 266)
        kwargs_352636 = {}
        # Getting the type of 'compute_frequency' (line 266)
        compute_frequency_352633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 266)
        compute_frequency_call_result_352637 = invoke(stypy.reporting.localization.Localization(__file__, 266, 16), compute_frequency_352633, *[t_352634, phase_352635], **kwargs_352636)
        
        # Obtaining the member '__getitem__' of a type (line 266)
        getitem___352638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), compute_frequency_call_result_352637, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 266)
        subscript_call_result_352639 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), getitem___352638, int_352632)
        
        # Assigning a type to the variable 'tuple_var_assignment_351438' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'tuple_var_assignment_351438', subscript_call_result_352639)
        
        # Assigning a Subscript to a Name (line 266):
        
        # Obtaining the type of the subscript
        int_352640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 8), 'int')
        
        # Call to compute_frequency(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 't' (line 266)
        t_352642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 34), 't', False)
        # Getting the type of 'phase' (line 266)
        phase_352643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 37), 'phase', False)
        # Processing the call keyword arguments (line 266)
        kwargs_352644 = {}
        # Getting the type of 'compute_frequency' (line 266)
        compute_frequency_352641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 266)
        compute_frequency_call_result_352645 = invoke(stypy.reporting.localization.Localization(__file__, 266, 16), compute_frequency_352641, *[t_352642, phase_352643], **kwargs_352644)
        
        # Obtaining the member '__getitem__' of a type (line 266)
        getitem___352646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), compute_frequency_call_result_352645, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 266)
        subscript_call_result_352647 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), getitem___352646, int_352640)
        
        # Assigning a type to the variable 'tuple_var_assignment_351439' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'tuple_var_assignment_351439', subscript_call_result_352647)
        
        # Assigning a Name to a Name (line 266):
        # Getting the type of 'tuple_var_assignment_351438' (line 266)
        tuple_var_assignment_351438_352648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'tuple_var_assignment_351438')
        # Assigning a type to the variable 'tf' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'tf', tuple_var_assignment_351438_352648)
        
        # Assigning a Name to a Name (line 266):
        # Getting the type of 'tuple_var_assignment_351439' (line 266)
        tuple_var_assignment_351439_352649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'tuple_var_assignment_351439')
        # Assigning a type to the variable 'f' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'f', tuple_var_assignment_351439_352649)
        
        # Assigning a Call to a Name (line 267):
        
        # Assigning a Call to a Name (line 267):
        
        # Call to p(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'tf' (line 267)
        tf_352651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 21), 'tf', False)
        # Processing the call keyword arguments (line 267)
        kwargs_352652 = {}
        # Getting the type of 'p' (line 267)
        p_352650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 19), 'p', False)
        # Calling p(args, kwargs) (line 267)
        p_call_result_352653 = invoke(stypy.reporting.localization.Localization(__file__, 267, 19), p_352650, *[tf_352651], **kwargs_352652)
        
        # Assigning a type to the variable 'expected' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'expected', p_call_result_352653)
        
        # Assigning a Call to a Name (line 268):
        
        # Assigning a Call to a Name (line 268):
        
        # Call to max(...): (line 268)
        # Processing the call arguments (line 268)
        
        # Call to abs(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'f' (line 268)
        f_352658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 31), 'f', False)
        # Getting the type of 'expected' (line 268)
        expected_352659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 35), 'expected', False)
        # Applying the binary operator '-' (line 268)
        result_sub_352660 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 31), '-', f_352658, expected_352659)
        
        # Processing the call keyword arguments (line 268)
        kwargs_352661 = {}
        # Getting the type of 'np' (line 268)
        np_352656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 24), 'np', False)
        # Obtaining the member 'abs' of a type (line 268)
        abs_352657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 24), np_352656, 'abs')
        # Calling abs(args, kwargs) (line 268)
        abs_call_result_352662 = invoke(stypy.reporting.localization.Localization(__file__, 268, 24), abs_352657, *[result_sub_352660], **kwargs_352661)
        
        # Processing the call keyword arguments (line 268)
        kwargs_352663 = {}
        # Getting the type of 'np' (line 268)
        np_352654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 17), 'np', False)
        # Obtaining the member 'max' of a type (line 268)
        max_352655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 17), np_352654, 'max')
        # Calling max(args, kwargs) (line 268)
        max_call_result_352664 = invoke(stypy.reporting.localization.Localization(__file__, 268, 17), max_352655, *[abs_call_result_352662], **kwargs_352663)
        
        # Assigning a type to the variable 'abserr' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'abserr', max_call_result_352664)
        
        # Call to assert_(...): (line 269)
        # Processing the call arguments (line 269)
        
        # Getting the type of 'abserr' (line 269)
        abserr_352666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'abserr', False)
        float_352667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 25), 'float')
        # Applying the binary operator '<' (line 269)
        result_lt_352668 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 16), '<', abserr_352666, float_352667)
        
        # Processing the call keyword arguments (line 269)
        kwargs_352669 = {}
        # Getting the type of 'assert_' (line 269)
        assert__352665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 269)
        assert__call_result_352670 = invoke(stypy.reporting.localization.Localization(__file__, 269, 8), assert__352665, *[result_lt_352668], **kwargs_352669)
        
        
        # ################# End of 'test_sweep_poly_cubic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sweep_poly_cubic' in the type store
        # Getting the type of 'stypy_return_type' (line 262)
        stypy_return_type_352671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352671)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sweep_poly_cubic'
        return stypy_return_type_352671


    @norecursion
    def test_sweep_poly_cubic2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sweep_poly_cubic2'
        module_type_store = module_type_store.open_function_context('test_sweep_poly_cubic2', 271, 4, False)
        # Assigning a type to the variable 'self' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSweepPoly.test_sweep_poly_cubic2.__dict__.__setitem__('stypy_localization', localization)
        TestSweepPoly.test_sweep_poly_cubic2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSweepPoly.test_sweep_poly_cubic2.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSweepPoly.test_sweep_poly_cubic2.__dict__.__setitem__('stypy_function_name', 'TestSweepPoly.test_sweep_poly_cubic2')
        TestSweepPoly.test_sweep_poly_cubic2.__dict__.__setitem__('stypy_param_names_list', [])
        TestSweepPoly.test_sweep_poly_cubic2.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSweepPoly.test_sweep_poly_cubic2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSweepPoly.test_sweep_poly_cubic2.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSweepPoly.test_sweep_poly_cubic2.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSweepPoly.test_sweep_poly_cubic2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSweepPoly.test_sweep_poly_cubic2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSweepPoly.test_sweep_poly_cubic2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sweep_poly_cubic2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sweep_poly_cubic2(...)' code ##################

        str_352672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 8), 'str', 'Use an array of coefficients instead of a poly1d.')
        
        # Assigning a Call to a Name (line 273):
        
        # Assigning a Call to a Name (line 273):
        
        # Call to array(...): (line 273)
        # Processing the call arguments (line 273)
        
        # Obtaining an instance of the builtin type 'list' (line 273)
        list_352675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 273)
        # Adding element type (line 273)
        float_352676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 21), list_352675, float_352676)
        # Adding element type (line 273)
        float_352677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 27), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 21), list_352675, float_352677)
        # Adding element type (line 273)
        float_352678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 32), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 21), list_352675, float_352678)
        # Adding element type (line 273)
        float_352679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 21), list_352675, float_352679)
        
        # Processing the call keyword arguments (line 273)
        kwargs_352680 = {}
        # Getting the type of 'np' (line 273)
        np_352673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 273)
        array_352674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 12), np_352673, 'array')
        # Calling array(args, kwargs) (line 273)
        array_call_result_352681 = invoke(stypy.reporting.localization.Localization(__file__, 273, 12), array_352674, *[list_352675], **kwargs_352680)
        
        # Assigning a type to the variable 'p' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'p', array_call_result_352681)
        
        # Assigning a Call to a Name (line 274):
        
        # Assigning a Call to a Name (line 274):
        
        # Call to linspace(...): (line 274)
        # Processing the call arguments (line 274)
        int_352684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 24), 'int')
        float_352685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 27), 'float')
        int_352686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 32), 'int')
        # Processing the call keyword arguments (line 274)
        kwargs_352687 = {}
        # Getting the type of 'np' (line 274)
        np_352682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 274)
        linspace_352683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 12), np_352682, 'linspace')
        # Calling linspace(args, kwargs) (line 274)
        linspace_call_result_352688 = invoke(stypy.reporting.localization.Localization(__file__, 274, 12), linspace_352683, *[int_352684, float_352685, int_352686], **kwargs_352687)
        
        # Assigning a type to the variable 't' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 't', linspace_call_result_352688)
        
        # Assigning a Call to a Name (line 275):
        
        # Assigning a Call to a Name (line 275):
        
        # Call to _sweep_poly_phase(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 't' (line 275)
        t_352691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 44), 't', False)
        # Getting the type of 'p' (line 275)
        p_352692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 47), 'p', False)
        # Processing the call keyword arguments (line 275)
        kwargs_352693 = {}
        # Getting the type of 'waveforms' (line 275)
        waveforms_352689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'waveforms', False)
        # Obtaining the member '_sweep_poly_phase' of a type (line 275)
        _sweep_poly_phase_352690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 16), waveforms_352689, '_sweep_poly_phase')
        # Calling _sweep_poly_phase(args, kwargs) (line 275)
        _sweep_poly_phase_call_result_352694 = invoke(stypy.reporting.localization.Localization(__file__, 275, 16), _sweep_poly_phase_352690, *[t_352691, p_352692], **kwargs_352693)
        
        # Assigning a type to the variable 'phase' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'phase', _sweep_poly_phase_call_result_352694)
        
        # Assigning a Call to a Tuple (line 276):
        
        # Assigning a Subscript to a Name (line 276):
        
        # Obtaining the type of the subscript
        int_352695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 8), 'int')
        
        # Call to compute_frequency(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 't' (line 276)
        t_352697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 34), 't', False)
        # Getting the type of 'phase' (line 276)
        phase_352698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 37), 'phase', False)
        # Processing the call keyword arguments (line 276)
        kwargs_352699 = {}
        # Getting the type of 'compute_frequency' (line 276)
        compute_frequency_352696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 276)
        compute_frequency_call_result_352700 = invoke(stypy.reporting.localization.Localization(__file__, 276, 16), compute_frequency_352696, *[t_352697, phase_352698], **kwargs_352699)
        
        # Obtaining the member '__getitem__' of a type (line 276)
        getitem___352701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), compute_frequency_call_result_352700, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 276)
        subscript_call_result_352702 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), getitem___352701, int_352695)
        
        # Assigning a type to the variable 'tuple_var_assignment_351440' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'tuple_var_assignment_351440', subscript_call_result_352702)
        
        # Assigning a Subscript to a Name (line 276):
        
        # Obtaining the type of the subscript
        int_352703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 8), 'int')
        
        # Call to compute_frequency(...): (line 276)
        # Processing the call arguments (line 276)
        # Getting the type of 't' (line 276)
        t_352705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 34), 't', False)
        # Getting the type of 'phase' (line 276)
        phase_352706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 37), 'phase', False)
        # Processing the call keyword arguments (line 276)
        kwargs_352707 = {}
        # Getting the type of 'compute_frequency' (line 276)
        compute_frequency_352704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 276)
        compute_frequency_call_result_352708 = invoke(stypy.reporting.localization.Localization(__file__, 276, 16), compute_frequency_352704, *[t_352705, phase_352706], **kwargs_352707)
        
        # Obtaining the member '__getitem__' of a type (line 276)
        getitem___352709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), compute_frequency_call_result_352708, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 276)
        subscript_call_result_352710 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), getitem___352709, int_352703)
        
        # Assigning a type to the variable 'tuple_var_assignment_351441' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'tuple_var_assignment_351441', subscript_call_result_352710)
        
        # Assigning a Name to a Name (line 276):
        # Getting the type of 'tuple_var_assignment_351440' (line 276)
        tuple_var_assignment_351440_352711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'tuple_var_assignment_351440')
        # Assigning a type to the variable 'tf' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'tf', tuple_var_assignment_351440_352711)
        
        # Assigning a Name to a Name (line 276):
        # Getting the type of 'tuple_var_assignment_351441' (line 276)
        tuple_var_assignment_351441_352712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'tuple_var_assignment_351441')
        # Assigning a type to the variable 'f' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'f', tuple_var_assignment_351441_352712)
        
        # Assigning a Call to a Name (line 277):
        
        # Assigning a Call to a Name (line 277):
        
        # Call to (...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'tf' (line 277)
        tf_352718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 32), 'tf', False)
        # Processing the call keyword arguments (line 277)
        kwargs_352719 = {}
        
        # Call to poly1d(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'p' (line 277)
        p_352715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 29), 'p', False)
        # Processing the call keyword arguments (line 277)
        kwargs_352716 = {}
        # Getting the type of 'np' (line 277)
        np_352713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 19), 'np', False)
        # Obtaining the member 'poly1d' of a type (line 277)
        poly1d_352714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 19), np_352713, 'poly1d')
        # Calling poly1d(args, kwargs) (line 277)
        poly1d_call_result_352717 = invoke(stypy.reporting.localization.Localization(__file__, 277, 19), poly1d_352714, *[p_352715], **kwargs_352716)
        
        # Calling (args, kwargs) (line 277)
        _call_result_352720 = invoke(stypy.reporting.localization.Localization(__file__, 277, 19), poly1d_call_result_352717, *[tf_352718], **kwargs_352719)
        
        # Assigning a type to the variable 'expected' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'expected', _call_result_352720)
        
        # Assigning a Call to a Name (line 278):
        
        # Assigning a Call to a Name (line 278):
        
        # Call to max(...): (line 278)
        # Processing the call arguments (line 278)
        
        # Call to abs(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'f' (line 278)
        f_352725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 31), 'f', False)
        # Getting the type of 'expected' (line 278)
        expected_352726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 35), 'expected', False)
        # Applying the binary operator '-' (line 278)
        result_sub_352727 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 31), '-', f_352725, expected_352726)
        
        # Processing the call keyword arguments (line 278)
        kwargs_352728 = {}
        # Getting the type of 'np' (line 278)
        np_352723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 24), 'np', False)
        # Obtaining the member 'abs' of a type (line 278)
        abs_352724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 24), np_352723, 'abs')
        # Calling abs(args, kwargs) (line 278)
        abs_call_result_352729 = invoke(stypy.reporting.localization.Localization(__file__, 278, 24), abs_352724, *[result_sub_352727], **kwargs_352728)
        
        # Processing the call keyword arguments (line 278)
        kwargs_352730 = {}
        # Getting the type of 'np' (line 278)
        np_352721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 17), 'np', False)
        # Obtaining the member 'max' of a type (line 278)
        max_352722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 17), np_352721, 'max')
        # Calling max(args, kwargs) (line 278)
        max_call_result_352731 = invoke(stypy.reporting.localization.Localization(__file__, 278, 17), max_352722, *[abs_call_result_352729], **kwargs_352730)
        
        # Assigning a type to the variable 'abserr' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'abserr', max_call_result_352731)
        
        # Call to assert_(...): (line 279)
        # Processing the call arguments (line 279)
        
        # Getting the type of 'abserr' (line 279)
        abserr_352733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 16), 'abserr', False)
        float_352734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 25), 'float')
        # Applying the binary operator '<' (line 279)
        result_lt_352735 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 16), '<', abserr_352733, float_352734)
        
        # Processing the call keyword arguments (line 279)
        kwargs_352736 = {}
        # Getting the type of 'assert_' (line 279)
        assert__352732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 279)
        assert__call_result_352737 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), assert__352732, *[result_lt_352735], **kwargs_352736)
        
        
        # ################# End of 'test_sweep_poly_cubic2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sweep_poly_cubic2' in the type store
        # Getting the type of 'stypy_return_type' (line 271)
        stypy_return_type_352738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352738)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sweep_poly_cubic2'
        return stypy_return_type_352738


    @norecursion
    def test_sweep_poly_cubic3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sweep_poly_cubic3'
        module_type_store = module_type_store.open_function_context('test_sweep_poly_cubic3', 281, 4, False)
        # Assigning a type to the variable 'self' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestSweepPoly.test_sweep_poly_cubic3.__dict__.__setitem__('stypy_localization', localization)
        TestSweepPoly.test_sweep_poly_cubic3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestSweepPoly.test_sweep_poly_cubic3.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestSweepPoly.test_sweep_poly_cubic3.__dict__.__setitem__('stypy_function_name', 'TestSweepPoly.test_sweep_poly_cubic3')
        TestSweepPoly.test_sweep_poly_cubic3.__dict__.__setitem__('stypy_param_names_list', [])
        TestSweepPoly.test_sweep_poly_cubic3.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestSweepPoly.test_sweep_poly_cubic3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestSweepPoly.test_sweep_poly_cubic3.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestSweepPoly.test_sweep_poly_cubic3.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestSweepPoly.test_sweep_poly_cubic3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestSweepPoly.test_sweep_poly_cubic3.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSweepPoly.test_sweep_poly_cubic3', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sweep_poly_cubic3', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sweep_poly_cubic3(...)' code ##################

        str_352739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 8), 'str', 'Use a list of coefficients instead of a poly1d.')
        
        # Assigning a List to a Name (line 283):
        
        # Assigning a List to a Name (line 283):
        
        # Obtaining an instance of the builtin type 'list' (line 283)
        list_352740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 283)
        # Adding element type (line 283)
        float_352741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 13), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 12), list_352740, float_352741)
        # Adding element type (line 283)
        float_352742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 12), list_352740, float_352742)
        # Adding element type (line 283)
        float_352743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 23), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 12), list_352740, float_352743)
        # Adding element type (line 283)
        float_352744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 12), list_352740, float_352744)
        
        # Assigning a type to the variable 'p' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'p', list_352740)
        
        # Assigning a Call to a Name (line 284):
        
        # Assigning a Call to a Name (line 284):
        
        # Call to linspace(...): (line 284)
        # Processing the call arguments (line 284)
        int_352747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 24), 'int')
        float_352748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 27), 'float')
        int_352749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 32), 'int')
        # Processing the call keyword arguments (line 284)
        kwargs_352750 = {}
        # Getting the type of 'np' (line 284)
        np_352745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'np', False)
        # Obtaining the member 'linspace' of a type (line 284)
        linspace_352746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 12), np_352745, 'linspace')
        # Calling linspace(args, kwargs) (line 284)
        linspace_call_result_352751 = invoke(stypy.reporting.localization.Localization(__file__, 284, 12), linspace_352746, *[int_352747, float_352748, int_352749], **kwargs_352750)
        
        # Assigning a type to the variable 't' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 't', linspace_call_result_352751)
        
        # Assigning a Call to a Name (line 285):
        
        # Assigning a Call to a Name (line 285):
        
        # Call to _sweep_poly_phase(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 't' (line 285)
        t_352754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 44), 't', False)
        # Getting the type of 'p' (line 285)
        p_352755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 47), 'p', False)
        # Processing the call keyword arguments (line 285)
        kwargs_352756 = {}
        # Getting the type of 'waveforms' (line 285)
        waveforms_352752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 16), 'waveforms', False)
        # Obtaining the member '_sweep_poly_phase' of a type (line 285)
        _sweep_poly_phase_352753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 16), waveforms_352752, '_sweep_poly_phase')
        # Calling _sweep_poly_phase(args, kwargs) (line 285)
        _sweep_poly_phase_call_result_352757 = invoke(stypy.reporting.localization.Localization(__file__, 285, 16), _sweep_poly_phase_352753, *[t_352754, p_352755], **kwargs_352756)
        
        # Assigning a type to the variable 'phase' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'phase', _sweep_poly_phase_call_result_352757)
        
        # Assigning a Call to a Tuple (line 286):
        
        # Assigning a Subscript to a Name (line 286):
        
        # Obtaining the type of the subscript
        int_352758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 8), 'int')
        
        # Call to compute_frequency(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 't' (line 286)
        t_352760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 34), 't', False)
        # Getting the type of 'phase' (line 286)
        phase_352761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 37), 'phase', False)
        # Processing the call keyword arguments (line 286)
        kwargs_352762 = {}
        # Getting the type of 'compute_frequency' (line 286)
        compute_frequency_352759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 286)
        compute_frequency_call_result_352763 = invoke(stypy.reporting.localization.Localization(__file__, 286, 16), compute_frequency_352759, *[t_352760, phase_352761], **kwargs_352762)
        
        # Obtaining the member '__getitem__' of a type (line 286)
        getitem___352764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), compute_frequency_call_result_352763, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 286)
        subscript_call_result_352765 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), getitem___352764, int_352758)
        
        # Assigning a type to the variable 'tuple_var_assignment_351442' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_351442', subscript_call_result_352765)
        
        # Assigning a Subscript to a Name (line 286):
        
        # Obtaining the type of the subscript
        int_352766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 8), 'int')
        
        # Call to compute_frequency(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 't' (line 286)
        t_352768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 34), 't', False)
        # Getting the type of 'phase' (line 286)
        phase_352769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 37), 'phase', False)
        # Processing the call keyword arguments (line 286)
        kwargs_352770 = {}
        # Getting the type of 'compute_frequency' (line 286)
        compute_frequency_352767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'compute_frequency', False)
        # Calling compute_frequency(args, kwargs) (line 286)
        compute_frequency_call_result_352771 = invoke(stypy.reporting.localization.Localization(__file__, 286, 16), compute_frequency_352767, *[t_352768, phase_352769], **kwargs_352770)
        
        # Obtaining the member '__getitem__' of a type (line 286)
        getitem___352772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), compute_frequency_call_result_352771, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 286)
        subscript_call_result_352773 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), getitem___352772, int_352766)
        
        # Assigning a type to the variable 'tuple_var_assignment_351443' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_351443', subscript_call_result_352773)
        
        # Assigning a Name to a Name (line 286):
        # Getting the type of 'tuple_var_assignment_351442' (line 286)
        tuple_var_assignment_351442_352774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_351442')
        # Assigning a type to the variable 'tf' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tf', tuple_var_assignment_351442_352774)
        
        # Assigning a Name to a Name (line 286):
        # Getting the type of 'tuple_var_assignment_351443' (line 286)
        tuple_var_assignment_351443_352775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'tuple_var_assignment_351443')
        # Assigning a type to the variable 'f' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'f', tuple_var_assignment_351443_352775)
        
        # Assigning a Call to a Name (line 287):
        
        # Assigning a Call to a Name (line 287):
        
        # Call to (...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'tf' (line 287)
        tf_352781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 32), 'tf', False)
        # Processing the call keyword arguments (line 287)
        kwargs_352782 = {}
        
        # Call to poly1d(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'p' (line 287)
        p_352778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 29), 'p', False)
        # Processing the call keyword arguments (line 287)
        kwargs_352779 = {}
        # Getting the type of 'np' (line 287)
        np_352776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 19), 'np', False)
        # Obtaining the member 'poly1d' of a type (line 287)
        poly1d_352777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 19), np_352776, 'poly1d')
        # Calling poly1d(args, kwargs) (line 287)
        poly1d_call_result_352780 = invoke(stypy.reporting.localization.Localization(__file__, 287, 19), poly1d_352777, *[p_352778], **kwargs_352779)
        
        # Calling (args, kwargs) (line 287)
        _call_result_352783 = invoke(stypy.reporting.localization.Localization(__file__, 287, 19), poly1d_call_result_352780, *[tf_352781], **kwargs_352782)
        
        # Assigning a type to the variable 'expected' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'expected', _call_result_352783)
        
        # Assigning a Call to a Name (line 288):
        
        # Assigning a Call to a Name (line 288):
        
        # Call to max(...): (line 288)
        # Processing the call arguments (line 288)
        
        # Call to abs(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'f' (line 288)
        f_352788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 31), 'f', False)
        # Getting the type of 'expected' (line 288)
        expected_352789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 35), 'expected', False)
        # Applying the binary operator '-' (line 288)
        result_sub_352790 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 31), '-', f_352788, expected_352789)
        
        # Processing the call keyword arguments (line 288)
        kwargs_352791 = {}
        # Getting the type of 'np' (line 288)
        np_352786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'np', False)
        # Obtaining the member 'abs' of a type (line 288)
        abs_352787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 24), np_352786, 'abs')
        # Calling abs(args, kwargs) (line 288)
        abs_call_result_352792 = invoke(stypy.reporting.localization.Localization(__file__, 288, 24), abs_352787, *[result_sub_352790], **kwargs_352791)
        
        # Processing the call keyword arguments (line 288)
        kwargs_352793 = {}
        # Getting the type of 'np' (line 288)
        np_352784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 17), 'np', False)
        # Obtaining the member 'max' of a type (line 288)
        max_352785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 17), np_352784, 'max')
        # Calling max(args, kwargs) (line 288)
        max_call_result_352794 = invoke(stypy.reporting.localization.Localization(__file__, 288, 17), max_352785, *[abs_call_result_352792], **kwargs_352793)
        
        # Assigning a type to the variable 'abserr' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'abserr', max_call_result_352794)
        
        # Call to assert_(...): (line 289)
        # Processing the call arguments (line 289)
        
        # Getting the type of 'abserr' (line 289)
        abserr_352796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'abserr', False)
        float_352797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 25), 'float')
        # Applying the binary operator '<' (line 289)
        result_lt_352798 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 16), '<', abserr_352796, float_352797)
        
        # Processing the call keyword arguments (line 289)
        kwargs_352799 = {}
        # Getting the type of 'assert_' (line 289)
        assert__352795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 289)
        assert__call_result_352800 = invoke(stypy.reporting.localization.Localization(__file__, 289, 8), assert__352795, *[result_lt_352798], **kwargs_352799)
        
        
        # ################# End of 'test_sweep_poly_cubic3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sweep_poly_cubic3' in the type store
        # Getting the type of 'stypy_return_type' (line 281)
        stypy_return_type_352801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352801)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sweep_poly_cubic3'
        return stypy_return_type_352801


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 224, 0, False)
        # Assigning a type to the variable 'self' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestSweepPoly.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'TestSweepPoly' (line 224)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), 'TestSweepPoly', TestSweepPoly)
# Declaration of the 'TestGaussPulse' class

class TestGaussPulse(object, ):

    @norecursion
    def test_integer_fc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_integer_fc'
        module_type_store = module_type_store.open_function_context('test_integer_fc', 294, 4, False)
        # Assigning a type to the variable 'self' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGaussPulse.test_integer_fc.__dict__.__setitem__('stypy_localization', localization)
        TestGaussPulse.test_integer_fc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGaussPulse.test_integer_fc.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGaussPulse.test_integer_fc.__dict__.__setitem__('stypy_function_name', 'TestGaussPulse.test_integer_fc')
        TestGaussPulse.test_integer_fc.__dict__.__setitem__('stypy_param_names_list', [])
        TestGaussPulse.test_integer_fc.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGaussPulse.test_integer_fc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGaussPulse.test_integer_fc.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGaussPulse.test_integer_fc.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGaussPulse.test_integer_fc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGaussPulse.test_integer_fc.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGaussPulse.test_integer_fc', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_integer_fc', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_integer_fc(...)' code ##################

        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to gausspulse(...): (line 295)
        # Processing the call arguments (line 295)
        str_352804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 44), 'str', 'cutoff')
        # Processing the call keyword arguments (line 295)
        float_352805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 57), 'float')
        keyword_352806 = float_352805
        kwargs_352807 = {'fc': keyword_352806}
        # Getting the type of 'waveforms' (line 295)
        waveforms_352802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 23), 'waveforms', False)
        # Obtaining the member 'gausspulse' of a type (line 295)
        gausspulse_352803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 23), waveforms_352802, 'gausspulse')
        # Calling gausspulse(args, kwargs) (line 295)
        gausspulse_call_result_352808 = invoke(stypy.reporting.localization.Localization(__file__, 295, 23), gausspulse_352803, *[str_352804], **kwargs_352807)
        
        # Assigning a type to the variable 'float_result' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'float_result', gausspulse_call_result_352808)
        
        # Assigning a Call to a Name (line 296):
        
        # Assigning a Call to a Name (line 296):
        
        # Call to gausspulse(...): (line 296)
        # Processing the call arguments (line 296)
        str_352811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 42), 'str', 'cutoff')
        # Processing the call keyword arguments (line 296)
        int_352812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 55), 'int')
        keyword_352813 = int_352812
        kwargs_352814 = {'fc': keyword_352813}
        # Getting the type of 'waveforms' (line 296)
        waveforms_352809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 21), 'waveforms', False)
        # Obtaining the member 'gausspulse' of a type (line 296)
        gausspulse_352810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 21), waveforms_352809, 'gausspulse')
        # Calling gausspulse(args, kwargs) (line 296)
        gausspulse_call_result_352815 = invoke(stypy.reporting.localization.Localization(__file__, 296, 21), gausspulse_352810, *[str_352811], **kwargs_352814)
        
        # Assigning a type to the variable 'int_result' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'int_result', gausspulse_call_result_352815)
        
        # Assigning a Str to a Name (line 297):
        
        # Assigning a Str to a Name (line 297):
        str_352816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 18), 'str', "Integer input 'fc=1000' gives wrong result")
        # Assigning a type to the variable 'err_msg' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'err_msg', str_352816)
        
        # Call to assert_equal(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'int_result' (line 298)
        int_result_352818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 21), 'int_result', False)
        # Getting the type of 'float_result' (line 298)
        float_result_352819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 33), 'float_result', False)
        # Processing the call keyword arguments (line 298)
        # Getting the type of 'err_msg' (line 298)
        err_msg_352820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 55), 'err_msg', False)
        keyword_352821 = err_msg_352820
        kwargs_352822 = {'err_msg': keyword_352821}
        # Getting the type of 'assert_equal' (line 298)
        assert_equal_352817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 298)
        assert_equal_call_result_352823 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), assert_equal_352817, *[int_result_352818, float_result_352819], **kwargs_352822)
        
        
        # ################# End of 'test_integer_fc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_integer_fc' in the type store
        # Getting the type of 'stypy_return_type' (line 294)
        stypy_return_type_352824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352824)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_integer_fc'
        return stypy_return_type_352824


    @norecursion
    def test_integer_bw(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_integer_bw'
        module_type_store = module_type_store.open_function_context('test_integer_bw', 300, 4, False)
        # Assigning a type to the variable 'self' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGaussPulse.test_integer_bw.__dict__.__setitem__('stypy_localization', localization)
        TestGaussPulse.test_integer_bw.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGaussPulse.test_integer_bw.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGaussPulse.test_integer_bw.__dict__.__setitem__('stypy_function_name', 'TestGaussPulse.test_integer_bw')
        TestGaussPulse.test_integer_bw.__dict__.__setitem__('stypy_param_names_list', [])
        TestGaussPulse.test_integer_bw.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGaussPulse.test_integer_bw.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGaussPulse.test_integer_bw.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGaussPulse.test_integer_bw.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGaussPulse.test_integer_bw.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGaussPulse.test_integer_bw.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGaussPulse.test_integer_bw', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_integer_bw', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_integer_bw(...)' code ##################

        
        # Assigning a Call to a Name (line 301):
        
        # Assigning a Call to a Name (line 301):
        
        # Call to gausspulse(...): (line 301)
        # Processing the call arguments (line 301)
        str_352827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 44), 'str', 'cutoff')
        # Processing the call keyword arguments (line 301)
        float_352828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 57), 'float')
        keyword_352829 = float_352828
        kwargs_352830 = {'bw': keyword_352829}
        # Getting the type of 'waveforms' (line 301)
        waveforms_352825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 23), 'waveforms', False)
        # Obtaining the member 'gausspulse' of a type (line 301)
        gausspulse_352826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 23), waveforms_352825, 'gausspulse')
        # Calling gausspulse(args, kwargs) (line 301)
        gausspulse_call_result_352831 = invoke(stypy.reporting.localization.Localization(__file__, 301, 23), gausspulse_352826, *[str_352827], **kwargs_352830)
        
        # Assigning a type to the variable 'float_result' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'float_result', gausspulse_call_result_352831)
        
        # Assigning a Call to a Name (line 302):
        
        # Assigning a Call to a Name (line 302):
        
        # Call to gausspulse(...): (line 302)
        # Processing the call arguments (line 302)
        str_352834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 42), 'str', 'cutoff')
        # Processing the call keyword arguments (line 302)
        int_352835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 55), 'int')
        keyword_352836 = int_352835
        kwargs_352837 = {'bw': keyword_352836}
        # Getting the type of 'waveforms' (line 302)
        waveforms_352832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 21), 'waveforms', False)
        # Obtaining the member 'gausspulse' of a type (line 302)
        gausspulse_352833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 21), waveforms_352832, 'gausspulse')
        # Calling gausspulse(args, kwargs) (line 302)
        gausspulse_call_result_352838 = invoke(stypy.reporting.localization.Localization(__file__, 302, 21), gausspulse_352833, *[str_352834], **kwargs_352837)
        
        # Assigning a type to the variable 'int_result' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'int_result', gausspulse_call_result_352838)
        
        # Assigning a Str to a Name (line 303):
        
        # Assigning a Str to a Name (line 303):
        str_352839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 18), 'str', "Integer input 'bw=1' gives wrong result")
        # Assigning a type to the variable 'err_msg' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'err_msg', str_352839)
        
        # Call to assert_equal(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'int_result' (line 304)
        int_result_352841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 21), 'int_result', False)
        # Getting the type of 'float_result' (line 304)
        float_result_352842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 33), 'float_result', False)
        # Processing the call keyword arguments (line 304)
        # Getting the type of 'err_msg' (line 304)
        err_msg_352843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 55), 'err_msg', False)
        keyword_352844 = err_msg_352843
        kwargs_352845 = {'err_msg': keyword_352844}
        # Getting the type of 'assert_equal' (line 304)
        assert_equal_352840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 304)
        assert_equal_call_result_352846 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), assert_equal_352840, *[int_result_352841, float_result_352842], **kwargs_352845)
        
        
        # ################# End of 'test_integer_bw(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_integer_bw' in the type store
        # Getting the type of 'stypy_return_type' (line 300)
        stypy_return_type_352847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352847)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_integer_bw'
        return stypy_return_type_352847


    @norecursion
    def test_integer_bwr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_integer_bwr'
        module_type_store = module_type_store.open_function_context('test_integer_bwr', 306, 4, False)
        # Assigning a type to the variable 'self' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGaussPulse.test_integer_bwr.__dict__.__setitem__('stypy_localization', localization)
        TestGaussPulse.test_integer_bwr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGaussPulse.test_integer_bwr.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGaussPulse.test_integer_bwr.__dict__.__setitem__('stypy_function_name', 'TestGaussPulse.test_integer_bwr')
        TestGaussPulse.test_integer_bwr.__dict__.__setitem__('stypy_param_names_list', [])
        TestGaussPulse.test_integer_bwr.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGaussPulse.test_integer_bwr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGaussPulse.test_integer_bwr.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGaussPulse.test_integer_bwr.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGaussPulse.test_integer_bwr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGaussPulse.test_integer_bwr.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGaussPulse.test_integer_bwr', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_integer_bwr', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_integer_bwr(...)' code ##################

        
        # Assigning a Call to a Name (line 307):
        
        # Assigning a Call to a Name (line 307):
        
        # Call to gausspulse(...): (line 307)
        # Processing the call arguments (line 307)
        str_352850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 44), 'str', 'cutoff')
        # Processing the call keyword arguments (line 307)
        float_352851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 58), 'float')
        keyword_352852 = float_352851
        kwargs_352853 = {'bwr': keyword_352852}
        # Getting the type of 'waveforms' (line 307)
        waveforms_352848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 23), 'waveforms', False)
        # Obtaining the member 'gausspulse' of a type (line 307)
        gausspulse_352849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 23), waveforms_352848, 'gausspulse')
        # Calling gausspulse(args, kwargs) (line 307)
        gausspulse_call_result_352854 = invoke(stypy.reporting.localization.Localization(__file__, 307, 23), gausspulse_352849, *[str_352850], **kwargs_352853)
        
        # Assigning a type to the variable 'float_result' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'float_result', gausspulse_call_result_352854)
        
        # Assigning a Call to a Name (line 308):
        
        # Assigning a Call to a Name (line 308):
        
        # Call to gausspulse(...): (line 308)
        # Processing the call arguments (line 308)
        str_352857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 42), 'str', 'cutoff')
        # Processing the call keyword arguments (line 308)
        int_352858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 56), 'int')
        keyword_352859 = int_352858
        kwargs_352860 = {'bwr': keyword_352859}
        # Getting the type of 'waveforms' (line 308)
        waveforms_352855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 21), 'waveforms', False)
        # Obtaining the member 'gausspulse' of a type (line 308)
        gausspulse_352856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 21), waveforms_352855, 'gausspulse')
        # Calling gausspulse(args, kwargs) (line 308)
        gausspulse_call_result_352861 = invoke(stypy.reporting.localization.Localization(__file__, 308, 21), gausspulse_352856, *[str_352857], **kwargs_352860)
        
        # Assigning a type to the variable 'int_result' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'int_result', gausspulse_call_result_352861)
        
        # Assigning a Str to a Name (line 309):
        
        # Assigning a Str to a Name (line 309):
        str_352862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 18), 'str', "Integer input 'bwr=-6' gives wrong result")
        # Assigning a type to the variable 'err_msg' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'err_msg', str_352862)
        
        # Call to assert_equal(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'int_result' (line 310)
        int_result_352864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 21), 'int_result', False)
        # Getting the type of 'float_result' (line 310)
        float_result_352865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 33), 'float_result', False)
        # Processing the call keyword arguments (line 310)
        # Getting the type of 'err_msg' (line 310)
        err_msg_352866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 55), 'err_msg', False)
        keyword_352867 = err_msg_352866
        kwargs_352868 = {'err_msg': keyword_352867}
        # Getting the type of 'assert_equal' (line 310)
        assert_equal_352863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 310)
        assert_equal_call_result_352869 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), assert_equal_352863, *[int_result_352864, float_result_352865], **kwargs_352868)
        
        
        # ################# End of 'test_integer_bwr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_integer_bwr' in the type store
        # Getting the type of 'stypy_return_type' (line 306)
        stypy_return_type_352870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352870)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_integer_bwr'
        return stypy_return_type_352870


    @norecursion
    def test_integer_tpr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_integer_tpr'
        module_type_store = module_type_store.open_function_context('test_integer_tpr', 312, 4, False)
        # Assigning a type to the variable 'self' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGaussPulse.test_integer_tpr.__dict__.__setitem__('stypy_localization', localization)
        TestGaussPulse.test_integer_tpr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGaussPulse.test_integer_tpr.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGaussPulse.test_integer_tpr.__dict__.__setitem__('stypy_function_name', 'TestGaussPulse.test_integer_tpr')
        TestGaussPulse.test_integer_tpr.__dict__.__setitem__('stypy_param_names_list', [])
        TestGaussPulse.test_integer_tpr.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGaussPulse.test_integer_tpr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGaussPulse.test_integer_tpr.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGaussPulse.test_integer_tpr.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGaussPulse.test_integer_tpr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGaussPulse.test_integer_tpr.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGaussPulse.test_integer_tpr', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_integer_tpr', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_integer_tpr(...)' code ##################

        
        # Assigning a Call to a Name (line 313):
        
        # Assigning a Call to a Name (line 313):
        
        # Call to gausspulse(...): (line 313)
        # Processing the call arguments (line 313)
        str_352873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 44), 'str', 'cutoff')
        # Processing the call keyword arguments (line 313)
        float_352874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 58), 'float')
        keyword_352875 = float_352874
        kwargs_352876 = {'tpr': keyword_352875}
        # Getting the type of 'waveforms' (line 313)
        waveforms_352871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 23), 'waveforms', False)
        # Obtaining the member 'gausspulse' of a type (line 313)
        gausspulse_352872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 23), waveforms_352871, 'gausspulse')
        # Calling gausspulse(args, kwargs) (line 313)
        gausspulse_call_result_352877 = invoke(stypy.reporting.localization.Localization(__file__, 313, 23), gausspulse_352872, *[str_352873], **kwargs_352876)
        
        # Assigning a type to the variable 'float_result' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'float_result', gausspulse_call_result_352877)
        
        # Assigning a Call to a Name (line 314):
        
        # Assigning a Call to a Name (line 314):
        
        # Call to gausspulse(...): (line 314)
        # Processing the call arguments (line 314)
        str_352880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 42), 'str', 'cutoff')
        # Processing the call keyword arguments (line 314)
        int_352881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 56), 'int')
        keyword_352882 = int_352881
        kwargs_352883 = {'tpr': keyword_352882}
        # Getting the type of 'waveforms' (line 314)
        waveforms_352878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 21), 'waveforms', False)
        # Obtaining the member 'gausspulse' of a type (line 314)
        gausspulse_352879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 21), waveforms_352878, 'gausspulse')
        # Calling gausspulse(args, kwargs) (line 314)
        gausspulse_call_result_352884 = invoke(stypy.reporting.localization.Localization(__file__, 314, 21), gausspulse_352879, *[str_352880], **kwargs_352883)
        
        # Assigning a type to the variable 'int_result' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'int_result', gausspulse_call_result_352884)
        
        # Assigning a Str to a Name (line 315):
        
        # Assigning a Str to a Name (line 315):
        str_352885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 18), 'str', "Integer input 'tpr=-60' gives wrong result")
        # Assigning a type to the variable 'err_msg' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'err_msg', str_352885)
        
        # Call to assert_equal(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'int_result' (line 316)
        int_result_352887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 21), 'int_result', False)
        # Getting the type of 'float_result' (line 316)
        float_result_352888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 33), 'float_result', False)
        # Processing the call keyword arguments (line 316)
        # Getting the type of 'err_msg' (line 316)
        err_msg_352889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 55), 'err_msg', False)
        keyword_352890 = err_msg_352889
        kwargs_352891 = {'err_msg': keyword_352890}
        # Getting the type of 'assert_equal' (line 316)
        assert_equal_352886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 316)
        assert_equal_call_result_352892 = invoke(stypy.reporting.localization.Localization(__file__, 316, 8), assert_equal_352886, *[int_result_352887, float_result_352888], **kwargs_352891)
        
        
        # ################# End of 'test_integer_tpr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_integer_tpr' in the type store
        # Getting the type of 'stypy_return_type' (line 312)
        stypy_return_type_352893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352893)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_integer_tpr'
        return stypy_return_type_352893


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 292, 0, False)
        # Assigning a type to the variable 'self' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGaussPulse.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'TestGaussPulse' (line 292)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 0), 'TestGaussPulse', TestGaussPulse)
# Declaration of the 'TestUnitImpulse' class

class TestUnitImpulse(object, ):

    @norecursion
    def test_no_index(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_no_index'
        module_type_store = module_type_store.open_function_context('test_no_index', 321, 4, False)
        # Assigning a type to the variable 'self' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestUnitImpulse.test_no_index.__dict__.__setitem__('stypy_localization', localization)
        TestUnitImpulse.test_no_index.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestUnitImpulse.test_no_index.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestUnitImpulse.test_no_index.__dict__.__setitem__('stypy_function_name', 'TestUnitImpulse.test_no_index')
        TestUnitImpulse.test_no_index.__dict__.__setitem__('stypy_param_names_list', [])
        TestUnitImpulse.test_no_index.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestUnitImpulse.test_no_index.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestUnitImpulse.test_no_index.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestUnitImpulse.test_no_index.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestUnitImpulse.test_no_index.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestUnitImpulse.test_no_index.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestUnitImpulse.test_no_index', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_no_index', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_no_index(...)' code ##################

        
        # Call to assert_array_equal(...): (line 322)
        # Processing the call arguments (line 322)
        
        # Call to unit_impulse(...): (line 322)
        # Processing the call arguments (line 322)
        int_352897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 50), 'int')
        # Processing the call keyword arguments (line 322)
        kwargs_352898 = {}
        # Getting the type of 'waveforms' (line 322)
        waveforms_352895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 27), 'waveforms', False)
        # Obtaining the member 'unit_impulse' of a type (line 322)
        unit_impulse_352896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 27), waveforms_352895, 'unit_impulse')
        # Calling unit_impulse(args, kwargs) (line 322)
        unit_impulse_call_result_352899 = invoke(stypy.reporting.localization.Localization(__file__, 322, 27), unit_impulse_352896, *[int_352897], **kwargs_352898)
        
        
        # Obtaining an instance of the builtin type 'list' (line 322)
        list_352900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 322)
        # Adding element type (line 322)
        int_352901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 54), list_352900, int_352901)
        # Adding element type (line 322)
        int_352902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 54), list_352900, int_352902)
        # Adding element type (line 322)
        int_352903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 61), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 54), list_352900, int_352903)
        # Adding element type (line 322)
        int_352904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 54), list_352900, int_352904)
        # Adding element type (line 322)
        int_352905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 67), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 54), list_352900, int_352905)
        # Adding element type (line 322)
        int_352906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 70), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 54), list_352900, int_352906)
        # Adding element type (line 322)
        int_352907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 73), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 54), list_352900, int_352907)
        
        # Processing the call keyword arguments (line 322)
        kwargs_352908 = {}
        # Getting the type of 'assert_array_equal' (line 322)
        assert_array_equal_352894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 322)
        assert_array_equal_call_result_352909 = invoke(stypy.reporting.localization.Localization(__file__, 322, 8), assert_array_equal_352894, *[unit_impulse_call_result_352899, list_352900], **kwargs_352908)
        
        
        # Call to assert_array_equal(...): (line 323)
        # Processing the call arguments (line 323)
        
        # Call to unit_impulse(...): (line 323)
        # Processing the call arguments (line 323)
        
        # Obtaining an instance of the builtin type 'tuple' (line 323)
        tuple_352913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 323)
        # Adding element type (line 323)
        int_352914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 51), tuple_352913, int_352914)
        # Adding element type (line 323)
        int_352915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 51), tuple_352913, int_352915)
        
        # Processing the call keyword arguments (line 323)
        kwargs_352916 = {}
        # Getting the type of 'waveforms' (line 323)
        waveforms_352911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 27), 'waveforms', False)
        # Obtaining the member 'unit_impulse' of a type (line 323)
        unit_impulse_352912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 27), waveforms_352911, 'unit_impulse')
        # Calling unit_impulse(args, kwargs) (line 323)
        unit_impulse_call_result_352917 = invoke(stypy.reporting.localization.Localization(__file__, 323, 27), unit_impulse_352912, *[tuple_352913], **kwargs_352916)
        
        
        # Obtaining an instance of the builtin type 'list' (line 324)
        list_352918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 324)
        # Adding element type (line 324)
        
        # Obtaining an instance of the builtin type 'list' (line 324)
        list_352919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 324)
        # Adding element type (line 324)
        int_352920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 28), list_352919, int_352920)
        # Adding element type (line 324)
        int_352921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 28), list_352919, int_352921)
        # Adding element type (line 324)
        int_352922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 28), list_352919, int_352922)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 27), list_352918, list_352919)
        # Adding element type (line 324)
        
        # Obtaining an instance of the builtin type 'list' (line 324)
        list_352923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 324)
        # Adding element type (line 324)
        int_352924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 39), list_352923, int_352924)
        # Adding element type (line 324)
        int_352925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 39), list_352923, int_352925)
        # Adding element type (line 324)
        int_352926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 39), list_352923, int_352926)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 27), list_352918, list_352923)
        # Adding element type (line 324)
        
        # Obtaining an instance of the builtin type 'list' (line 324)
        list_352927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 324)
        # Adding element type (line 324)
        int_352928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 50), list_352927, int_352928)
        # Adding element type (line 324)
        int_352929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 50), list_352927, int_352929)
        # Adding element type (line 324)
        int_352930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 50), list_352927, int_352930)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 27), list_352918, list_352927)
        
        # Processing the call keyword arguments (line 323)
        kwargs_352931 = {}
        # Getting the type of 'assert_array_equal' (line 323)
        assert_array_equal_352910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 323)
        assert_array_equal_call_result_352932 = invoke(stypy.reporting.localization.Localization(__file__, 323, 8), assert_array_equal_352910, *[unit_impulse_call_result_352917, list_352918], **kwargs_352931)
        
        
        # ################# End of 'test_no_index(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_no_index' in the type store
        # Getting the type of 'stypy_return_type' (line 321)
        stypy_return_type_352933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_352933)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_no_index'
        return stypy_return_type_352933


    @norecursion
    def test_index(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_index'
        module_type_store = module_type_store.open_function_context('test_index', 326, 4, False)
        # Assigning a type to the variable 'self' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestUnitImpulse.test_index.__dict__.__setitem__('stypy_localization', localization)
        TestUnitImpulse.test_index.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestUnitImpulse.test_index.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestUnitImpulse.test_index.__dict__.__setitem__('stypy_function_name', 'TestUnitImpulse.test_index')
        TestUnitImpulse.test_index.__dict__.__setitem__('stypy_param_names_list', [])
        TestUnitImpulse.test_index.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestUnitImpulse.test_index.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestUnitImpulse.test_index.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestUnitImpulse.test_index.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestUnitImpulse.test_index.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestUnitImpulse.test_index.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestUnitImpulse.test_index', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_index', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_index(...)' code ##################

        
        # Call to assert_array_equal(...): (line 327)
        # Processing the call arguments (line 327)
        
        # Call to unit_impulse(...): (line 327)
        # Processing the call arguments (line 327)
        int_352937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 50), 'int')
        int_352938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 54), 'int')
        # Processing the call keyword arguments (line 327)
        kwargs_352939 = {}
        # Getting the type of 'waveforms' (line 327)
        waveforms_352935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 27), 'waveforms', False)
        # Obtaining the member 'unit_impulse' of a type (line 327)
        unit_impulse_352936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 27), waveforms_352935, 'unit_impulse')
        # Calling unit_impulse(args, kwargs) (line 327)
        unit_impulse_call_result_352940 = invoke(stypy.reporting.localization.Localization(__file__, 327, 27), unit_impulse_352936, *[int_352937, int_352938], **kwargs_352939)
        
        
        # Obtaining an instance of the builtin type 'list' (line 328)
        list_352941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 328)
        # Adding element type (line 328)
        int_352942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 27), list_352941, int_352942)
        # Adding element type (line 328)
        int_352943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 27), list_352941, int_352943)
        # Adding element type (line 328)
        int_352944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 27), list_352941, int_352944)
        # Adding element type (line 328)
        int_352945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 27), list_352941, int_352945)
        # Adding element type (line 328)
        int_352946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 27), list_352941, int_352946)
        # Adding element type (line 328)
        int_352947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 27), list_352941, int_352947)
        # Adding element type (line 328)
        int_352948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 27), list_352941, int_352948)
        # Adding element type (line 328)
        int_352949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 27), list_352941, int_352949)
        # Adding element type (line 328)
        int_352950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 27), list_352941, int_352950)
        # Adding element type (line 328)
        int_352951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 328, 27), list_352941, int_352951)
        
        # Processing the call keyword arguments (line 327)
        kwargs_352952 = {}
        # Getting the type of 'assert_array_equal' (line 327)
        assert_array_equal_352934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 327)
        assert_array_equal_call_result_352953 = invoke(stypy.reporting.localization.Localization(__file__, 327, 8), assert_array_equal_352934, *[unit_impulse_call_result_352940, list_352941], **kwargs_352952)
        
        
        # Call to assert_array_equal(...): (line 329)
        # Processing the call arguments (line 329)
        
        # Call to unit_impulse(...): (line 329)
        # Processing the call arguments (line 329)
        
        # Obtaining an instance of the builtin type 'tuple' (line 329)
        tuple_352957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 329)
        # Adding element type (line 329)
        int_352958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 51), tuple_352957, int_352958)
        # Adding element type (line 329)
        int_352959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 51), tuple_352957, int_352959)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 329)
        tuple_352960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 59), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 329)
        # Adding element type (line 329)
        int_352961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 59), tuple_352960, int_352961)
        # Adding element type (line 329)
        int_352962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 59), tuple_352960, int_352962)
        
        # Processing the call keyword arguments (line 329)
        kwargs_352963 = {}
        # Getting the type of 'waveforms' (line 329)
        waveforms_352955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 27), 'waveforms', False)
        # Obtaining the member 'unit_impulse' of a type (line 329)
        unit_impulse_352956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 27), waveforms_352955, 'unit_impulse')
        # Calling unit_impulse(args, kwargs) (line 329)
        unit_impulse_call_result_352964 = invoke(stypy.reporting.localization.Localization(__file__, 329, 27), unit_impulse_352956, *[tuple_352957, tuple_352960], **kwargs_352963)
        
        
        # Obtaining an instance of the builtin type 'list' (line 330)
        list_352965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 330)
        # Adding element type (line 330)
        
        # Obtaining an instance of the builtin type 'list' (line 330)
        list_352966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 330)
        # Adding element type (line 330)
        int_352967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 28), list_352966, int_352967)
        # Adding element type (line 330)
        int_352968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 28), list_352966, int_352968)
        # Adding element type (line 330)
        int_352969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 28), list_352966, int_352969)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 27), list_352965, list_352966)
        # Adding element type (line 330)
        
        # Obtaining an instance of the builtin type 'list' (line 330)
        list_352970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 330)
        # Adding element type (line 330)
        int_352971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 39), list_352970, int_352971)
        # Adding element type (line 330)
        int_352972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 39), list_352970, int_352972)
        # Adding element type (line 330)
        int_352973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 39), list_352970, int_352973)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 27), list_352965, list_352970)
        # Adding element type (line 330)
        
        # Obtaining an instance of the builtin type 'list' (line 330)
        list_352974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 330)
        # Adding element type (line 330)
        int_352975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 50), list_352974, int_352975)
        # Adding element type (line 330)
        int_352976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 50), list_352974, int_352976)
        # Adding element type (line 330)
        int_352977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 50), list_352974, int_352977)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 330, 27), list_352965, list_352974)
        
        # Processing the call keyword arguments (line 329)
        kwargs_352978 = {}
        # Getting the type of 'assert_array_equal' (line 329)
        assert_array_equal_352954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 329)
        assert_array_equal_call_result_352979 = invoke(stypy.reporting.localization.Localization(__file__, 329, 8), assert_array_equal_352954, *[unit_impulse_call_result_352964, list_352965], **kwargs_352978)
        
        
        # Assigning a Call to a Name (line 333):
        
        # Assigning a Call to a Name (line 333):
        
        # Call to unit_impulse(...): (line 333)
        # Processing the call arguments (line 333)
        
        # Obtaining an instance of the builtin type 'tuple' (line 333)
        tuple_352982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 333)
        # Adding element type (line 333)
        int_352983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 38), tuple_352982, int_352983)
        # Adding element type (line 333)
        int_352984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 38), tuple_352982, int_352984)
        
        int_352985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 45), 'int')
        # Processing the call keyword arguments (line 333)
        kwargs_352986 = {}
        # Getting the type of 'waveforms' (line 333)
        waveforms_352980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 14), 'waveforms', False)
        # Obtaining the member 'unit_impulse' of a type (line 333)
        unit_impulse_352981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 14), waveforms_352980, 'unit_impulse')
        # Calling unit_impulse(args, kwargs) (line 333)
        unit_impulse_call_result_352987 = invoke(stypy.reporting.localization.Localization(__file__, 333, 14), unit_impulse_352981, *[tuple_352982, int_352985], **kwargs_352986)
        
        # Assigning a type to the variable 'imp' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'imp', unit_impulse_call_result_352987)
        
        # Call to assert_array_equal(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'imp' (line 334)
        imp_352989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 27), 'imp', False)
        
        # Call to array(...): (line 334)
        # Processing the call arguments (line 334)
        
        # Obtaining an instance of the builtin type 'list' (line 334)
        list_352992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 334)
        # Adding element type (line 334)
        
        # Obtaining an instance of the builtin type 'list' (line 334)
        list_352993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 334)
        # Adding element type (line 334)
        int_352994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 42), list_352993, int_352994)
        # Adding element type (line 334)
        int_352995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 42), list_352993, int_352995)
        # Adding element type (line 334)
        int_352996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 42), list_352993, int_352996)
        # Adding element type (line 334)
        int_352997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 42), list_352993, int_352997)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 41), list_352992, list_352993)
        # Adding element type (line 334)
        
        # Obtaining an instance of the builtin type 'list' (line 335)
        list_352998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 335)
        # Adding element type (line 335)
        int_352999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 42), list_352998, int_352999)
        # Adding element type (line 335)
        int_353000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 42), list_352998, int_353000)
        # Adding element type (line 335)
        int_353001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 42), list_352998, int_353001)
        # Adding element type (line 335)
        int_353002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 335, 42), list_352998, int_353002)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 41), list_352992, list_352998)
        # Adding element type (line 334)
        
        # Obtaining an instance of the builtin type 'list' (line 336)
        list_353003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 336)
        # Adding element type (line 336)
        int_353004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 42), list_353003, int_353004)
        # Adding element type (line 336)
        int_353005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 42), list_353003, int_353005)
        # Adding element type (line 336)
        int_353006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 42), list_353003, int_353006)
        # Adding element type (line 336)
        int_353007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 336, 42), list_353003, int_353007)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 41), list_352992, list_353003)
        # Adding element type (line 334)
        
        # Obtaining an instance of the builtin type 'list' (line 337)
        list_353008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 337)
        # Adding element type (line 337)
        int_353009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 42), list_353008, int_353009)
        # Adding element type (line 337)
        int_353010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 42), list_353008, int_353010)
        # Adding element type (line 337)
        int_353011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 42), list_353008, int_353011)
        # Adding element type (line 337)
        int_353012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 42), list_353008, int_353012)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 41), list_352992, list_353008)
        
        # Processing the call keyword arguments (line 334)
        kwargs_353013 = {}
        # Getting the type of 'np' (line 334)
        np_352990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 32), 'np', False)
        # Obtaining the member 'array' of a type (line 334)
        array_352991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 32), np_352990, 'array')
        # Calling array(args, kwargs) (line 334)
        array_call_result_353014 = invoke(stypy.reporting.localization.Localization(__file__, 334, 32), array_352991, *[list_352992], **kwargs_353013)
        
        # Processing the call keyword arguments (line 334)
        kwargs_353015 = {}
        # Getting the type of 'assert_array_equal' (line 334)
        assert_array_equal_352988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 334)
        assert_array_equal_call_result_353016 = invoke(stypy.reporting.localization.Localization(__file__, 334, 8), assert_array_equal_352988, *[imp_352989, array_call_result_353014], **kwargs_353015)
        
        
        # ################# End of 'test_index(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_index' in the type store
        # Getting the type of 'stypy_return_type' (line 326)
        stypy_return_type_353017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_353017)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_index'
        return stypy_return_type_353017


    @norecursion
    def test_mid(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_mid'
        module_type_store = module_type_store.open_function_context('test_mid', 339, 4, False)
        # Assigning a type to the variable 'self' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestUnitImpulse.test_mid.__dict__.__setitem__('stypy_localization', localization)
        TestUnitImpulse.test_mid.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestUnitImpulse.test_mid.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestUnitImpulse.test_mid.__dict__.__setitem__('stypy_function_name', 'TestUnitImpulse.test_mid')
        TestUnitImpulse.test_mid.__dict__.__setitem__('stypy_param_names_list', [])
        TestUnitImpulse.test_mid.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestUnitImpulse.test_mid.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestUnitImpulse.test_mid.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestUnitImpulse.test_mid.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestUnitImpulse.test_mid.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestUnitImpulse.test_mid.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestUnitImpulse.test_mid', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_mid', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_mid(...)' code ##################

        
        # Call to assert_array_equal(...): (line 340)
        # Processing the call arguments (line 340)
        
        # Call to unit_impulse(...): (line 340)
        # Processing the call arguments (line 340)
        
        # Obtaining an instance of the builtin type 'tuple' (line 340)
        tuple_353021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 51), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 340)
        # Adding element type (line 340)
        int_353022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 51), tuple_353021, int_353022)
        # Adding element type (line 340)
        int_353023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 51), tuple_353021, int_353023)
        
        str_353024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 58), 'str', 'mid')
        # Processing the call keyword arguments (line 340)
        kwargs_353025 = {}
        # Getting the type of 'waveforms' (line 340)
        waveforms_353019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 27), 'waveforms', False)
        # Obtaining the member 'unit_impulse' of a type (line 340)
        unit_impulse_353020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 27), waveforms_353019, 'unit_impulse')
        # Calling unit_impulse(args, kwargs) (line 340)
        unit_impulse_call_result_353026 = invoke(stypy.reporting.localization.Localization(__file__, 340, 27), unit_impulse_353020, *[tuple_353021, str_353024], **kwargs_353025)
        
        
        # Obtaining an instance of the builtin type 'list' (line 341)
        list_353027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 341)
        # Adding element type (line 341)
        
        # Obtaining an instance of the builtin type 'list' (line 341)
        list_353028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 341)
        # Adding element type (line 341)
        int_353029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 28), list_353028, int_353029)
        # Adding element type (line 341)
        int_353030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 28), list_353028, int_353030)
        # Adding element type (line 341)
        int_353031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 28), list_353028, int_353031)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 27), list_353027, list_353028)
        # Adding element type (line 341)
        
        # Obtaining an instance of the builtin type 'list' (line 341)
        list_353032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 341)
        # Adding element type (line 341)
        int_353033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 39), list_353032, int_353033)
        # Adding element type (line 341)
        int_353034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 39), list_353032, int_353034)
        # Adding element type (line 341)
        int_353035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 39), list_353032, int_353035)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 27), list_353027, list_353032)
        # Adding element type (line 341)
        
        # Obtaining an instance of the builtin type 'list' (line 341)
        list_353036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 341)
        # Adding element type (line 341)
        int_353037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 50), list_353036, int_353037)
        # Adding element type (line 341)
        int_353038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 50), list_353036, int_353038)
        # Adding element type (line 341)
        int_353039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 50), list_353036, int_353039)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 341, 27), list_353027, list_353036)
        
        # Processing the call keyword arguments (line 340)
        kwargs_353040 = {}
        # Getting the type of 'assert_array_equal' (line 340)
        assert_array_equal_353018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 340)
        assert_array_equal_call_result_353041 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), assert_array_equal_353018, *[unit_impulse_call_result_353026, list_353027], **kwargs_353040)
        
        
        # Call to assert_array_equal(...): (line 342)
        # Processing the call arguments (line 342)
        
        # Call to unit_impulse(...): (line 342)
        # Processing the call arguments (line 342)
        int_353045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 50), 'int')
        str_353046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 53), 'str', 'mid')
        # Processing the call keyword arguments (line 342)
        kwargs_353047 = {}
        # Getting the type of 'waveforms' (line 342)
        waveforms_353043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 27), 'waveforms', False)
        # Obtaining the member 'unit_impulse' of a type (line 342)
        unit_impulse_353044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 27), waveforms_353043, 'unit_impulse')
        # Calling unit_impulse(args, kwargs) (line 342)
        unit_impulse_call_result_353048 = invoke(stypy.reporting.localization.Localization(__file__, 342, 27), unit_impulse_353044, *[int_353045, str_353046], **kwargs_353047)
        
        
        # Obtaining an instance of the builtin type 'list' (line 343)
        list_353049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 343)
        # Adding element type (line 343)
        int_353050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 27), list_353049, int_353050)
        # Adding element type (line 343)
        int_353051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 27), list_353049, int_353051)
        # Adding element type (line 343)
        int_353052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 27), list_353049, int_353052)
        # Adding element type (line 343)
        int_353053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 27), list_353049, int_353053)
        # Adding element type (line 343)
        int_353054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 27), list_353049, int_353054)
        # Adding element type (line 343)
        int_353055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 27), list_353049, int_353055)
        # Adding element type (line 343)
        int_353056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 27), list_353049, int_353056)
        # Adding element type (line 343)
        int_353057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 27), list_353049, int_353057)
        # Adding element type (line 343)
        int_353058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 27), list_353049, int_353058)
        
        # Processing the call keyword arguments (line 342)
        kwargs_353059 = {}
        # Getting the type of 'assert_array_equal' (line 342)
        assert_array_equal_353042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'assert_array_equal', False)
        # Calling assert_array_equal(args, kwargs) (line 342)
        assert_array_equal_call_result_353060 = invoke(stypy.reporting.localization.Localization(__file__, 342, 8), assert_array_equal_353042, *[unit_impulse_call_result_353048, list_353049], **kwargs_353059)
        
        
        # ################# End of 'test_mid(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_mid' in the type store
        # Getting the type of 'stypy_return_type' (line 339)
        stypy_return_type_353061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_353061)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_mid'
        return stypy_return_type_353061


    @norecursion
    def test_dtype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dtype'
        module_type_store = module_type_store.open_function_context('test_dtype', 345, 4, False)
        # Assigning a type to the variable 'self' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestUnitImpulse.test_dtype.__dict__.__setitem__('stypy_localization', localization)
        TestUnitImpulse.test_dtype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestUnitImpulse.test_dtype.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestUnitImpulse.test_dtype.__dict__.__setitem__('stypy_function_name', 'TestUnitImpulse.test_dtype')
        TestUnitImpulse.test_dtype.__dict__.__setitem__('stypy_param_names_list', [])
        TestUnitImpulse.test_dtype.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestUnitImpulse.test_dtype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestUnitImpulse.test_dtype.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestUnitImpulse.test_dtype.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestUnitImpulse.test_dtype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestUnitImpulse.test_dtype.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestUnitImpulse.test_dtype', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dtype', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dtype(...)' code ##################

        
        # Assigning a Call to a Name (line 346):
        
        # Assigning a Call to a Name (line 346):
        
        # Call to unit_impulse(...): (line 346)
        # Processing the call arguments (line 346)
        int_353064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 37), 'int')
        # Processing the call keyword arguments (line 346)
        kwargs_353065 = {}
        # Getting the type of 'waveforms' (line 346)
        waveforms_353062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 14), 'waveforms', False)
        # Obtaining the member 'unit_impulse' of a type (line 346)
        unit_impulse_353063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 14), waveforms_353062, 'unit_impulse')
        # Calling unit_impulse(args, kwargs) (line 346)
        unit_impulse_call_result_353066 = invoke(stypy.reporting.localization.Localization(__file__, 346, 14), unit_impulse_353063, *[int_353064], **kwargs_353065)
        
        # Assigning a type to the variable 'imp' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'imp', unit_impulse_call_result_353066)
        
        # Call to assert_(...): (line 347)
        # Processing the call arguments (line 347)
        
        # Call to issubdtype(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 'imp' (line 347)
        imp_353070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 30), 'imp', False)
        # Obtaining the member 'dtype' of a type (line 347)
        dtype_353071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 30), imp_353070, 'dtype')
        # Getting the type of 'np' (line 347)
        np_353072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 41), 'np', False)
        # Obtaining the member 'floating' of a type (line 347)
        floating_353073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 41), np_353072, 'floating')
        # Processing the call keyword arguments (line 347)
        kwargs_353074 = {}
        # Getting the type of 'np' (line 347)
        np_353068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 347)
        issubdtype_353069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 16), np_353068, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 347)
        issubdtype_call_result_353075 = invoke(stypy.reporting.localization.Localization(__file__, 347, 16), issubdtype_353069, *[dtype_353071, floating_353073], **kwargs_353074)
        
        # Processing the call keyword arguments (line 347)
        kwargs_353076 = {}
        # Getting the type of 'assert_' (line 347)
        assert__353067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 347)
        assert__call_result_353077 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), assert__353067, *[issubdtype_call_result_353075], **kwargs_353076)
        
        
        # Assigning a Call to a Name (line 349):
        
        # Assigning a Call to a Name (line 349):
        
        # Call to unit_impulse(...): (line 349)
        # Processing the call arguments (line 349)
        int_353080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 37), 'int')
        int_353081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 40), 'int')
        # Processing the call keyword arguments (line 349)
        # Getting the type of 'int' (line 349)
        int_353082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 49), 'int', False)
        keyword_353083 = int_353082
        kwargs_353084 = {'dtype': keyword_353083}
        # Getting the type of 'waveforms' (line 349)
        waveforms_353078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 14), 'waveforms', False)
        # Obtaining the member 'unit_impulse' of a type (line 349)
        unit_impulse_353079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 14), waveforms_353078, 'unit_impulse')
        # Calling unit_impulse(args, kwargs) (line 349)
        unit_impulse_call_result_353085 = invoke(stypy.reporting.localization.Localization(__file__, 349, 14), unit_impulse_353079, *[int_353080, int_353081], **kwargs_353084)
        
        # Assigning a type to the variable 'imp' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'imp', unit_impulse_call_result_353085)
        
        # Call to assert_(...): (line 350)
        # Processing the call arguments (line 350)
        
        # Call to issubdtype(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'imp' (line 350)
        imp_353089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 30), 'imp', False)
        # Obtaining the member 'dtype' of a type (line 350)
        dtype_353090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 30), imp_353089, 'dtype')
        # Getting the type of 'np' (line 350)
        np_353091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 41), 'np', False)
        # Obtaining the member 'integer' of a type (line 350)
        integer_353092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 41), np_353091, 'integer')
        # Processing the call keyword arguments (line 350)
        kwargs_353093 = {}
        # Getting the type of 'np' (line 350)
        np_353087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 16), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 350)
        issubdtype_353088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 16), np_353087, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 350)
        issubdtype_call_result_353094 = invoke(stypy.reporting.localization.Localization(__file__, 350, 16), issubdtype_353088, *[dtype_353090, integer_353092], **kwargs_353093)
        
        # Processing the call keyword arguments (line 350)
        kwargs_353095 = {}
        # Getting the type of 'assert_' (line 350)
        assert__353086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 350)
        assert__call_result_353096 = invoke(stypy.reporting.localization.Localization(__file__, 350, 8), assert__353086, *[issubdtype_call_result_353094], **kwargs_353095)
        
        
        # Assigning a Call to a Name (line 352):
        
        # Assigning a Call to a Name (line 352):
        
        # Call to unit_impulse(...): (line 352)
        # Processing the call arguments (line 352)
        
        # Obtaining an instance of the builtin type 'tuple' (line 352)
        tuple_353099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 352)
        # Adding element type (line 352)
        int_353100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 38), tuple_353099, int_353100)
        # Adding element type (line 352)
        int_353101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 38), tuple_353099, int_353101)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 352)
        tuple_353102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 352)
        # Adding element type (line 352)
        int_353103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 46), tuple_353102, int_353103)
        # Adding element type (line 352)
        int_353104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 352, 46), tuple_353102, int_353104)
        
        # Processing the call keyword arguments (line 352)
        # Getting the type of 'complex' (line 352)
        complex_353105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 59), 'complex', False)
        keyword_353106 = complex_353105
        kwargs_353107 = {'dtype': keyword_353106}
        # Getting the type of 'waveforms' (line 352)
        waveforms_353097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 14), 'waveforms', False)
        # Obtaining the member 'unit_impulse' of a type (line 352)
        unit_impulse_353098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 14), waveforms_353097, 'unit_impulse')
        # Calling unit_impulse(args, kwargs) (line 352)
        unit_impulse_call_result_353108 = invoke(stypy.reporting.localization.Localization(__file__, 352, 14), unit_impulse_353098, *[tuple_353099, tuple_353102], **kwargs_353107)
        
        # Assigning a type to the variable 'imp' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 'imp', unit_impulse_call_result_353108)
        
        # Call to assert_(...): (line 353)
        # Processing the call arguments (line 353)
        
        # Call to issubdtype(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'imp' (line 353)
        imp_353112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 30), 'imp', False)
        # Obtaining the member 'dtype' of a type (line 353)
        dtype_353113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 30), imp_353112, 'dtype')
        # Getting the type of 'np' (line 353)
        np_353114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 41), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 353)
        complexfloating_353115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 41), np_353114, 'complexfloating')
        # Processing the call keyword arguments (line 353)
        kwargs_353116 = {}
        # Getting the type of 'np' (line 353)
        np_353110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 16), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 353)
        issubdtype_353111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 16), np_353110, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 353)
        issubdtype_call_result_353117 = invoke(stypy.reporting.localization.Localization(__file__, 353, 16), issubdtype_353111, *[dtype_353113, complexfloating_353115], **kwargs_353116)
        
        # Processing the call keyword arguments (line 353)
        kwargs_353118 = {}
        # Getting the type of 'assert_' (line 353)
        assert__353109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 353)
        assert__call_result_353119 = invoke(stypy.reporting.localization.Localization(__file__, 353, 8), assert__353109, *[issubdtype_call_result_353117], **kwargs_353118)
        
        
        # ################# End of 'test_dtype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dtype' in the type store
        # Getting the type of 'stypy_return_type' (line 345)
        stypy_return_type_353120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_353120)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dtype'
        return stypy_return_type_353120


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 319, 0, False)
        # Assigning a type to the variable 'self' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestUnitImpulse.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'TestUnitImpulse' (line 319)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 0), 'TestUnitImpulse', TestUnitImpulse)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
