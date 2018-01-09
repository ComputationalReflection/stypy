
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Created by Pearu Peterson, September 2002
2: 
3: from __future__ import division, print_function, absolute_import
4: 
5: __usage__ = '''
6: Build fftpack:
7:   python setup_fftpack.py build
8: Run tests if scipy is installed:
9:   python -c 'import scipy;scipy.fftpack.test(<level>)'
10: Run tests if fftpack is not installed:
11:   python tests/test_pseudo_diffs.py [<level>]
12: '''
13: 
14: from numpy.testing import (assert_equal, assert_almost_equal,
15:                            assert_array_almost_equal)
16: from scipy.fftpack import (diff, fft, ifft, tilbert, itilbert, hilbert,
17:                            ihilbert, shift, fftfreq, cs_diff, sc_diff,
18:                            ss_diff, cc_diff)
19: 
20: import numpy as np
21: from numpy import arange, sin, cos, pi, exp, tanh, sum, sign
22: from numpy.random import random
23: 
24: 
25: def direct_diff(x,k=1,period=None):
26:     fx = fft(x)
27:     n = len(fx)
28:     if period is None:
29:         period = 2*pi
30:     w = fftfreq(n)*2j*pi/period*n
31:     if k < 0:
32:         w = 1 / w**k
33:         w[0] = 0.0
34:     else:
35:         w = w**k
36:     if n > 2000:
37:         w[250:n-250] = 0.0
38:     return ifft(w*fx).real
39: 
40: 
41: def direct_tilbert(x,h=1,period=None):
42:     fx = fft(x)
43:     n = len(fx)
44:     if period is None:
45:         period = 2*pi
46:     w = fftfreq(n)*h*2*pi/period*n
47:     w[0] = 1
48:     w = 1j/tanh(w)
49:     w[0] = 0j
50:     return ifft(w*fx)
51: 
52: 
53: def direct_itilbert(x,h=1,period=None):
54:     fx = fft(x)
55:     n = len(fx)
56:     if period is None:
57:         period = 2*pi
58:     w = fftfreq(n)*h*2*pi/period*n
59:     w = -1j*tanh(w)
60:     return ifft(w*fx)
61: 
62: 
63: def direct_hilbert(x):
64:     fx = fft(x)
65:     n = len(fx)
66:     w = fftfreq(n)*n
67:     w = 1j*sign(w)
68:     return ifft(w*fx)
69: 
70: 
71: def direct_ihilbert(x):
72:     return -direct_hilbert(x)
73: 
74: 
75: def direct_shift(x,a,period=None):
76:     n = len(x)
77:     if period is None:
78:         k = fftfreq(n)*1j*n
79:     else:
80:         k = fftfreq(n)*2j*pi/period*n
81:     return ifft(fft(x)*exp(k*a)).real
82: 
83: 
84: class TestDiff(object):
85: 
86:     def test_definition(self):
87:         for n in [16,17,64,127,32]:
88:             x = arange(n)*2*pi/n
89:             assert_array_almost_equal(diff(sin(x)),direct_diff(sin(x)))
90:             assert_array_almost_equal(diff(sin(x),2),direct_diff(sin(x),2))
91:             assert_array_almost_equal(diff(sin(x),3),direct_diff(sin(x),3))
92:             assert_array_almost_equal(diff(sin(x),4),direct_diff(sin(x),4))
93:             assert_array_almost_equal(diff(sin(x),5),direct_diff(sin(x),5))
94:             assert_array_almost_equal(diff(sin(2*x),3),direct_diff(sin(2*x),3))
95:             assert_array_almost_equal(diff(sin(2*x),4),direct_diff(sin(2*x),4))
96:             assert_array_almost_equal(diff(cos(x)),direct_diff(cos(x)))
97:             assert_array_almost_equal(diff(cos(x),2),direct_diff(cos(x),2))
98:             assert_array_almost_equal(diff(cos(x),3),direct_diff(cos(x),3))
99:             assert_array_almost_equal(diff(cos(x),4),direct_diff(cos(x),4))
100:             assert_array_almost_equal(diff(cos(2*x)),direct_diff(cos(2*x)))
101:             assert_array_almost_equal(diff(sin(x*n/8)),direct_diff(sin(x*n/8)))
102:             assert_array_almost_equal(diff(cos(x*n/8)),direct_diff(cos(x*n/8)))
103:             for k in range(5):
104:                 assert_array_almost_equal(diff(sin(4*x),k),direct_diff(sin(4*x),k))
105:                 assert_array_almost_equal(diff(cos(4*x),k),direct_diff(cos(4*x),k))
106: 
107:     def test_period(self):
108:         for n in [17,64]:
109:             x = arange(n)/float(n)
110:             assert_array_almost_equal(diff(sin(2*pi*x),period=1),
111:                                       2*pi*cos(2*pi*x))
112:             assert_array_almost_equal(diff(sin(2*pi*x),3,period=1),
113:                                       -(2*pi)**3*cos(2*pi*x))
114: 
115:     def test_sin(self):
116:         for n in [32,64,77]:
117:             x = arange(n)*2*pi/n
118:             assert_array_almost_equal(diff(sin(x)),cos(x))
119:             assert_array_almost_equal(diff(cos(x)),-sin(x))
120:             assert_array_almost_equal(diff(sin(x),2),-sin(x))
121:             assert_array_almost_equal(diff(sin(x),4),sin(x))
122:             assert_array_almost_equal(diff(sin(4*x)),4*cos(4*x))
123:             assert_array_almost_equal(diff(sin(sin(x))),cos(x)*cos(sin(x)))
124: 
125:     def test_expr(self):
126:         for n in [64,77,100,128,256,512,1024,2048,4096,8192][:5]:
127:             x = arange(n)*2*pi/n
128:             f = sin(x)*cos(4*x)+exp(sin(3*x))
129:             df = cos(x)*cos(4*x)-4*sin(x)*sin(4*x)+3*cos(3*x)*exp(sin(3*x))
130:             ddf = -17*sin(x)*cos(4*x)-8*cos(x)*sin(4*x)\
131:                  - 9*sin(3*x)*exp(sin(3*x))+9*cos(3*x)**2*exp(sin(3*x))
132:             d1 = diff(f)
133:             assert_array_almost_equal(d1,df)
134:             assert_array_almost_equal(diff(df),ddf)
135:             assert_array_almost_equal(diff(f,2),ddf)
136:             assert_array_almost_equal(diff(ddf,-1),df)
137: 
138:     def test_expr_large(self):
139:         for n in [2048,4096]:
140:             x = arange(n)*2*pi/n
141:             f = sin(x)*cos(4*x)+exp(sin(3*x))
142:             df = cos(x)*cos(4*x)-4*sin(x)*sin(4*x)+3*cos(3*x)*exp(sin(3*x))
143:             ddf = -17*sin(x)*cos(4*x)-8*cos(x)*sin(4*x)\
144:                  - 9*sin(3*x)*exp(sin(3*x))+9*cos(3*x)**2*exp(sin(3*x))
145:             assert_array_almost_equal(diff(f),df)
146:             assert_array_almost_equal(diff(df),ddf)
147:             assert_array_almost_equal(diff(ddf,-1),df)
148:             assert_array_almost_equal(diff(f,2),ddf)
149: 
150:     def test_int(self):
151:         n = 64
152:         x = arange(n)*2*pi/n
153:         assert_array_almost_equal(diff(sin(x),-1),-cos(x))
154:         assert_array_almost_equal(diff(sin(x),-2),-sin(x))
155:         assert_array_almost_equal(diff(sin(x),-4),sin(x))
156:         assert_array_almost_equal(diff(2*cos(2*x),-1),sin(2*x))
157: 
158:     def test_random_even(self):
159:         for k in [0,2,4,6]:
160:             for n in [60,32,64,56,55]:
161:                 f = random((n,))
162:                 af = sum(f,axis=0)/n
163:                 f = f-af
164:                 # zeroing Nyquist mode:
165:                 f = diff(diff(f,1),-1)
166:                 assert_almost_equal(sum(f,axis=0),0.0)
167:                 assert_array_almost_equal(diff(diff(f,k),-k),f)
168:                 assert_array_almost_equal(diff(diff(f,-k),k),f)
169: 
170:     def test_random_odd(self):
171:         for k in [0,1,2,3,4,5,6]:
172:             for n in [33,65,55]:
173:                 f = random((n,))
174:                 af = sum(f,axis=0)/n
175:                 f = f-af
176:                 assert_almost_equal(sum(f,axis=0),0.0)
177:                 assert_array_almost_equal(diff(diff(f,k),-k),f)
178:                 assert_array_almost_equal(diff(diff(f,-k),k),f)
179: 
180:     def test_zero_nyquist(self):
181:         for k in [0,1,2,3,4,5,6]:
182:             for n in [32,33,64,56,55]:
183:                 f = random((n,))
184:                 af = sum(f,axis=0)/n
185:                 f = f-af
186:                 # zeroing Nyquist mode:
187:                 f = diff(diff(f,1),-1)
188:                 assert_almost_equal(sum(f,axis=0),0.0)
189:                 assert_array_almost_equal(diff(diff(f,k),-k),f)
190:                 assert_array_almost_equal(diff(diff(f,-k),k),f)
191: 
192: 
193: class TestTilbert(object):
194: 
195:     def test_definition(self):
196:         for h in [0.1,0.5,1,5.5,10]:
197:             for n in [16,17,64,127]:
198:                 x = arange(n)*2*pi/n
199:                 y = tilbert(sin(x),h)
200:                 y1 = direct_tilbert(sin(x),h)
201:                 assert_array_almost_equal(y,y1)
202:                 assert_array_almost_equal(tilbert(sin(x),h),
203:                                           direct_tilbert(sin(x),h))
204:                 assert_array_almost_equal(tilbert(sin(2*x),h),
205:                                           direct_tilbert(sin(2*x),h))
206: 
207:     def test_random_even(self):
208:         for h in [0.1,0.5,1,5.5,10]:
209:             for n in [32,64,56]:
210:                 f = random((n,))
211:                 af = sum(f,axis=0)/n
212:                 f = f-af
213:                 assert_almost_equal(sum(f,axis=0),0.0)
214:                 assert_array_almost_equal(direct_tilbert(direct_itilbert(f,h),h),f)
215: 
216:     def test_random_odd(self):
217:         for h in [0.1,0.5,1,5.5,10]:
218:             for n in [33,65,55]:
219:                 f = random((n,))
220:                 af = sum(f,axis=0)/n
221:                 f = f-af
222:                 assert_almost_equal(sum(f,axis=0),0.0)
223:                 assert_array_almost_equal(itilbert(tilbert(f,h),h),f)
224:                 assert_array_almost_equal(tilbert(itilbert(f,h),h),f)
225: 
226: 
227: class TestITilbert(object):
228: 
229:     def test_definition(self):
230:         for h in [0.1,0.5,1,5.5,10]:
231:             for n in [16,17,64,127]:
232:                 x = arange(n)*2*pi/n
233:                 y = itilbert(sin(x),h)
234:                 y1 = direct_itilbert(sin(x),h)
235:                 assert_array_almost_equal(y,y1)
236:                 assert_array_almost_equal(itilbert(sin(x),h),
237:                                           direct_itilbert(sin(x),h))
238:                 assert_array_almost_equal(itilbert(sin(2*x),h),
239:                                           direct_itilbert(sin(2*x),h))
240: 
241: 
242: class TestHilbert(object):
243: 
244:     def test_definition(self):
245:         for n in [16,17,64,127]:
246:             x = arange(n)*2*pi/n
247:             y = hilbert(sin(x))
248:             y1 = direct_hilbert(sin(x))
249:             assert_array_almost_equal(y,y1)
250:             assert_array_almost_equal(hilbert(sin(2*x)),
251:                                       direct_hilbert(sin(2*x)))
252: 
253:     def test_tilbert_relation(self):
254:         for n in [16,17,64,127]:
255:             x = arange(n)*2*pi/n
256:             f = sin(x)+cos(2*x)*sin(x)
257:             y = hilbert(f)
258:             y1 = direct_hilbert(f)
259:             assert_array_almost_equal(y,y1)
260:             y2 = tilbert(f,h=10)
261:             assert_array_almost_equal(y,y2)
262: 
263:     def test_random_odd(self):
264:         for n in [33,65,55]:
265:             f = random((n,))
266:             af = sum(f,axis=0)/n
267:             f = f-af
268:             assert_almost_equal(sum(f,axis=0),0.0)
269:             assert_array_almost_equal(ihilbert(hilbert(f)),f)
270:             assert_array_almost_equal(hilbert(ihilbert(f)),f)
271: 
272:     def test_random_even(self):
273:         for n in [32,64,56]:
274:             f = random((n,))
275:             af = sum(f,axis=0)/n
276:             f = f-af
277:             # zeroing Nyquist mode:
278:             f = diff(diff(f,1),-1)
279:             assert_almost_equal(sum(f,axis=0),0.0)
280:             assert_array_almost_equal(direct_hilbert(direct_ihilbert(f)),f)
281:             assert_array_almost_equal(hilbert(ihilbert(f)),f)
282: 
283: 
284: class TestIHilbert(object):
285: 
286:     def test_definition(self):
287:         for n in [16,17,64,127]:
288:             x = arange(n)*2*pi/n
289:             y = ihilbert(sin(x))
290:             y1 = direct_ihilbert(sin(x))
291:             assert_array_almost_equal(y,y1)
292:             assert_array_almost_equal(ihilbert(sin(2*x)),
293:                                       direct_ihilbert(sin(2*x)))
294: 
295:     def test_itilbert_relation(self):
296:         for n in [16,17,64,127]:
297:             x = arange(n)*2*pi/n
298:             f = sin(x)+cos(2*x)*sin(x)
299:             y = ihilbert(f)
300:             y1 = direct_ihilbert(f)
301:             assert_array_almost_equal(y,y1)
302:             y2 = itilbert(f,h=10)
303:             assert_array_almost_equal(y,y2)
304: 
305: 
306: class TestShift(object):
307: 
308:     def test_definition(self):
309:         for n in [18,17,64,127,32,2048,256]:
310:             x = arange(n)*2*pi/n
311:             for a in [0.1,3]:
312:                 assert_array_almost_equal(shift(sin(x),a),direct_shift(sin(x),a))
313:                 assert_array_almost_equal(shift(sin(x),a),sin(x+a))
314:                 assert_array_almost_equal(shift(cos(x),a),cos(x+a))
315:                 assert_array_almost_equal(shift(cos(2*x)+sin(x),a),
316:                                           cos(2*(x+a))+sin(x+a))
317:                 assert_array_almost_equal(shift(exp(sin(x)),a),exp(sin(x+a)))
318:             assert_array_almost_equal(shift(sin(x),2*pi),sin(x))
319:             assert_array_almost_equal(shift(sin(x),pi),-sin(x))
320:             assert_array_almost_equal(shift(sin(x),pi/2),cos(x))
321: 
322: 
323: class TestOverwrite(object):
324:     '''Check input overwrite behavior '''
325: 
326:     real_dtypes = [np.float32, np.float64]
327:     dtypes = real_dtypes + [np.complex64, np.complex128]
328: 
329:     def _check(self, x, routine, *args, **kwargs):
330:         x2 = x.copy()
331:         routine(x2, *args, **kwargs)
332:         sig = routine.__name__
333:         if args:
334:             sig += repr(args)
335:         if kwargs:
336:             sig += repr(kwargs)
337:         assert_equal(x2, x, err_msg="spurious overwrite in %s" % sig)
338: 
339:     def _check_1d(self, routine, dtype, shape, *args, **kwargs):
340:         np.random.seed(1234)
341:         if np.issubdtype(dtype, np.complexfloating):
342:             data = np.random.randn(*shape) + 1j*np.random.randn(*shape)
343:         else:
344:             data = np.random.randn(*shape)
345:         data = data.astype(dtype)
346:         self._check(data, routine, *args, **kwargs)
347: 
348:     def test_diff(self):
349:         for dtype in self.dtypes:
350:             self._check_1d(diff, dtype, (16,))
351: 
352:     def test_tilbert(self):
353:         for dtype in self.dtypes:
354:             self._check_1d(tilbert, dtype, (16,), 1.6)
355: 
356:     def test_itilbert(self):
357:         for dtype in self.dtypes:
358:             self._check_1d(itilbert, dtype, (16,), 1.6)
359: 
360:     def test_hilbert(self):
361:         for dtype in self.dtypes:
362:             self._check_1d(hilbert, dtype, (16,))
363: 
364:     def test_cs_diff(self):
365:         for dtype in self.dtypes:
366:             self._check_1d(cs_diff, dtype, (16,), 1.0, 4.0)
367: 
368:     def test_sc_diff(self):
369:         for dtype in self.dtypes:
370:             self._check_1d(sc_diff, dtype, (16,), 1.0, 4.0)
371: 
372:     def test_ss_diff(self):
373:         for dtype in self.dtypes:
374:             self._check_1d(ss_diff, dtype, (16,), 1.0, 4.0)
375: 
376:     def test_cc_diff(self):
377:         for dtype in self.dtypes:
378:             self._check_1d(cc_diff, dtype, (16,), 1.0, 4.0)
379: 
380:     def test_shift(self):
381:         for dtype in self.dtypes:
382:             self._check_1d(shift, dtype, (16,), 1.0)
383: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 5):
str_24260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, (-1)), 'str', "\nBuild fftpack:\n  python setup_fftpack.py build\nRun tests if scipy is installed:\n  python -c 'import scipy;scipy.fftpack.test(<level>)'\nRun tests if fftpack is not installed:\n  python tests/test_pseudo_diffs.py [<level>]\n")
# Assigning a type to the variable '__usage__' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__usage__', str_24260)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy.testing import assert_equal, assert_almost_equal, assert_array_almost_equal' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_24261 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing')

if (type(import_24261) is not StypyTypeError):

    if (import_24261 != 'pyd_module'):
        __import__(import_24261)
        sys_modules_24262 = sys.modules[import_24261]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', sys_modules_24262.module_type_store, module_type_store, ['assert_equal', 'assert_almost_equal', 'assert_array_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_24262, sys_modules_24262.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_almost_equal, assert_array_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_almost_equal', 'assert_array_almost_equal'], [assert_equal, assert_almost_equal, assert_array_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.testing', import_24261)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from scipy.fftpack import diff, fft, ifft, tilbert, itilbert, hilbert, ihilbert, shift, fftfreq, cs_diff, sc_diff, ss_diff, cc_diff' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_24263 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.fftpack')

if (type(import_24263) is not StypyTypeError):

    if (import_24263 != 'pyd_module'):
        __import__(import_24263)
        sys_modules_24264 = sys.modules[import_24263]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.fftpack', sys_modules_24264.module_type_store, module_type_store, ['diff', 'fft', 'ifft', 'tilbert', 'itilbert', 'hilbert', 'ihilbert', 'shift', 'fftfreq', 'cs_diff', 'sc_diff', 'ss_diff', 'cc_diff'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_24264, sys_modules_24264.module_type_store, module_type_store)
    else:
        from scipy.fftpack import diff, fft, ifft, tilbert, itilbert, hilbert, ihilbert, shift, fftfreq, cs_diff, sc_diff, ss_diff, cc_diff

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.fftpack', None, module_type_store, ['diff', 'fft', 'ifft', 'tilbert', 'itilbert', 'hilbert', 'ihilbert', 'shift', 'fftfreq', 'cs_diff', 'sc_diff', 'ss_diff', 'cc_diff'], [diff, fft, ifft, tilbert, itilbert, hilbert, ihilbert, shift, fftfreq, cs_diff, sc_diff, ss_diff, cc_diff])

else:
    # Assigning a type to the variable 'scipy.fftpack' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'scipy.fftpack', import_24263)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import numpy' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_24265 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy')

if (type(import_24265) is not StypyTypeError):

    if (import_24265 != 'pyd_module'):
        __import__(import_24265)
        sys_modules_24266 = sys.modules[import_24265]
        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'np', sys_modules_24266.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'numpy', import_24265)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from numpy import arange, sin, cos, pi, exp, tanh, sum, sign' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_24267 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy')

if (type(import_24267) is not StypyTypeError):

    if (import_24267 != 'pyd_module'):
        __import__(import_24267)
        sys_modules_24268 = sys.modules[import_24267]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy', sys_modules_24268.module_type_store, module_type_store, ['arange', 'sin', 'cos', 'pi', 'exp', 'tanh', 'sum', 'sign'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_24268, sys_modules_24268.module_type_store, module_type_store)
    else:
        from numpy import arange, sin, cos, pi, exp, tanh, sum, sign

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy', None, module_type_store, ['arange', 'sin', 'cos', 'pi', 'exp', 'tanh', 'sum', 'sign'], [arange, sin, cos, pi, exp, tanh, sum, sign])

else:
    # Assigning a type to the variable 'numpy' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'numpy', import_24267)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from numpy.random import random' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/fftpack/tests/')
import_24269 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.random')

if (type(import_24269) is not StypyTypeError):

    if (import_24269 != 'pyd_module'):
        __import__(import_24269)
        sys_modules_24270 = sys.modules[import_24269]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.random', sys_modules_24270.module_type_store, module_type_store, ['random'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_24270, sys_modules_24270.module_type_store, module_type_store)
    else:
        from numpy.random import random

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.random', None, module_type_store, ['random'], [random])

else:
    # Assigning a type to the variable 'numpy.random' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'numpy.random', import_24269)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/fftpack/tests/')


@norecursion
def direct_diff(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_24271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'int')
    # Getting the type of 'None' (line 25)
    None_24272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 29), 'None')
    defaults = [int_24271, None_24272]
    # Create a new context for function 'direct_diff'
    module_type_store = module_type_store.open_function_context('direct_diff', 25, 0, False)
    
    # Passed parameters checking function
    direct_diff.stypy_localization = localization
    direct_diff.stypy_type_of_self = None
    direct_diff.stypy_type_store = module_type_store
    direct_diff.stypy_function_name = 'direct_diff'
    direct_diff.stypy_param_names_list = ['x', 'k', 'period']
    direct_diff.stypy_varargs_param_name = None
    direct_diff.stypy_kwargs_param_name = None
    direct_diff.stypy_call_defaults = defaults
    direct_diff.stypy_call_varargs = varargs
    direct_diff.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'direct_diff', ['x', 'k', 'period'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'direct_diff', localization, ['x', 'k', 'period'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'direct_diff(...)' code ##################

    
    # Assigning a Call to a Name (line 26):
    
    # Call to fft(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'x' (line 26)
    x_24274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 13), 'x', False)
    # Processing the call keyword arguments (line 26)
    kwargs_24275 = {}
    # Getting the type of 'fft' (line 26)
    fft_24273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 9), 'fft', False)
    # Calling fft(args, kwargs) (line 26)
    fft_call_result_24276 = invoke(stypy.reporting.localization.Localization(__file__, 26, 9), fft_24273, *[x_24274], **kwargs_24275)
    
    # Assigning a type to the variable 'fx' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'fx', fft_call_result_24276)
    
    # Assigning a Call to a Name (line 27):
    
    # Call to len(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'fx' (line 27)
    fx_24278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'fx', False)
    # Processing the call keyword arguments (line 27)
    kwargs_24279 = {}
    # Getting the type of 'len' (line 27)
    len_24277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'len', False)
    # Calling len(args, kwargs) (line 27)
    len_call_result_24280 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), len_24277, *[fx_24278], **kwargs_24279)
    
    # Assigning a type to the variable 'n' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'n', len_call_result_24280)
    
    # Type idiom detected: calculating its left and rigth part (line 28)
    # Getting the type of 'period' (line 28)
    period_24281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 7), 'period')
    # Getting the type of 'None' (line 28)
    None_24282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'None')
    
    (may_be_24283, more_types_in_union_24284) = may_be_none(period_24281, None_24282)

    if may_be_24283:

        if more_types_in_union_24284:
            # Runtime conditional SSA (line 28)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 29):
        int_24285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 17), 'int')
        # Getting the type of 'pi' (line 29)
        pi_24286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'pi')
        # Applying the binary operator '*' (line 29)
        result_mul_24287 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 17), '*', int_24285, pi_24286)
        
        # Assigning a type to the variable 'period' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'period', result_mul_24287)

        if more_types_in_union_24284:
            # SSA join for if statement (line 28)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 30):
    
    # Call to fftfreq(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'n' (line 30)
    n_24289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'n', False)
    # Processing the call keyword arguments (line 30)
    kwargs_24290 = {}
    # Getting the type of 'fftfreq' (line 30)
    fftfreq_24288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'fftfreq', False)
    # Calling fftfreq(args, kwargs) (line 30)
    fftfreq_call_result_24291 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), fftfreq_24288, *[n_24289], **kwargs_24290)
    
    complex_24292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 19), 'complex')
    # Applying the binary operator '*' (line 30)
    result_mul_24293 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 8), '*', fftfreq_call_result_24291, complex_24292)
    
    # Getting the type of 'pi' (line 30)
    pi_24294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'pi')
    # Applying the binary operator '*' (line 30)
    result_mul_24295 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 21), '*', result_mul_24293, pi_24294)
    
    # Getting the type of 'period' (line 30)
    period_24296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), 'period')
    # Applying the binary operator 'div' (line 30)
    result_div_24297 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 24), 'div', result_mul_24295, period_24296)
    
    # Getting the type of 'n' (line 30)
    n_24298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 32), 'n')
    # Applying the binary operator '*' (line 30)
    result_mul_24299 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 31), '*', result_div_24297, n_24298)
    
    # Assigning a type to the variable 'w' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'w', result_mul_24299)
    
    
    # Getting the type of 'k' (line 31)
    k_24300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 7), 'k')
    int_24301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 11), 'int')
    # Applying the binary operator '<' (line 31)
    result_lt_24302 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 7), '<', k_24300, int_24301)
    
    # Testing the type of an if condition (line 31)
    if_condition_24303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 4), result_lt_24302)
    # Assigning a type to the variable 'if_condition_24303' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'if_condition_24303', if_condition_24303)
    # SSA begins for if statement (line 31)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 32):
    int_24304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 12), 'int')
    # Getting the type of 'w' (line 32)
    w_24305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'w')
    # Getting the type of 'k' (line 32)
    k_24306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'k')
    # Applying the binary operator '**' (line 32)
    result_pow_24307 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 16), '**', w_24305, k_24306)
    
    # Applying the binary operator 'div' (line 32)
    result_div_24308 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 12), 'div', int_24304, result_pow_24307)
    
    # Assigning a type to the variable 'w' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'w', result_div_24308)
    
    # Assigning a Num to a Subscript (line 33):
    float_24309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'float')
    # Getting the type of 'w' (line 33)
    w_24310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'w')
    int_24311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 10), 'int')
    # Storing an element on a container (line 33)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 8), w_24310, (int_24311, float_24309))
    # SSA branch for the else part of an if statement (line 31)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 35):
    # Getting the type of 'w' (line 35)
    w_24312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'w')
    # Getting the type of 'k' (line 35)
    k_24313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'k')
    # Applying the binary operator '**' (line 35)
    result_pow_24314 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 12), '**', w_24312, k_24313)
    
    # Assigning a type to the variable 'w' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'w', result_pow_24314)
    # SSA join for if statement (line 31)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'n' (line 36)
    n_24315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 7), 'n')
    int_24316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 11), 'int')
    # Applying the binary operator '>' (line 36)
    result_gt_24317 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 7), '>', n_24315, int_24316)
    
    # Testing the type of an if condition (line 36)
    if_condition_24318 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 4), result_gt_24317)
    # Assigning a type to the variable 'if_condition_24318' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'if_condition_24318', if_condition_24318)
    # SSA begins for if statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 37):
    float_24319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 23), 'float')
    # Getting the type of 'w' (line 37)
    w_24320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'w')
    int_24321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 10), 'int')
    # Getting the type of 'n' (line 37)
    n_24322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 14), 'n')
    int_24323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 16), 'int')
    # Applying the binary operator '-' (line 37)
    result_sub_24324 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 14), '-', n_24322, int_24323)
    
    slice_24325 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 37, 8), int_24321, result_sub_24324, None)
    # Storing an element on a container (line 37)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 8), w_24320, (slice_24325, float_24319))
    # SSA join for if statement (line 36)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to ifft(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'w' (line 38)
    w_24327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'w', False)
    # Getting the type of 'fx' (line 38)
    fx_24328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 18), 'fx', False)
    # Applying the binary operator '*' (line 38)
    result_mul_24329 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 16), '*', w_24327, fx_24328)
    
    # Processing the call keyword arguments (line 38)
    kwargs_24330 = {}
    # Getting the type of 'ifft' (line 38)
    ifft_24326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'ifft', False)
    # Calling ifft(args, kwargs) (line 38)
    ifft_call_result_24331 = invoke(stypy.reporting.localization.Localization(__file__, 38, 11), ifft_24326, *[result_mul_24329], **kwargs_24330)
    
    # Obtaining the member 'real' of a type (line 38)
    real_24332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 11), ifft_call_result_24331, 'real')
    # Assigning a type to the variable 'stypy_return_type' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type', real_24332)
    
    # ################# End of 'direct_diff(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'direct_diff' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_24333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24333)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'direct_diff'
    return stypy_return_type_24333

# Assigning a type to the variable 'direct_diff' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'direct_diff', direct_diff)

@norecursion
def direct_tilbert(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_24334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 23), 'int')
    # Getting the type of 'None' (line 41)
    None_24335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 32), 'None')
    defaults = [int_24334, None_24335]
    # Create a new context for function 'direct_tilbert'
    module_type_store = module_type_store.open_function_context('direct_tilbert', 41, 0, False)
    
    # Passed parameters checking function
    direct_tilbert.stypy_localization = localization
    direct_tilbert.stypy_type_of_self = None
    direct_tilbert.stypy_type_store = module_type_store
    direct_tilbert.stypy_function_name = 'direct_tilbert'
    direct_tilbert.stypy_param_names_list = ['x', 'h', 'period']
    direct_tilbert.stypy_varargs_param_name = None
    direct_tilbert.stypy_kwargs_param_name = None
    direct_tilbert.stypy_call_defaults = defaults
    direct_tilbert.stypy_call_varargs = varargs
    direct_tilbert.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'direct_tilbert', ['x', 'h', 'period'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'direct_tilbert', localization, ['x', 'h', 'period'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'direct_tilbert(...)' code ##################

    
    # Assigning a Call to a Name (line 42):
    
    # Call to fft(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'x' (line 42)
    x_24337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'x', False)
    # Processing the call keyword arguments (line 42)
    kwargs_24338 = {}
    # Getting the type of 'fft' (line 42)
    fft_24336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 9), 'fft', False)
    # Calling fft(args, kwargs) (line 42)
    fft_call_result_24339 = invoke(stypy.reporting.localization.Localization(__file__, 42, 9), fft_24336, *[x_24337], **kwargs_24338)
    
    # Assigning a type to the variable 'fx' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'fx', fft_call_result_24339)
    
    # Assigning a Call to a Name (line 43):
    
    # Call to len(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'fx' (line 43)
    fx_24341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'fx', False)
    # Processing the call keyword arguments (line 43)
    kwargs_24342 = {}
    # Getting the type of 'len' (line 43)
    len_24340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'len', False)
    # Calling len(args, kwargs) (line 43)
    len_call_result_24343 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), len_24340, *[fx_24341], **kwargs_24342)
    
    # Assigning a type to the variable 'n' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'n', len_call_result_24343)
    
    # Type idiom detected: calculating its left and rigth part (line 44)
    # Getting the type of 'period' (line 44)
    period_24344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 7), 'period')
    # Getting the type of 'None' (line 44)
    None_24345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 17), 'None')
    
    (may_be_24346, more_types_in_union_24347) = may_be_none(period_24344, None_24345)

    if may_be_24346:

        if more_types_in_union_24347:
            # Runtime conditional SSA (line 44)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 45):
        int_24348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 17), 'int')
        # Getting the type of 'pi' (line 45)
        pi_24349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'pi')
        # Applying the binary operator '*' (line 45)
        result_mul_24350 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 17), '*', int_24348, pi_24349)
        
        # Assigning a type to the variable 'period' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'period', result_mul_24350)

        if more_types_in_union_24347:
            # SSA join for if statement (line 44)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 46):
    
    # Call to fftfreq(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'n' (line 46)
    n_24352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'n', False)
    # Processing the call keyword arguments (line 46)
    kwargs_24353 = {}
    # Getting the type of 'fftfreq' (line 46)
    fftfreq_24351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'fftfreq', False)
    # Calling fftfreq(args, kwargs) (line 46)
    fftfreq_call_result_24354 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), fftfreq_24351, *[n_24352], **kwargs_24353)
    
    # Getting the type of 'h' (line 46)
    h_24355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 19), 'h')
    # Applying the binary operator '*' (line 46)
    result_mul_24356 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 8), '*', fftfreq_call_result_24354, h_24355)
    
    int_24357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 21), 'int')
    # Applying the binary operator '*' (line 46)
    result_mul_24358 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 20), '*', result_mul_24356, int_24357)
    
    # Getting the type of 'pi' (line 46)
    pi_24359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 23), 'pi')
    # Applying the binary operator '*' (line 46)
    result_mul_24360 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 22), '*', result_mul_24358, pi_24359)
    
    # Getting the type of 'period' (line 46)
    period_24361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'period')
    # Applying the binary operator 'div' (line 46)
    result_div_24362 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 25), 'div', result_mul_24360, period_24361)
    
    # Getting the type of 'n' (line 46)
    n_24363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 33), 'n')
    # Applying the binary operator '*' (line 46)
    result_mul_24364 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 32), '*', result_div_24362, n_24363)
    
    # Assigning a type to the variable 'w' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'w', result_mul_24364)
    
    # Assigning a Num to a Subscript (line 47):
    int_24365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 11), 'int')
    # Getting the type of 'w' (line 47)
    w_24366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'w')
    int_24367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 6), 'int')
    # Storing an element on a container (line 47)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 4), w_24366, (int_24367, int_24365))
    
    # Assigning a BinOp to a Name (line 48):
    complex_24368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'complex')
    
    # Call to tanh(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'w' (line 48)
    w_24370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'w', False)
    # Processing the call keyword arguments (line 48)
    kwargs_24371 = {}
    # Getting the type of 'tanh' (line 48)
    tanh_24369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'tanh', False)
    # Calling tanh(args, kwargs) (line 48)
    tanh_call_result_24372 = invoke(stypy.reporting.localization.Localization(__file__, 48, 11), tanh_24369, *[w_24370], **kwargs_24371)
    
    # Applying the binary operator 'div' (line 48)
    result_div_24373 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 8), 'div', complex_24368, tanh_call_result_24372)
    
    # Assigning a type to the variable 'w' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'w', result_div_24373)
    
    # Assigning a Num to a Subscript (line 49):
    complex_24374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 11), 'complex')
    # Getting the type of 'w' (line 49)
    w_24375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'w')
    int_24376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 6), 'int')
    # Storing an element on a container (line 49)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 4), w_24375, (int_24376, complex_24374))
    
    # Call to ifft(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'w' (line 50)
    w_24378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 16), 'w', False)
    # Getting the type of 'fx' (line 50)
    fx_24379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 18), 'fx', False)
    # Applying the binary operator '*' (line 50)
    result_mul_24380 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 16), '*', w_24378, fx_24379)
    
    # Processing the call keyword arguments (line 50)
    kwargs_24381 = {}
    # Getting the type of 'ifft' (line 50)
    ifft_24377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'ifft', False)
    # Calling ifft(args, kwargs) (line 50)
    ifft_call_result_24382 = invoke(stypy.reporting.localization.Localization(__file__, 50, 11), ifft_24377, *[result_mul_24380], **kwargs_24381)
    
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type', ifft_call_result_24382)
    
    # ################# End of 'direct_tilbert(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'direct_tilbert' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_24383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24383)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'direct_tilbert'
    return stypy_return_type_24383

# Assigning a type to the variable 'direct_tilbert' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'direct_tilbert', direct_tilbert)

@norecursion
def direct_itilbert(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_24384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 24), 'int')
    # Getting the type of 'None' (line 53)
    None_24385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 33), 'None')
    defaults = [int_24384, None_24385]
    # Create a new context for function 'direct_itilbert'
    module_type_store = module_type_store.open_function_context('direct_itilbert', 53, 0, False)
    
    # Passed parameters checking function
    direct_itilbert.stypy_localization = localization
    direct_itilbert.stypy_type_of_self = None
    direct_itilbert.stypy_type_store = module_type_store
    direct_itilbert.stypy_function_name = 'direct_itilbert'
    direct_itilbert.stypy_param_names_list = ['x', 'h', 'period']
    direct_itilbert.stypy_varargs_param_name = None
    direct_itilbert.stypy_kwargs_param_name = None
    direct_itilbert.stypy_call_defaults = defaults
    direct_itilbert.stypy_call_varargs = varargs
    direct_itilbert.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'direct_itilbert', ['x', 'h', 'period'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'direct_itilbert', localization, ['x', 'h', 'period'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'direct_itilbert(...)' code ##################

    
    # Assigning a Call to a Name (line 54):
    
    # Call to fft(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'x' (line 54)
    x_24387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'x', False)
    # Processing the call keyword arguments (line 54)
    kwargs_24388 = {}
    # Getting the type of 'fft' (line 54)
    fft_24386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 9), 'fft', False)
    # Calling fft(args, kwargs) (line 54)
    fft_call_result_24389 = invoke(stypy.reporting.localization.Localization(__file__, 54, 9), fft_24386, *[x_24387], **kwargs_24388)
    
    # Assigning a type to the variable 'fx' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'fx', fft_call_result_24389)
    
    # Assigning a Call to a Name (line 55):
    
    # Call to len(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'fx' (line 55)
    fx_24391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'fx', False)
    # Processing the call keyword arguments (line 55)
    kwargs_24392 = {}
    # Getting the type of 'len' (line 55)
    len_24390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'len', False)
    # Calling len(args, kwargs) (line 55)
    len_call_result_24393 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), len_24390, *[fx_24391], **kwargs_24392)
    
    # Assigning a type to the variable 'n' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'n', len_call_result_24393)
    
    # Type idiom detected: calculating its left and rigth part (line 56)
    # Getting the type of 'period' (line 56)
    period_24394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 7), 'period')
    # Getting the type of 'None' (line 56)
    None_24395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 17), 'None')
    
    (may_be_24396, more_types_in_union_24397) = may_be_none(period_24394, None_24395)

    if may_be_24396:

        if more_types_in_union_24397:
            # Runtime conditional SSA (line 56)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 57):
        int_24398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 17), 'int')
        # Getting the type of 'pi' (line 57)
        pi_24399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'pi')
        # Applying the binary operator '*' (line 57)
        result_mul_24400 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 17), '*', int_24398, pi_24399)
        
        # Assigning a type to the variable 'period' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'period', result_mul_24400)

        if more_types_in_union_24397:
            # SSA join for if statement (line 56)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 58):
    
    # Call to fftfreq(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'n' (line 58)
    n_24402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'n', False)
    # Processing the call keyword arguments (line 58)
    kwargs_24403 = {}
    # Getting the type of 'fftfreq' (line 58)
    fftfreq_24401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'fftfreq', False)
    # Calling fftfreq(args, kwargs) (line 58)
    fftfreq_call_result_24404 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), fftfreq_24401, *[n_24402], **kwargs_24403)
    
    # Getting the type of 'h' (line 58)
    h_24405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'h')
    # Applying the binary operator '*' (line 58)
    result_mul_24406 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 8), '*', fftfreq_call_result_24404, h_24405)
    
    int_24407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 21), 'int')
    # Applying the binary operator '*' (line 58)
    result_mul_24408 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 20), '*', result_mul_24406, int_24407)
    
    # Getting the type of 'pi' (line 58)
    pi_24409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 23), 'pi')
    # Applying the binary operator '*' (line 58)
    result_mul_24410 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 22), '*', result_mul_24408, pi_24409)
    
    # Getting the type of 'period' (line 58)
    period_24411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), 'period')
    # Applying the binary operator 'div' (line 58)
    result_div_24412 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 25), 'div', result_mul_24410, period_24411)
    
    # Getting the type of 'n' (line 58)
    n_24413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 33), 'n')
    # Applying the binary operator '*' (line 58)
    result_mul_24414 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 32), '*', result_div_24412, n_24413)
    
    # Assigning a type to the variable 'w' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'w', result_mul_24414)
    
    # Assigning a BinOp to a Name (line 59):
    complex_24415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 8), 'complex')
    
    # Call to tanh(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'w' (line 59)
    w_24417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 17), 'w', False)
    # Processing the call keyword arguments (line 59)
    kwargs_24418 = {}
    # Getting the type of 'tanh' (line 59)
    tanh_24416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'tanh', False)
    # Calling tanh(args, kwargs) (line 59)
    tanh_call_result_24419 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), tanh_24416, *[w_24417], **kwargs_24418)
    
    # Applying the binary operator '*' (line 59)
    result_mul_24420 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 8), '*', complex_24415, tanh_call_result_24419)
    
    # Assigning a type to the variable 'w' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'w', result_mul_24420)
    
    # Call to ifft(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'w' (line 60)
    w_24422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'w', False)
    # Getting the type of 'fx' (line 60)
    fx_24423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 18), 'fx', False)
    # Applying the binary operator '*' (line 60)
    result_mul_24424 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 16), '*', w_24422, fx_24423)
    
    # Processing the call keyword arguments (line 60)
    kwargs_24425 = {}
    # Getting the type of 'ifft' (line 60)
    ifft_24421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'ifft', False)
    # Calling ifft(args, kwargs) (line 60)
    ifft_call_result_24426 = invoke(stypy.reporting.localization.Localization(__file__, 60, 11), ifft_24421, *[result_mul_24424], **kwargs_24425)
    
    # Assigning a type to the variable 'stypy_return_type' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type', ifft_call_result_24426)
    
    # ################# End of 'direct_itilbert(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'direct_itilbert' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_24427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24427)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'direct_itilbert'
    return stypy_return_type_24427

# Assigning a type to the variable 'direct_itilbert' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'direct_itilbert', direct_itilbert)

@norecursion
def direct_hilbert(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'direct_hilbert'
    module_type_store = module_type_store.open_function_context('direct_hilbert', 63, 0, False)
    
    # Passed parameters checking function
    direct_hilbert.stypy_localization = localization
    direct_hilbert.stypy_type_of_self = None
    direct_hilbert.stypy_type_store = module_type_store
    direct_hilbert.stypy_function_name = 'direct_hilbert'
    direct_hilbert.stypy_param_names_list = ['x']
    direct_hilbert.stypy_varargs_param_name = None
    direct_hilbert.stypy_kwargs_param_name = None
    direct_hilbert.stypy_call_defaults = defaults
    direct_hilbert.stypy_call_varargs = varargs
    direct_hilbert.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'direct_hilbert', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'direct_hilbert', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'direct_hilbert(...)' code ##################

    
    # Assigning a Call to a Name (line 64):
    
    # Call to fft(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'x' (line 64)
    x_24429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 13), 'x', False)
    # Processing the call keyword arguments (line 64)
    kwargs_24430 = {}
    # Getting the type of 'fft' (line 64)
    fft_24428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 9), 'fft', False)
    # Calling fft(args, kwargs) (line 64)
    fft_call_result_24431 = invoke(stypy.reporting.localization.Localization(__file__, 64, 9), fft_24428, *[x_24429], **kwargs_24430)
    
    # Assigning a type to the variable 'fx' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'fx', fft_call_result_24431)
    
    # Assigning a Call to a Name (line 65):
    
    # Call to len(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'fx' (line 65)
    fx_24433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'fx', False)
    # Processing the call keyword arguments (line 65)
    kwargs_24434 = {}
    # Getting the type of 'len' (line 65)
    len_24432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'len', False)
    # Calling len(args, kwargs) (line 65)
    len_call_result_24435 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), len_24432, *[fx_24433], **kwargs_24434)
    
    # Assigning a type to the variable 'n' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'n', len_call_result_24435)
    
    # Assigning a BinOp to a Name (line 66):
    
    # Call to fftfreq(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'n' (line 66)
    n_24437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'n', False)
    # Processing the call keyword arguments (line 66)
    kwargs_24438 = {}
    # Getting the type of 'fftfreq' (line 66)
    fftfreq_24436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'fftfreq', False)
    # Calling fftfreq(args, kwargs) (line 66)
    fftfreq_call_result_24439 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), fftfreq_24436, *[n_24437], **kwargs_24438)
    
    # Getting the type of 'n' (line 66)
    n_24440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), 'n')
    # Applying the binary operator '*' (line 66)
    result_mul_24441 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 8), '*', fftfreq_call_result_24439, n_24440)
    
    # Assigning a type to the variable 'w' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'w', result_mul_24441)
    
    # Assigning a BinOp to a Name (line 67):
    complex_24442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 8), 'complex')
    
    # Call to sign(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'w' (line 67)
    w_24444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'w', False)
    # Processing the call keyword arguments (line 67)
    kwargs_24445 = {}
    # Getting the type of 'sign' (line 67)
    sign_24443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'sign', False)
    # Calling sign(args, kwargs) (line 67)
    sign_call_result_24446 = invoke(stypy.reporting.localization.Localization(__file__, 67, 11), sign_24443, *[w_24444], **kwargs_24445)
    
    # Applying the binary operator '*' (line 67)
    result_mul_24447 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 8), '*', complex_24442, sign_call_result_24446)
    
    # Assigning a type to the variable 'w' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'w', result_mul_24447)
    
    # Call to ifft(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'w' (line 68)
    w_24449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'w', False)
    # Getting the type of 'fx' (line 68)
    fx_24450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 18), 'fx', False)
    # Applying the binary operator '*' (line 68)
    result_mul_24451 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 16), '*', w_24449, fx_24450)
    
    # Processing the call keyword arguments (line 68)
    kwargs_24452 = {}
    # Getting the type of 'ifft' (line 68)
    ifft_24448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'ifft', False)
    # Calling ifft(args, kwargs) (line 68)
    ifft_call_result_24453 = invoke(stypy.reporting.localization.Localization(__file__, 68, 11), ifft_24448, *[result_mul_24451], **kwargs_24452)
    
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type', ifft_call_result_24453)
    
    # ################# End of 'direct_hilbert(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'direct_hilbert' in the type store
    # Getting the type of 'stypy_return_type' (line 63)
    stypy_return_type_24454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24454)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'direct_hilbert'
    return stypy_return_type_24454

# Assigning a type to the variable 'direct_hilbert' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'direct_hilbert', direct_hilbert)

@norecursion
def direct_ihilbert(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'direct_ihilbert'
    module_type_store = module_type_store.open_function_context('direct_ihilbert', 71, 0, False)
    
    # Passed parameters checking function
    direct_ihilbert.stypy_localization = localization
    direct_ihilbert.stypy_type_of_self = None
    direct_ihilbert.stypy_type_store = module_type_store
    direct_ihilbert.stypy_function_name = 'direct_ihilbert'
    direct_ihilbert.stypy_param_names_list = ['x']
    direct_ihilbert.stypy_varargs_param_name = None
    direct_ihilbert.stypy_kwargs_param_name = None
    direct_ihilbert.stypy_call_defaults = defaults
    direct_ihilbert.stypy_call_varargs = varargs
    direct_ihilbert.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'direct_ihilbert', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'direct_ihilbert', localization, ['x'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'direct_ihilbert(...)' code ##################

    
    
    # Call to direct_hilbert(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'x' (line 72)
    x_24456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), 'x', False)
    # Processing the call keyword arguments (line 72)
    kwargs_24457 = {}
    # Getting the type of 'direct_hilbert' (line 72)
    direct_hilbert_24455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'direct_hilbert', False)
    # Calling direct_hilbert(args, kwargs) (line 72)
    direct_hilbert_call_result_24458 = invoke(stypy.reporting.localization.Localization(__file__, 72, 12), direct_hilbert_24455, *[x_24456], **kwargs_24457)
    
    # Applying the 'usub' unary operator (line 72)
    result___neg___24459 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 11), 'usub', direct_hilbert_call_result_24458)
    
    # Assigning a type to the variable 'stypy_return_type' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type', result___neg___24459)
    
    # ################# End of 'direct_ihilbert(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'direct_ihilbert' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_24460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24460)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'direct_ihilbert'
    return stypy_return_type_24460

# Assigning a type to the variable 'direct_ihilbert' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'direct_ihilbert', direct_ihilbert)

@norecursion
def direct_shift(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 75)
    None_24461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 28), 'None')
    defaults = [None_24461]
    # Create a new context for function 'direct_shift'
    module_type_store = module_type_store.open_function_context('direct_shift', 75, 0, False)
    
    # Passed parameters checking function
    direct_shift.stypy_localization = localization
    direct_shift.stypy_type_of_self = None
    direct_shift.stypy_type_store = module_type_store
    direct_shift.stypy_function_name = 'direct_shift'
    direct_shift.stypy_param_names_list = ['x', 'a', 'period']
    direct_shift.stypy_varargs_param_name = None
    direct_shift.stypy_kwargs_param_name = None
    direct_shift.stypy_call_defaults = defaults
    direct_shift.stypy_call_varargs = varargs
    direct_shift.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'direct_shift', ['x', 'a', 'period'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'direct_shift', localization, ['x', 'a', 'period'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'direct_shift(...)' code ##################

    
    # Assigning a Call to a Name (line 76):
    
    # Call to len(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'x' (line 76)
    x_24463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'x', False)
    # Processing the call keyword arguments (line 76)
    kwargs_24464 = {}
    # Getting the type of 'len' (line 76)
    len_24462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'len', False)
    # Calling len(args, kwargs) (line 76)
    len_call_result_24465 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), len_24462, *[x_24463], **kwargs_24464)
    
    # Assigning a type to the variable 'n' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'n', len_call_result_24465)
    
    # Type idiom detected: calculating its left and rigth part (line 77)
    # Getting the type of 'period' (line 77)
    period_24466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 7), 'period')
    # Getting the type of 'None' (line 77)
    None_24467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 17), 'None')
    
    (may_be_24468, more_types_in_union_24469) = may_be_none(period_24466, None_24467)

    if may_be_24468:

        if more_types_in_union_24469:
            # Runtime conditional SSA (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 78):
        
        # Call to fftfreq(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'n' (line 78)
        n_24471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'n', False)
        # Processing the call keyword arguments (line 78)
        kwargs_24472 = {}
        # Getting the type of 'fftfreq' (line 78)
        fftfreq_24470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'fftfreq', False)
        # Calling fftfreq(args, kwargs) (line 78)
        fftfreq_call_result_24473 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), fftfreq_24470, *[n_24471], **kwargs_24472)
        
        complex_24474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 23), 'complex')
        # Applying the binary operator '*' (line 78)
        result_mul_24475 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 12), '*', fftfreq_call_result_24473, complex_24474)
        
        # Getting the type of 'n' (line 78)
        n_24476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'n')
        # Applying the binary operator '*' (line 78)
        result_mul_24477 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 25), '*', result_mul_24475, n_24476)
        
        # Assigning a type to the variable 'k' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'k', result_mul_24477)

        if more_types_in_union_24469:
            # Runtime conditional SSA for else branch (line 77)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_24468) or more_types_in_union_24469):
        
        # Assigning a BinOp to a Name (line 80):
        
        # Call to fftfreq(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'n' (line 80)
        n_24479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'n', False)
        # Processing the call keyword arguments (line 80)
        kwargs_24480 = {}
        # Getting the type of 'fftfreq' (line 80)
        fftfreq_24478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'fftfreq', False)
        # Calling fftfreq(args, kwargs) (line 80)
        fftfreq_call_result_24481 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), fftfreq_24478, *[n_24479], **kwargs_24480)
        
        complex_24482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 23), 'complex')
        # Applying the binary operator '*' (line 80)
        result_mul_24483 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 12), '*', fftfreq_call_result_24481, complex_24482)
        
        # Getting the type of 'pi' (line 80)
        pi_24484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 26), 'pi')
        # Applying the binary operator '*' (line 80)
        result_mul_24485 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 25), '*', result_mul_24483, pi_24484)
        
        # Getting the type of 'period' (line 80)
        period_24486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 29), 'period')
        # Applying the binary operator 'div' (line 80)
        result_div_24487 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 28), 'div', result_mul_24485, period_24486)
        
        # Getting the type of 'n' (line 80)
        n_24488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 36), 'n')
        # Applying the binary operator '*' (line 80)
        result_mul_24489 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 35), '*', result_div_24487, n_24488)
        
        # Assigning a type to the variable 'k' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'k', result_mul_24489)

        if (may_be_24468 and more_types_in_union_24469):
            # SSA join for if statement (line 77)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to ifft(...): (line 81)
    # Processing the call arguments (line 81)
    
    # Call to fft(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'x' (line 81)
    x_24492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'x', False)
    # Processing the call keyword arguments (line 81)
    kwargs_24493 = {}
    # Getting the type of 'fft' (line 81)
    fft_24491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'fft', False)
    # Calling fft(args, kwargs) (line 81)
    fft_call_result_24494 = invoke(stypy.reporting.localization.Localization(__file__, 81, 16), fft_24491, *[x_24492], **kwargs_24493)
    
    
    # Call to exp(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'k' (line 81)
    k_24496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 27), 'k', False)
    # Getting the type of 'a' (line 81)
    a_24497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 29), 'a', False)
    # Applying the binary operator '*' (line 81)
    result_mul_24498 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 27), '*', k_24496, a_24497)
    
    # Processing the call keyword arguments (line 81)
    kwargs_24499 = {}
    # Getting the type of 'exp' (line 81)
    exp_24495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 23), 'exp', False)
    # Calling exp(args, kwargs) (line 81)
    exp_call_result_24500 = invoke(stypy.reporting.localization.Localization(__file__, 81, 23), exp_24495, *[result_mul_24498], **kwargs_24499)
    
    # Applying the binary operator '*' (line 81)
    result_mul_24501 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 16), '*', fft_call_result_24494, exp_call_result_24500)
    
    # Processing the call keyword arguments (line 81)
    kwargs_24502 = {}
    # Getting the type of 'ifft' (line 81)
    ifft_24490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'ifft', False)
    # Calling ifft(args, kwargs) (line 81)
    ifft_call_result_24503 = invoke(stypy.reporting.localization.Localization(__file__, 81, 11), ifft_24490, *[result_mul_24501], **kwargs_24502)
    
    # Obtaining the member 'real' of a type (line 81)
    real_24504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 11), ifft_call_result_24503, 'real')
    # Assigning a type to the variable 'stypy_return_type' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type', real_24504)
    
    # ################# End of 'direct_shift(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'direct_shift' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_24505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24505)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'direct_shift'
    return stypy_return_type_24505

# Assigning a type to the variable 'direct_shift' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'direct_shift', direct_shift)
# Declaration of the 'TestDiff' class

class TestDiff(object, ):

    @norecursion
    def test_definition(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_definition'
        module_type_store = module_type_store.open_function_context('test_definition', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiff.test_definition.__dict__.__setitem__('stypy_localization', localization)
        TestDiff.test_definition.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiff.test_definition.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiff.test_definition.__dict__.__setitem__('stypy_function_name', 'TestDiff.test_definition')
        TestDiff.test_definition.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiff.test_definition.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiff.test_definition.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiff.test_definition.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiff.test_definition.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiff.test_definition.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiff.test_definition.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiff.test_definition', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_definition', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_definition(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 87)
        list_24506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 87)
        # Adding element type (line 87)
        int_24507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 17), list_24506, int_24507)
        # Adding element type (line 87)
        int_24508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 17), list_24506, int_24508)
        # Adding element type (line 87)
        int_24509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 17), list_24506, int_24509)
        # Adding element type (line 87)
        int_24510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 17), list_24506, int_24510)
        # Adding element type (line 87)
        int_24511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 17), list_24506, int_24511)
        
        # Testing the type of a for loop iterable (line 87)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 87, 8), list_24506)
        # Getting the type of the for loop variable (line 87)
        for_loop_var_24512 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 87, 8), list_24506)
        # Assigning a type to the variable 'n' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'n', for_loop_var_24512)
        # SSA begins for a for statement (line 87)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 88):
        
        # Call to arange(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'n' (line 88)
        n_24514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'n', False)
        # Processing the call keyword arguments (line 88)
        kwargs_24515 = {}
        # Getting the type of 'arange' (line 88)
        arange_24513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'arange', False)
        # Calling arange(args, kwargs) (line 88)
        arange_call_result_24516 = invoke(stypy.reporting.localization.Localization(__file__, 88, 16), arange_24513, *[n_24514], **kwargs_24515)
        
        int_24517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 26), 'int')
        # Applying the binary operator '*' (line 88)
        result_mul_24518 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 16), '*', arange_call_result_24516, int_24517)
        
        # Getting the type of 'pi' (line 88)
        pi_24519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 28), 'pi')
        # Applying the binary operator '*' (line 88)
        result_mul_24520 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 27), '*', result_mul_24518, pi_24519)
        
        # Getting the type of 'n' (line 88)
        n_24521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 31), 'n')
        # Applying the binary operator 'div' (line 88)
        result_div_24522 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 30), 'div', result_mul_24520, n_24521)
        
        # Assigning a type to the variable 'x' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'x', result_div_24522)
        
        # Call to assert_array_almost_equal(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Call to diff(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Call to sin(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'x' (line 89)
        x_24526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 47), 'x', False)
        # Processing the call keyword arguments (line 89)
        kwargs_24527 = {}
        # Getting the type of 'sin' (line 89)
        sin_24525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 43), 'sin', False)
        # Calling sin(args, kwargs) (line 89)
        sin_call_result_24528 = invoke(stypy.reporting.localization.Localization(__file__, 89, 43), sin_24525, *[x_24526], **kwargs_24527)
        
        # Processing the call keyword arguments (line 89)
        kwargs_24529 = {}
        # Getting the type of 'diff' (line 89)
        diff_24524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 89)
        diff_call_result_24530 = invoke(stypy.reporting.localization.Localization(__file__, 89, 38), diff_24524, *[sin_call_result_24528], **kwargs_24529)
        
        
        # Call to direct_diff(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Call to sin(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'x' (line 89)
        x_24533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 67), 'x', False)
        # Processing the call keyword arguments (line 89)
        kwargs_24534 = {}
        # Getting the type of 'sin' (line 89)
        sin_24532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 63), 'sin', False)
        # Calling sin(args, kwargs) (line 89)
        sin_call_result_24535 = invoke(stypy.reporting.localization.Localization(__file__, 89, 63), sin_24532, *[x_24533], **kwargs_24534)
        
        # Processing the call keyword arguments (line 89)
        kwargs_24536 = {}
        # Getting the type of 'direct_diff' (line 89)
        direct_diff_24531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 51), 'direct_diff', False)
        # Calling direct_diff(args, kwargs) (line 89)
        direct_diff_call_result_24537 = invoke(stypy.reporting.localization.Localization(__file__, 89, 51), direct_diff_24531, *[sin_call_result_24535], **kwargs_24536)
        
        # Processing the call keyword arguments (line 89)
        kwargs_24538 = {}
        # Getting the type of 'assert_array_almost_equal' (line 89)
        assert_array_almost_equal_24523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 89)
        assert_array_almost_equal_call_result_24539 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), assert_array_almost_equal_24523, *[diff_call_result_24530, direct_diff_call_result_24537], **kwargs_24538)
        
        
        # Call to assert_array_almost_equal(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Call to diff(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Call to sin(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'x' (line 90)
        x_24543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 47), 'x', False)
        # Processing the call keyword arguments (line 90)
        kwargs_24544 = {}
        # Getting the type of 'sin' (line 90)
        sin_24542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 43), 'sin', False)
        # Calling sin(args, kwargs) (line 90)
        sin_call_result_24545 = invoke(stypy.reporting.localization.Localization(__file__, 90, 43), sin_24542, *[x_24543], **kwargs_24544)
        
        int_24546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 50), 'int')
        # Processing the call keyword arguments (line 90)
        kwargs_24547 = {}
        # Getting the type of 'diff' (line 90)
        diff_24541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 90)
        diff_call_result_24548 = invoke(stypy.reporting.localization.Localization(__file__, 90, 38), diff_24541, *[sin_call_result_24545, int_24546], **kwargs_24547)
        
        
        # Call to direct_diff(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Call to sin(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'x' (line 90)
        x_24551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 69), 'x', False)
        # Processing the call keyword arguments (line 90)
        kwargs_24552 = {}
        # Getting the type of 'sin' (line 90)
        sin_24550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 65), 'sin', False)
        # Calling sin(args, kwargs) (line 90)
        sin_call_result_24553 = invoke(stypy.reporting.localization.Localization(__file__, 90, 65), sin_24550, *[x_24551], **kwargs_24552)
        
        int_24554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 72), 'int')
        # Processing the call keyword arguments (line 90)
        kwargs_24555 = {}
        # Getting the type of 'direct_diff' (line 90)
        direct_diff_24549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 53), 'direct_diff', False)
        # Calling direct_diff(args, kwargs) (line 90)
        direct_diff_call_result_24556 = invoke(stypy.reporting.localization.Localization(__file__, 90, 53), direct_diff_24549, *[sin_call_result_24553, int_24554], **kwargs_24555)
        
        # Processing the call keyword arguments (line 90)
        kwargs_24557 = {}
        # Getting the type of 'assert_array_almost_equal' (line 90)
        assert_array_almost_equal_24540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 90)
        assert_array_almost_equal_call_result_24558 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), assert_array_almost_equal_24540, *[diff_call_result_24548, direct_diff_call_result_24556], **kwargs_24557)
        
        
        # Call to assert_array_almost_equal(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Call to diff(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Call to sin(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'x' (line 91)
        x_24562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 47), 'x', False)
        # Processing the call keyword arguments (line 91)
        kwargs_24563 = {}
        # Getting the type of 'sin' (line 91)
        sin_24561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'sin', False)
        # Calling sin(args, kwargs) (line 91)
        sin_call_result_24564 = invoke(stypy.reporting.localization.Localization(__file__, 91, 43), sin_24561, *[x_24562], **kwargs_24563)
        
        int_24565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 50), 'int')
        # Processing the call keyword arguments (line 91)
        kwargs_24566 = {}
        # Getting the type of 'diff' (line 91)
        diff_24560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 91)
        diff_call_result_24567 = invoke(stypy.reporting.localization.Localization(__file__, 91, 38), diff_24560, *[sin_call_result_24564, int_24565], **kwargs_24566)
        
        
        # Call to direct_diff(...): (line 91)
        # Processing the call arguments (line 91)
        
        # Call to sin(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'x' (line 91)
        x_24570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 69), 'x', False)
        # Processing the call keyword arguments (line 91)
        kwargs_24571 = {}
        # Getting the type of 'sin' (line 91)
        sin_24569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 65), 'sin', False)
        # Calling sin(args, kwargs) (line 91)
        sin_call_result_24572 = invoke(stypy.reporting.localization.Localization(__file__, 91, 65), sin_24569, *[x_24570], **kwargs_24571)
        
        int_24573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 72), 'int')
        # Processing the call keyword arguments (line 91)
        kwargs_24574 = {}
        # Getting the type of 'direct_diff' (line 91)
        direct_diff_24568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 53), 'direct_diff', False)
        # Calling direct_diff(args, kwargs) (line 91)
        direct_diff_call_result_24575 = invoke(stypy.reporting.localization.Localization(__file__, 91, 53), direct_diff_24568, *[sin_call_result_24572, int_24573], **kwargs_24574)
        
        # Processing the call keyword arguments (line 91)
        kwargs_24576 = {}
        # Getting the type of 'assert_array_almost_equal' (line 91)
        assert_array_almost_equal_24559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 91)
        assert_array_almost_equal_call_result_24577 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), assert_array_almost_equal_24559, *[diff_call_result_24567, direct_diff_call_result_24575], **kwargs_24576)
        
        
        # Call to assert_array_almost_equal(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Call to diff(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Call to sin(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'x' (line 92)
        x_24581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 47), 'x', False)
        # Processing the call keyword arguments (line 92)
        kwargs_24582 = {}
        # Getting the type of 'sin' (line 92)
        sin_24580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 43), 'sin', False)
        # Calling sin(args, kwargs) (line 92)
        sin_call_result_24583 = invoke(stypy.reporting.localization.Localization(__file__, 92, 43), sin_24580, *[x_24581], **kwargs_24582)
        
        int_24584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 50), 'int')
        # Processing the call keyword arguments (line 92)
        kwargs_24585 = {}
        # Getting the type of 'diff' (line 92)
        diff_24579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 92)
        diff_call_result_24586 = invoke(stypy.reporting.localization.Localization(__file__, 92, 38), diff_24579, *[sin_call_result_24583, int_24584], **kwargs_24585)
        
        
        # Call to direct_diff(...): (line 92)
        # Processing the call arguments (line 92)
        
        # Call to sin(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'x' (line 92)
        x_24589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 69), 'x', False)
        # Processing the call keyword arguments (line 92)
        kwargs_24590 = {}
        # Getting the type of 'sin' (line 92)
        sin_24588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 65), 'sin', False)
        # Calling sin(args, kwargs) (line 92)
        sin_call_result_24591 = invoke(stypy.reporting.localization.Localization(__file__, 92, 65), sin_24588, *[x_24589], **kwargs_24590)
        
        int_24592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 72), 'int')
        # Processing the call keyword arguments (line 92)
        kwargs_24593 = {}
        # Getting the type of 'direct_diff' (line 92)
        direct_diff_24587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 53), 'direct_diff', False)
        # Calling direct_diff(args, kwargs) (line 92)
        direct_diff_call_result_24594 = invoke(stypy.reporting.localization.Localization(__file__, 92, 53), direct_diff_24587, *[sin_call_result_24591, int_24592], **kwargs_24593)
        
        # Processing the call keyword arguments (line 92)
        kwargs_24595 = {}
        # Getting the type of 'assert_array_almost_equal' (line 92)
        assert_array_almost_equal_24578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 92)
        assert_array_almost_equal_call_result_24596 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), assert_array_almost_equal_24578, *[diff_call_result_24586, direct_diff_call_result_24594], **kwargs_24595)
        
        
        # Call to assert_array_almost_equal(...): (line 93)
        # Processing the call arguments (line 93)
        
        # Call to diff(...): (line 93)
        # Processing the call arguments (line 93)
        
        # Call to sin(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'x' (line 93)
        x_24600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 47), 'x', False)
        # Processing the call keyword arguments (line 93)
        kwargs_24601 = {}
        # Getting the type of 'sin' (line 93)
        sin_24599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 43), 'sin', False)
        # Calling sin(args, kwargs) (line 93)
        sin_call_result_24602 = invoke(stypy.reporting.localization.Localization(__file__, 93, 43), sin_24599, *[x_24600], **kwargs_24601)
        
        int_24603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 50), 'int')
        # Processing the call keyword arguments (line 93)
        kwargs_24604 = {}
        # Getting the type of 'diff' (line 93)
        diff_24598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 93)
        diff_call_result_24605 = invoke(stypy.reporting.localization.Localization(__file__, 93, 38), diff_24598, *[sin_call_result_24602, int_24603], **kwargs_24604)
        
        
        # Call to direct_diff(...): (line 93)
        # Processing the call arguments (line 93)
        
        # Call to sin(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'x' (line 93)
        x_24608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 69), 'x', False)
        # Processing the call keyword arguments (line 93)
        kwargs_24609 = {}
        # Getting the type of 'sin' (line 93)
        sin_24607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 65), 'sin', False)
        # Calling sin(args, kwargs) (line 93)
        sin_call_result_24610 = invoke(stypy.reporting.localization.Localization(__file__, 93, 65), sin_24607, *[x_24608], **kwargs_24609)
        
        int_24611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 72), 'int')
        # Processing the call keyword arguments (line 93)
        kwargs_24612 = {}
        # Getting the type of 'direct_diff' (line 93)
        direct_diff_24606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 53), 'direct_diff', False)
        # Calling direct_diff(args, kwargs) (line 93)
        direct_diff_call_result_24613 = invoke(stypy.reporting.localization.Localization(__file__, 93, 53), direct_diff_24606, *[sin_call_result_24610, int_24611], **kwargs_24612)
        
        # Processing the call keyword arguments (line 93)
        kwargs_24614 = {}
        # Getting the type of 'assert_array_almost_equal' (line 93)
        assert_array_almost_equal_24597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 93)
        assert_array_almost_equal_call_result_24615 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), assert_array_almost_equal_24597, *[diff_call_result_24605, direct_diff_call_result_24613], **kwargs_24614)
        
        
        # Call to assert_array_almost_equal(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Call to diff(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Call to sin(...): (line 94)
        # Processing the call arguments (line 94)
        int_24619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 47), 'int')
        # Getting the type of 'x' (line 94)
        x_24620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 49), 'x', False)
        # Applying the binary operator '*' (line 94)
        result_mul_24621 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 47), '*', int_24619, x_24620)
        
        # Processing the call keyword arguments (line 94)
        kwargs_24622 = {}
        # Getting the type of 'sin' (line 94)
        sin_24618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 43), 'sin', False)
        # Calling sin(args, kwargs) (line 94)
        sin_call_result_24623 = invoke(stypy.reporting.localization.Localization(__file__, 94, 43), sin_24618, *[result_mul_24621], **kwargs_24622)
        
        int_24624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 52), 'int')
        # Processing the call keyword arguments (line 94)
        kwargs_24625 = {}
        # Getting the type of 'diff' (line 94)
        diff_24617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 94)
        diff_call_result_24626 = invoke(stypy.reporting.localization.Localization(__file__, 94, 38), diff_24617, *[sin_call_result_24623, int_24624], **kwargs_24625)
        
        
        # Call to direct_diff(...): (line 94)
        # Processing the call arguments (line 94)
        
        # Call to sin(...): (line 94)
        # Processing the call arguments (line 94)
        int_24629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 71), 'int')
        # Getting the type of 'x' (line 94)
        x_24630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 73), 'x', False)
        # Applying the binary operator '*' (line 94)
        result_mul_24631 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 71), '*', int_24629, x_24630)
        
        # Processing the call keyword arguments (line 94)
        kwargs_24632 = {}
        # Getting the type of 'sin' (line 94)
        sin_24628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 67), 'sin', False)
        # Calling sin(args, kwargs) (line 94)
        sin_call_result_24633 = invoke(stypy.reporting.localization.Localization(__file__, 94, 67), sin_24628, *[result_mul_24631], **kwargs_24632)
        
        int_24634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 76), 'int')
        # Processing the call keyword arguments (line 94)
        kwargs_24635 = {}
        # Getting the type of 'direct_diff' (line 94)
        direct_diff_24627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 55), 'direct_diff', False)
        # Calling direct_diff(args, kwargs) (line 94)
        direct_diff_call_result_24636 = invoke(stypy.reporting.localization.Localization(__file__, 94, 55), direct_diff_24627, *[sin_call_result_24633, int_24634], **kwargs_24635)
        
        # Processing the call keyword arguments (line 94)
        kwargs_24637 = {}
        # Getting the type of 'assert_array_almost_equal' (line 94)
        assert_array_almost_equal_24616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 94)
        assert_array_almost_equal_call_result_24638 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), assert_array_almost_equal_24616, *[diff_call_result_24626, direct_diff_call_result_24636], **kwargs_24637)
        
        
        # Call to assert_array_almost_equal(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Call to diff(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Call to sin(...): (line 95)
        # Processing the call arguments (line 95)
        int_24642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 47), 'int')
        # Getting the type of 'x' (line 95)
        x_24643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 49), 'x', False)
        # Applying the binary operator '*' (line 95)
        result_mul_24644 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 47), '*', int_24642, x_24643)
        
        # Processing the call keyword arguments (line 95)
        kwargs_24645 = {}
        # Getting the type of 'sin' (line 95)
        sin_24641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 43), 'sin', False)
        # Calling sin(args, kwargs) (line 95)
        sin_call_result_24646 = invoke(stypy.reporting.localization.Localization(__file__, 95, 43), sin_24641, *[result_mul_24644], **kwargs_24645)
        
        int_24647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 52), 'int')
        # Processing the call keyword arguments (line 95)
        kwargs_24648 = {}
        # Getting the type of 'diff' (line 95)
        diff_24640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 95)
        diff_call_result_24649 = invoke(stypy.reporting.localization.Localization(__file__, 95, 38), diff_24640, *[sin_call_result_24646, int_24647], **kwargs_24648)
        
        
        # Call to direct_diff(...): (line 95)
        # Processing the call arguments (line 95)
        
        # Call to sin(...): (line 95)
        # Processing the call arguments (line 95)
        int_24652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 71), 'int')
        # Getting the type of 'x' (line 95)
        x_24653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 73), 'x', False)
        # Applying the binary operator '*' (line 95)
        result_mul_24654 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 71), '*', int_24652, x_24653)
        
        # Processing the call keyword arguments (line 95)
        kwargs_24655 = {}
        # Getting the type of 'sin' (line 95)
        sin_24651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 67), 'sin', False)
        # Calling sin(args, kwargs) (line 95)
        sin_call_result_24656 = invoke(stypy.reporting.localization.Localization(__file__, 95, 67), sin_24651, *[result_mul_24654], **kwargs_24655)
        
        int_24657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 76), 'int')
        # Processing the call keyword arguments (line 95)
        kwargs_24658 = {}
        # Getting the type of 'direct_diff' (line 95)
        direct_diff_24650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 55), 'direct_diff', False)
        # Calling direct_diff(args, kwargs) (line 95)
        direct_diff_call_result_24659 = invoke(stypy.reporting.localization.Localization(__file__, 95, 55), direct_diff_24650, *[sin_call_result_24656, int_24657], **kwargs_24658)
        
        # Processing the call keyword arguments (line 95)
        kwargs_24660 = {}
        # Getting the type of 'assert_array_almost_equal' (line 95)
        assert_array_almost_equal_24639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 95)
        assert_array_almost_equal_call_result_24661 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), assert_array_almost_equal_24639, *[diff_call_result_24649, direct_diff_call_result_24659], **kwargs_24660)
        
        
        # Call to assert_array_almost_equal(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to diff(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to cos(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'x' (line 96)
        x_24665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 47), 'x', False)
        # Processing the call keyword arguments (line 96)
        kwargs_24666 = {}
        # Getting the type of 'cos' (line 96)
        cos_24664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 43), 'cos', False)
        # Calling cos(args, kwargs) (line 96)
        cos_call_result_24667 = invoke(stypy.reporting.localization.Localization(__file__, 96, 43), cos_24664, *[x_24665], **kwargs_24666)
        
        # Processing the call keyword arguments (line 96)
        kwargs_24668 = {}
        # Getting the type of 'diff' (line 96)
        diff_24663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 96)
        diff_call_result_24669 = invoke(stypy.reporting.localization.Localization(__file__, 96, 38), diff_24663, *[cos_call_result_24667], **kwargs_24668)
        
        
        # Call to direct_diff(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to cos(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'x' (line 96)
        x_24672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 67), 'x', False)
        # Processing the call keyword arguments (line 96)
        kwargs_24673 = {}
        # Getting the type of 'cos' (line 96)
        cos_24671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 63), 'cos', False)
        # Calling cos(args, kwargs) (line 96)
        cos_call_result_24674 = invoke(stypy.reporting.localization.Localization(__file__, 96, 63), cos_24671, *[x_24672], **kwargs_24673)
        
        # Processing the call keyword arguments (line 96)
        kwargs_24675 = {}
        # Getting the type of 'direct_diff' (line 96)
        direct_diff_24670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 51), 'direct_diff', False)
        # Calling direct_diff(args, kwargs) (line 96)
        direct_diff_call_result_24676 = invoke(stypy.reporting.localization.Localization(__file__, 96, 51), direct_diff_24670, *[cos_call_result_24674], **kwargs_24675)
        
        # Processing the call keyword arguments (line 96)
        kwargs_24677 = {}
        # Getting the type of 'assert_array_almost_equal' (line 96)
        assert_array_almost_equal_24662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 96)
        assert_array_almost_equal_call_result_24678 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), assert_array_almost_equal_24662, *[diff_call_result_24669, direct_diff_call_result_24676], **kwargs_24677)
        
        
        # Call to assert_array_almost_equal(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Call to diff(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Call to cos(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'x' (line 97)
        x_24682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 47), 'x', False)
        # Processing the call keyword arguments (line 97)
        kwargs_24683 = {}
        # Getting the type of 'cos' (line 97)
        cos_24681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 43), 'cos', False)
        # Calling cos(args, kwargs) (line 97)
        cos_call_result_24684 = invoke(stypy.reporting.localization.Localization(__file__, 97, 43), cos_24681, *[x_24682], **kwargs_24683)
        
        int_24685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 50), 'int')
        # Processing the call keyword arguments (line 97)
        kwargs_24686 = {}
        # Getting the type of 'diff' (line 97)
        diff_24680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 97)
        diff_call_result_24687 = invoke(stypy.reporting.localization.Localization(__file__, 97, 38), diff_24680, *[cos_call_result_24684, int_24685], **kwargs_24686)
        
        
        # Call to direct_diff(...): (line 97)
        # Processing the call arguments (line 97)
        
        # Call to cos(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'x' (line 97)
        x_24690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 69), 'x', False)
        # Processing the call keyword arguments (line 97)
        kwargs_24691 = {}
        # Getting the type of 'cos' (line 97)
        cos_24689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 65), 'cos', False)
        # Calling cos(args, kwargs) (line 97)
        cos_call_result_24692 = invoke(stypy.reporting.localization.Localization(__file__, 97, 65), cos_24689, *[x_24690], **kwargs_24691)
        
        int_24693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 72), 'int')
        # Processing the call keyword arguments (line 97)
        kwargs_24694 = {}
        # Getting the type of 'direct_diff' (line 97)
        direct_diff_24688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 53), 'direct_diff', False)
        # Calling direct_diff(args, kwargs) (line 97)
        direct_diff_call_result_24695 = invoke(stypy.reporting.localization.Localization(__file__, 97, 53), direct_diff_24688, *[cos_call_result_24692, int_24693], **kwargs_24694)
        
        # Processing the call keyword arguments (line 97)
        kwargs_24696 = {}
        # Getting the type of 'assert_array_almost_equal' (line 97)
        assert_array_almost_equal_24679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 97)
        assert_array_almost_equal_call_result_24697 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), assert_array_almost_equal_24679, *[diff_call_result_24687, direct_diff_call_result_24695], **kwargs_24696)
        
        
        # Call to assert_array_almost_equal(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Call to diff(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Call to cos(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'x' (line 98)
        x_24701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 47), 'x', False)
        # Processing the call keyword arguments (line 98)
        kwargs_24702 = {}
        # Getting the type of 'cos' (line 98)
        cos_24700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 43), 'cos', False)
        # Calling cos(args, kwargs) (line 98)
        cos_call_result_24703 = invoke(stypy.reporting.localization.Localization(__file__, 98, 43), cos_24700, *[x_24701], **kwargs_24702)
        
        int_24704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 50), 'int')
        # Processing the call keyword arguments (line 98)
        kwargs_24705 = {}
        # Getting the type of 'diff' (line 98)
        diff_24699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 98)
        diff_call_result_24706 = invoke(stypy.reporting.localization.Localization(__file__, 98, 38), diff_24699, *[cos_call_result_24703, int_24704], **kwargs_24705)
        
        
        # Call to direct_diff(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Call to cos(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'x' (line 98)
        x_24709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 69), 'x', False)
        # Processing the call keyword arguments (line 98)
        kwargs_24710 = {}
        # Getting the type of 'cos' (line 98)
        cos_24708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 65), 'cos', False)
        # Calling cos(args, kwargs) (line 98)
        cos_call_result_24711 = invoke(stypy.reporting.localization.Localization(__file__, 98, 65), cos_24708, *[x_24709], **kwargs_24710)
        
        int_24712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 72), 'int')
        # Processing the call keyword arguments (line 98)
        kwargs_24713 = {}
        # Getting the type of 'direct_diff' (line 98)
        direct_diff_24707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 53), 'direct_diff', False)
        # Calling direct_diff(args, kwargs) (line 98)
        direct_diff_call_result_24714 = invoke(stypy.reporting.localization.Localization(__file__, 98, 53), direct_diff_24707, *[cos_call_result_24711, int_24712], **kwargs_24713)
        
        # Processing the call keyword arguments (line 98)
        kwargs_24715 = {}
        # Getting the type of 'assert_array_almost_equal' (line 98)
        assert_array_almost_equal_24698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 98)
        assert_array_almost_equal_call_result_24716 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), assert_array_almost_equal_24698, *[diff_call_result_24706, direct_diff_call_result_24714], **kwargs_24715)
        
        
        # Call to assert_array_almost_equal(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Call to diff(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Call to cos(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'x' (line 99)
        x_24720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 47), 'x', False)
        # Processing the call keyword arguments (line 99)
        kwargs_24721 = {}
        # Getting the type of 'cos' (line 99)
        cos_24719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 43), 'cos', False)
        # Calling cos(args, kwargs) (line 99)
        cos_call_result_24722 = invoke(stypy.reporting.localization.Localization(__file__, 99, 43), cos_24719, *[x_24720], **kwargs_24721)
        
        int_24723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 50), 'int')
        # Processing the call keyword arguments (line 99)
        kwargs_24724 = {}
        # Getting the type of 'diff' (line 99)
        diff_24718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 99)
        diff_call_result_24725 = invoke(stypy.reporting.localization.Localization(__file__, 99, 38), diff_24718, *[cos_call_result_24722, int_24723], **kwargs_24724)
        
        
        # Call to direct_diff(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Call to cos(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'x' (line 99)
        x_24728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 69), 'x', False)
        # Processing the call keyword arguments (line 99)
        kwargs_24729 = {}
        # Getting the type of 'cos' (line 99)
        cos_24727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 65), 'cos', False)
        # Calling cos(args, kwargs) (line 99)
        cos_call_result_24730 = invoke(stypy.reporting.localization.Localization(__file__, 99, 65), cos_24727, *[x_24728], **kwargs_24729)
        
        int_24731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 72), 'int')
        # Processing the call keyword arguments (line 99)
        kwargs_24732 = {}
        # Getting the type of 'direct_diff' (line 99)
        direct_diff_24726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 53), 'direct_diff', False)
        # Calling direct_diff(args, kwargs) (line 99)
        direct_diff_call_result_24733 = invoke(stypy.reporting.localization.Localization(__file__, 99, 53), direct_diff_24726, *[cos_call_result_24730, int_24731], **kwargs_24732)
        
        # Processing the call keyword arguments (line 99)
        kwargs_24734 = {}
        # Getting the type of 'assert_array_almost_equal' (line 99)
        assert_array_almost_equal_24717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 99)
        assert_array_almost_equal_call_result_24735 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), assert_array_almost_equal_24717, *[diff_call_result_24725, direct_diff_call_result_24733], **kwargs_24734)
        
        
        # Call to assert_array_almost_equal(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Call to diff(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Call to cos(...): (line 100)
        # Processing the call arguments (line 100)
        int_24739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 47), 'int')
        # Getting the type of 'x' (line 100)
        x_24740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 49), 'x', False)
        # Applying the binary operator '*' (line 100)
        result_mul_24741 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 47), '*', int_24739, x_24740)
        
        # Processing the call keyword arguments (line 100)
        kwargs_24742 = {}
        # Getting the type of 'cos' (line 100)
        cos_24738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 43), 'cos', False)
        # Calling cos(args, kwargs) (line 100)
        cos_call_result_24743 = invoke(stypy.reporting.localization.Localization(__file__, 100, 43), cos_24738, *[result_mul_24741], **kwargs_24742)
        
        # Processing the call keyword arguments (line 100)
        kwargs_24744 = {}
        # Getting the type of 'diff' (line 100)
        diff_24737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 100)
        diff_call_result_24745 = invoke(stypy.reporting.localization.Localization(__file__, 100, 38), diff_24737, *[cos_call_result_24743], **kwargs_24744)
        
        
        # Call to direct_diff(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Call to cos(...): (line 100)
        # Processing the call arguments (line 100)
        int_24748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 69), 'int')
        # Getting the type of 'x' (line 100)
        x_24749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 71), 'x', False)
        # Applying the binary operator '*' (line 100)
        result_mul_24750 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 69), '*', int_24748, x_24749)
        
        # Processing the call keyword arguments (line 100)
        kwargs_24751 = {}
        # Getting the type of 'cos' (line 100)
        cos_24747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 65), 'cos', False)
        # Calling cos(args, kwargs) (line 100)
        cos_call_result_24752 = invoke(stypy.reporting.localization.Localization(__file__, 100, 65), cos_24747, *[result_mul_24750], **kwargs_24751)
        
        # Processing the call keyword arguments (line 100)
        kwargs_24753 = {}
        # Getting the type of 'direct_diff' (line 100)
        direct_diff_24746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 53), 'direct_diff', False)
        # Calling direct_diff(args, kwargs) (line 100)
        direct_diff_call_result_24754 = invoke(stypy.reporting.localization.Localization(__file__, 100, 53), direct_diff_24746, *[cos_call_result_24752], **kwargs_24753)
        
        # Processing the call keyword arguments (line 100)
        kwargs_24755 = {}
        # Getting the type of 'assert_array_almost_equal' (line 100)
        assert_array_almost_equal_24736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 100)
        assert_array_almost_equal_call_result_24756 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), assert_array_almost_equal_24736, *[diff_call_result_24745, direct_diff_call_result_24754], **kwargs_24755)
        
        
        # Call to assert_array_almost_equal(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Call to diff(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Call to sin(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'x' (line 101)
        x_24760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 47), 'x', False)
        # Getting the type of 'n' (line 101)
        n_24761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 49), 'n', False)
        # Applying the binary operator '*' (line 101)
        result_mul_24762 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 47), '*', x_24760, n_24761)
        
        int_24763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 51), 'int')
        # Applying the binary operator 'div' (line 101)
        result_div_24764 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 50), 'div', result_mul_24762, int_24763)
        
        # Processing the call keyword arguments (line 101)
        kwargs_24765 = {}
        # Getting the type of 'sin' (line 101)
        sin_24759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 43), 'sin', False)
        # Calling sin(args, kwargs) (line 101)
        sin_call_result_24766 = invoke(stypy.reporting.localization.Localization(__file__, 101, 43), sin_24759, *[result_div_24764], **kwargs_24765)
        
        # Processing the call keyword arguments (line 101)
        kwargs_24767 = {}
        # Getting the type of 'diff' (line 101)
        diff_24758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 101)
        diff_call_result_24768 = invoke(stypy.reporting.localization.Localization(__file__, 101, 38), diff_24758, *[sin_call_result_24766], **kwargs_24767)
        
        
        # Call to direct_diff(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Call to sin(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'x' (line 101)
        x_24771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 71), 'x', False)
        # Getting the type of 'n' (line 101)
        n_24772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 73), 'n', False)
        # Applying the binary operator '*' (line 101)
        result_mul_24773 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 71), '*', x_24771, n_24772)
        
        int_24774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 75), 'int')
        # Applying the binary operator 'div' (line 101)
        result_div_24775 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 74), 'div', result_mul_24773, int_24774)
        
        # Processing the call keyword arguments (line 101)
        kwargs_24776 = {}
        # Getting the type of 'sin' (line 101)
        sin_24770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 67), 'sin', False)
        # Calling sin(args, kwargs) (line 101)
        sin_call_result_24777 = invoke(stypy.reporting.localization.Localization(__file__, 101, 67), sin_24770, *[result_div_24775], **kwargs_24776)
        
        # Processing the call keyword arguments (line 101)
        kwargs_24778 = {}
        # Getting the type of 'direct_diff' (line 101)
        direct_diff_24769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 55), 'direct_diff', False)
        # Calling direct_diff(args, kwargs) (line 101)
        direct_diff_call_result_24779 = invoke(stypy.reporting.localization.Localization(__file__, 101, 55), direct_diff_24769, *[sin_call_result_24777], **kwargs_24778)
        
        # Processing the call keyword arguments (line 101)
        kwargs_24780 = {}
        # Getting the type of 'assert_array_almost_equal' (line 101)
        assert_array_almost_equal_24757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 101)
        assert_array_almost_equal_call_result_24781 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), assert_array_almost_equal_24757, *[diff_call_result_24768, direct_diff_call_result_24779], **kwargs_24780)
        
        
        # Call to assert_array_almost_equal(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Call to diff(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Call to cos(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'x' (line 102)
        x_24785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 47), 'x', False)
        # Getting the type of 'n' (line 102)
        n_24786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 49), 'n', False)
        # Applying the binary operator '*' (line 102)
        result_mul_24787 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 47), '*', x_24785, n_24786)
        
        int_24788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 51), 'int')
        # Applying the binary operator 'div' (line 102)
        result_div_24789 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 50), 'div', result_mul_24787, int_24788)
        
        # Processing the call keyword arguments (line 102)
        kwargs_24790 = {}
        # Getting the type of 'cos' (line 102)
        cos_24784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 43), 'cos', False)
        # Calling cos(args, kwargs) (line 102)
        cos_call_result_24791 = invoke(stypy.reporting.localization.Localization(__file__, 102, 43), cos_24784, *[result_div_24789], **kwargs_24790)
        
        # Processing the call keyword arguments (line 102)
        kwargs_24792 = {}
        # Getting the type of 'diff' (line 102)
        diff_24783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 102)
        diff_call_result_24793 = invoke(stypy.reporting.localization.Localization(__file__, 102, 38), diff_24783, *[cos_call_result_24791], **kwargs_24792)
        
        
        # Call to direct_diff(...): (line 102)
        # Processing the call arguments (line 102)
        
        # Call to cos(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'x' (line 102)
        x_24796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 71), 'x', False)
        # Getting the type of 'n' (line 102)
        n_24797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 73), 'n', False)
        # Applying the binary operator '*' (line 102)
        result_mul_24798 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 71), '*', x_24796, n_24797)
        
        int_24799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 75), 'int')
        # Applying the binary operator 'div' (line 102)
        result_div_24800 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 74), 'div', result_mul_24798, int_24799)
        
        # Processing the call keyword arguments (line 102)
        kwargs_24801 = {}
        # Getting the type of 'cos' (line 102)
        cos_24795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 67), 'cos', False)
        # Calling cos(args, kwargs) (line 102)
        cos_call_result_24802 = invoke(stypy.reporting.localization.Localization(__file__, 102, 67), cos_24795, *[result_div_24800], **kwargs_24801)
        
        # Processing the call keyword arguments (line 102)
        kwargs_24803 = {}
        # Getting the type of 'direct_diff' (line 102)
        direct_diff_24794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 55), 'direct_diff', False)
        # Calling direct_diff(args, kwargs) (line 102)
        direct_diff_call_result_24804 = invoke(stypy.reporting.localization.Localization(__file__, 102, 55), direct_diff_24794, *[cos_call_result_24802], **kwargs_24803)
        
        # Processing the call keyword arguments (line 102)
        kwargs_24805 = {}
        # Getting the type of 'assert_array_almost_equal' (line 102)
        assert_array_almost_equal_24782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 102)
        assert_array_almost_equal_call_result_24806 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), assert_array_almost_equal_24782, *[diff_call_result_24793, direct_diff_call_result_24804], **kwargs_24805)
        
        
        
        # Call to range(...): (line 103)
        # Processing the call arguments (line 103)
        int_24808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 27), 'int')
        # Processing the call keyword arguments (line 103)
        kwargs_24809 = {}
        # Getting the type of 'range' (line 103)
        range_24807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 21), 'range', False)
        # Calling range(args, kwargs) (line 103)
        range_call_result_24810 = invoke(stypy.reporting.localization.Localization(__file__, 103, 21), range_24807, *[int_24808], **kwargs_24809)
        
        # Testing the type of a for loop iterable (line 103)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 103, 12), range_call_result_24810)
        # Getting the type of the for loop variable (line 103)
        for_loop_var_24811 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 103, 12), range_call_result_24810)
        # Assigning a type to the variable 'k' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'k', for_loop_var_24811)
        # SSA begins for a for statement (line 103)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_array_almost_equal(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Call to diff(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Call to sin(...): (line 104)
        # Processing the call arguments (line 104)
        int_24815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 51), 'int')
        # Getting the type of 'x' (line 104)
        x_24816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 53), 'x', False)
        # Applying the binary operator '*' (line 104)
        result_mul_24817 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 51), '*', int_24815, x_24816)
        
        # Processing the call keyword arguments (line 104)
        kwargs_24818 = {}
        # Getting the type of 'sin' (line 104)
        sin_24814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 47), 'sin', False)
        # Calling sin(args, kwargs) (line 104)
        sin_call_result_24819 = invoke(stypy.reporting.localization.Localization(__file__, 104, 47), sin_24814, *[result_mul_24817], **kwargs_24818)
        
        # Getting the type of 'k' (line 104)
        k_24820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 56), 'k', False)
        # Processing the call keyword arguments (line 104)
        kwargs_24821 = {}
        # Getting the type of 'diff' (line 104)
        diff_24813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 42), 'diff', False)
        # Calling diff(args, kwargs) (line 104)
        diff_call_result_24822 = invoke(stypy.reporting.localization.Localization(__file__, 104, 42), diff_24813, *[sin_call_result_24819, k_24820], **kwargs_24821)
        
        
        # Call to direct_diff(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Call to sin(...): (line 104)
        # Processing the call arguments (line 104)
        int_24825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 75), 'int')
        # Getting the type of 'x' (line 104)
        x_24826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 77), 'x', False)
        # Applying the binary operator '*' (line 104)
        result_mul_24827 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 75), '*', int_24825, x_24826)
        
        # Processing the call keyword arguments (line 104)
        kwargs_24828 = {}
        # Getting the type of 'sin' (line 104)
        sin_24824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 71), 'sin', False)
        # Calling sin(args, kwargs) (line 104)
        sin_call_result_24829 = invoke(stypy.reporting.localization.Localization(__file__, 104, 71), sin_24824, *[result_mul_24827], **kwargs_24828)
        
        # Getting the type of 'k' (line 104)
        k_24830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 80), 'k', False)
        # Processing the call keyword arguments (line 104)
        kwargs_24831 = {}
        # Getting the type of 'direct_diff' (line 104)
        direct_diff_24823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 59), 'direct_diff', False)
        # Calling direct_diff(args, kwargs) (line 104)
        direct_diff_call_result_24832 = invoke(stypy.reporting.localization.Localization(__file__, 104, 59), direct_diff_24823, *[sin_call_result_24829, k_24830], **kwargs_24831)
        
        # Processing the call keyword arguments (line 104)
        kwargs_24833 = {}
        # Getting the type of 'assert_array_almost_equal' (line 104)
        assert_array_almost_equal_24812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 104)
        assert_array_almost_equal_call_result_24834 = invoke(stypy.reporting.localization.Localization(__file__, 104, 16), assert_array_almost_equal_24812, *[diff_call_result_24822, direct_diff_call_result_24832], **kwargs_24833)
        
        
        # Call to assert_array_almost_equal(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Call to diff(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Call to cos(...): (line 105)
        # Processing the call arguments (line 105)
        int_24838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 51), 'int')
        # Getting the type of 'x' (line 105)
        x_24839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 53), 'x', False)
        # Applying the binary operator '*' (line 105)
        result_mul_24840 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 51), '*', int_24838, x_24839)
        
        # Processing the call keyword arguments (line 105)
        kwargs_24841 = {}
        # Getting the type of 'cos' (line 105)
        cos_24837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 47), 'cos', False)
        # Calling cos(args, kwargs) (line 105)
        cos_call_result_24842 = invoke(stypy.reporting.localization.Localization(__file__, 105, 47), cos_24837, *[result_mul_24840], **kwargs_24841)
        
        # Getting the type of 'k' (line 105)
        k_24843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 56), 'k', False)
        # Processing the call keyword arguments (line 105)
        kwargs_24844 = {}
        # Getting the type of 'diff' (line 105)
        diff_24836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 42), 'diff', False)
        # Calling diff(args, kwargs) (line 105)
        diff_call_result_24845 = invoke(stypy.reporting.localization.Localization(__file__, 105, 42), diff_24836, *[cos_call_result_24842, k_24843], **kwargs_24844)
        
        
        # Call to direct_diff(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Call to cos(...): (line 105)
        # Processing the call arguments (line 105)
        int_24848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 75), 'int')
        # Getting the type of 'x' (line 105)
        x_24849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 77), 'x', False)
        # Applying the binary operator '*' (line 105)
        result_mul_24850 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 75), '*', int_24848, x_24849)
        
        # Processing the call keyword arguments (line 105)
        kwargs_24851 = {}
        # Getting the type of 'cos' (line 105)
        cos_24847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 71), 'cos', False)
        # Calling cos(args, kwargs) (line 105)
        cos_call_result_24852 = invoke(stypy.reporting.localization.Localization(__file__, 105, 71), cos_24847, *[result_mul_24850], **kwargs_24851)
        
        # Getting the type of 'k' (line 105)
        k_24853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 80), 'k', False)
        # Processing the call keyword arguments (line 105)
        kwargs_24854 = {}
        # Getting the type of 'direct_diff' (line 105)
        direct_diff_24846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 59), 'direct_diff', False)
        # Calling direct_diff(args, kwargs) (line 105)
        direct_diff_call_result_24855 = invoke(stypy.reporting.localization.Localization(__file__, 105, 59), direct_diff_24846, *[cos_call_result_24852, k_24853], **kwargs_24854)
        
        # Processing the call keyword arguments (line 105)
        kwargs_24856 = {}
        # Getting the type of 'assert_array_almost_equal' (line 105)
        assert_array_almost_equal_24835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 105)
        assert_array_almost_equal_call_result_24857 = invoke(stypy.reporting.localization.Localization(__file__, 105, 16), assert_array_almost_equal_24835, *[diff_call_result_24845, direct_diff_call_result_24855], **kwargs_24856)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_definition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_definition' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_24858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24858)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_definition'
        return stypy_return_type_24858


    @norecursion
    def test_period(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_period'
        module_type_store = module_type_store.open_function_context('test_period', 107, 4, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiff.test_period.__dict__.__setitem__('stypy_localization', localization)
        TestDiff.test_period.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiff.test_period.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiff.test_period.__dict__.__setitem__('stypy_function_name', 'TestDiff.test_period')
        TestDiff.test_period.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiff.test_period.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiff.test_period.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiff.test_period.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiff.test_period.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiff.test_period.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiff.test_period.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiff.test_period', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_period', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_period(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_24859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        int_24860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 17), list_24859, int_24860)
        # Adding element type (line 108)
        int_24861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 17), list_24859, int_24861)
        
        # Testing the type of a for loop iterable (line 108)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 8), list_24859)
        # Getting the type of the for loop variable (line 108)
        for_loop_var_24862 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 8), list_24859)
        # Assigning a type to the variable 'n' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'n', for_loop_var_24862)
        # SSA begins for a for statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 109):
        
        # Call to arange(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'n' (line 109)
        n_24864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 23), 'n', False)
        # Processing the call keyword arguments (line 109)
        kwargs_24865 = {}
        # Getting the type of 'arange' (line 109)
        arange_24863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'arange', False)
        # Calling arange(args, kwargs) (line 109)
        arange_call_result_24866 = invoke(stypy.reporting.localization.Localization(__file__, 109, 16), arange_24863, *[n_24864], **kwargs_24865)
        
        
        # Call to float(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'n' (line 109)
        n_24868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 32), 'n', False)
        # Processing the call keyword arguments (line 109)
        kwargs_24869 = {}
        # Getting the type of 'float' (line 109)
        float_24867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 26), 'float', False)
        # Calling float(args, kwargs) (line 109)
        float_call_result_24870 = invoke(stypy.reporting.localization.Localization(__file__, 109, 26), float_24867, *[n_24868], **kwargs_24869)
        
        # Applying the binary operator 'div' (line 109)
        result_div_24871 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 16), 'div', arange_call_result_24866, float_call_result_24870)
        
        # Assigning a type to the variable 'x' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'x', result_div_24871)
        
        # Call to assert_array_almost_equal(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Call to diff(...): (line 110)
        # Processing the call arguments (line 110)
        
        # Call to sin(...): (line 110)
        # Processing the call arguments (line 110)
        int_24875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 47), 'int')
        # Getting the type of 'pi' (line 110)
        pi_24876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 49), 'pi', False)
        # Applying the binary operator '*' (line 110)
        result_mul_24877 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 47), '*', int_24875, pi_24876)
        
        # Getting the type of 'x' (line 110)
        x_24878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 52), 'x', False)
        # Applying the binary operator '*' (line 110)
        result_mul_24879 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 51), '*', result_mul_24877, x_24878)
        
        # Processing the call keyword arguments (line 110)
        kwargs_24880 = {}
        # Getting the type of 'sin' (line 110)
        sin_24874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 43), 'sin', False)
        # Calling sin(args, kwargs) (line 110)
        sin_call_result_24881 = invoke(stypy.reporting.localization.Localization(__file__, 110, 43), sin_24874, *[result_mul_24879], **kwargs_24880)
        
        # Processing the call keyword arguments (line 110)
        int_24882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 62), 'int')
        keyword_24883 = int_24882
        kwargs_24884 = {'period': keyword_24883}
        # Getting the type of 'diff' (line 110)
        diff_24873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 110)
        diff_call_result_24885 = invoke(stypy.reporting.localization.Localization(__file__, 110, 38), diff_24873, *[sin_call_result_24881], **kwargs_24884)
        
        int_24886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 38), 'int')
        # Getting the type of 'pi' (line 111)
        pi_24887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 40), 'pi', False)
        # Applying the binary operator '*' (line 111)
        result_mul_24888 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 38), '*', int_24886, pi_24887)
        
        
        # Call to cos(...): (line 111)
        # Processing the call arguments (line 111)
        int_24890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 47), 'int')
        # Getting the type of 'pi' (line 111)
        pi_24891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 49), 'pi', False)
        # Applying the binary operator '*' (line 111)
        result_mul_24892 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 47), '*', int_24890, pi_24891)
        
        # Getting the type of 'x' (line 111)
        x_24893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 52), 'x', False)
        # Applying the binary operator '*' (line 111)
        result_mul_24894 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 51), '*', result_mul_24892, x_24893)
        
        # Processing the call keyword arguments (line 111)
        kwargs_24895 = {}
        # Getting the type of 'cos' (line 111)
        cos_24889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 43), 'cos', False)
        # Calling cos(args, kwargs) (line 111)
        cos_call_result_24896 = invoke(stypy.reporting.localization.Localization(__file__, 111, 43), cos_24889, *[result_mul_24894], **kwargs_24895)
        
        # Applying the binary operator '*' (line 111)
        result_mul_24897 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 42), '*', result_mul_24888, cos_call_result_24896)
        
        # Processing the call keyword arguments (line 110)
        kwargs_24898 = {}
        # Getting the type of 'assert_array_almost_equal' (line 110)
        assert_array_almost_equal_24872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 110)
        assert_array_almost_equal_call_result_24899 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), assert_array_almost_equal_24872, *[diff_call_result_24885, result_mul_24897], **kwargs_24898)
        
        
        # Call to assert_array_almost_equal(...): (line 112)
        # Processing the call arguments (line 112)
        
        # Call to diff(...): (line 112)
        # Processing the call arguments (line 112)
        
        # Call to sin(...): (line 112)
        # Processing the call arguments (line 112)
        int_24903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 47), 'int')
        # Getting the type of 'pi' (line 112)
        pi_24904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 49), 'pi', False)
        # Applying the binary operator '*' (line 112)
        result_mul_24905 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 47), '*', int_24903, pi_24904)
        
        # Getting the type of 'x' (line 112)
        x_24906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 52), 'x', False)
        # Applying the binary operator '*' (line 112)
        result_mul_24907 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 51), '*', result_mul_24905, x_24906)
        
        # Processing the call keyword arguments (line 112)
        kwargs_24908 = {}
        # Getting the type of 'sin' (line 112)
        sin_24902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 43), 'sin', False)
        # Calling sin(args, kwargs) (line 112)
        sin_call_result_24909 = invoke(stypy.reporting.localization.Localization(__file__, 112, 43), sin_24902, *[result_mul_24907], **kwargs_24908)
        
        int_24910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 55), 'int')
        # Processing the call keyword arguments (line 112)
        int_24911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 64), 'int')
        keyword_24912 = int_24911
        kwargs_24913 = {'period': keyword_24912}
        # Getting the type of 'diff' (line 112)
        diff_24901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 112)
        diff_call_result_24914 = invoke(stypy.reporting.localization.Localization(__file__, 112, 38), diff_24901, *[sin_call_result_24909, int_24910], **kwargs_24913)
        
        
        int_24915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 40), 'int')
        # Getting the type of 'pi' (line 113)
        pi_24916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 42), 'pi', False)
        # Applying the binary operator '*' (line 113)
        result_mul_24917 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 40), '*', int_24915, pi_24916)
        
        int_24918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 47), 'int')
        # Applying the binary operator '**' (line 113)
        result_pow_24919 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 39), '**', result_mul_24917, int_24918)
        
        # Applying the 'usub' unary operator (line 113)
        result___neg___24920 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 38), 'usub', result_pow_24919)
        
        
        # Call to cos(...): (line 113)
        # Processing the call arguments (line 113)
        int_24922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 53), 'int')
        # Getting the type of 'pi' (line 113)
        pi_24923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 55), 'pi', False)
        # Applying the binary operator '*' (line 113)
        result_mul_24924 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 53), '*', int_24922, pi_24923)
        
        # Getting the type of 'x' (line 113)
        x_24925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 58), 'x', False)
        # Applying the binary operator '*' (line 113)
        result_mul_24926 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 57), '*', result_mul_24924, x_24925)
        
        # Processing the call keyword arguments (line 113)
        kwargs_24927 = {}
        # Getting the type of 'cos' (line 113)
        cos_24921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 49), 'cos', False)
        # Calling cos(args, kwargs) (line 113)
        cos_call_result_24928 = invoke(stypy.reporting.localization.Localization(__file__, 113, 49), cos_24921, *[result_mul_24926], **kwargs_24927)
        
        # Applying the binary operator '*' (line 113)
        result_mul_24929 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 38), '*', result___neg___24920, cos_call_result_24928)
        
        # Processing the call keyword arguments (line 112)
        kwargs_24930 = {}
        # Getting the type of 'assert_array_almost_equal' (line 112)
        assert_array_almost_equal_24900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 112)
        assert_array_almost_equal_call_result_24931 = invoke(stypy.reporting.localization.Localization(__file__, 112, 12), assert_array_almost_equal_24900, *[diff_call_result_24914, result_mul_24929], **kwargs_24930)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_period(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_period' in the type store
        # Getting the type of 'stypy_return_type' (line 107)
        stypy_return_type_24932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24932)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_period'
        return stypy_return_type_24932


    @norecursion
    def test_sin(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sin'
        module_type_store = module_type_store.open_function_context('test_sin', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiff.test_sin.__dict__.__setitem__('stypy_localization', localization)
        TestDiff.test_sin.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiff.test_sin.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiff.test_sin.__dict__.__setitem__('stypy_function_name', 'TestDiff.test_sin')
        TestDiff.test_sin.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiff.test_sin.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiff.test_sin.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiff.test_sin.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiff.test_sin.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiff.test_sin.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiff.test_sin.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiff.test_sin', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sin', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sin(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_24933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        # Adding element type (line 116)
        int_24934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 17), list_24933, int_24934)
        # Adding element type (line 116)
        int_24935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 17), list_24933, int_24935)
        # Adding element type (line 116)
        int_24936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 17), list_24933, int_24936)
        
        # Testing the type of a for loop iterable (line 116)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 116, 8), list_24933)
        # Getting the type of the for loop variable (line 116)
        for_loop_var_24937 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 116, 8), list_24933)
        # Assigning a type to the variable 'n' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'n', for_loop_var_24937)
        # SSA begins for a for statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 117):
        
        # Call to arange(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'n' (line 117)
        n_24939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'n', False)
        # Processing the call keyword arguments (line 117)
        kwargs_24940 = {}
        # Getting the type of 'arange' (line 117)
        arange_24938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'arange', False)
        # Calling arange(args, kwargs) (line 117)
        arange_call_result_24941 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), arange_24938, *[n_24939], **kwargs_24940)
        
        int_24942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 26), 'int')
        # Applying the binary operator '*' (line 117)
        result_mul_24943 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 16), '*', arange_call_result_24941, int_24942)
        
        # Getting the type of 'pi' (line 117)
        pi_24944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 28), 'pi')
        # Applying the binary operator '*' (line 117)
        result_mul_24945 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 27), '*', result_mul_24943, pi_24944)
        
        # Getting the type of 'n' (line 117)
        n_24946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 31), 'n')
        # Applying the binary operator 'div' (line 117)
        result_div_24947 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 30), 'div', result_mul_24945, n_24946)
        
        # Assigning a type to the variable 'x' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'x', result_div_24947)
        
        # Call to assert_array_almost_equal(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Call to diff(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Call to sin(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'x' (line 118)
        x_24951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 47), 'x', False)
        # Processing the call keyword arguments (line 118)
        kwargs_24952 = {}
        # Getting the type of 'sin' (line 118)
        sin_24950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 43), 'sin', False)
        # Calling sin(args, kwargs) (line 118)
        sin_call_result_24953 = invoke(stypy.reporting.localization.Localization(__file__, 118, 43), sin_24950, *[x_24951], **kwargs_24952)
        
        # Processing the call keyword arguments (line 118)
        kwargs_24954 = {}
        # Getting the type of 'diff' (line 118)
        diff_24949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 118)
        diff_call_result_24955 = invoke(stypy.reporting.localization.Localization(__file__, 118, 38), diff_24949, *[sin_call_result_24953], **kwargs_24954)
        
        
        # Call to cos(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'x' (line 118)
        x_24957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 55), 'x', False)
        # Processing the call keyword arguments (line 118)
        kwargs_24958 = {}
        # Getting the type of 'cos' (line 118)
        cos_24956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 51), 'cos', False)
        # Calling cos(args, kwargs) (line 118)
        cos_call_result_24959 = invoke(stypy.reporting.localization.Localization(__file__, 118, 51), cos_24956, *[x_24957], **kwargs_24958)
        
        # Processing the call keyword arguments (line 118)
        kwargs_24960 = {}
        # Getting the type of 'assert_array_almost_equal' (line 118)
        assert_array_almost_equal_24948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 118)
        assert_array_almost_equal_call_result_24961 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), assert_array_almost_equal_24948, *[diff_call_result_24955, cos_call_result_24959], **kwargs_24960)
        
        
        # Call to assert_array_almost_equal(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Call to diff(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Call to cos(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'x' (line 119)
        x_24965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 47), 'x', False)
        # Processing the call keyword arguments (line 119)
        kwargs_24966 = {}
        # Getting the type of 'cos' (line 119)
        cos_24964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 43), 'cos', False)
        # Calling cos(args, kwargs) (line 119)
        cos_call_result_24967 = invoke(stypy.reporting.localization.Localization(__file__, 119, 43), cos_24964, *[x_24965], **kwargs_24966)
        
        # Processing the call keyword arguments (line 119)
        kwargs_24968 = {}
        # Getting the type of 'diff' (line 119)
        diff_24963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 119)
        diff_call_result_24969 = invoke(stypy.reporting.localization.Localization(__file__, 119, 38), diff_24963, *[cos_call_result_24967], **kwargs_24968)
        
        
        
        # Call to sin(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'x' (line 119)
        x_24971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 56), 'x', False)
        # Processing the call keyword arguments (line 119)
        kwargs_24972 = {}
        # Getting the type of 'sin' (line 119)
        sin_24970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 52), 'sin', False)
        # Calling sin(args, kwargs) (line 119)
        sin_call_result_24973 = invoke(stypy.reporting.localization.Localization(__file__, 119, 52), sin_24970, *[x_24971], **kwargs_24972)
        
        # Applying the 'usub' unary operator (line 119)
        result___neg___24974 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 51), 'usub', sin_call_result_24973)
        
        # Processing the call keyword arguments (line 119)
        kwargs_24975 = {}
        # Getting the type of 'assert_array_almost_equal' (line 119)
        assert_array_almost_equal_24962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 119)
        assert_array_almost_equal_call_result_24976 = invoke(stypy.reporting.localization.Localization(__file__, 119, 12), assert_array_almost_equal_24962, *[diff_call_result_24969, result___neg___24974], **kwargs_24975)
        
        
        # Call to assert_array_almost_equal(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Call to diff(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Call to sin(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'x' (line 120)
        x_24980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 47), 'x', False)
        # Processing the call keyword arguments (line 120)
        kwargs_24981 = {}
        # Getting the type of 'sin' (line 120)
        sin_24979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 43), 'sin', False)
        # Calling sin(args, kwargs) (line 120)
        sin_call_result_24982 = invoke(stypy.reporting.localization.Localization(__file__, 120, 43), sin_24979, *[x_24980], **kwargs_24981)
        
        int_24983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 50), 'int')
        # Processing the call keyword arguments (line 120)
        kwargs_24984 = {}
        # Getting the type of 'diff' (line 120)
        diff_24978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 120)
        diff_call_result_24985 = invoke(stypy.reporting.localization.Localization(__file__, 120, 38), diff_24978, *[sin_call_result_24982, int_24983], **kwargs_24984)
        
        
        
        # Call to sin(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'x' (line 120)
        x_24987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 58), 'x', False)
        # Processing the call keyword arguments (line 120)
        kwargs_24988 = {}
        # Getting the type of 'sin' (line 120)
        sin_24986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 54), 'sin', False)
        # Calling sin(args, kwargs) (line 120)
        sin_call_result_24989 = invoke(stypy.reporting.localization.Localization(__file__, 120, 54), sin_24986, *[x_24987], **kwargs_24988)
        
        # Applying the 'usub' unary operator (line 120)
        result___neg___24990 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 53), 'usub', sin_call_result_24989)
        
        # Processing the call keyword arguments (line 120)
        kwargs_24991 = {}
        # Getting the type of 'assert_array_almost_equal' (line 120)
        assert_array_almost_equal_24977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 120)
        assert_array_almost_equal_call_result_24992 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), assert_array_almost_equal_24977, *[diff_call_result_24985, result___neg___24990], **kwargs_24991)
        
        
        # Call to assert_array_almost_equal(...): (line 121)
        # Processing the call arguments (line 121)
        
        # Call to diff(...): (line 121)
        # Processing the call arguments (line 121)
        
        # Call to sin(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'x' (line 121)
        x_24996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 47), 'x', False)
        # Processing the call keyword arguments (line 121)
        kwargs_24997 = {}
        # Getting the type of 'sin' (line 121)
        sin_24995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 43), 'sin', False)
        # Calling sin(args, kwargs) (line 121)
        sin_call_result_24998 = invoke(stypy.reporting.localization.Localization(__file__, 121, 43), sin_24995, *[x_24996], **kwargs_24997)
        
        int_24999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 50), 'int')
        # Processing the call keyword arguments (line 121)
        kwargs_25000 = {}
        # Getting the type of 'diff' (line 121)
        diff_24994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 121)
        diff_call_result_25001 = invoke(stypy.reporting.localization.Localization(__file__, 121, 38), diff_24994, *[sin_call_result_24998, int_24999], **kwargs_25000)
        
        
        # Call to sin(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'x' (line 121)
        x_25003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 57), 'x', False)
        # Processing the call keyword arguments (line 121)
        kwargs_25004 = {}
        # Getting the type of 'sin' (line 121)
        sin_25002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 53), 'sin', False)
        # Calling sin(args, kwargs) (line 121)
        sin_call_result_25005 = invoke(stypy.reporting.localization.Localization(__file__, 121, 53), sin_25002, *[x_25003], **kwargs_25004)
        
        # Processing the call keyword arguments (line 121)
        kwargs_25006 = {}
        # Getting the type of 'assert_array_almost_equal' (line 121)
        assert_array_almost_equal_24993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 121)
        assert_array_almost_equal_call_result_25007 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), assert_array_almost_equal_24993, *[diff_call_result_25001, sin_call_result_25005], **kwargs_25006)
        
        
        # Call to assert_array_almost_equal(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Call to diff(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Call to sin(...): (line 122)
        # Processing the call arguments (line 122)
        int_25011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 47), 'int')
        # Getting the type of 'x' (line 122)
        x_25012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 49), 'x', False)
        # Applying the binary operator '*' (line 122)
        result_mul_25013 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 47), '*', int_25011, x_25012)
        
        # Processing the call keyword arguments (line 122)
        kwargs_25014 = {}
        # Getting the type of 'sin' (line 122)
        sin_25010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 43), 'sin', False)
        # Calling sin(args, kwargs) (line 122)
        sin_call_result_25015 = invoke(stypy.reporting.localization.Localization(__file__, 122, 43), sin_25010, *[result_mul_25013], **kwargs_25014)
        
        # Processing the call keyword arguments (line 122)
        kwargs_25016 = {}
        # Getting the type of 'diff' (line 122)
        diff_25009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 122)
        diff_call_result_25017 = invoke(stypy.reporting.localization.Localization(__file__, 122, 38), diff_25009, *[sin_call_result_25015], **kwargs_25016)
        
        int_25018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 53), 'int')
        
        # Call to cos(...): (line 122)
        # Processing the call arguments (line 122)
        int_25020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 59), 'int')
        # Getting the type of 'x' (line 122)
        x_25021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 61), 'x', False)
        # Applying the binary operator '*' (line 122)
        result_mul_25022 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 59), '*', int_25020, x_25021)
        
        # Processing the call keyword arguments (line 122)
        kwargs_25023 = {}
        # Getting the type of 'cos' (line 122)
        cos_25019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 55), 'cos', False)
        # Calling cos(args, kwargs) (line 122)
        cos_call_result_25024 = invoke(stypy.reporting.localization.Localization(__file__, 122, 55), cos_25019, *[result_mul_25022], **kwargs_25023)
        
        # Applying the binary operator '*' (line 122)
        result_mul_25025 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 53), '*', int_25018, cos_call_result_25024)
        
        # Processing the call keyword arguments (line 122)
        kwargs_25026 = {}
        # Getting the type of 'assert_array_almost_equal' (line 122)
        assert_array_almost_equal_25008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 122)
        assert_array_almost_equal_call_result_25027 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), assert_array_almost_equal_25008, *[diff_call_result_25017, result_mul_25025], **kwargs_25026)
        
        
        # Call to assert_array_almost_equal(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Call to diff(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Call to sin(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Call to sin(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'x' (line 123)
        x_25032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 51), 'x', False)
        # Processing the call keyword arguments (line 123)
        kwargs_25033 = {}
        # Getting the type of 'sin' (line 123)
        sin_25031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 47), 'sin', False)
        # Calling sin(args, kwargs) (line 123)
        sin_call_result_25034 = invoke(stypy.reporting.localization.Localization(__file__, 123, 47), sin_25031, *[x_25032], **kwargs_25033)
        
        # Processing the call keyword arguments (line 123)
        kwargs_25035 = {}
        # Getting the type of 'sin' (line 123)
        sin_25030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 43), 'sin', False)
        # Calling sin(args, kwargs) (line 123)
        sin_call_result_25036 = invoke(stypy.reporting.localization.Localization(__file__, 123, 43), sin_25030, *[sin_call_result_25034], **kwargs_25035)
        
        # Processing the call keyword arguments (line 123)
        kwargs_25037 = {}
        # Getting the type of 'diff' (line 123)
        diff_25029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 123)
        diff_call_result_25038 = invoke(stypy.reporting.localization.Localization(__file__, 123, 38), diff_25029, *[sin_call_result_25036], **kwargs_25037)
        
        
        # Call to cos(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'x' (line 123)
        x_25040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 60), 'x', False)
        # Processing the call keyword arguments (line 123)
        kwargs_25041 = {}
        # Getting the type of 'cos' (line 123)
        cos_25039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 56), 'cos', False)
        # Calling cos(args, kwargs) (line 123)
        cos_call_result_25042 = invoke(stypy.reporting.localization.Localization(__file__, 123, 56), cos_25039, *[x_25040], **kwargs_25041)
        
        
        # Call to cos(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Call to sin(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'x' (line 123)
        x_25045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 71), 'x', False)
        # Processing the call keyword arguments (line 123)
        kwargs_25046 = {}
        # Getting the type of 'sin' (line 123)
        sin_25044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 67), 'sin', False)
        # Calling sin(args, kwargs) (line 123)
        sin_call_result_25047 = invoke(stypy.reporting.localization.Localization(__file__, 123, 67), sin_25044, *[x_25045], **kwargs_25046)
        
        # Processing the call keyword arguments (line 123)
        kwargs_25048 = {}
        # Getting the type of 'cos' (line 123)
        cos_25043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 63), 'cos', False)
        # Calling cos(args, kwargs) (line 123)
        cos_call_result_25049 = invoke(stypy.reporting.localization.Localization(__file__, 123, 63), cos_25043, *[sin_call_result_25047], **kwargs_25048)
        
        # Applying the binary operator '*' (line 123)
        result_mul_25050 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 56), '*', cos_call_result_25042, cos_call_result_25049)
        
        # Processing the call keyword arguments (line 123)
        kwargs_25051 = {}
        # Getting the type of 'assert_array_almost_equal' (line 123)
        assert_array_almost_equal_25028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 123)
        assert_array_almost_equal_call_result_25052 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), assert_array_almost_equal_25028, *[diff_call_result_25038, result_mul_25050], **kwargs_25051)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_sin(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sin' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_25053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25053)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sin'
        return stypy_return_type_25053


    @norecursion
    def test_expr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_expr'
        module_type_store = module_type_store.open_function_context('test_expr', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiff.test_expr.__dict__.__setitem__('stypy_localization', localization)
        TestDiff.test_expr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiff.test_expr.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiff.test_expr.__dict__.__setitem__('stypy_function_name', 'TestDiff.test_expr')
        TestDiff.test_expr.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiff.test_expr.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiff.test_expr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiff.test_expr.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiff.test_expr.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiff.test_expr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiff.test_expr.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiff.test_expr', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_expr', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_expr(...)' code ##################

        
        
        # Obtaining the type of the subscript
        int_25054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 62), 'int')
        slice_25055 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 126, 17), None, int_25054, None)
        
        # Obtaining an instance of the builtin type 'list' (line 126)
        list_25056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 126)
        # Adding element type (line 126)
        int_25057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 17), list_25056, int_25057)
        # Adding element type (line 126)
        int_25058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 17), list_25056, int_25058)
        # Adding element type (line 126)
        int_25059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 17), list_25056, int_25059)
        # Adding element type (line 126)
        int_25060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 17), list_25056, int_25060)
        # Adding element type (line 126)
        int_25061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 17), list_25056, int_25061)
        # Adding element type (line 126)
        int_25062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 17), list_25056, int_25062)
        # Adding element type (line 126)
        int_25063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 17), list_25056, int_25063)
        # Adding element type (line 126)
        int_25064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 17), list_25056, int_25064)
        # Adding element type (line 126)
        int_25065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 17), list_25056, int_25065)
        # Adding element type (line 126)
        int_25066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 17), list_25056, int_25066)
        
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___25067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 17), list_25056, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 126)
        subscript_call_result_25068 = invoke(stypy.reporting.localization.Localization(__file__, 126, 17), getitem___25067, slice_25055)
        
        # Testing the type of a for loop iterable (line 126)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 126, 8), subscript_call_result_25068)
        # Getting the type of the for loop variable (line 126)
        for_loop_var_25069 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 126, 8), subscript_call_result_25068)
        # Assigning a type to the variable 'n' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'n', for_loop_var_25069)
        # SSA begins for a for statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 127):
        
        # Call to arange(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'n' (line 127)
        n_25071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'n', False)
        # Processing the call keyword arguments (line 127)
        kwargs_25072 = {}
        # Getting the type of 'arange' (line 127)
        arange_25070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'arange', False)
        # Calling arange(args, kwargs) (line 127)
        arange_call_result_25073 = invoke(stypy.reporting.localization.Localization(__file__, 127, 16), arange_25070, *[n_25071], **kwargs_25072)
        
        int_25074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 26), 'int')
        # Applying the binary operator '*' (line 127)
        result_mul_25075 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 16), '*', arange_call_result_25073, int_25074)
        
        # Getting the type of 'pi' (line 127)
        pi_25076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 28), 'pi')
        # Applying the binary operator '*' (line 127)
        result_mul_25077 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 27), '*', result_mul_25075, pi_25076)
        
        # Getting the type of 'n' (line 127)
        n_25078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 31), 'n')
        # Applying the binary operator 'div' (line 127)
        result_div_25079 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 30), 'div', result_mul_25077, n_25078)
        
        # Assigning a type to the variable 'x' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'x', result_div_25079)
        
        # Assigning a BinOp to a Name (line 128):
        
        # Call to sin(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'x' (line 128)
        x_25081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 20), 'x', False)
        # Processing the call keyword arguments (line 128)
        kwargs_25082 = {}
        # Getting the type of 'sin' (line 128)
        sin_25080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'sin', False)
        # Calling sin(args, kwargs) (line 128)
        sin_call_result_25083 = invoke(stypy.reporting.localization.Localization(__file__, 128, 16), sin_25080, *[x_25081], **kwargs_25082)
        
        
        # Call to cos(...): (line 128)
        # Processing the call arguments (line 128)
        int_25085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 27), 'int')
        # Getting the type of 'x' (line 128)
        x_25086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 29), 'x', False)
        # Applying the binary operator '*' (line 128)
        result_mul_25087 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 27), '*', int_25085, x_25086)
        
        # Processing the call keyword arguments (line 128)
        kwargs_25088 = {}
        # Getting the type of 'cos' (line 128)
        cos_25084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'cos', False)
        # Calling cos(args, kwargs) (line 128)
        cos_call_result_25089 = invoke(stypy.reporting.localization.Localization(__file__, 128, 23), cos_25084, *[result_mul_25087], **kwargs_25088)
        
        # Applying the binary operator '*' (line 128)
        result_mul_25090 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 16), '*', sin_call_result_25083, cos_call_result_25089)
        
        
        # Call to exp(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Call to sin(...): (line 128)
        # Processing the call arguments (line 128)
        int_25093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 40), 'int')
        # Getting the type of 'x' (line 128)
        x_25094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 42), 'x', False)
        # Applying the binary operator '*' (line 128)
        result_mul_25095 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 40), '*', int_25093, x_25094)
        
        # Processing the call keyword arguments (line 128)
        kwargs_25096 = {}
        # Getting the type of 'sin' (line 128)
        sin_25092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 36), 'sin', False)
        # Calling sin(args, kwargs) (line 128)
        sin_call_result_25097 = invoke(stypy.reporting.localization.Localization(__file__, 128, 36), sin_25092, *[result_mul_25095], **kwargs_25096)
        
        # Processing the call keyword arguments (line 128)
        kwargs_25098 = {}
        # Getting the type of 'exp' (line 128)
        exp_25091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 32), 'exp', False)
        # Calling exp(args, kwargs) (line 128)
        exp_call_result_25099 = invoke(stypy.reporting.localization.Localization(__file__, 128, 32), exp_25091, *[sin_call_result_25097], **kwargs_25098)
        
        # Applying the binary operator '+' (line 128)
        result_add_25100 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 16), '+', result_mul_25090, exp_call_result_25099)
        
        # Assigning a type to the variable 'f' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'f', result_add_25100)
        
        # Assigning a BinOp to a Name (line 129):
        
        # Call to cos(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'x' (line 129)
        x_25102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'x', False)
        # Processing the call keyword arguments (line 129)
        kwargs_25103 = {}
        # Getting the type of 'cos' (line 129)
        cos_25101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 17), 'cos', False)
        # Calling cos(args, kwargs) (line 129)
        cos_call_result_25104 = invoke(stypy.reporting.localization.Localization(__file__, 129, 17), cos_25101, *[x_25102], **kwargs_25103)
        
        
        # Call to cos(...): (line 129)
        # Processing the call arguments (line 129)
        int_25106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 28), 'int')
        # Getting the type of 'x' (line 129)
        x_25107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 30), 'x', False)
        # Applying the binary operator '*' (line 129)
        result_mul_25108 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 28), '*', int_25106, x_25107)
        
        # Processing the call keyword arguments (line 129)
        kwargs_25109 = {}
        # Getting the type of 'cos' (line 129)
        cos_25105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 24), 'cos', False)
        # Calling cos(args, kwargs) (line 129)
        cos_call_result_25110 = invoke(stypy.reporting.localization.Localization(__file__, 129, 24), cos_25105, *[result_mul_25108], **kwargs_25109)
        
        # Applying the binary operator '*' (line 129)
        result_mul_25111 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 17), '*', cos_call_result_25104, cos_call_result_25110)
        
        int_25112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 33), 'int')
        
        # Call to sin(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'x' (line 129)
        x_25114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 39), 'x', False)
        # Processing the call keyword arguments (line 129)
        kwargs_25115 = {}
        # Getting the type of 'sin' (line 129)
        sin_25113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 35), 'sin', False)
        # Calling sin(args, kwargs) (line 129)
        sin_call_result_25116 = invoke(stypy.reporting.localization.Localization(__file__, 129, 35), sin_25113, *[x_25114], **kwargs_25115)
        
        # Applying the binary operator '*' (line 129)
        result_mul_25117 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 33), '*', int_25112, sin_call_result_25116)
        
        
        # Call to sin(...): (line 129)
        # Processing the call arguments (line 129)
        int_25119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 46), 'int')
        # Getting the type of 'x' (line 129)
        x_25120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 48), 'x', False)
        # Applying the binary operator '*' (line 129)
        result_mul_25121 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 46), '*', int_25119, x_25120)
        
        # Processing the call keyword arguments (line 129)
        kwargs_25122 = {}
        # Getting the type of 'sin' (line 129)
        sin_25118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 42), 'sin', False)
        # Calling sin(args, kwargs) (line 129)
        sin_call_result_25123 = invoke(stypy.reporting.localization.Localization(__file__, 129, 42), sin_25118, *[result_mul_25121], **kwargs_25122)
        
        # Applying the binary operator '*' (line 129)
        result_mul_25124 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 41), '*', result_mul_25117, sin_call_result_25123)
        
        # Applying the binary operator '-' (line 129)
        result_sub_25125 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 17), '-', result_mul_25111, result_mul_25124)
        
        int_25126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 51), 'int')
        
        # Call to cos(...): (line 129)
        # Processing the call arguments (line 129)
        int_25128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 57), 'int')
        # Getting the type of 'x' (line 129)
        x_25129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 59), 'x', False)
        # Applying the binary operator '*' (line 129)
        result_mul_25130 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 57), '*', int_25128, x_25129)
        
        # Processing the call keyword arguments (line 129)
        kwargs_25131 = {}
        # Getting the type of 'cos' (line 129)
        cos_25127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 53), 'cos', False)
        # Calling cos(args, kwargs) (line 129)
        cos_call_result_25132 = invoke(stypy.reporting.localization.Localization(__file__, 129, 53), cos_25127, *[result_mul_25130], **kwargs_25131)
        
        # Applying the binary operator '*' (line 129)
        result_mul_25133 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 51), '*', int_25126, cos_call_result_25132)
        
        
        # Call to exp(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Call to sin(...): (line 129)
        # Processing the call arguments (line 129)
        int_25136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 70), 'int')
        # Getting the type of 'x' (line 129)
        x_25137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 72), 'x', False)
        # Applying the binary operator '*' (line 129)
        result_mul_25138 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 70), '*', int_25136, x_25137)
        
        # Processing the call keyword arguments (line 129)
        kwargs_25139 = {}
        # Getting the type of 'sin' (line 129)
        sin_25135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 66), 'sin', False)
        # Calling sin(args, kwargs) (line 129)
        sin_call_result_25140 = invoke(stypy.reporting.localization.Localization(__file__, 129, 66), sin_25135, *[result_mul_25138], **kwargs_25139)
        
        # Processing the call keyword arguments (line 129)
        kwargs_25141 = {}
        # Getting the type of 'exp' (line 129)
        exp_25134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 62), 'exp', False)
        # Calling exp(args, kwargs) (line 129)
        exp_call_result_25142 = invoke(stypy.reporting.localization.Localization(__file__, 129, 62), exp_25134, *[sin_call_result_25140], **kwargs_25141)
        
        # Applying the binary operator '*' (line 129)
        result_mul_25143 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 61), '*', result_mul_25133, exp_call_result_25142)
        
        # Applying the binary operator '+' (line 129)
        result_add_25144 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 50), '+', result_sub_25125, result_mul_25143)
        
        # Assigning a type to the variable 'df' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'df', result_add_25144)
        
        # Assigning a BinOp to a Name (line 130):
        int_25145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 18), 'int')
        
        # Call to sin(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'x' (line 130)
        x_25147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 26), 'x', False)
        # Processing the call keyword arguments (line 130)
        kwargs_25148 = {}
        # Getting the type of 'sin' (line 130)
        sin_25146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 22), 'sin', False)
        # Calling sin(args, kwargs) (line 130)
        sin_call_result_25149 = invoke(stypy.reporting.localization.Localization(__file__, 130, 22), sin_25146, *[x_25147], **kwargs_25148)
        
        # Applying the binary operator '*' (line 130)
        result_mul_25150 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 18), '*', int_25145, sin_call_result_25149)
        
        
        # Call to cos(...): (line 130)
        # Processing the call arguments (line 130)
        int_25152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 33), 'int')
        # Getting the type of 'x' (line 130)
        x_25153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 35), 'x', False)
        # Applying the binary operator '*' (line 130)
        result_mul_25154 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 33), '*', int_25152, x_25153)
        
        # Processing the call keyword arguments (line 130)
        kwargs_25155 = {}
        # Getting the type of 'cos' (line 130)
        cos_25151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 29), 'cos', False)
        # Calling cos(args, kwargs) (line 130)
        cos_call_result_25156 = invoke(stypy.reporting.localization.Localization(__file__, 130, 29), cos_25151, *[result_mul_25154], **kwargs_25155)
        
        # Applying the binary operator '*' (line 130)
        result_mul_25157 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 28), '*', result_mul_25150, cos_call_result_25156)
        
        int_25158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 38), 'int')
        
        # Call to cos(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'x' (line 130)
        x_25160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 44), 'x', False)
        # Processing the call keyword arguments (line 130)
        kwargs_25161 = {}
        # Getting the type of 'cos' (line 130)
        cos_25159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 40), 'cos', False)
        # Calling cos(args, kwargs) (line 130)
        cos_call_result_25162 = invoke(stypy.reporting.localization.Localization(__file__, 130, 40), cos_25159, *[x_25160], **kwargs_25161)
        
        # Applying the binary operator '*' (line 130)
        result_mul_25163 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 38), '*', int_25158, cos_call_result_25162)
        
        
        # Call to sin(...): (line 130)
        # Processing the call arguments (line 130)
        int_25165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 51), 'int')
        # Getting the type of 'x' (line 130)
        x_25166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 53), 'x', False)
        # Applying the binary operator '*' (line 130)
        result_mul_25167 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 51), '*', int_25165, x_25166)
        
        # Processing the call keyword arguments (line 130)
        kwargs_25168 = {}
        # Getting the type of 'sin' (line 130)
        sin_25164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 47), 'sin', False)
        # Calling sin(args, kwargs) (line 130)
        sin_call_result_25169 = invoke(stypy.reporting.localization.Localization(__file__, 130, 47), sin_25164, *[result_mul_25167], **kwargs_25168)
        
        # Applying the binary operator '*' (line 130)
        result_mul_25170 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 46), '*', result_mul_25163, sin_call_result_25169)
        
        # Applying the binary operator '-' (line 130)
        result_sub_25171 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 18), '-', result_mul_25157, result_mul_25170)
        
        int_25172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 19), 'int')
        
        # Call to sin(...): (line 131)
        # Processing the call arguments (line 131)
        int_25174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 25), 'int')
        # Getting the type of 'x' (line 131)
        x_25175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 27), 'x', False)
        # Applying the binary operator '*' (line 131)
        result_mul_25176 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 25), '*', int_25174, x_25175)
        
        # Processing the call keyword arguments (line 131)
        kwargs_25177 = {}
        # Getting the type of 'sin' (line 131)
        sin_25173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 21), 'sin', False)
        # Calling sin(args, kwargs) (line 131)
        sin_call_result_25178 = invoke(stypy.reporting.localization.Localization(__file__, 131, 21), sin_25173, *[result_mul_25176], **kwargs_25177)
        
        # Applying the binary operator '*' (line 131)
        result_mul_25179 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 19), '*', int_25172, sin_call_result_25178)
        
        
        # Call to exp(...): (line 131)
        # Processing the call arguments (line 131)
        
        # Call to sin(...): (line 131)
        # Processing the call arguments (line 131)
        int_25182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 38), 'int')
        # Getting the type of 'x' (line 131)
        x_25183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 40), 'x', False)
        # Applying the binary operator '*' (line 131)
        result_mul_25184 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 38), '*', int_25182, x_25183)
        
        # Processing the call keyword arguments (line 131)
        kwargs_25185 = {}
        # Getting the type of 'sin' (line 131)
        sin_25181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 34), 'sin', False)
        # Calling sin(args, kwargs) (line 131)
        sin_call_result_25186 = invoke(stypy.reporting.localization.Localization(__file__, 131, 34), sin_25181, *[result_mul_25184], **kwargs_25185)
        
        # Processing the call keyword arguments (line 131)
        kwargs_25187 = {}
        # Getting the type of 'exp' (line 131)
        exp_25180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 30), 'exp', False)
        # Calling exp(args, kwargs) (line 131)
        exp_call_result_25188 = invoke(stypy.reporting.localization.Localization(__file__, 131, 30), exp_25180, *[sin_call_result_25186], **kwargs_25187)
        
        # Applying the binary operator '*' (line 131)
        result_mul_25189 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 29), '*', result_mul_25179, exp_call_result_25188)
        
        # Applying the binary operator '-' (line 131)
        result_sub_25190 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 17), '-', result_sub_25171, result_mul_25189)
        
        int_25191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 44), 'int')
        
        # Call to cos(...): (line 131)
        # Processing the call arguments (line 131)
        int_25193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 50), 'int')
        # Getting the type of 'x' (line 131)
        x_25194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 52), 'x', False)
        # Applying the binary operator '*' (line 131)
        result_mul_25195 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 50), '*', int_25193, x_25194)
        
        # Processing the call keyword arguments (line 131)
        kwargs_25196 = {}
        # Getting the type of 'cos' (line 131)
        cos_25192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 46), 'cos', False)
        # Calling cos(args, kwargs) (line 131)
        cos_call_result_25197 = invoke(stypy.reporting.localization.Localization(__file__, 131, 46), cos_25192, *[result_mul_25195], **kwargs_25196)
        
        int_25198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 56), 'int')
        # Applying the binary operator '**' (line 131)
        result_pow_25199 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 46), '**', cos_call_result_25197, int_25198)
        
        # Applying the binary operator '*' (line 131)
        result_mul_25200 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 44), '*', int_25191, result_pow_25199)
        
        
        # Call to exp(...): (line 131)
        # Processing the call arguments (line 131)
        
        # Call to sin(...): (line 131)
        # Processing the call arguments (line 131)
        int_25203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 66), 'int')
        # Getting the type of 'x' (line 131)
        x_25204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 68), 'x', False)
        # Applying the binary operator '*' (line 131)
        result_mul_25205 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 66), '*', int_25203, x_25204)
        
        # Processing the call keyword arguments (line 131)
        kwargs_25206 = {}
        # Getting the type of 'sin' (line 131)
        sin_25202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 62), 'sin', False)
        # Calling sin(args, kwargs) (line 131)
        sin_call_result_25207 = invoke(stypy.reporting.localization.Localization(__file__, 131, 62), sin_25202, *[result_mul_25205], **kwargs_25206)
        
        # Processing the call keyword arguments (line 131)
        kwargs_25208 = {}
        # Getting the type of 'exp' (line 131)
        exp_25201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 58), 'exp', False)
        # Calling exp(args, kwargs) (line 131)
        exp_call_result_25209 = invoke(stypy.reporting.localization.Localization(__file__, 131, 58), exp_25201, *[sin_call_result_25207], **kwargs_25208)
        
        # Applying the binary operator '*' (line 131)
        result_mul_25210 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 57), '*', result_mul_25200, exp_call_result_25209)
        
        # Applying the binary operator '+' (line 131)
        result_add_25211 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 43), '+', result_sub_25190, result_mul_25210)
        
        # Assigning a type to the variable 'ddf' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'ddf', result_add_25211)
        
        # Assigning a Call to a Name (line 132):
        
        # Call to diff(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'f' (line 132)
        f_25213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 22), 'f', False)
        # Processing the call keyword arguments (line 132)
        kwargs_25214 = {}
        # Getting the type of 'diff' (line 132)
        diff_25212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 17), 'diff', False)
        # Calling diff(args, kwargs) (line 132)
        diff_call_result_25215 = invoke(stypy.reporting.localization.Localization(__file__, 132, 17), diff_25212, *[f_25213], **kwargs_25214)
        
        # Assigning a type to the variable 'd1' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'd1', diff_call_result_25215)
        
        # Call to assert_array_almost_equal(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'd1' (line 133)
        d1_25217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 38), 'd1', False)
        # Getting the type of 'df' (line 133)
        df_25218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 41), 'df', False)
        # Processing the call keyword arguments (line 133)
        kwargs_25219 = {}
        # Getting the type of 'assert_array_almost_equal' (line 133)
        assert_array_almost_equal_25216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 133)
        assert_array_almost_equal_call_result_25220 = invoke(stypy.reporting.localization.Localization(__file__, 133, 12), assert_array_almost_equal_25216, *[d1_25217, df_25218], **kwargs_25219)
        
        
        # Call to assert_array_almost_equal(...): (line 134)
        # Processing the call arguments (line 134)
        
        # Call to diff(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'df' (line 134)
        df_25223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 43), 'df', False)
        # Processing the call keyword arguments (line 134)
        kwargs_25224 = {}
        # Getting the type of 'diff' (line 134)
        diff_25222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 134)
        diff_call_result_25225 = invoke(stypy.reporting.localization.Localization(__file__, 134, 38), diff_25222, *[df_25223], **kwargs_25224)
        
        # Getting the type of 'ddf' (line 134)
        ddf_25226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 47), 'ddf', False)
        # Processing the call keyword arguments (line 134)
        kwargs_25227 = {}
        # Getting the type of 'assert_array_almost_equal' (line 134)
        assert_array_almost_equal_25221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 134)
        assert_array_almost_equal_call_result_25228 = invoke(stypy.reporting.localization.Localization(__file__, 134, 12), assert_array_almost_equal_25221, *[diff_call_result_25225, ddf_25226], **kwargs_25227)
        
        
        # Call to assert_array_almost_equal(...): (line 135)
        # Processing the call arguments (line 135)
        
        # Call to diff(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'f' (line 135)
        f_25231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 43), 'f', False)
        int_25232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 45), 'int')
        # Processing the call keyword arguments (line 135)
        kwargs_25233 = {}
        # Getting the type of 'diff' (line 135)
        diff_25230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 135)
        diff_call_result_25234 = invoke(stypy.reporting.localization.Localization(__file__, 135, 38), diff_25230, *[f_25231, int_25232], **kwargs_25233)
        
        # Getting the type of 'ddf' (line 135)
        ddf_25235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 48), 'ddf', False)
        # Processing the call keyword arguments (line 135)
        kwargs_25236 = {}
        # Getting the type of 'assert_array_almost_equal' (line 135)
        assert_array_almost_equal_25229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 135)
        assert_array_almost_equal_call_result_25237 = invoke(stypy.reporting.localization.Localization(__file__, 135, 12), assert_array_almost_equal_25229, *[diff_call_result_25234, ddf_25235], **kwargs_25236)
        
        
        # Call to assert_array_almost_equal(...): (line 136)
        # Processing the call arguments (line 136)
        
        # Call to diff(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'ddf' (line 136)
        ddf_25240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 43), 'ddf', False)
        int_25241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 47), 'int')
        # Processing the call keyword arguments (line 136)
        kwargs_25242 = {}
        # Getting the type of 'diff' (line 136)
        diff_25239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 136)
        diff_call_result_25243 = invoke(stypy.reporting.localization.Localization(__file__, 136, 38), diff_25239, *[ddf_25240, int_25241], **kwargs_25242)
        
        # Getting the type of 'df' (line 136)
        df_25244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 51), 'df', False)
        # Processing the call keyword arguments (line 136)
        kwargs_25245 = {}
        # Getting the type of 'assert_array_almost_equal' (line 136)
        assert_array_almost_equal_25238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 136)
        assert_array_almost_equal_call_result_25246 = invoke(stypy.reporting.localization.Localization(__file__, 136, 12), assert_array_almost_equal_25238, *[diff_call_result_25243, df_25244], **kwargs_25245)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_expr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_expr' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_25247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25247)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_expr'
        return stypy_return_type_25247


    @norecursion
    def test_expr_large(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_expr_large'
        module_type_store = module_type_store.open_function_context('test_expr_large', 138, 4, False)
        # Assigning a type to the variable 'self' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiff.test_expr_large.__dict__.__setitem__('stypy_localization', localization)
        TestDiff.test_expr_large.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiff.test_expr_large.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiff.test_expr_large.__dict__.__setitem__('stypy_function_name', 'TestDiff.test_expr_large')
        TestDiff.test_expr_large.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiff.test_expr_large.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiff.test_expr_large.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiff.test_expr_large.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiff.test_expr_large.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiff.test_expr_large.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiff.test_expr_large.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiff.test_expr_large', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_expr_large', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_expr_large(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 139)
        list_25248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 139)
        # Adding element type (line 139)
        int_25249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 17), list_25248, int_25249)
        # Adding element type (line 139)
        int_25250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 17), list_25248, int_25250)
        
        # Testing the type of a for loop iterable (line 139)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 139, 8), list_25248)
        # Getting the type of the for loop variable (line 139)
        for_loop_var_25251 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 139, 8), list_25248)
        # Assigning a type to the variable 'n' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'n', for_loop_var_25251)
        # SSA begins for a for statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 140):
        
        # Call to arange(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'n' (line 140)
        n_25253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 23), 'n', False)
        # Processing the call keyword arguments (line 140)
        kwargs_25254 = {}
        # Getting the type of 'arange' (line 140)
        arange_25252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'arange', False)
        # Calling arange(args, kwargs) (line 140)
        arange_call_result_25255 = invoke(stypy.reporting.localization.Localization(__file__, 140, 16), arange_25252, *[n_25253], **kwargs_25254)
        
        int_25256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 26), 'int')
        # Applying the binary operator '*' (line 140)
        result_mul_25257 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 16), '*', arange_call_result_25255, int_25256)
        
        # Getting the type of 'pi' (line 140)
        pi_25258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 28), 'pi')
        # Applying the binary operator '*' (line 140)
        result_mul_25259 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 27), '*', result_mul_25257, pi_25258)
        
        # Getting the type of 'n' (line 140)
        n_25260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 31), 'n')
        # Applying the binary operator 'div' (line 140)
        result_div_25261 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 30), 'div', result_mul_25259, n_25260)
        
        # Assigning a type to the variable 'x' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'x', result_div_25261)
        
        # Assigning a BinOp to a Name (line 141):
        
        # Call to sin(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'x' (line 141)
        x_25263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'x', False)
        # Processing the call keyword arguments (line 141)
        kwargs_25264 = {}
        # Getting the type of 'sin' (line 141)
        sin_25262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'sin', False)
        # Calling sin(args, kwargs) (line 141)
        sin_call_result_25265 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), sin_25262, *[x_25263], **kwargs_25264)
        
        
        # Call to cos(...): (line 141)
        # Processing the call arguments (line 141)
        int_25267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 27), 'int')
        # Getting the type of 'x' (line 141)
        x_25268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 29), 'x', False)
        # Applying the binary operator '*' (line 141)
        result_mul_25269 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 27), '*', int_25267, x_25268)
        
        # Processing the call keyword arguments (line 141)
        kwargs_25270 = {}
        # Getting the type of 'cos' (line 141)
        cos_25266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 23), 'cos', False)
        # Calling cos(args, kwargs) (line 141)
        cos_call_result_25271 = invoke(stypy.reporting.localization.Localization(__file__, 141, 23), cos_25266, *[result_mul_25269], **kwargs_25270)
        
        # Applying the binary operator '*' (line 141)
        result_mul_25272 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 16), '*', sin_call_result_25265, cos_call_result_25271)
        
        
        # Call to exp(...): (line 141)
        # Processing the call arguments (line 141)
        
        # Call to sin(...): (line 141)
        # Processing the call arguments (line 141)
        int_25275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 40), 'int')
        # Getting the type of 'x' (line 141)
        x_25276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 42), 'x', False)
        # Applying the binary operator '*' (line 141)
        result_mul_25277 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 40), '*', int_25275, x_25276)
        
        # Processing the call keyword arguments (line 141)
        kwargs_25278 = {}
        # Getting the type of 'sin' (line 141)
        sin_25274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 36), 'sin', False)
        # Calling sin(args, kwargs) (line 141)
        sin_call_result_25279 = invoke(stypy.reporting.localization.Localization(__file__, 141, 36), sin_25274, *[result_mul_25277], **kwargs_25278)
        
        # Processing the call keyword arguments (line 141)
        kwargs_25280 = {}
        # Getting the type of 'exp' (line 141)
        exp_25273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 32), 'exp', False)
        # Calling exp(args, kwargs) (line 141)
        exp_call_result_25281 = invoke(stypy.reporting.localization.Localization(__file__, 141, 32), exp_25273, *[sin_call_result_25279], **kwargs_25280)
        
        # Applying the binary operator '+' (line 141)
        result_add_25282 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 16), '+', result_mul_25272, exp_call_result_25281)
        
        # Assigning a type to the variable 'f' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'f', result_add_25282)
        
        # Assigning a BinOp to a Name (line 142):
        
        # Call to cos(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'x' (line 142)
        x_25284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 21), 'x', False)
        # Processing the call keyword arguments (line 142)
        kwargs_25285 = {}
        # Getting the type of 'cos' (line 142)
        cos_25283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 17), 'cos', False)
        # Calling cos(args, kwargs) (line 142)
        cos_call_result_25286 = invoke(stypy.reporting.localization.Localization(__file__, 142, 17), cos_25283, *[x_25284], **kwargs_25285)
        
        
        # Call to cos(...): (line 142)
        # Processing the call arguments (line 142)
        int_25288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 28), 'int')
        # Getting the type of 'x' (line 142)
        x_25289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 30), 'x', False)
        # Applying the binary operator '*' (line 142)
        result_mul_25290 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 28), '*', int_25288, x_25289)
        
        # Processing the call keyword arguments (line 142)
        kwargs_25291 = {}
        # Getting the type of 'cos' (line 142)
        cos_25287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 24), 'cos', False)
        # Calling cos(args, kwargs) (line 142)
        cos_call_result_25292 = invoke(stypy.reporting.localization.Localization(__file__, 142, 24), cos_25287, *[result_mul_25290], **kwargs_25291)
        
        # Applying the binary operator '*' (line 142)
        result_mul_25293 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 17), '*', cos_call_result_25286, cos_call_result_25292)
        
        int_25294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 33), 'int')
        
        # Call to sin(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'x' (line 142)
        x_25296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 39), 'x', False)
        # Processing the call keyword arguments (line 142)
        kwargs_25297 = {}
        # Getting the type of 'sin' (line 142)
        sin_25295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 35), 'sin', False)
        # Calling sin(args, kwargs) (line 142)
        sin_call_result_25298 = invoke(stypy.reporting.localization.Localization(__file__, 142, 35), sin_25295, *[x_25296], **kwargs_25297)
        
        # Applying the binary operator '*' (line 142)
        result_mul_25299 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 33), '*', int_25294, sin_call_result_25298)
        
        
        # Call to sin(...): (line 142)
        # Processing the call arguments (line 142)
        int_25301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 46), 'int')
        # Getting the type of 'x' (line 142)
        x_25302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 48), 'x', False)
        # Applying the binary operator '*' (line 142)
        result_mul_25303 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 46), '*', int_25301, x_25302)
        
        # Processing the call keyword arguments (line 142)
        kwargs_25304 = {}
        # Getting the type of 'sin' (line 142)
        sin_25300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), 'sin', False)
        # Calling sin(args, kwargs) (line 142)
        sin_call_result_25305 = invoke(stypy.reporting.localization.Localization(__file__, 142, 42), sin_25300, *[result_mul_25303], **kwargs_25304)
        
        # Applying the binary operator '*' (line 142)
        result_mul_25306 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 41), '*', result_mul_25299, sin_call_result_25305)
        
        # Applying the binary operator '-' (line 142)
        result_sub_25307 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 17), '-', result_mul_25293, result_mul_25306)
        
        int_25308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 51), 'int')
        
        # Call to cos(...): (line 142)
        # Processing the call arguments (line 142)
        int_25310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 57), 'int')
        # Getting the type of 'x' (line 142)
        x_25311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 59), 'x', False)
        # Applying the binary operator '*' (line 142)
        result_mul_25312 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 57), '*', int_25310, x_25311)
        
        # Processing the call keyword arguments (line 142)
        kwargs_25313 = {}
        # Getting the type of 'cos' (line 142)
        cos_25309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 53), 'cos', False)
        # Calling cos(args, kwargs) (line 142)
        cos_call_result_25314 = invoke(stypy.reporting.localization.Localization(__file__, 142, 53), cos_25309, *[result_mul_25312], **kwargs_25313)
        
        # Applying the binary operator '*' (line 142)
        result_mul_25315 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 51), '*', int_25308, cos_call_result_25314)
        
        
        # Call to exp(...): (line 142)
        # Processing the call arguments (line 142)
        
        # Call to sin(...): (line 142)
        # Processing the call arguments (line 142)
        int_25318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 70), 'int')
        # Getting the type of 'x' (line 142)
        x_25319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 72), 'x', False)
        # Applying the binary operator '*' (line 142)
        result_mul_25320 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 70), '*', int_25318, x_25319)
        
        # Processing the call keyword arguments (line 142)
        kwargs_25321 = {}
        # Getting the type of 'sin' (line 142)
        sin_25317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 66), 'sin', False)
        # Calling sin(args, kwargs) (line 142)
        sin_call_result_25322 = invoke(stypy.reporting.localization.Localization(__file__, 142, 66), sin_25317, *[result_mul_25320], **kwargs_25321)
        
        # Processing the call keyword arguments (line 142)
        kwargs_25323 = {}
        # Getting the type of 'exp' (line 142)
        exp_25316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 62), 'exp', False)
        # Calling exp(args, kwargs) (line 142)
        exp_call_result_25324 = invoke(stypy.reporting.localization.Localization(__file__, 142, 62), exp_25316, *[sin_call_result_25322], **kwargs_25323)
        
        # Applying the binary operator '*' (line 142)
        result_mul_25325 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 61), '*', result_mul_25315, exp_call_result_25324)
        
        # Applying the binary operator '+' (line 142)
        result_add_25326 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 50), '+', result_sub_25307, result_mul_25325)
        
        # Assigning a type to the variable 'df' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'df', result_add_25326)
        
        # Assigning a BinOp to a Name (line 143):
        int_25327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 18), 'int')
        
        # Call to sin(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'x' (line 143)
        x_25329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 26), 'x', False)
        # Processing the call keyword arguments (line 143)
        kwargs_25330 = {}
        # Getting the type of 'sin' (line 143)
        sin_25328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 22), 'sin', False)
        # Calling sin(args, kwargs) (line 143)
        sin_call_result_25331 = invoke(stypy.reporting.localization.Localization(__file__, 143, 22), sin_25328, *[x_25329], **kwargs_25330)
        
        # Applying the binary operator '*' (line 143)
        result_mul_25332 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 18), '*', int_25327, sin_call_result_25331)
        
        
        # Call to cos(...): (line 143)
        # Processing the call arguments (line 143)
        int_25334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 33), 'int')
        # Getting the type of 'x' (line 143)
        x_25335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 35), 'x', False)
        # Applying the binary operator '*' (line 143)
        result_mul_25336 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 33), '*', int_25334, x_25335)
        
        # Processing the call keyword arguments (line 143)
        kwargs_25337 = {}
        # Getting the type of 'cos' (line 143)
        cos_25333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 29), 'cos', False)
        # Calling cos(args, kwargs) (line 143)
        cos_call_result_25338 = invoke(stypy.reporting.localization.Localization(__file__, 143, 29), cos_25333, *[result_mul_25336], **kwargs_25337)
        
        # Applying the binary operator '*' (line 143)
        result_mul_25339 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 28), '*', result_mul_25332, cos_call_result_25338)
        
        int_25340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 38), 'int')
        
        # Call to cos(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'x' (line 143)
        x_25342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 44), 'x', False)
        # Processing the call keyword arguments (line 143)
        kwargs_25343 = {}
        # Getting the type of 'cos' (line 143)
        cos_25341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 40), 'cos', False)
        # Calling cos(args, kwargs) (line 143)
        cos_call_result_25344 = invoke(stypy.reporting.localization.Localization(__file__, 143, 40), cos_25341, *[x_25342], **kwargs_25343)
        
        # Applying the binary operator '*' (line 143)
        result_mul_25345 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 38), '*', int_25340, cos_call_result_25344)
        
        
        # Call to sin(...): (line 143)
        # Processing the call arguments (line 143)
        int_25347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 51), 'int')
        # Getting the type of 'x' (line 143)
        x_25348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 53), 'x', False)
        # Applying the binary operator '*' (line 143)
        result_mul_25349 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 51), '*', int_25347, x_25348)
        
        # Processing the call keyword arguments (line 143)
        kwargs_25350 = {}
        # Getting the type of 'sin' (line 143)
        sin_25346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 47), 'sin', False)
        # Calling sin(args, kwargs) (line 143)
        sin_call_result_25351 = invoke(stypy.reporting.localization.Localization(__file__, 143, 47), sin_25346, *[result_mul_25349], **kwargs_25350)
        
        # Applying the binary operator '*' (line 143)
        result_mul_25352 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 46), '*', result_mul_25345, sin_call_result_25351)
        
        # Applying the binary operator '-' (line 143)
        result_sub_25353 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 18), '-', result_mul_25339, result_mul_25352)
        
        int_25354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 19), 'int')
        
        # Call to sin(...): (line 144)
        # Processing the call arguments (line 144)
        int_25356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 25), 'int')
        # Getting the type of 'x' (line 144)
        x_25357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 27), 'x', False)
        # Applying the binary operator '*' (line 144)
        result_mul_25358 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 25), '*', int_25356, x_25357)
        
        # Processing the call keyword arguments (line 144)
        kwargs_25359 = {}
        # Getting the type of 'sin' (line 144)
        sin_25355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 21), 'sin', False)
        # Calling sin(args, kwargs) (line 144)
        sin_call_result_25360 = invoke(stypy.reporting.localization.Localization(__file__, 144, 21), sin_25355, *[result_mul_25358], **kwargs_25359)
        
        # Applying the binary operator '*' (line 144)
        result_mul_25361 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 19), '*', int_25354, sin_call_result_25360)
        
        
        # Call to exp(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Call to sin(...): (line 144)
        # Processing the call arguments (line 144)
        int_25364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 38), 'int')
        # Getting the type of 'x' (line 144)
        x_25365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 40), 'x', False)
        # Applying the binary operator '*' (line 144)
        result_mul_25366 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 38), '*', int_25364, x_25365)
        
        # Processing the call keyword arguments (line 144)
        kwargs_25367 = {}
        # Getting the type of 'sin' (line 144)
        sin_25363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 34), 'sin', False)
        # Calling sin(args, kwargs) (line 144)
        sin_call_result_25368 = invoke(stypy.reporting.localization.Localization(__file__, 144, 34), sin_25363, *[result_mul_25366], **kwargs_25367)
        
        # Processing the call keyword arguments (line 144)
        kwargs_25369 = {}
        # Getting the type of 'exp' (line 144)
        exp_25362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 30), 'exp', False)
        # Calling exp(args, kwargs) (line 144)
        exp_call_result_25370 = invoke(stypy.reporting.localization.Localization(__file__, 144, 30), exp_25362, *[sin_call_result_25368], **kwargs_25369)
        
        # Applying the binary operator '*' (line 144)
        result_mul_25371 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 29), '*', result_mul_25361, exp_call_result_25370)
        
        # Applying the binary operator '-' (line 144)
        result_sub_25372 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 17), '-', result_sub_25353, result_mul_25371)
        
        int_25373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 44), 'int')
        
        # Call to cos(...): (line 144)
        # Processing the call arguments (line 144)
        int_25375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 50), 'int')
        # Getting the type of 'x' (line 144)
        x_25376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 52), 'x', False)
        # Applying the binary operator '*' (line 144)
        result_mul_25377 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 50), '*', int_25375, x_25376)
        
        # Processing the call keyword arguments (line 144)
        kwargs_25378 = {}
        # Getting the type of 'cos' (line 144)
        cos_25374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 46), 'cos', False)
        # Calling cos(args, kwargs) (line 144)
        cos_call_result_25379 = invoke(stypy.reporting.localization.Localization(__file__, 144, 46), cos_25374, *[result_mul_25377], **kwargs_25378)
        
        int_25380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 56), 'int')
        # Applying the binary operator '**' (line 144)
        result_pow_25381 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 46), '**', cos_call_result_25379, int_25380)
        
        # Applying the binary operator '*' (line 144)
        result_mul_25382 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 44), '*', int_25373, result_pow_25381)
        
        
        # Call to exp(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Call to sin(...): (line 144)
        # Processing the call arguments (line 144)
        int_25385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 66), 'int')
        # Getting the type of 'x' (line 144)
        x_25386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 68), 'x', False)
        # Applying the binary operator '*' (line 144)
        result_mul_25387 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 66), '*', int_25385, x_25386)
        
        # Processing the call keyword arguments (line 144)
        kwargs_25388 = {}
        # Getting the type of 'sin' (line 144)
        sin_25384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 62), 'sin', False)
        # Calling sin(args, kwargs) (line 144)
        sin_call_result_25389 = invoke(stypy.reporting.localization.Localization(__file__, 144, 62), sin_25384, *[result_mul_25387], **kwargs_25388)
        
        # Processing the call keyword arguments (line 144)
        kwargs_25390 = {}
        # Getting the type of 'exp' (line 144)
        exp_25383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 58), 'exp', False)
        # Calling exp(args, kwargs) (line 144)
        exp_call_result_25391 = invoke(stypy.reporting.localization.Localization(__file__, 144, 58), exp_25383, *[sin_call_result_25389], **kwargs_25390)
        
        # Applying the binary operator '*' (line 144)
        result_mul_25392 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 57), '*', result_mul_25382, exp_call_result_25391)
        
        # Applying the binary operator '+' (line 144)
        result_add_25393 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 43), '+', result_sub_25372, result_mul_25392)
        
        # Assigning a type to the variable 'ddf' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'ddf', result_add_25393)
        
        # Call to assert_array_almost_equal(...): (line 145)
        # Processing the call arguments (line 145)
        
        # Call to diff(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'f' (line 145)
        f_25396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 43), 'f', False)
        # Processing the call keyword arguments (line 145)
        kwargs_25397 = {}
        # Getting the type of 'diff' (line 145)
        diff_25395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 145)
        diff_call_result_25398 = invoke(stypy.reporting.localization.Localization(__file__, 145, 38), diff_25395, *[f_25396], **kwargs_25397)
        
        # Getting the type of 'df' (line 145)
        df_25399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 46), 'df', False)
        # Processing the call keyword arguments (line 145)
        kwargs_25400 = {}
        # Getting the type of 'assert_array_almost_equal' (line 145)
        assert_array_almost_equal_25394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 145)
        assert_array_almost_equal_call_result_25401 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), assert_array_almost_equal_25394, *[diff_call_result_25398, df_25399], **kwargs_25400)
        
        
        # Call to assert_array_almost_equal(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Call to diff(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'df' (line 146)
        df_25404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 43), 'df', False)
        # Processing the call keyword arguments (line 146)
        kwargs_25405 = {}
        # Getting the type of 'diff' (line 146)
        diff_25403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 146)
        diff_call_result_25406 = invoke(stypy.reporting.localization.Localization(__file__, 146, 38), diff_25403, *[df_25404], **kwargs_25405)
        
        # Getting the type of 'ddf' (line 146)
        ddf_25407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 47), 'ddf', False)
        # Processing the call keyword arguments (line 146)
        kwargs_25408 = {}
        # Getting the type of 'assert_array_almost_equal' (line 146)
        assert_array_almost_equal_25402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 146)
        assert_array_almost_equal_call_result_25409 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), assert_array_almost_equal_25402, *[diff_call_result_25406, ddf_25407], **kwargs_25408)
        
        
        # Call to assert_array_almost_equal(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Call to diff(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'ddf' (line 147)
        ddf_25412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 43), 'ddf', False)
        int_25413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 47), 'int')
        # Processing the call keyword arguments (line 147)
        kwargs_25414 = {}
        # Getting the type of 'diff' (line 147)
        diff_25411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 147)
        diff_call_result_25415 = invoke(stypy.reporting.localization.Localization(__file__, 147, 38), diff_25411, *[ddf_25412, int_25413], **kwargs_25414)
        
        # Getting the type of 'df' (line 147)
        df_25416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 51), 'df', False)
        # Processing the call keyword arguments (line 147)
        kwargs_25417 = {}
        # Getting the type of 'assert_array_almost_equal' (line 147)
        assert_array_almost_equal_25410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 147)
        assert_array_almost_equal_call_result_25418 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), assert_array_almost_equal_25410, *[diff_call_result_25415, df_25416], **kwargs_25417)
        
        
        # Call to assert_array_almost_equal(...): (line 148)
        # Processing the call arguments (line 148)
        
        # Call to diff(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'f' (line 148)
        f_25421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 43), 'f', False)
        int_25422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 45), 'int')
        # Processing the call keyword arguments (line 148)
        kwargs_25423 = {}
        # Getting the type of 'diff' (line 148)
        diff_25420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 38), 'diff', False)
        # Calling diff(args, kwargs) (line 148)
        diff_call_result_25424 = invoke(stypy.reporting.localization.Localization(__file__, 148, 38), diff_25420, *[f_25421, int_25422], **kwargs_25423)
        
        # Getting the type of 'ddf' (line 148)
        ddf_25425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 48), 'ddf', False)
        # Processing the call keyword arguments (line 148)
        kwargs_25426 = {}
        # Getting the type of 'assert_array_almost_equal' (line 148)
        assert_array_almost_equal_25419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 148)
        assert_array_almost_equal_call_result_25427 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), assert_array_almost_equal_25419, *[diff_call_result_25424, ddf_25425], **kwargs_25426)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_expr_large(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_expr_large' in the type store
        # Getting the type of 'stypy_return_type' (line 138)
        stypy_return_type_25428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25428)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_expr_large'
        return stypy_return_type_25428


    @norecursion
    def test_int(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_int'
        module_type_store = module_type_store.open_function_context('test_int', 150, 4, False)
        # Assigning a type to the variable 'self' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiff.test_int.__dict__.__setitem__('stypy_localization', localization)
        TestDiff.test_int.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiff.test_int.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiff.test_int.__dict__.__setitem__('stypy_function_name', 'TestDiff.test_int')
        TestDiff.test_int.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiff.test_int.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiff.test_int.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiff.test_int.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiff.test_int.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiff.test_int.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiff.test_int.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiff.test_int', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_int', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_int(...)' code ##################

        
        # Assigning a Num to a Name (line 151):
        int_25429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 12), 'int')
        # Assigning a type to the variable 'n' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'n', int_25429)
        
        # Assigning a BinOp to a Name (line 152):
        
        # Call to arange(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'n' (line 152)
        n_25431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 19), 'n', False)
        # Processing the call keyword arguments (line 152)
        kwargs_25432 = {}
        # Getting the type of 'arange' (line 152)
        arange_25430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'arange', False)
        # Calling arange(args, kwargs) (line 152)
        arange_call_result_25433 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), arange_25430, *[n_25431], **kwargs_25432)
        
        int_25434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 22), 'int')
        # Applying the binary operator '*' (line 152)
        result_mul_25435 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 12), '*', arange_call_result_25433, int_25434)
        
        # Getting the type of 'pi' (line 152)
        pi_25436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'pi')
        # Applying the binary operator '*' (line 152)
        result_mul_25437 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 23), '*', result_mul_25435, pi_25436)
        
        # Getting the type of 'n' (line 152)
        n_25438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 27), 'n')
        # Applying the binary operator 'div' (line 152)
        result_div_25439 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 26), 'div', result_mul_25437, n_25438)
        
        # Assigning a type to the variable 'x' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'x', result_div_25439)
        
        # Call to assert_array_almost_equal(...): (line 153)
        # Processing the call arguments (line 153)
        
        # Call to diff(...): (line 153)
        # Processing the call arguments (line 153)
        
        # Call to sin(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'x' (line 153)
        x_25443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 43), 'x', False)
        # Processing the call keyword arguments (line 153)
        kwargs_25444 = {}
        # Getting the type of 'sin' (line 153)
        sin_25442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 39), 'sin', False)
        # Calling sin(args, kwargs) (line 153)
        sin_call_result_25445 = invoke(stypy.reporting.localization.Localization(__file__, 153, 39), sin_25442, *[x_25443], **kwargs_25444)
        
        int_25446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 46), 'int')
        # Processing the call keyword arguments (line 153)
        kwargs_25447 = {}
        # Getting the type of 'diff' (line 153)
        diff_25441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 34), 'diff', False)
        # Calling diff(args, kwargs) (line 153)
        diff_call_result_25448 = invoke(stypy.reporting.localization.Localization(__file__, 153, 34), diff_25441, *[sin_call_result_25445, int_25446], **kwargs_25447)
        
        
        
        # Call to cos(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'x' (line 153)
        x_25450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 55), 'x', False)
        # Processing the call keyword arguments (line 153)
        kwargs_25451 = {}
        # Getting the type of 'cos' (line 153)
        cos_25449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 51), 'cos', False)
        # Calling cos(args, kwargs) (line 153)
        cos_call_result_25452 = invoke(stypy.reporting.localization.Localization(__file__, 153, 51), cos_25449, *[x_25450], **kwargs_25451)
        
        # Applying the 'usub' unary operator (line 153)
        result___neg___25453 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 50), 'usub', cos_call_result_25452)
        
        # Processing the call keyword arguments (line 153)
        kwargs_25454 = {}
        # Getting the type of 'assert_array_almost_equal' (line 153)
        assert_array_almost_equal_25440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 153)
        assert_array_almost_equal_call_result_25455 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), assert_array_almost_equal_25440, *[diff_call_result_25448, result___neg___25453], **kwargs_25454)
        
        
        # Call to assert_array_almost_equal(...): (line 154)
        # Processing the call arguments (line 154)
        
        # Call to diff(...): (line 154)
        # Processing the call arguments (line 154)
        
        # Call to sin(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'x' (line 154)
        x_25459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 43), 'x', False)
        # Processing the call keyword arguments (line 154)
        kwargs_25460 = {}
        # Getting the type of 'sin' (line 154)
        sin_25458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 39), 'sin', False)
        # Calling sin(args, kwargs) (line 154)
        sin_call_result_25461 = invoke(stypy.reporting.localization.Localization(__file__, 154, 39), sin_25458, *[x_25459], **kwargs_25460)
        
        int_25462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 46), 'int')
        # Processing the call keyword arguments (line 154)
        kwargs_25463 = {}
        # Getting the type of 'diff' (line 154)
        diff_25457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 34), 'diff', False)
        # Calling diff(args, kwargs) (line 154)
        diff_call_result_25464 = invoke(stypy.reporting.localization.Localization(__file__, 154, 34), diff_25457, *[sin_call_result_25461, int_25462], **kwargs_25463)
        
        
        
        # Call to sin(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'x' (line 154)
        x_25466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 55), 'x', False)
        # Processing the call keyword arguments (line 154)
        kwargs_25467 = {}
        # Getting the type of 'sin' (line 154)
        sin_25465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 51), 'sin', False)
        # Calling sin(args, kwargs) (line 154)
        sin_call_result_25468 = invoke(stypy.reporting.localization.Localization(__file__, 154, 51), sin_25465, *[x_25466], **kwargs_25467)
        
        # Applying the 'usub' unary operator (line 154)
        result___neg___25469 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 50), 'usub', sin_call_result_25468)
        
        # Processing the call keyword arguments (line 154)
        kwargs_25470 = {}
        # Getting the type of 'assert_array_almost_equal' (line 154)
        assert_array_almost_equal_25456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 154)
        assert_array_almost_equal_call_result_25471 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), assert_array_almost_equal_25456, *[diff_call_result_25464, result___neg___25469], **kwargs_25470)
        
        
        # Call to assert_array_almost_equal(...): (line 155)
        # Processing the call arguments (line 155)
        
        # Call to diff(...): (line 155)
        # Processing the call arguments (line 155)
        
        # Call to sin(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'x' (line 155)
        x_25475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 43), 'x', False)
        # Processing the call keyword arguments (line 155)
        kwargs_25476 = {}
        # Getting the type of 'sin' (line 155)
        sin_25474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 39), 'sin', False)
        # Calling sin(args, kwargs) (line 155)
        sin_call_result_25477 = invoke(stypy.reporting.localization.Localization(__file__, 155, 39), sin_25474, *[x_25475], **kwargs_25476)
        
        int_25478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 46), 'int')
        # Processing the call keyword arguments (line 155)
        kwargs_25479 = {}
        # Getting the type of 'diff' (line 155)
        diff_25473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), 'diff', False)
        # Calling diff(args, kwargs) (line 155)
        diff_call_result_25480 = invoke(stypy.reporting.localization.Localization(__file__, 155, 34), diff_25473, *[sin_call_result_25477, int_25478], **kwargs_25479)
        
        
        # Call to sin(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'x' (line 155)
        x_25482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 54), 'x', False)
        # Processing the call keyword arguments (line 155)
        kwargs_25483 = {}
        # Getting the type of 'sin' (line 155)
        sin_25481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 50), 'sin', False)
        # Calling sin(args, kwargs) (line 155)
        sin_call_result_25484 = invoke(stypy.reporting.localization.Localization(__file__, 155, 50), sin_25481, *[x_25482], **kwargs_25483)
        
        # Processing the call keyword arguments (line 155)
        kwargs_25485 = {}
        # Getting the type of 'assert_array_almost_equal' (line 155)
        assert_array_almost_equal_25472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 155)
        assert_array_almost_equal_call_result_25486 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), assert_array_almost_equal_25472, *[diff_call_result_25480, sin_call_result_25484], **kwargs_25485)
        
        
        # Call to assert_array_almost_equal(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Call to diff(...): (line 156)
        # Processing the call arguments (line 156)
        int_25489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 39), 'int')
        
        # Call to cos(...): (line 156)
        # Processing the call arguments (line 156)
        int_25491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 45), 'int')
        # Getting the type of 'x' (line 156)
        x_25492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 47), 'x', False)
        # Applying the binary operator '*' (line 156)
        result_mul_25493 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 45), '*', int_25491, x_25492)
        
        # Processing the call keyword arguments (line 156)
        kwargs_25494 = {}
        # Getting the type of 'cos' (line 156)
        cos_25490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 41), 'cos', False)
        # Calling cos(args, kwargs) (line 156)
        cos_call_result_25495 = invoke(stypy.reporting.localization.Localization(__file__, 156, 41), cos_25490, *[result_mul_25493], **kwargs_25494)
        
        # Applying the binary operator '*' (line 156)
        result_mul_25496 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 39), '*', int_25489, cos_call_result_25495)
        
        int_25497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 50), 'int')
        # Processing the call keyword arguments (line 156)
        kwargs_25498 = {}
        # Getting the type of 'diff' (line 156)
        diff_25488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 34), 'diff', False)
        # Calling diff(args, kwargs) (line 156)
        diff_call_result_25499 = invoke(stypy.reporting.localization.Localization(__file__, 156, 34), diff_25488, *[result_mul_25496, int_25497], **kwargs_25498)
        
        
        # Call to sin(...): (line 156)
        # Processing the call arguments (line 156)
        int_25501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 58), 'int')
        # Getting the type of 'x' (line 156)
        x_25502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 60), 'x', False)
        # Applying the binary operator '*' (line 156)
        result_mul_25503 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 58), '*', int_25501, x_25502)
        
        # Processing the call keyword arguments (line 156)
        kwargs_25504 = {}
        # Getting the type of 'sin' (line 156)
        sin_25500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 54), 'sin', False)
        # Calling sin(args, kwargs) (line 156)
        sin_call_result_25505 = invoke(stypy.reporting.localization.Localization(__file__, 156, 54), sin_25500, *[result_mul_25503], **kwargs_25504)
        
        # Processing the call keyword arguments (line 156)
        kwargs_25506 = {}
        # Getting the type of 'assert_array_almost_equal' (line 156)
        assert_array_almost_equal_25487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 156)
        assert_array_almost_equal_call_result_25507 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), assert_array_almost_equal_25487, *[diff_call_result_25499, sin_call_result_25505], **kwargs_25506)
        
        
        # ################# End of 'test_int(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_int' in the type store
        # Getting the type of 'stypy_return_type' (line 150)
        stypy_return_type_25508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25508)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_int'
        return stypy_return_type_25508


    @norecursion
    def test_random_even(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_random_even'
        module_type_store = module_type_store.open_function_context('test_random_even', 158, 4, False)
        # Assigning a type to the variable 'self' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiff.test_random_even.__dict__.__setitem__('stypy_localization', localization)
        TestDiff.test_random_even.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiff.test_random_even.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiff.test_random_even.__dict__.__setitem__('stypy_function_name', 'TestDiff.test_random_even')
        TestDiff.test_random_even.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiff.test_random_even.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiff.test_random_even.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiff.test_random_even.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiff.test_random_even.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiff.test_random_even.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiff.test_random_even.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiff.test_random_even', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_random_even', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_random_even(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 159)
        list_25509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 159)
        # Adding element type (line 159)
        int_25510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 17), list_25509, int_25510)
        # Adding element type (line 159)
        int_25511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 17), list_25509, int_25511)
        # Adding element type (line 159)
        int_25512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 17), list_25509, int_25512)
        # Adding element type (line 159)
        int_25513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 17), list_25509, int_25513)
        
        # Testing the type of a for loop iterable (line 159)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 159, 8), list_25509)
        # Getting the type of the for loop variable (line 159)
        for_loop_var_25514 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 159, 8), list_25509)
        # Assigning a type to the variable 'k' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'k', for_loop_var_25514)
        # SSA begins for a for statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'list' (line 160)
        list_25515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 160)
        # Adding element type (line 160)
        int_25516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 21), list_25515, int_25516)
        # Adding element type (line 160)
        int_25517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 21), list_25515, int_25517)
        # Adding element type (line 160)
        int_25518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 21), list_25515, int_25518)
        # Adding element type (line 160)
        int_25519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 21), list_25515, int_25519)
        # Adding element type (line 160)
        int_25520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 21), list_25515, int_25520)
        
        # Testing the type of a for loop iterable (line 160)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 160, 12), list_25515)
        # Getting the type of the for loop variable (line 160)
        for_loop_var_25521 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 160, 12), list_25515)
        # Assigning a type to the variable 'n' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'n', for_loop_var_25521)
        # SSA begins for a for statement (line 160)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 161):
        
        # Call to random(...): (line 161)
        # Processing the call arguments (line 161)
        
        # Obtaining an instance of the builtin type 'tuple' (line 161)
        tuple_25523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 161)
        # Adding element type (line 161)
        # Getting the type of 'n' (line 161)
        n_25524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 28), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 28), tuple_25523, n_25524)
        
        # Processing the call keyword arguments (line 161)
        kwargs_25525 = {}
        # Getting the type of 'random' (line 161)
        random_25522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'random', False)
        # Calling random(args, kwargs) (line 161)
        random_call_result_25526 = invoke(stypy.reporting.localization.Localization(__file__, 161, 20), random_25522, *[tuple_25523], **kwargs_25525)
        
        # Assigning a type to the variable 'f' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'f', random_call_result_25526)
        
        # Assigning a BinOp to a Name (line 162):
        
        # Call to sum(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'f' (line 162)
        f_25528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 25), 'f', False)
        # Processing the call keyword arguments (line 162)
        int_25529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 32), 'int')
        keyword_25530 = int_25529
        kwargs_25531 = {'axis': keyword_25530}
        # Getting the type of 'sum' (line 162)
        sum_25527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 21), 'sum', False)
        # Calling sum(args, kwargs) (line 162)
        sum_call_result_25532 = invoke(stypy.reporting.localization.Localization(__file__, 162, 21), sum_25527, *[f_25528], **kwargs_25531)
        
        # Getting the type of 'n' (line 162)
        n_25533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 35), 'n')
        # Applying the binary operator 'div' (line 162)
        result_div_25534 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 21), 'div', sum_call_result_25532, n_25533)
        
        # Assigning a type to the variable 'af' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'af', result_div_25534)
        
        # Assigning a BinOp to a Name (line 163):
        # Getting the type of 'f' (line 163)
        f_25535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'f')
        # Getting the type of 'af' (line 163)
        af_25536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 22), 'af')
        # Applying the binary operator '-' (line 163)
        result_sub_25537 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 20), '-', f_25535, af_25536)
        
        # Assigning a type to the variable 'f' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), 'f', result_sub_25537)
        
        # Assigning a Call to a Name (line 165):
        
        # Call to diff(...): (line 165)
        # Processing the call arguments (line 165)
        
        # Call to diff(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'f' (line 165)
        f_25540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 30), 'f', False)
        int_25541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 32), 'int')
        # Processing the call keyword arguments (line 165)
        kwargs_25542 = {}
        # Getting the type of 'diff' (line 165)
        diff_25539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 25), 'diff', False)
        # Calling diff(args, kwargs) (line 165)
        diff_call_result_25543 = invoke(stypy.reporting.localization.Localization(__file__, 165, 25), diff_25539, *[f_25540, int_25541], **kwargs_25542)
        
        int_25544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 35), 'int')
        # Processing the call keyword arguments (line 165)
        kwargs_25545 = {}
        # Getting the type of 'diff' (line 165)
        diff_25538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 20), 'diff', False)
        # Calling diff(args, kwargs) (line 165)
        diff_call_result_25546 = invoke(stypy.reporting.localization.Localization(__file__, 165, 20), diff_25538, *[diff_call_result_25543, int_25544], **kwargs_25545)
        
        # Assigning a type to the variable 'f' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'f', diff_call_result_25546)
        
        # Call to assert_almost_equal(...): (line 166)
        # Processing the call arguments (line 166)
        
        # Call to sum(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'f' (line 166)
        f_25549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 40), 'f', False)
        # Processing the call keyword arguments (line 166)
        int_25550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 47), 'int')
        keyword_25551 = int_25550
        kwargs_25552 = {'axis': keyword_25551}
        # Getting the type of 'sum' (line 166)
        sum_25548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 36), 'sum', False)
        # Calling sum(args, kwargs) (line 166)
        sum_call_result_25553 = invoke(stypy.reporting.localization.Localization(__file__, 166, 36), sum_25548, *[f_25549], **kwargs_25552)
        
        float_25554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 50), 'float')
        # Processing the call keyword arguments (line 166)
        kwargs_25555 = {}
        # Getting the type of 'assert_almost_equal' (line 166)
        assert_almost_equal_25547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 166)
        assert_almost_equal_call_result_25556 = invoke(stypy.reporting.localization.Localization(__file__, 166, 16), assert_almost_equal_25547, *[sum_call_result_25553, float_25554], **kwargs_25555)
        
        
        # Call to assert_array_almost_equal(...): (line 167)
        # Processing the call arguments (line 167)
        
        # Call to diff(...): (line 167)
        # Processing the call arguments (line 167)
        
        # Call to diff(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'f' (line 167)
        f_25560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 52), 'f', False)
        # Getting the type of 'k' (line 167)
        k_25561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 54), 'k', False)
        # Processing the call keyword arguments (line 167)
        kwargs_25562 = {}
        # Getting the type of 'diff' (line 167)
        diff_25559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 47), 'diff', False)
        # Calling diff(args, kwargs) (line 167)
        diff_call_result_25563 = invoke(stypy.reporting.localization.Localization(__file__, 167, 47), diff_25559, *[f_25560, k_25561], **kwargs_25562)
        
        
        # Getting the type of 'k' (line 167)
        k_25564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 58), 'k', False)
        # Applying the 'usub' unary operator (line 167)
        result___neg___25565 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 57), 'usub', k_25564)
        
        # Processing the call keyword arguments (line 167)
        kwargs_25566 = {}
        # Getting the type of 'diff' (line 167)
        diff_25558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 42), 'diff', False)
        # Calling diff(args, kwargs) (line 167)
        diff_call_result_25567 = invoke(stypy.reporting.localization.Localization(__file__, 167, 42), diff_25558, *[diff_call_result_25563, result___neg___25565], **kwargs_25566)
        
        # Getting the type of 'f' (line 167)
        f_25568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 61), 'f', False)
        # Processing the call keyword arguments (line 167)
        kwargs_25569 = {}
        # Getting the type of 'assert_array_almost_equal' (line 167)
        assert_array_almost_equal_25557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 167)
        assert_array_almost_equal_call_result_25570 = invoke(stypy.reporting.localization.Localization(__file__, 167, 16), assert_array_almost_equal_25557, *[diff_call_result_25567, f_25568], **kwargs_25569)
        
        
        # Call to assert_array_almost_equal(...): (line 168)
        # Processing the call arguments (line 168)
        
        # Call to diff(...): (line 168)
        # Processing the call arguments (line 168)
        
        # Call to diff(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'f' (line 168)
        f_25574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 52), 'f', False)
        
        # Getting the type of 'k' (line 168)
        k_25575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 55), 'k', False)
        # Applying the 'usub' unary operator (line 168)
        result___neg___25576 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 54), 'usub', k_25575)
        
        # Processing the call keyword arguments (line 168)
        kwargs_25577 = {}
        # Getting the type of 'diff' (line 168)
        diff_25573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 47), 'diff', False)
        # Calling diff(args, kwargs) (line 168)
        diff_call_result_25578 = invoke(stypy.reporting.localization.Localization(__file__, 168, 47), diff_25573, *[f_25574, result___neg___25576], **kwargs_25577)
        
        # Getting the type of 'k' (line 168)
        k_25579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 58), 'k', False)
        # Processing the call keyword arguments (line 168)
        kwargs_25580 = {}
        # Getting the type of 'diff' (line 168)
        diff_25572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 42), 'diff', False)
        # Calling diff(args, kwargs) (line 168)
        diff_call_result_25581 = invoke(stypy.reporting.localization.Localization(__file__, 168, 42), diff_25572, *[diff_call_result_25578, k_25579], **kwargs_25580)
        
        # Getting the type of 'f' (line 168)
        f_25582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 61), 'f', False)
        # Processing the call keyword arguments (line 168)
        kwargs_25583 = {}
        # Getting the type of 'assert_array_almost_equal' (line 168)
        assert_array_almost_equal_25571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 168)
        assert_array_almost_equal_call_result_25584 = invoke(stypy.reporting.localization.Localization(__file__, 168, 16), assert_array_almost_equal_25571, *[diff_call_result_25581, f_25582], **kwargs_25583)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_random_even(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_random_even' in the type store
        # Getting the type of 'stypy_return_type' (line 158)
        stypy_return_type_25585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25585)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_random_even'
        return stypy_return_type_25585


    @norecursion
    def test_random_odd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_random_odd'
        module_type_store = module_type_store.open_function_context('test_random_odd', 170, 4, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiff.test_random_odd.__dict__.__setitem__('stypy_localization', localization)
        TestDiff.test_random_odd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiff.test_random_odd.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiff.test_random_odd.__dict__.__setitem__('stypy_function_name', 'TestDiff.test_random_odd')
        TestDiff.test_random_odd.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiff.test_random_odd.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiff.test_random_odd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiff.test_random_odd.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiff.test_random_odd.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiff.test_random_odd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiff.test_random_odd.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiff.test_random_odd', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_random_odd', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_random_odd(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 171)
        list_25586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 171)
        # Adding element type (line 171)
        int_25587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 17), list_25586, int_25587)
        # Adding element type (line 171)
        int_25588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 17), list_25586, int_25588)
        # Adding element type (line 171)
        int_25589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 17), list_25586, int_25589)
        # Adding element type (line 171)
        int_25590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 17), list_25586, int_25590)
        # Adding element type (line 171)
        int_25591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 17), list_25586, int_25591)
        # Adding element type (line 171)
        int_25592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 17), list_25586, int_25592)
        # Adding element type (line 171)
        int_25593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 17), list_25586, int_25593)
        
        # Testing the type of a for loop iterable (line 171)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 171, 8), list_25586)
        # Getting the type of the for loop variable (line 171)
        for_loop_var_25594 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 171, 8), list_25586)
        # Assigning a type to the variable 'k' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'k', for_loop_var_25594)
        # SSA begins for a for statement (line 171)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'list' (line 172)
        list_25595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 172)
        # Adding element type (line 172)
        int_25596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 21), list_25595, int_25596)
        # Adding element type (line 172)
        int_25597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 21), list_25595, int_25597)
        # Adding element type (line 172)
        int_25598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 21), list_25595, int_25598)
        
        # Testing the type of a for loop iterable (line 172)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 172, 12), list_25595)
        # Getting the type of the for loop variable (line 172)
        for_loop_var_25599 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 172, 12), list_25595)
        # Assigning a type to the variable 'n' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'n', for_loop_var_25599)
        # SSA begins for a for statement (line 172)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 173):
        
        # Call to random(...): (line 173)
        # Processing the call arguments (line 173)
        
        # Obtaining an instance of the builtin type 'tuple' (line 173)
        tuple_25601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 173)
        # Adding element type (line 173)
        # Getting the type of 'n' (line 173)
        n_25602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 28), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 28), tuple_25601, n_25602)
        
        # Processing the call keyword arguments (line 173)
        kwargs_25603 = {}
        # Getting the type of 'random' (line 173)
        random_25600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 20), 'random', False)
        # Calling random(args, kwargs) (line 173)
        random_call_result_25604 = invoke(stypy.reporting.localization.Localization(__file__, 173, 20), random_25600, *[tuple_25601], **kwargs_25603)
        
        # Assigning a type to the variable 'f' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'f', random_call_result_25604)
        
        # Assigning a BinOp to a Name (line 174):
        
        # Call to sum(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'f' (line 174)
        f_25606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 25), 'f', False)
        # Processing the call keyword arguments (line 174)
        int_25607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 32), 'int')
        keyword_25608 = int_25607
        kwargs_25609 = {'axis': keyword_25608}
        # Getting the type of 'sum' (line 174)
        sum_25605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 21), 'sum', False)
        # Calling sum(args, kwargs) (line 174)
        sum_call_result_25610 = invoke(stypy.reporting.localization.Localization(__file__, 174, 21), sum_25605, *[f_25606], **kwargs_25609)
        
        # Getting the type of 'n' (line 174)
        n_25611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 35), 'n')
        # Applying the binary operator 'div' (line 174)
        result_div_25612 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 21), 'div', sum_call_result_25610, n_25611)
        
        # Assigning a type to the variable 'af' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'af', result_div_25612)
        
        # Assigning a BinOp to a Name (line 175):
        # Getting the type of 'f' (line 175)
        f_25613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 20), 'f')
        # Getting the type of 'af' (line 175)
        af_25614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 22), 'af')
        # Applying the binary operator '-' (line 175)
        result_sub_25615 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 20), '-', f_25613, af_25614)
        
        # Assigning a type to the variable 'f' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 16), 'f', result_sub_25615)
        
        # Call to assert_almost_equal(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Call to sum(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'f' (line 176)
        f_25618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 40), 'f', False)
        # Processing the call keyword arguments (line 176)
        int_25619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 47), 'int')
        keyword_25620 = int_25619
        kwargs_25621 = {'axis': keyword_25620}
        # Getting the type of 'sum' (line 176)
        sum_25617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 36), 'sum', False)
        # Calling sum(args, kwargs) (line 176)
        sum_call_result_25622 = invoke(stypy.reporting.localization.Localization(__file__, 176, 36), sum_25617, *[f_25618], **kwargs_25621)
        
        float_25623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 50), 'float')
        # Processing the call keyword arguments (line 176)
        kwargs_25624 = {}
        # Getting the type of 'assert_almost_equal' (line 176)
        assert_almost_equal_25616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 176)
        assert_almost_equal_call_result_25625 = invoke(stypy.reporting.localization.Localization(__file__, 176, 16), assert_almost_equal_25616, *[sum_call_result_25622, float_25623], **kwargs_25624)
        
        
        # Call to assert_array_almost_equal(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Call to diff(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Call to diff(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'f' (line 177)
        f_25629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 52), 'f', False)
        # Getting the type of 'k' (line 177)
        k_25630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 54), 'k', False)
        # Processing the call keyword arguments (line 177)
        kwargs_25631 = {}
        # Getting the type of 'diff' (line 177)
        diff_25628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 47), 'diff', False)
        # Calling diff(args, kwargs) (line 177)
        diff_call_result_25632 = invoke(stypy.reporting.localization.Localization(__file__, 177, 47), diff_25628, *[f_25629, k_25630], **kwargs_25631)
        
        
        # Getting the type of 'k' (line 177)
        k_25633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 58), 'k', False)
        # Applying the 'usub' unary operator (line 177)
        result___neg___25634 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 57), 'usub', k_25633)
        
        # Processing the call keyword arguments (line 177)
        kwargs_25635 = {}
        # Getting the type of 'diff' (line 177)
        diff_25627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 42), 'diff', False)
        # Calling diff(args, kwargs) (line 177)
        diff_call_result_25636 = invoke(stypy.reporting.localization.Localization(__file__, 177, 42), diff_25627, *[diff_call_result_25632, result___neg___25634], **kwargs_25635)
        
        # Getting the type of 'f' (line 177)
        f_25637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 61), 'f', False)
        # Processing the call keyword arguments (line 177)
        kwargs_25638 = {}
        # Getting the type of 'assert_array_almost_equal' (line 177)
        assert_array_almost_equal_25626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 177)
        assert_array_almost_equal_call_result_25639 = invoke(stypy.reporting.localization.Localization(__file__, 177, 16), assert_array_almost_equal_25626, *[diff_call_result_25636, f_25637], **kwargs_25638)
        
        
        # Call to assert_array_almost_equal(...): (line 178)
        # Processing the call arguments (line 178)
        
        # Call to diff(...): (line 178)
        # Processing the call arguments (line 178)
        
        # Call to diff(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'f' (line 178)
        f_25643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 52), 'f', False)
        
        # Getting the type of 'k' (line 178)
        k_25644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 55), 'k', False)
        # Applying the 'usub' unary operator (line 178)
        result___neg___25645 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 54), 'usub', k_25644)
        
        # Processing the call keyword arguments (line 178)
        kwargs_25646 = {}
        # Getting the type of 'diff' (line 178)
        diff_25642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 47), 'diff', False)
        # Calling diff(args, kwargs) (line 178)
        diff_call_result_25647 = invoke(stypy.reporting.localization.Localization(__file__, 178, 47), diff_25642, *[f_25643, result___neg___25645], **kwargs_25646)
        
        # Getting the type of 'k' (line 178)
        k_25648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 58), 'k', False)
        # Processing the call keyword arguments (line 178)
        kwargs_25649 = {}
        # Getting the type of 'diff' (line 178)
        diff_25641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 42), 'diff', False)
        # Calling diff(args, kwargs) (line 178)
        diff_call_result_25650 = invoke(stypy.reporting.localization.Localization(__file__, 178, 42), diff_25641, *[diff_call_result_25647, k_25648], **kwargs_25649)
        
        # Getting the type of 'f' (line 178)
        f_25651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 61), 'f', False)
        # Processing the call keyword arguments (line 178)
        kwargs_25652 = {}
        # Getting the type of 'assert_array_almost_equal' (line 178)
        assert_array_almost_equal_25640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 178)
        assert_array_almost_equal_call_result_25653 = invoke(stypy.reporting.localization.Localization(__file__, 178, 16), assert_array_almost_equal_25640, *[diff_call_result_25650, f_25651], **kwargs_25652)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_random_odd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_random_odd' in the type store
        # Getting the type of 'stypy_return_type' (line 170)
        stypy_return_type_25654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25654)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_random_odd'
        return stypy_return_type_25654


    @norecursion
    def test_zero_nyquist(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_zero_nyquist'
        module_type_store = module_type_store.open_function_context('test_zero_nyquist', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestDiff.test_zero_nyquist.__dict__.__setitem__('stypy_localization', localization)
        TestDiff.test_zero_nyquist.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestDiff.test_zero_nyquist.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestDiff.test_zero_nyquist.__dict__.__setitem__('stypy_function_name', 'TestDiff.test_zero_nyquist')
        TestDiff.test_zero_nyquist.__dict__.__setitem__('stypy_param_names_list', [])
        TestDiff.test_zero_nyquist.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestDiff.test_zero_nyquist.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestDiff.test_zero_nyquist.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestDiff.test_zero_nyquist.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestDiff.test_zero_nyquist.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestDiff.test_zero_nyquist.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiff.test_zero_nyquist', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_zero_nyquist', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_zero_nyquist(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 181)
        list_25655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 181)
        # Adding element type (line 181)
        int_25656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 17), list_25655, int_25656)
        # Adding element type (line 181)
        int_25657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 17), list_25655, int_25657)
        # Adding element type (line 181)
        int_25658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 17), list_25655, int_25658)
        # Adding element type (line 181)
        int_25659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 17), list_25655, int_25659)
        # Adding element type (line 181)
        int_25660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 17), list_25655, int_25660)
        # Adding element type (line 181)
        int_25661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 17), list_25655, int_25661)
        # Adding element type (line 181)
        int_25662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 17), list_25655, int_25662)
        
        # Testing the type of a for loop iterable (line 181)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 181, 8), list_25655)
        # Getting the type of the for loop variable (line 181)
        for_loop_var_25663 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 181, 8), list_25655)
        # Assigning a type to the variable 'k' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'k', for_loop_var_25663)
        # SSA begins for a for statement (line 181)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'list' (line 182)
        list_25664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 182)
        # Adding element type (line 182)
        int_25665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 21), list_25664, int_25665)
        # Adding element type (line 182)
        int_25666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 21), list_25664, int_25666)
        # Adding element type (line 182)
        int_25667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 21), list_25664, int_25667)
        # Adding element type (line 182)
        int_25668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 21), list_25664, int_25668)
        # Adding element type (line 182)
        int_25669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 21), list_25664, int_25669)
        
        # Testing the type of a for loop iterable (line 182)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 182, 12), list_25664)
        # Getting the type of the for loop variable (line 182)
        for_loop_var_25670 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 182, 12), list_25664)
        # Assigning a type to the variable 'n' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'n', for_loop_var_25670)
        # SSA begins for a for statement (line 182)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 183):
        
        # Call to random(...): (line 183)
        # Processing the call arguments (line 183)
        
        # Obtaining an instance of the builtin type 'tuple' (line 183)
        tuple_25672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 183)
        # Adding element type (line 183)
        # Getting the type of 'n' (line 183)
        n_25673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 28), tuple_25672, n_25673)
        
        # Processing the call keyword arguments (line 183)
        kwargs_25674 = {}
        # Getting the type of 'random' (line 183)
        random_25671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 20), 'random', False)
        # Calling random(args, kwargs) (line 183)
        random_call_result_25675 = invoke(stypy.reporting.localization.Localization(__file__, 183, 20), random_25671, *[tuple_25672], **kwargs_25674)
        
        # Assigning a type to the variable 'f' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'f', random_call_result_25675)
        
        # Assigning a BinOp to a Name (line 184):
        
        # Call to sum(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'f' (line 184)
        f_25677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 25), 'f', False)
        # Processing the call keyword arguments (line 184)
        int_25678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 32), 'int')
        keyword_25679 = int_25678
        kwargs_25680 = {'axis': keyword_25679}
        # Getting the type of 'sum' (line 184)
        sum_25676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), 'sum', False)
        # Calling sum(args, kwargs) (line 184)
        sum_call_result_25681 = invoke(stypy.reporting.localization.Localization(__file__, 184, 21), sum_25676, *[f_25677], **kwargs_25680)
        
        # Getting the type of 'n' (line 184)
        n_25682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 35), 'n')
        # Applying the binary operator 'div' (line 184)
        result_div_25683 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 21), 'div', sum_call_result_25681, n_25682)
        
        # Assigning a type to the variable 'af' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'af', result_div_25683)
        
        # Assigning a BinOp to a Name (line 185):
        # Getting the type of 'f' (line 185)
        f_25684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 20), 'f')
        # Getting the type of 'af' (line 185)
        af_25685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 22), 'af')
        # Applying the binary operator '-' (line 185)
        result_sub_25686 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 20), '-', f_25684, af_25685)
        
        # Assigning a type to the variable 'f' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'f', result_sub_25686)
        
        # Assigning a Call to a Name (line 187):
        
        # Call to diff(...): (line 187)
        # Processing the call arguments (line 187)
        
        # Call to diff(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'f' (line 187)
        f_25689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 30), 'f', False)
        int_25690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 32), 'int')
        # Processing the call keyword arguments (line 187)
        kwargs_25691 = {}
        # Getting the type of 'diff' (line 187)
        diff_25688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 25), 'diff', False)
        # Calling diff(args, kwargs) (line 187)
        diff_call_result_25692 = invoke(stypy.reporting.localization.Localization(__file__, 187, 25), diff_25688, *[f_25689, int_25690], **kwargs_25691)
        
        int_25693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 35), 'int')
        # Processing the call keyword arguments (line 187)
        kwargs_25694 = {}
        # Getting the type of 'diff' (line 187)
        diff_25687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 20), 'diff', False)
        # Calling diff(args, kwargs) (line 187)
        diff_call_result_25695 = invoke(stypy.reporting.localization.Localization(__file__, 187, 20), diff_25687, *[diff_call_result_25692, int_25693], **kwargs_25694)
        
        # Assigning a type to the variable 'f' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'f', diff_call_result_25695)
        
        # Call to assert_almost_equal(...): (line 188)
        # Processing the call arguments (line 188)
        
        # Call to sum(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'f' (line 188)
        f_25698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 40), 'f', False)
        # Processing the call keyword arguments (line 188)
        int_25699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 47), 'int')
        keyword_25700 = int_25699
        kwargs_25701 = {'axis': keyword_25700}
        # Getting the type of 'sum' (line 188)
        sum_25697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 36), 'sum', False)
        # Calling sum(args, kwargs) (line 188)
        sum_call_result_25702 = invoke(stypy.reporting.localization.Localization(__file__, 188, 36), sum_25697, *[f_25698], **kwargs_25701)
        
        float_25703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 50), 'float')
        # Processing the call keyword arguments (line 188)
        kwargs_25704 = {}
        # Getting the type of 'assert_almost_equal' (line 188)
        assert_almost_equal_25696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 188)
        assert_almost_equal_call_result_25705 = invoke(stypy.reporting.localization.Localization(__file__, 188, 16), assert_almost_equal_25696, *[sum_call_result_25702, float_25703], **kwargs_25704)
        
        
        # Call to assert_array_almost_equal(...): (line 189)
        # Processing the call arguments (line 189)
        
        # Call to diff(...): (line 189)
        # Processing the call arguments (line 189)
        
        # Call to diff(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'f' (line 189)
        f_25709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 52), 'f', False)
        # Getting the type of 'k' (line 189)
        k_25710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 54), 'k', False)
        # Processing the call keyword arguments (line 189)
        kwargs_25711 = {}
        # Getting the type of 'diff' (line 189)
        diff_25708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 47), 'diff', False)
        # Calling diff(args, kwargs) (line 189)
        diff_call_result_25712 = invoke(stypy.reporting.localization.Localization(__file__, 189, 47), diff_25708, *[f_25709, k_25710], **kwargs_25711)
        
        
        # Getting the type of 'k' (line 189)
        k_25713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 58), 'k', False)
        # Applying the 'usub' unary operator (line 189)
        result___neg___25714 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 57), 'usub', k_25713)
        
        # Processing the call keyword arguments (line 189)
        kwargs_25715 = {}
        # Getting the type of 'diff' (line 189)
        diff_25707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 42), 'diff', False)
        # Calling diff(args, kwargs) (line 189)
        diff_call_result_25716 = invoke(stypy.reporting.localization.Localization(__file__, 189, 42), diff_25707, *[diff_call_result_25712, result___neg___25714], **kwargs_25715)
        
        # Getting the type of 'f' (line 189)
        f_25717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 61), 'f', False)
        # Processing the call keyword arguments (line 189)
        kwargs_25718 = {}
        # Getting the type of 'assert_array_almost_equal' (line 189)
        assert_array_almost_equal_25706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 189)
        assert_array_almost_equal_call_result_25719 = invoke(stypy.reporting.localization.Localization(__file__, 189, 16), assert_array_almost_equal_25706, *[diff_call_result_25716, f_25717], **kwargs_25718)
        
        
        # Call to assert_array_almost_equal(...): (line 190)
        # Processing the call arguments (line 190)
        
        # Call to diff(...): (line 190)
        # Processing the call arguments (line 190)
        
        # Call to diff(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'f' (line 190)
        f_25723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 52), 'f', False)
        
        # Getting the type of 'k' (line 190)
        k_25724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 55), 'k', False)
        # Applying the 'usub' unary operator (line 190)
        result___neg___25725 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 54), 'usub', k_25724)
        
        # Processing the call keyword arguments (line 190)
        kwargs_25726 = {}
        # Getting the type of 'diff' (line 190)
        diff_25722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 47), 'diff', False)
        # Calling diff(args, kwargs) (line 190)
        diff_call_result_25727 = invoke(stypy.reporting.localization.Localization(__file__, 190, 47), diff_25722, *[f_25723, result___neg___25725], **kwargs_25726)
        
        # Getting the type of 'k' (line 190)
        k_25728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 58), 'k', False)
        # Processing the call keyword arguments (line 190)
        kwargs_25729 = {}
        # Getting the type of 'diff' (line 190)
        diff_25721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 42), 'diff', False)
        # Calling diff(args, kwargs) (line 190)
        diff_call_result_25730 = invoke(stypy.reporting.localization.Localization(__file__, 190, 42), diff_25721, *[diff_call_result_25727, k_25728], **kwargs_25729)
        
        # Getting the type of 'f' (line 190)
        f_25731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 61), 'f', False)
        # Processing the call keyword arguments (line 190)
        kwargs_25732 = {}
        # Getting the type of 'assert_array_almost_equal' (line 190)
        assert_array_almost_equal_25720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 190)
        assert_array_almost_equal_call_result_25733 = invoke(stypy.reporting.localization.Localization(__file__, 190, 16), assert_array_almost_equal_25720, *[diff_call_result_25730, f_25731], **kwargs_25732)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_zero_nyquist(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_zero_nyquist' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_25734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25734)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_zero_nyquist'
        return stypy_return_type_25734


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 84, 0, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestDiff.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestDiff' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'TestDiff', TestDiff)
# Declaration of the 'TestTilbert' class

class TestTilbert(object, ):

    @norecursion
    def test_definition(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_definition'
        module_type_store = module_type_store.open_function_context('test_definition', 195, 4, False)
        # Assigning a type to the variable 'self' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTilbert.test_definition.__dict__.__setitem__('stypy_localization', localization)
        TestTilbert.test_definition.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTilbert.test_definition.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTilbert.test_definition.__dict__.__setitem__('stypy_function_name', 'TestTilbert.test_definition')
        TestTilbert.test_definition.__dict__.__setitem__('stypy_param_names_list', [])
        TestTilbert.test_definition.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTilbert.test_definition.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTilbert.test_definition.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTilbert.test_definition.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTilbert.test_definition.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTilbert.test_definition.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTilbert.test_definition', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_definition', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_definition(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 196)
        list_25735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 196)
        # Adding element type (line 196)
        float_25736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 17), list_25735, float_25736)
        # Adding element type (line 196)
        float_25737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 17), list_25735, float_25737)
        # Adding element type (line 196)
        int_25738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 17), list_25735, int_25738)
        # Adding element type (line 196)
        float_25739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 17), list_25735, float_25739)
        # Adding element type (line 196)
        int_25740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 196, 17), list_25735, int_25740)
        
        # Testing the type of a for loop iterable (line 196)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 196, 8), list_25735)
        # Getting the type of the for loop variable (line 196)
        for_loop_var_25741 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 196, 8), list_25735)
        # Assigning a type to the variable 'h' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'h', for_loop_var_25741)
        # SSA begins for a for statement (line 196)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'list' (line 197)
        list_25742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 197)
        # Adding element type (line 197)
        int_25743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 21), list_25742, int_25743)
        # Adding element type (line 197)
        int_25744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 21), list_25742, int_25744)
        # Adding element type (line 197)
        int_25745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 21), list_25742, int_25745)
        # Adding element type (line 197)
        int_25746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 21), list_25742, int_25746)
        
        # Testing the type of a for loop iterable (line 197)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 197, 12), list_25742)
        # Getting the type of the for loop variable (line 197)
        for_loop_var_25747 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 197, 12), list_25742)
        # Assigning a type to the variable 'n' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'n', for_loop_var_25747)
        # SSA begins for a for statement (line 197)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 198):
        
        # Call to arange(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'n' (line 198)
        n_25749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 27), 'n', False)
        # Processing the call keyword arguments (line 198)
        kwargs_25750 = {}
        # Getting the type of 'arange' (line 198)
        arange_25748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 'arange', False)
        # Calling arange(args, kwargs) (line 198)
        arange_call_result_25751 = invoke(stypy.reporting.localization.Localization(__file__, 198, 20), arange_25748, *[n_25749], **kwargs_25750)
        
        int_25752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 30), 'int')
        # Applying the binary operator '*' (line 198)
        result_mul_25753 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 20), '*', arange_call_result_25751, int_25752)
        
        # Getting the type of 'pi' (line 198)
        pi_25754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 32), 'pi')
        # Applying the binary operator '*' (line 198)
        result_mul_25755 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 31), '*', result_mul_25753, pi_25754)
        
        # Getting the type of 'n' (line 198)
        n_25756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 35), 'n')
        # Applying the binary operator 'div' (line 198)
        result_div_25757 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 34), 'div', result_mul_25755, n_25756)
        
        # Assigning a type to the variable 'x' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'x', result_div_25757)
        
        # Assigning a Call to a Name (line 199):
        
        # Call to tilbert(...): (line 199)
        # Processing the call arguments (line 199)
        
        # Call to sin(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'x' (line 199)
        x_25760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 32), 'x', False)
        # Processing the call keyword arguments (line 199)
        kwargs_25761 = {}
        # Getting the type of 'sin' (line 199)
        sin_25759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'sin', False)
        # Calling sin(args, kwargs) (line 199)
        sin_call_result_25762 = invoke(stypy.reporting.localization.Localization(__file__, 199, 28), sin_25759, *[x_25760], **kwargs_25761)
        
        # Getting the type of 'h' (line 199)
        h_25763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 35), 'h', False)
        # Processing the call keyword arguments (line 199)
        kwargs_25764 = {}
        # Getting the type of 'tilbert' (line 199)
        tilbert_25758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'tilbert', False)
        # Calling tilbert(args, kwargs) (line 199)
        tilbert_call_result_25765 = invoke(stypy.reporting.localization.Localization(__file__, 199, 20), tilbert_25758, *[sin_call_result_25762, h_25763], **kwargs_25764)
        
        # Assigning a type to the variable 'y' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'y', tilbert_call_result_25765)
        
        # Assigning a Call to a Name (line 200):
        
        # Call to direct_tilbert(...): (line 200)
        # Processing the call arguments (line 200)
        
        # Call to sin(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'x' (line 200)
        x_25768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 40), 'x', False)
        # Processing the call keyword arguments (line 200)
        kwargs_25769 = {}
        # Getting the type of 'sin' (line 200)
        sin_25767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), 'sin', False)
        # Calling sin(args, kwargs) (line 200)
        sin_call_result_25770 = invoke(stypy.reporting.localization.Localization(__file__, 200, 36), sin_25767, *[x_25768], **kwargs_25769)
        
        # Getting the type of 'h' (line 200)
        h_25771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 43), 'h', False)
        # Processing the call keyword arguments (line 200)
        kwargs_25772 = {}
        # Getting the type of 'direct_tilbert' (line 200)
        direct_tilbert_25766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 'direct_tilbert', False)
        # Calling direct_tilbert(args, kwargs) (line 200)
        direct_tilbert_call_result_25773 = invoke(stypy.reporting.localization.Localization(__file__, 200, 21), direct_tilbert_25766, *[sin_call_result_25770, h_25771], **kwargs_25772)
        
        # Assigning a type to the variable 'y1' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 16), 'y1', direct_tilbert_call_result_25773)
        
        # Call to assert_array_almost_equal(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'y' (line 201)
        y_25775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 42), 'y', False)
        # Getting the type of 'y1' (line 201)
        y1_25776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 44), 'y1', False)
        # Processing the call keyword arguments (line 201)
        kwargs_25777 = {}
        # Getting the type of 'assert_array_almost_equal' (line 201)
        assert_array_almost_equal_25774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 201)
        assert_array_almost_equal_call_result_25778 = invoke(stypy.reporting.localization.Localization(__file__, 201, 16), assert_array_almost_equal_25774, *[y_25775, y1_25776], **kwargs_25777)
        
        
        # Call to assert_array_almost_equal(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Call to tilbert(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Call to sin(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'x' (line 202)
        x_25782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 54), 'x', False)
        # Processing the call keyword arguments (line 202)
        kwargs_25783 = {}
        # Getting the type of 'sin' (line 202)
        sin_25781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 50), 'sin', False)
        # Calling sin(args, kwargs) (line 202)
        sin_call_result_25784 = invoke(stypy.reporting.localization.Localization(__file__, 202, 50), sin_25781, *[x_25782], **kwargs_25783)
        
        # Getting the type of 'h' (line 202)
        h_25785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 57), 'h', False)
        # Processing the call keyword arguments (line 202)
        kwargs_25786 = {}
        # Getting the type of 'tilbert' (line 202)
        tilbert_25780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 42), 'tilbert', False)
        # Calling tilbert(args, kwargs) (line 202)
        tilbert_call_result_25787 = invoke(stypy.reporting.localization.Localization(__file__, 202, 42), tilbert_25780, *[sin_call_result_25784, h_25785], **kwargs_25786)
        
        
        # Call to direct_tilbert(...): (line 203)
        # Processing the call arguments (line 203)
        
        # Call to sin(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'x' (line 203)
        x_25790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 61), 'x', False)
        # Processing the call keyword arguments (line 203)
        kwargs_25791 = {}
        # Getting the type of 'sin' (line 203)
        sin_25789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 57), 'sin', False)
        # Calling sin(args, kwargs) (line 203)
        sin_call_result_25792 = invoke(stypy.reporting.localization.Localization(__file__, 203, 57), sin_25789, *[x_25790], **kwargs_25791)
        
        # Getting the type of 'h' (line 203)
        h_25793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 64), 'h', False)
        # Processing the call keyword arguments (line 203)
        kwargs_25794 = {}
        # Getting the type of 'direct_tilbert' (line 203)
        direct_tilbert_25788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 42), 'direct_tilbert', False)
        # Calling direct_tilbert(args, kwargs) (line 203)
        direct_tilbert_call_result_25795 = invoke(stypy.reporting.localization.Localization(__file__, 203, 42), direct_tilbert_25788, *[sin_call_result_25792, h_25793], **kwargs_25794)
        
        # Processing the call keyword arguments (line 202)
        kwargs_25796 = {}
        # Getting the type of 'assert_array_almost_equal' (line 202)
        assert_array_almost_equal_25779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 202)
        assert_array_almost_equal_call_result_25797 = invoke(stypy.reporting.localization.Localization(__file__, 202, 16), assert_array_almost_equal_25779, *[tilbert_call_result_25787, direct_tilbert_call_result_25795], **kwargs_25796)
        
        
        # Call to assert_array_almost_equal(...): (line 204)
        # Processing the call arguments (line 204)
        
        # Call to tilbert(...): (line 204)
        # Processing the call arguments (line 204)
        
        # Call to sin(...): (line 204)
        # Processing the call arguments (line 204)
        int_25801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 54), 'int')
        # Getting the type of 'x' (line 204)
        x_25802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 56), 'x', False)
        # Applying the binary operator '*' (line 204)
        result_mul_25803 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 54), '*', int_25801, x_25802)
        
        # Processing the call keyword arguments (line 204)
        kwargs_25804 = {}
        # Getting the type of 'sin' (line 204)
        sin_25800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 50), 'sin', False)
        # Calling sin(args, kwargs) (line 204)
        sin_call_result_25805 = invoke(stypy.reporting.localization.Localization(__file__, 204, 50), sin_25800, *[result_mul_25803], **kwargs_25804)
        
        # Getting the type of 'h' (line 204)
        h_25806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 59), 'h', False)
        # Processing the call keyword arguments (line 204)
        kwargs_25807 = {}
        # Getting the type of 'tilbert' (line 204)
        tilbert_25799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 42), 'tilbert', False)
        # Calling tilbert(args, kwargs) (line 204)
        tilbert_call_result_25808 = invoke(stypy.reporting.localization.Localization(__file__, 204, 42), tilbert_25799, *[sin_call_result_25805, h_25806], **kwargs_25807)
        
        
        # Call to direct_tilbert(...): (line 205)
        # Processing the call arguments (line 205)
        
        # Call to sin(...): (line 205)
        # Processing the call arguments (line 205)
        int_25811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 61), 'int')
        # Getting the type of 'x' (line 205)
        x_25812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 63), 'x', False)
        # Applying the binary operator '*' (line 205)
        result_mul_25813 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 61), '*', int_25811, x_25812)
        
        # Processing the call keyword arguments (line 205)
        kwargs_25814 = {}
        # Getting the type of 'sin' (line 205)
        sin_25810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 57), 'sin', False)
        # Calling sin(args, kwargs) (line 205)
        sin_call_result_25815 = invoke(stypy.reporting.localization.Localization(__file__, 205, 57), sin_25810, *[result_mul_25813], **kwargs_25814)
        
        # Getting the type of 'h' (line 205)
        h_25816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 66), 'h', False)
        # Processing the call keyword arguments (line 205)
        kwargs_25817 = {}
        # Getting the type of 'direct_tilbert' (line 205)
        direct_tilbert_25809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 42), 'direct_tilbert', False)
        # Calling direct_tilbert(args, kwargs) (line 205)
        direct_tilbert_call_result_25818 = invoke(stypy.reporting.localization.Localization(__file__, 205, 42), direct_tilbert_25809, *[sin_call_result_25815, h_25816], **kwargs_25817)
        
        # Processing the call keyword arguments (line 204)
        kwargs_25819 = {}
        # Getting the type of 'assert_array_almost_equal' (line 204)
        assert_array_almost_equal_25798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 204)
        assert_array_almost_equal_call_result_25820 = invoke(stypy.reporting.localization.Localization(__file__, 204, 16), assert_array_almost_equal_25798, *[tilbert_call_result_25808, direct_tilbert_call_result_25818], **kwargs_25819)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_definition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_definition' in the type store
        # Getting the type of 'stypy_return_type' (line 195)
        stypy_return_type_25821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25821)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_definition'
        return stypy_return_type_25821


    @norecursion
    def test_random_even(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_random_even'
        module_type_store = module_type_store.open_function_context('test_random_even', 207, 4, False)
        # Assigning a type to the variable 'self' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTilbert.test_random_even.__dict__.__setitem__('stypy_localization', localization)
        TestTilbert.test_random_even.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTilbert.test_random_even.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTilbert.test_random_even.__dict__.__setitem__('stypy_function_name', 'TestTilbert.test_random_even')
        TestTilbert.test_random_even.__dict__.__setitem__('stypy_param_names_list', [])
        TestTilbert.test_random_even.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTilbert.test_random_even.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTilbert.test_random_even.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTilbert.test_random_even.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTilbert.test_random_even.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTilbert.test_random_even.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTilbert.test_random_even', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_random_even', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_random_even(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 208)
        list_25822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 208)
        # Adding element type (line 208)
        float_25823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 17), list_25822, float_25823)
        # Adding element type (line 208)
        float_25824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 17), list_25822, float_25824)
        # Adding element type (line 208)
        int_25825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 17), list_25822, int_25825)
        # Adding element type (line 208)
        float_25826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 17), list_25822, float_25826)
        # Adding element type (line 208)
        int_25827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 17), list_25822, int_25827)
        
        # Testing the type of a for loop iterable (line 208)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 208, 8), list_25822)
        # Getting the type of the for loop variable (line 208)
        for_loop_var_25828 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 208, 8), list_25822)
        # Assigning a type to the variable 'h' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'h', for_loop_var_25828)
        # SSA begins for a for statement (line 208)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'list' (line 209)
        list_25829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 209)
        # Adding element type (line 209)
        int_25830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 21), list_25829, int_25830)
        # Adding element type (line 209)
        int_25831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 21), list_25829, int_25831)
        # Adding element type (line 209)
        int_25832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 21), list_25829, int_25832)
        
        # Testing the type of a for loop iterable (line 209)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 209, 12), list_25829)
        # Getting the type of the for loop variable (line 209)
        for_loop_var_25833 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 209, 12), list_25829)
        # Assigning a type to the variable 'n' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'n', for_loop_var_25833)
        # SSA begins for a for statement (line 209)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 210):
        
        # Call to random(...): (line 210)
        # Processing the call arguments (line 210)
        
        # Obtaining an instance of the builtin type 'tuple' (line 210)
        tuple_25835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 210)
        # Adding element type (line 210)
        # Getting the type of 'n' (line 210)
        n_25836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 28), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 28), tuple_25835, n_25836)
        
        # Processing the call keyword arguments (line 210)
        kwargs_25837 = {}
        # Getting the type of 'random' (line 210)
        random_25834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 20), 'random', False)
        # Calling random(args, kwargs) (line 210)
        random_call_result_25838 = invoke(stypy.reporting.localization.Localization(__file__, 210, 20), random_25834, *[tuple_25835], **kwargs_25837)
        
        # Assigning a type to the variable 'f' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'f', random_call_result_25838)
        
        # Assigning a BinOp to a Name (line 211):
        
        # Call to sum(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'f' (line 211)
        f_25840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 25), 'f', False)
        # Processing the call keyword arguments (line 211)
        int_25841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 32), 'int')
        keyword_25842 = int_25841
        kwargs_25843 = {'axis': keyword_25842}
        # Getting the type of 'sum' (line 211)
        sum_25839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 21), 'sum', False)
        # Calling sum(args, kwargs) (line 211)
        sum_call_result_25844 = invoke(stypy.reporting.localization.Localization(__file__, 211, 21), sum_25839, *[f_25840], **kwargs_25843)
        
        # Getting the type of 'n' (line 211)
        n_25845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 35), 'n')
        # Applying the binary operator 'div' (line 211)
        result_div_25846 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 21), 'div', sum_call_result_25844, n_25845)
        
        # Assigning a type to the variable 'af' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'af', result_div_25846)
        
        # Assigning a BinOp to a Name (line 212):
        # Getting the type of 'f' (line 212)
        f_25847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'f')
        # Getting the type of 'af' (line 212)
        af_25848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 22), 'af')
        # Applying the binary operator '-' (line 212)
        result_sub_25849 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 20), '-', f_25847, af_25848)
        
        # Assigning a type to the variable 'f' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), 'f', result_sub_25849)
        
        # Call to assert_almost_equal(...): (line 213)
        # Processing the call arguments (line 213)
        
        # Call to sum(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'f' (line 213)
        f_25852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 40), 'f', False)
        # Processing the call keyword arguments (line 213)
        int_25853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 47), 'int')
        keyword_25854 = int_25853
        kwargs_25855 = {'axis': keyword_25854}
        # Getting the type of 'sum' (line 213)
        sum_25851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 36), 'sum', False)
        # Calling sum(args, kwargs) (line 213)
        sum_call_result_25856 = invoke(stypy.reporting.localization.Localization(__file__, 213, 36), sum_25851, *[f_25852], **kwargs_25855)
        
        float_25857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 50), 'float')
        # Processing the call keyword arguments (line 213)
        kwargs_25858 = {}
        # Getting the type of 'assert_almost_equal' (line 213)
        assert_almost_equal_25850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 213)
        assert_almost_equal_call_result_25859 = invoke(stypy.reporting.localization.Localization(__file__, 213, 16), assert_almost_equal_25850, *[sum_call_result_25856, float_25857], **kwargs_25858)
        
        
        # Call to assert_array_almost_equal(...): (line 214)
        # Processing the call arguments (line 214)
        
        # Call to direct_tilbert(...): (line 214)
        # Processing the call arguments (line 214)
        
        # Call to direct_itilbert(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'f' (line 214)
        f_25863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 73), 'f', False)
        # Getting the type of 'h' (line 214)
        h_25864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 75), 'h', False)
        # Processing the call keyword arguments (line 214)
        kwargs_25865 = {}
        # Getting the type of 'direct_itilbert' (line 214)
        direct_itilbert_25862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 57), 'direct_itilbert', False)
        # Calling direct_itilbert(args, kwargs) (line 214)
        direct_itilbert_call_result_25866 = invoke(stypy.reporting.localization.Localization(__file__, 214, 57), direct_itilbert_25862, *[f_25863, h_25864], **kwargs_25865)
        
        # Getting the type of 'h' (line 214)
        h_25867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 78), 'h', False)
        # Processing the call keyword arguments (line 214)
        kwargs_25868 = {}
        # Getting the type of 'direct_tilbert' (line 214)
        direct_tilbert_25861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 42), 'direct_tilbert', False)
        # Calling direct_tilbert(args, kwargs) (line 214)
        direct_tilbert_call_result_25869 = invoke(stypy.reporting.localization.Localization(__file__, 214, 42), direct_tilbert_25861, *[direct_itilbert_call_result_25866, h_25867], **kwargs_25868)
        
        # Getting the type of 'f' (line 214)
        f_25870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 81), 'f', False)
        # Processing the call keyword arguments (line 214)
        kwargs_25871 = {}
        # Getting the type of 'assert_array_almost_equal' (line 214)
        assert_array_almost_equal_25860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 214)
        assert_array_almost_equal_call_result_25872 = invoke(stypy.reporting.localization.Localization(__file__, 214, 16), assert_array_almost_equal_25860, *[direct_tilbert_call_result_25869, f_25870], **kwargs_25871)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_random_even(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_random_even' in the type store
        # Getting the type of 'stypy_return_type' (line 207)
        stypy_return_type_25873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25873)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_random_even'
        return stypy_return_type_25873


    @norecursion
    def test_random_odd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_random_odd'
        module_type_store = module_type_store.open_function_context('test_random_odd', 216, 4, False)
        # Assigning a type to the variable 'self' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTilbert.test_random_odd.__dict__.__setitem__('stypy_localization', localization)
        TestTilbert.test_random_odd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTilbert.test_random_odd.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTilbert.test_random_odd.__dict__.__setitem__('stypy_function_name', 'TestTilbert.test_random_odd')
        TestTilbert.test_random_odd.__dict__.__setitem__('stypy_param_names_list', [])
        TestTilbert.test_random_odd.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTilbert.test_random_odd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTilbert.test_random_odd.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTilbert.test_random_odd.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTilbert.test_random_odd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTilbert.test_random_odd.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTilbert.test_random_odd', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_random_odd', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_random_odd(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 217)
        list_25874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 217)
        # Adding element type (line 217)
        float_25875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 17), list_25874, float_25875)
        # Adding element type (line 217)
        float_25876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 17), list_25874, float_25876)
        # Adding element type (line 217)
        int_25877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 17), list_25874, int_25877)
        # Adding element type (line 217)
        float_25878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 17), list_25874, float_25878)
        # Adding element type (line 217)
        int_25879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 17), list_25874, int_25879)
        
        # Testing the type of a for loop iterable (line 217)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 217, 8), list_25874)
        # Getting the type of the for loop variable (line 217)
        for_loop_var_25880 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 217, 8), list_25874)
        # Assigning a type to the variable 'h' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'h', for_loop_var_25880)
        # SSA begins for a for statement (line 217)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'list' (line 218)
        list_25881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 218)
        # Adding element type (line 218)
        int_25882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 21), list_25881, int_25882)
        # Adding element type (line 218)
        int_25883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 21), list_25881, int_25883)
        # Adding element type (line 218)
        int_25884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 21), list_25881, int_25884)
        
        # Testing the type of a for loop iterable (line 218)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 218, 12), list_25881)
        # Getting the type of the for loop variable (line 218)
        for_loop_var_25885 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 218, 12), list_25881)
        # Assigning a type to the variable 'n' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'n', for_loop_var_25885)
        # SSA begins for a for statement (line 218)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 219):
        
        # Call to random(...): (line 219)
        # Processing the call arguments (line 219)
        
        # Obtaining an instance of the builtin type 'tuple' (line 219)
        tuple_25887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 219)
        # Adding element type (line 219)
        # Getting the type of 'n' (line 219)
        n_25888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 28), tuple_25887, n_25888)
        
        # Processing the call keyword arguments (line 219)
        kwargs_25889 = {}
        # Getting the type of 'random' (line 219)
        random_25886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 20), 'random', False)
        # Calling random(args, kwargs) (line 219)
        random_call_result_25890 = invoke(stypy.reporting.localization.Localization(__file__, 219, 20), random_25886, *[tuple_25887], **kwargs_25889)
        
        # Assigning a type to the variable 'f' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'f', random_call_result_25890)
        
        # Assigning a BinOp to a Name (line 220):
        
        # Call to sum(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'f' (line 220)
        f_25892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 25), 'f', False)
        # Processing the call keyword arguments (line 220)
        int_25893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 32), 'int')
        keyword_25894 = int_25893
        kwargs_25895 = {'axis': keyword_25894}
        # Getting the type of 'sum' (line 220)
        sum_25891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 21), 'sum', False)
        # Calling sum(args, kwargs) (line 220)
        sum_call_result_25896 = invoke(stypy.reporting.localization.Localization(__file__, 220, 21), sum_25891, *[f_25892], **kwargs_25895)
        
        # Getting the type of 'n' (line 220)
        n_25897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 35), 'n')
        # Applying the binary operator 'div' (line 220)
        result_div_25898 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 21), 'div', sum_call_result_25896, n_25897)
        
        # Assigning a type to the variable 'af' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'af', result_div_25898)
        
        # Assigning a BinOp to a Name (line 221):
        # Getting the type of 'f' (line 221)
        f_25899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'f')
        # Getting the type of 'af' (line 221)
        af_25900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 22), 'af')
        # Applying the binary operator '-' (line 221)
        result_sub_25901 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 20), '-', f_25899, af_25900)
        
        # Assigning a type to the variable 'f' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'f', result_sub_25901)
        
        # Call to assert_almost_equal(...): (line 222)
        # Processing the call arguments (line 222)
        
        # Call to sum(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'f' (line 222)
        f_25904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 40), 'f', False)
        # Processing the call keyword arguments (line 222)
        int_25905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 47), 'int')
        keyword_25906 = int_25905
        kwargs_25907 = {'axis': keyword_25906}
        # Getting the type of 'sum' (line 222)
        sum_25903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 36), 'sum', False)
        # Calling sum(args, kwargs) (line 222)
        sum_call_result_25908 = invoke(stypy.reporting.localization.Localization(__file__, 222, 36), sum_25903, *[f_25904], **kwargs_25907)
        
        float_25909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 50), 'float')
        # Processing the call keyword arguments (line 222)
        kwargs_25910 = {}
        # Getting the type of 'assert_almost_equal' (line 222)
        assert_almost_equal_25902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 222)
        assert_almost_equal_call_result_25911 = invoke(stypy.reporting.localization.Localization(__file__, 222, 16), assert_almost_equal_25902, *[sum_call_result_25908, float_25909], **kwargs_25910)
        
        
        # Call to assert_array_almost_equal(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Call to itilbert(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Call to tilbert(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'f' (line 223)
        f_25915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 59), 'f', False)
        # Getting the type of 'h' (line 223)
        h_25916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 61), 'h', False)
        # Processing the call keyword arguments (line 223)
        kwargs_25917 = {}
        # Getting the type of 'tilbert' (line 223)
        tilbert_25914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 51), 'tilbert', False)
        # Calling tilbert(args, kwargs) (line 223)
        tilbert_call_result_25918 = invoke(stypy.reporting.localization.Localization(__file__, 223, 51), tilbert_25914, *[f_25915, h_25916], **kwargs_25917)
        
        # Getting the type of 'h' (line 223)
        h_25919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 64), 'h', False)
        # Processing the call keyword arguments (line 223)
        kwargs_25920 = {}
        # Getting the type of 'itilbert' (line 223)
        itilbert_25913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 42), 'itilbert', False)
        # Calling itilbert(args, kwargs) (line 223)
        itilbert_call_result_25921 = invoke(stypy.reporting.localization.Localization(__file__, 223, 42), itilbert_25913, *[tilbert_call_result_25918, h_25919], **kwargs_25920)
        
        # Getting the type of 'f' (line 223)
        f_25922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 67), 'f', False)
        # Processing the call keyword arguments (line 223)
        kwargs_25923 = {}
        # Getting the type of 'assert_array_almost_equal' (line 223)
        assert_array_almost_equal_25912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 223)
        assert_array_almost_equal_call_result_25924 = invoke(stypy.reporting.localization.Localization(__file__, 223, 16), assert_array_almost_equal_25912, *[itilbert_call_result_25921, f_25922], **kwargs_25923)
        
        
        # Call to assert_array_almost_equal(...): (line 224)
        # Processing the call arguments (line 224)
        
        # Call to tilbert(...): (line 224)
        # Processing the call arguments (line 224)
        
        # Call to itilbert(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'f' (line 224)
        f_25928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 59), 'f', False)
        # Getting the type of 'h' (line 224)
        h_25929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 61), 'h', False)
        # Processing the call keyword arguments (line 224)
        kwargs_25930 = {}
        # Getting the type of 'itilbert' (line 224)
        itilbert_25927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 50), 'itilbert', False)
        # Calling itilbert(args, kwargs) (line 224)
        itilbert_call_result_25931 = invoke(stypy.reporting.localization.Localization(__file__, 224, 50), itilbert_25927, *[f_25928, h_25929], **kwargs_25930)
        
        # Getting the type of 'h' (line 224)
        h_25932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 64), 'h', False)
        # Processing the call keyword arguments (line 224)
        kwargs_25933 = {}
        # Getting the type of 'tilbert' (line 224)
        tilbert_25926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 42), 'tilbert', False)
        # Calling tilbert(args, kwargs) (line 224)
        tilbert_call_result_25934 = invoke(stypy.reporting.localization.Localization(__file__, 224, 42), tilbert_25926, *[itilbert_call_result_25931, h_25932], **kwargs_25933)
        
        # Getting the type of 'f' (line 224)
        f_25935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 67), 'f', False)
        # Processing the call keyword arguments (line 224)
        kwargs_25936 = {}
        # Getting the type of 'assert_array_almost_equal' (line 224)
        assert_array_almost_equal_25925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 224)
        assert_array_almost_equal_call_result_25937 = invoke(stypy.reporting.localization.Localization(__file__, 224, 16), assert_array_almost_equal_25925, *[tilbert_call_result_25934, f_25935], **kwargs_25936)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_random_odd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_random_odd' in the type store
        # Getting the type of 'stypy_return_type' (line 216)
        stypy_return_type_25938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_25938)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_random_odd'
        return stypy_return_type_25938


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 193, 0, False)
        # Assigning a type to the variable 'self' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTilbert.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestTilbert' (line 193)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'TestTilbert', TestTilbert)
# Declaration of the 'TestITilbert' class

class TestITilbert(object, ):

    @norecursion
    def test_definition(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_definition'
        module_type_store = module_type_store.open_function_context('test_definition', 229, 4, False)
        # Assigning a type to the variable 'self' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestITilbert.test_definition.__dict__.__setitem__('stypy_localization', localization)
        TestITilbert.test_definition.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestITilbert.test_definition.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestITilbert.test_definition.__dict__.__setitem__('stypy_function_name', 'TestITilbert.test_definition')
        TestITilbert.test_definition.__dict__.__setitem__('stypy_param_names_list', [])
        TestITilbert.test_definition.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestITilbert.test_definition.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestITilbert.test_definition.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestITilbert.test_definition.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestITilbert.test_definition.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestITilbert.test_definition.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestITilbert.test_definition', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_definition', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_definition(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 230)
        list_25939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 230)
        # Adding element type (line 230)
        float_25940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 18), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 17), list_25939, float_25940)
        # Adding element type (line 230)
        float_25941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 17), list_25939, float_25941)
        # Adding element type (line 230)
        int_25942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 17), list_25939, int_25942)
        # Adding element type (line 230)
        float_25943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 28), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 17), list_25939, float_25943)
        # Adding element type (line 230)
        int_25944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 17), list_25939, int_25944)
        
        # Testing the type of a for loop iterable (line 230)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 230, 8), list_25939)
        # Getting the type of the for loop variable (line 230)
        for_loop_var_25945 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 230, 8), list_25939)
        # Assigning a type to the variable 'h' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'h', for_loop_var_25945)
        # SSA begins for a for statement (line 230)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_25946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        # Adding element type (line 231)
        int_25947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 21), list_25946, int_25947)
        # Adding element type (line 231)
        int_25948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 21), list_25946, int_25948)
        # Adding element type (line 231)
        int_25949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 21), list_25946, int_25949)
        # Adding element type (line 231)
        int_25950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 21), list_25946, int_25950)
        
        # Testing the type of a for loop iterable (line 231)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 231, 12), list_25946)
        # Getting the type of the for loop variable (line 231)
        for_loop_var_25951 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 231, 12), list_25946)
        # Assigning a type to the variable 'n' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'n', for_loop_var_25951)
        # SSA begins for a for statement (line 231)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 232):
        
        # Call to arange(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'n' (line 232)
        n_25953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 27), 'n', False)
        # Processing the call keyword arguments (line 232)
        kwargs_25954 = {}
        # Getting the type of 'arange' (line 232)
        arange_25952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 20), 'arange', False)
        # Calling arange(args, kwargs) (line 232)
        arange_call_result_25955 = invoke(stypy.reporting.localization.Localization(__file__, 232, 20), arange_25952, *[n_25953], **kwargs_25954)
        
        int_25956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 30), 'int')
        # Applying the binary operator '*' (line 232)
        result_mul_25957 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 20), '*', arange_call_result_25955, int_25956)
        
        # Getting the type of 'pi' (line 232)
        pi_25958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 32), 'pi')
        # Applying the binary operator '*' (line 232)
        result_mul_25959 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 31), '*', result_mul_25957, pi_25958)
        
        # Getting the type of 'n' (line 232)
        n_25960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 35), 'n')
        # Applying the binary operator 'div' (line 232)
        result_div_25961 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 34), 'div', result_mul_25959, n_25960)
        
        # Assigning a type to the variable 'x' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), 'x', result_div_25961)
        
        # Assigning a Call to a Name (line 233):
        
        # Call to itilbert(...): (line 233)
        # Processing the call arguments (line 233)
        
        # Call to sin(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'x' (line 233)
        x_25964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 33), 'x', False)
        # Processing the call keyword arguments (line 233)
        kwargs_25965 = {}
        # Getting the type of 'sin' (line 233)
        sin_25963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 29), 'sin', False)
        # Calling sin(args, kwargs) (line 233)
        sin_call_result_25966 = invoke(stypy.reporting.localization.Localization(__file__, 233, 29), sin_25963, *[x_25964], **kwargs_25965)
        
        # Getting the type of 'h' (line 233)
        h_25967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 36), 'h', False)
        # Processing the call keyword arguments (line 233)
        kwargs_25968 = {}
        # Getting the type of 'itilbert' (line 233)
        itilbert_25962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 20), 'itilbert', False)
        # Calling itilbert(args, kwargs) (line 233)
        itilbert_call_result_25969 = invoke(stypy.reporting.localization.Localization(__file__, 233, 20), itilbert_25962, *[sin_call_result_25966, h_25967], **kwargs_25968)
        
        # Assigning a type to the variable 'y' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'y', itilbert_call_result_25969)
        
        # Assigning a Call to a Name (line 234):
        
        # Call to direct_itilbert(...): (line 234)
        # Processing the call arguments (line 234)
        
        # Call to sin(...): (line 234)
        # Processing the call arguments (line 234)
        # Getting the type of 'x' (line 234)
        x_25972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 41), 'x', False)
        # Processing the call keyword arguments (line 234)
        kwargs_25973 = {}
        # Getting the type of 'sin' (line 234)
        sin_25971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 37), 'sin', False)
        # Calling sin(args, kwargs) (line 234)
        sin_call_result_25974 = invoke(stypy.reporting.localization.Localization(__file__, 234, 37), sin_25971, *[x_25972], **kwargs_25973)
        
        # Getting the type of 'h' (line 234)
        h_25975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 44), 'h', False)
        # Processing the call keyword arguments (line 234)
        kwargs_25976 = {}
        # Getting the type of 'direct_itilbert' (line 234)
        direct_itilbert_25970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 21), 'direct_itilbert', False)
        # Calling direct_itilbert(args, kwargs) (line 234)
        direct_itilbert_call_result_25977 = invoke(stypy.reporting.localization.Localization(__file__, 234, 21), direct_itilbert_25970, *[sin_call_result_25974, h_25975], **kwargs_25976)
        
        # Assigning a type to the variable 'y1' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'y1', direct_itilbert_call_result_25977)
        
        # Call to assert_array_almost_equal(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'y' (line 235)
        y_25979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 42), 'y', False)
        # Getting the type of 'y1' (line 235)
        y1_25980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 44), 'y1', False)
        # Processing the call keyword arguments (line 235)
        kwargs_25981 = {}
        # Getting the type of 'assert_array_almost_equal' (line 235)
        assert_array_almost_equal_25978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 235)
        assert_array_almost_equal_call_result_25982 = invoke(stypy.reporting.localization.Localization(__file__, 235, 16), assert_array_almost_equal_25978, *[y_25979, y1_25980], **kwargs_25981)
        
        
        # Call to assert_array_almost_equal(...): (line 236)
        # Processing the call arguments (line 236)
        
        # Call to itilbert(...): (line 236)
        # Processing the call arguments (line 236)
        
        # Call to sin(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'x' (line 236)
        x_25986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 55), 'x', False)
        # Processing the call keyword arguments (line 236)
        kwargs_25987 = {}
        # Getting the type of 'sin' (line 236)
        sin_25985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 51), 'sin', False)
        # Calling sin(args, kwargs) (line 236)
        sin_call_result_25988 = invoke(stypy.reporting.localization.Localization(__file__, 236, 51), sin_25985, *[x_25986], **kwargs_25987)
        
        # Getting the type of 'h' (line 236)
        h_25989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 58), 'h', False)
        # Processing the call keyword arguments (line 236)
        kwargs_25990 = {}
        # Getting the type of 'itilbert' (line 236)
        itilbert_25984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 42), 'itilbert', False)
        # Calling itilbert(args, kwargs) (line 236)
        itilbert_call_result_25991 = invoke(stypy.reporting.localization.Localization(__file__, 236, 42), itilbert_25984, *[sin_call_result_25988, h_25989], **kwargs_25990)
        
        
        # Call to direct_itilbert(...): (line 237)
        # Processing the call arguments (line 237)
        
        # Call to sin(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'x' (line 237)
        x_25994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 62), 'x', False)
        # Processing the call keyword arguments (line 237)
        kwargs_25995 = {}
        # Getting the type of 'sin' (line 237)
        sin_25993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 58), 'sin', False)
        # Calling sin(args, kwargs) (line 237)
        sin_call_result_25996 = invoke(stypy.reporting.localization.Localization(__file__, 237, 58), sin_25993, *[x_25994], **kwargs_25995)
        
        # Getting the type of 'h' (line 237)
        h_25997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 65), 'h', False)
        # Processing the call keyword arguments (line 237)
        kwargs_25998 = {}
        # Getting the type of 'direct_itilbert' (line 237)
        direct_itilbert_25992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 42), 'direct_itilbert', False)
        # Calling direct_itilbert(args, kwargs) (line 237)
        direct_itilbert_call_result_25999 = invoke(stypy.reporting.localization.Localization(__file__, 237, 42), direct_itilbert_25992, *[sin_call_result_25996, h_25997], **kwargs_25998)
        
        # Processing the call keyword arguments (line 236)
        kwargs_26000 = {}
        # Getting the type of 'assert_array_almost_equal' (line 236)
        assert_array_almost_equal_25983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 236)
        assert_array_almost_equal_call_result_26001 = invoke(stypy.reporting.localization.Localization(__file__, 236, 16), assert_array_almost_equal_25983, *[itilbert_call_result_25991, direct_itilbert_call_result_25999], **kwargs_26000)
        
        
        # Call to assert_array_almost_equal(...): (line 238)
        # Processing the call arguments (line 238)
        
        # Call to itilbert(...): (line 238)
        # Processing the call arguments (line 238)
        
        # Call to sin(...): (line 238)
        # Processing the call arguments (line 238)
        int_26005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 55), 'int')
        # Getting the type of 'x' (line 238)
        x_26006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 57), 'x', False)
        # Applying the binary operator '*' (line 238)
        result_mul_26007 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 55), '*', int_26005, x_26006)
        
        # Processing the call keyword arguments (line 238)
        kwargs_26008 = {}
        # Getting the type of 'sin' (line 238)
        sin_26004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 51), 'sin', False)
        # Calling sin(args, kwargs) (line 238)
        sin_call_result_26009 = invoke(stypy.reporting.localization.Localization(__file__, 238, 51), sin_26004, *[result_mul_26007], **kwargs_26008)
        
        # Getting the type of 'h' (line 238)
        h_26010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 60), 'h', False)
        # Processing the call keyword arguments (line 238)
        kwargs_26011 = {}
        # Getting the type of 'itilbert' (line 238)
        itilbert_26003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 42), 'itilbert', False)
        # Calling itilbert(args, kwargs) (line 238)
        itilbert_call_result_26012 = invoke(stypy.reporting.localization.Localization(__file__, 238, 42), itilbert_26003, *[sin_call_result_26009, h_26010], **kwargs_26011)
        
        
        # Call to direct_itilbert(...): (line 239)
        # Processing the call arguments (line 239)
        
        # Call to sin(...): (line 239)
        # Processing the call arguments (line 239)
        int_26015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 62), 'int')
        # Getting the type of 'x' (line 239)
        x_26016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 64), 'x', False)
        # Applying the binary operator '*' (line 239)
        result_mul_26017 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 62), '*', int_26015, x_26016)
        
        # Processing the call keyword arguments (line 239)
        kwargs_26018 = {}
        # Getting the type of 'sin' (line 239)
        sin_26014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 58), 'sin', False)
        # Calling sin(args, kwargs) (line 239)
        sin_call_result_26019 = invoke(stypy.reporting.localization.Localization(__file__, 239, 58), sin_26014, *[result_mul_26017], **kwargs_26018)
        
        # Getting the type of 'h' (line 239)
        h_26020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 67), 'h', False)
        # Processing the call keyword arguments (line 239)
        kwargs_26021 = {}
        # Getting the type of 'direct_itilbert' (line 239)
        direct_itilbert_26013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 42), 'direct_itilbert', False)
        # Calling direct_itilbert(args, kwargs) (line 239)
        direct_itilbert_call_result_26022 = invoke(stypy.reporting.localization.Localization(__file__, 239, 42), direct_itilbert_26013, *[sin_call_result_26019, h_26020], **kwargs_26021)
        
        # Processing the call keyword arguments (line 238)
        kwargs_26023 = {}
        # Getting the type of 'assert_array_almost_equal' (line 238)
        assert_array_almost_equal_26002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 238)
        assert_array_almost_equal_call_result_26024 = invoke(stypy.reporting.localization.Localization(__file__, 238, 16), assert_array_almost_equal_26002, *[itilbert_call_result_26012, direct_itilbert_call_result_26022], **kwargs_26023)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_definition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_definition' in the type store
        # Getting the type of 'stypy_return_type' (line 229)
        stypy_return_type_26025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26025)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_definition'
        return stypy_return_type_26025


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 227, 0, False)
        # Assigning a type to the variable 'self' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestITilbert.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestITilbert' (line 227)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 0), 'TestITilbert', TestITilbert)
# Declaration of the 'TestHilbert' class

class TestHilbert(object, ):

    @norecursion
    def test_definition(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_definition'
        module_type_store = module_type_store.open_function_context('test_definition', 244, 4, False)
        # Assigning a type to the variable 'self' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHilbert.test_definition.__dict__.__setitem__('stypy_localization', localization)
        TestHilbert.test_definition.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHilbert.test_definition.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHilbert.test_definition.__dict__.__setitem__('stypy_function_name', 'TestHilbert.test_definition')
        TestHilbert.test_definition.__dict__.__setitem__('stypy_param_names_list', [])
        TestHilbert.test_definition.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHilbert.test_definition.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHilbert.test_definition.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHilbert.test_definition.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHilbert.test_definition.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHilbert.test_definition.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHilbert.test_definition', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_definition', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_definition(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 245)
        list_26026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 245)
        # Adding element type (line 245)
        int_26027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 17), list_26026, int_26027)
        # Adding element type (line 245)
        int_26028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 17), list_26026, int_26028)
        # Adding element type (line 245)
        int_26029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 17), list_26026, int_26029)
        # Adding element type (line 245)
        int_26030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 17), list_26026, int_26030)
        
        # Testing the type of a for loop iterable (line 245)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 245, 8), list_26026)
        # Getting the type of the for loop variable (line 245)
        for_loop_var_26031 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 245, 8), list_26026)
        # Assigning a type to the variable 'n' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'n', for_loop_var_26031)
        # SSA begins for a for statement (line 245)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 246):
        
        # Call to arange(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'n' (line 246)
        n_26033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 23), 'n', False)
        # Processing the call keyword arguments (line 246)
        kwargs_26034 = {}
        # Getting the type of 'arange' (line 246)
        arange_26032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'arange', False)
        # Calling arange(args, kwargs) (line 246)
        arange_call_result_26035 = invoke(stypy.reporting.localization.Localization(__file__, 246, 16), arange_26032, *[n_26033], **kwargs_26034)
        
        int_26036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 26), 'int')
        # Applying the binary operator '*' (line 246)
        result_mul_26037 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 16), '*', arange_call_result_26035, int_26036)
        
        # Getting the type of 'pi' (line 246)
        pi_26038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 28), 'pi')
        # Applying the binary operator '*' (line 246)
        result_mul_26039 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 27), '*', result_mul_26037, pi_26038)
        
        # Getting the type of 'n' (line 246)
        n_26040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 31), 'n')
        # Applying the binary operator 'div' (line 246)
        result_div_26041 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 30), 'div', result_mul_26039, n_26040)
        
        # Assigning a type to the variable 'x' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'x', result_div_26041)
        
        # Assigning a Call to a Name (line 247):
        
        # Call to hilbert(...): (line 247)
        # Processing the call arguments (line 247)
        
        # Call to sin(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'x' (line 247)
        x_26044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 28), 'x', False)
        # Processing the call keyword arguments (line 247)
        kwargs_26045 = {}
        # Getting the type of 'sin' (line 247)
        sin_26043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 24), 'sin', False)
        # Calling sin(args, kwargs) (line 247)
        sin_call_result_26046 = invoke(stypy.reporting.localization.Localization(__file__, 247, 24), sin_26043, *[x_26044], **kwargs_26045)
        
        # Processing the call keyword arguments (line 247)
        kwargs_26047 = {}
        # Getting the type of 'hilbert' (line 247)
        hilbert_26042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'hilbert', False)
        # Calling hilbert(args, kwargs) (line 247)
        hilbert_call_result_26048 = invoke(stypy.reporting.localization.Localization(__file__, 247, 16), hilbert_26042, *[sin_call_result_26046], **kwargs_26047)
        
        # Assigning a type to the variable 'y' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'y', hilbert_call_result_26048)
        
        # Assigning a Call to a Name (line 248):
        
        # Call to direct_hilbert(...): (line 248)
        # Processing the call arguments (line 248)
        
        # Call to sin(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'x' (line 248)
        x_26051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 36), 'x', False)
        # Processing the call keyword arguments (line 248)
        kwargs_26052 = {}
        # Getting the type of 'sin' (line 248)
        sin_26050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 32), 'sin', False)
        # Calling sin(args, kwargs) (line 248)
        sin_call_result_26053 = invoke(stypy.reporting.localization.Localization(__file__, 248, 32), sin_26050, *[x_26051], **kwargs_26052)
        
        # Processing the call keyword arguments (line 248)
        kwargs_26054 = {}
        # Getting the type of 'direct_hilbert' (line 248)
        direct_hilbert_26049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 17), 'direct_hilbert', False)
        # Calling direct_hilbert(args, kwargs) (line 248)
        direct_hilbert_call_result_26055 = invoke(stypy.reporting.localization.Localization(__file__, 248, 17), direct_hilbert_26049, *[sin_call_result_26053], **kwargs_26054)
        
        # Assigning a type to the variable 'y1' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'y1', direct_hilbert_call_result_26055)
        
        # Call to assert_array_almost_equal(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'y' (line 249)
        y_26057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 38), 'y', False)
        # Getting the type of 'y1' (line 249)
        y1_26058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 40), 'y1', False)
        # Processing the call keyword arguments (line 249)
        kwargs_26059 = {}
        # Getting the type of 'assert_array_almost_equal' (line 249)
        assert_array_almost_equal_26056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 249)
        assert_array_almost_equal_call_result_26060 = invoke(stypy.reporting.localization.Localization(__file__, 249, 12), assert_array_almost_equal_26056, *[y_26057, y1_26058], **kwargs_26059)
        
        
        # Call to assert_array_almost_equal(...): (line 250)
        # Processing the call arguments (line 250)
        
        # Call to hilbert(...): (line 250)
        # Processing the call arguments (line 250)
        
        # Call to sin(...): (line 250)
        # Processing the call arguments (line 250)
        int_26064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 50), 'int')
        # Getting the type of 'x' (line 250)
        x_26065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 52), 'x', False)
        # Applying the binary operator '*' (line 250)
        result_mul_26066 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 50), '*', int_26064, x_26065)
        
        # Processing the call keyword arguments (line 250)
        kwargs_26067 = {}
        # Getting the type of 'sin' (line 250)
        sin_26063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 46), 'sin', False)
        # Calling sin(args, kwargs) (line 250)
        sin_call_result_26068 = invoke(stypy.reporting.localization.Localization(__file__, 250, 46), sin_26063, *[result_mul_26066], **kwargs_26067)
        
        # Processing the call keyword arguments (line 250)
        kwargs_26069 = {}
        # Getting the type of 'hilbert' (line 250)
        hilbert_26062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 38), 'hilbert', False)
        # Calling hilbert(args, kwargs) (line 250)
        hilbert_call_result_26070 = invoke(stypy.reporting.localization.Localization(__file__, 250, 38), hilbert_26062, *[sin_call_result_26068], **kwargs_26069)
        
        
        # Call to direct_hilbert(...): (line 251)
        # Processing the call arguments (line 251)
        
        # Call to sin(...): (line 251)
        # Processing the call arguments (line 251)
        int_26073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 57), 'int')
        # Getting the type of 'x' (line 251)
        x_26074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 59), 'x', False)
        # Applying the binary operator '*' (line 251)
        result_mul_26075 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 57), '*', int_26073, x_26074)
        
        # Processing the call keyword arguments (line 251)
        kwargs_26076 = {}
        # Getting the type of 'sin' (line 251)
        sin_26072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 53), 'sin', False)
        # Calling sin(args, kwargs) (line 251)
        sin_call_result_26077 = invoke(stypy.reporting.localization.Localization(__file__, 251, 53), sin_26072, *[result_mul_26075], **kwargs_26076)
        
        # Processing the call keyword arguments (line 251)
        kwargs_26078 = {}
        # Getting the type of 'direct_hilbert' (line 251)
        direct_hilbert_26071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 38), 'direct_hilbert', False)
        # Calling direct_hilbert(args, kwargs) (line 251)
        direct_hilbert_call_result_26079 = invoke(stypy.reporting.localization.Localization(__file__, 251, 38), direct_hilbert_26071, *[sin_call_result_26077], **kwargs_26078)
        
        # Processing the call keyword arguments (line 250)
        kwargs_26080 = {}
        # Getting the type of 'assert_array_almost_equal' (line 250)
        assert_array_almost_equal_26061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 250)
        assert_array_almost_equal_call_result_26081 = invoke(stypy.reporting.localization.Localization(__file__, 250, 12), assert_array_almost_equal_26061, *[hilbert_call_result_26070, direct_hilbert_call_result_26079], **kwargs_26080)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_definition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_definition' in the type store
        # Getting the type of 'stypy_return_type' (line 244)
        stypy_return_type_26082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26082)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_definition'
        return stypy_return_type_26082


    @norecursion
    def test_tilbert_relation(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tilbert_relation'
        module_type_store = module_type_store.open_function_context('test_tilbert_relation', 253, 4, False)
        # Assigning a type to the variable 'self' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHilbert.test_tilbert_relation.__dict__.__setitem__('stypy_localization', localization)
        TestHilbert.test_tilbert_relation.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHilbert.test_tilbert_relation.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHilbert.test_tilbert_relation.__dict__.__setitem__('stypy_function_name', 'TestHilbert.test_tilbert_relation')
        TestHilbert.test_tilbert_relation.__dict__.__setitem__('stypy_param_names_list', [])
        TestHilbert.test_tilbert_relation.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHilbert.test_tilbert_relation.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHilbert.test_tilbert_relation.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHilbert.test_tilbert_relation.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHilbert.test_tilbert_relation.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHilbert.test_tilbert_relation.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHilbert.test_tilbert_relation', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tilbert_relation', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tilbert_relation(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 254)
        list_26083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 254)
        # Adding element type (line 254)
        int_26084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 17), list_26083, int_26084)
        # Adding element type (line 254)
        int_26085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 17), list_26083, int_26085)
        # Adding element type (line 254)
        int_26086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 17), list_26083, int_26086)
        # Adding element type (line 254)
        int_26087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 17), list_26083, int_26087)
        
        # Testing the type of a for loop iterable (line 254)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 254, 8), list_26083)
        # Getting the type of the for loop variable (line 254)
        for_loop_var_26088 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 254, 8), list_26083)
        # Assigning a type to the variable 'n' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'n', for_loop_var_26088)
        # SSA begins for a for statement (line 254)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 255):
        
        # Call to arange(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'n' (line 255)
        n_26090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 23), 'n', False)
        # Processing the call keyword arguments (line 255)
        kwargs_26091 = {}
        # Getting the type of 'arange' (line 255)
        arange_26089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'arange', False)
        # Calling arange(args, kwargs) (line 255)
        arange_call_result_26092 = invoke(stypy.reporting.localization.Localization(__file__, 255, 16), arange_26089, *[n_26090], **kwargs_26091)
        
        int_26093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 26), 'int')
        # Applying the binary operator '*' (line 255)
        result_mul_26094 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 16), '*', arange_call_result_26092, int_26093)
        
        # Getting the type of 'pi' (line 255)
        pi_26095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 28), 'pi')
        # Applying the binary operator '*' (line 255)
        result_mul_26096 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 27), '*', result_mul_26094, pi_26095)
        
        # Getting the type of 'n' (line 255)
        n_26097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 31), 'n')
        # Applying the binary operator 'div' (line 255)
        result_div_26098 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 30), 'div', result_mul_26096, n_26097)
        
        # Assigning a type to the variable 'x' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'x', result_div_26098)
        
        # Assigning a BinOp to a Name (line 256):
        
        # Call to sin(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'x' (line 256)
        x_26100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 20), 'x', False)
        # Processing the call keyword arguments (line 256)
        kwargs_26101 = {}
        # Getting the type of 'sin' (line 256)
        sin_26099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 16), 'sin', False)
        # Calling sin(args, kwargs) (line 256)
        sin_call_result_26102 = invoke(stypy.reporting.localization.Localization(__file__, 256, 16), sin_26099, *[x_26100], **kwargs_26101)
        
        
        # Call to cos(...): (line 256)
        # Processing the call arguments (line 256)
        int_26104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 27), 'int')
        # Getting the type of 'x' (line 256)
        x_26105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 29), 'x', False)
        # Applying the binary operator '*' (line 256)
        result_mul_26106 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 27), '*', int_26104, x_26105)
        
        # Processing the call keyword arguments (line 256)
        kwargs_26107 = {}
        # Getting the type of 'cos' (line 256)
        cos_26103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 23), 'cos', False)
        # Calling cos(args, kwargs) (line 256)
        cos_call_result_26108 = invoke(stypy.reporting.localization.Localization(__file__, 256, 23), cos_26103, *[result_mul_26106], **kwargs_26107)
        
        
        # Call to sin(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'x' (line 256)
        x_26110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 36), 'x', False)
        # Processing the call keyword arguments (line 256)
        kwargs_26111 = {}
        # Getting the type of 'sin' (line 256)
        sin_26109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 32), 'sin', False)
        # Calling sin(args, kwargs) (line 256)
        sin_call_result_26112 = invoke(stypy.reporting.localization.Localization(__file__, 256, 32), sin_26109, *[x_26110], **kwargs_26111)
        
        # Applying the binary operator '*' (line 256)
        result_mul_26113 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 23), '*', cos_call_result_26108, sin_call_result_26112)
        
        # Applying the binary operator '+' (line 256)
        result_add_26114 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 16), '+', sin_call_result_26102, result_mul_26113)
        
        # Assigning a type to the variable 'f' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'f', result_add_26114)
        
        # Assigning a Call to a Name (line 257):
        
        # Call to hilbert(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'f' (line 257)
        f_26116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 24), 'f', False)
        # Processing the call keyword arguments (line 257)
        kwargs_26117 = {}
        # Getting the type of 'hilbert' (line 257)
        hilbert_26115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 16), 'hilbert', False)
        # Calling hilbert(args, kwargs) (line 257)
        hilbert_call_result_26118 = invoke(stypy.reporting.localization.Localization(__file__, 257, 16), hilbert_26115, *[f_26116], **kwargs_26117)
        
        # Assigning a type to the variable 'y' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'y', hilbert_call_result_26118)
        
        # Assigning a Call to a Name (line 258):
        
        # Call to direct_hilbert(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'f' (line 258)
        f_26120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 32), 'f', False)
        # Processing the call keyword arguments (line 258)
        kwargs_26121 = {}
        # Getting the type of 'direct_hilbert' (line 258)
        direct_hilbert_26119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 17), 'direct_hilbert', False)
        # Calling direct_hilbert(args, kwargs) (line 258)
        direct_hilbert_call_result_26122 = invoke(stypy.reporting.localization.Localization(__file__, 258, 17), direct_hilbert_26119, *[f_26120], **kwargs_26121)
        
        # Assigning a type to the variable 'y1' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'y1', direct_hilbert_call_result_26122)
        
        # Call to assert_array_almost_equal(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'y' (line 259)
        y_26124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 38), 'y', False)
        # Getting the type of 'y1' (line 259)
        y1_26125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 40), 'y1', False)
        # Processing the call keyword arguments (line 259)
        kwargs_26126 = {}
        # Getting the type of 'assert_array_almost_equal' (line 259)
        assert_array_almost_equal_26123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 259)
        assert_array_almost_equal_call_result_26127 = invoke(stypy.reporting.localization.Localization(__file__, 259, 12), assert_array_almost_equal_26123, *[y_26124, y1_26125], **kwargs_26126)
        
        
        # Assigning a Call to a Name (line 260):
        
        # Call to tilbert(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'f' (line 260)
        f_26129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 25), 'f', False)
        # Processing the call keyword arguments (line 260)
        int_26130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 29), 'int')
        keyword_26131 = int_26130
        kwargs_26132 = {'h': keyword_26131}
        # Getting the type of 'tilbert' (line 260)
        tilbert_26128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 17), 'tilbert', False)
        # Calling tilbert(args, kwargs) (line 260)
        tilbert_call_result_26133 = invoke(stypy.reporting.localization.Localization(__file__, 260, 17), tilbert_26128, *[f_26129], **kwargs_26132)
        
        # Assigning a type to the variable 'y2' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'y2', tilbert_call_result_26133)
        
        # Call to assert_array_almost_equal(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'y' (line 261)
        y_26135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 38), 'y', False)
        # Getting the type of 'y2' (line 261)
        y2_26136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 40), 'y2', False)
        # Processing the call keyword arguments (line 261)
        kwargs_26137 = {}
        # Getting the type of 'assert_array_almost_equal' (line 261)
        assert_array_almost_equal_26134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 261)
        assert_array_almost_equal_call_result_26138 = invoke(stypy.reporting.localization.Localization(__file__, 261, 12), assert_array_almost_equal_26134, *[y_26135, y2_26136], **kwargs_26137)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_tilbert_relation(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tilbert_relation' in the type store
        # Getting the type of 'stypy_return_type' (line 253)
        stypy_return_type_26139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26139)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tilbert_relation'
        return stypy_return_type_26139


    @norecursion
    def test_random_odd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_random_odd'
        module_type_store = module_type_store.open_function_context('test_random_odd', 263, 4, False)
        # Assigning a type to the variable 'self' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHilbert.test_random_odd.__dict__.__setitem__('stypy_localization', localization)
        TestHilbert.test_random_odd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHilbert.test_random_odd.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHilbert.test_random_odd.__dict__.__setitem__('stypy_function_name', 'TestHilbert.test_random_odd')
        TestHilbert.test_random_odd.__dict__.__setitem__('stypy_param_names_list', [])
        TestHilbert.test_random_odd.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHilbert.test_random_odd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHilbert.test_random_odd.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHilbert.test_random_odd.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHilbert.test_random_odd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHilbert.test_random_odd.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHilbert.test_random_odd', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_random_odd', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_random_odd(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 264)
        list_26140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 264)
        # Adding element type (line 264)
        int_26141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 17), list_26140, int_26141)
        # Adding element type (line 264)
        int_26142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 17), list_26140, int_26142)
        # Adding element type (line 264)
        int_26143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 17), list_26140, int_26143)
        
        # Testing the type of a for loop iterable (line 264)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 264, 8), list_26140)
        # Getting the type of the for loop variable (line 264)
        for_loop_var_26144 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 264, 8), list_26140)
        # Assigning a type to the variable 'n' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'n', for_loop_var_26144)
        # SSA begins for a for statement (line 264)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 265):
        
        # Call to random(...): (line 265)
        # Processing the call arguments (line 265)
        
        # Obtaining an instance of the builtin type 'tuple' (line 265)
        tuple_26146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 265)
        # Adding element type (line 265)
        # Getting the type of 'n' (line 265)
        n_26147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 24), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 24), tuple_26146, n_26147)
        
        # Processing the call keyword arguments (line 265)
        kwargs_26148 = {}
        # Getting the type of 'random' (line 265)
        random_26145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 16), 'random', False)
        # Calling random(args, kwargs) (line 265)
        random_call_result_26149 = invoke(stypy.reporting.localization.Localization(__file__, 265, 16), random_26145, *[tuple_26146], **kwargs_26148)
        
        # Assigning a type to the variable 'f' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'f', random_call_result_26149)
        
        # Assigning a BinOp to a Name (line 266):
        
        # Call to sum(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'f' (line 266)
        f_26151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 21), 'f', False)
        # Processing the call keyword arguments (line 266)
        int_26152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 28), 'int')
        keyword_26153 = int_26152
        kwargs_26154 = {'axis': keyword_26153}
        # Getting the type of 'sum' (line 266)
        sum_26150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 17), 'sum', False)
        # Calling sum(args, kwargs) (line 266)
        sum_call_result_26155 = invoke(stypy.reporting.localization.Localization(__file__, 266, 17), sum_26150, *[f_26151], **kwargs_26154)
        
        # Getting the type of 'n' (line 266)
        n_26156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 31), 'n')
        # Applying the binary operator 'div' (line 266)
        result_div_26157 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 17), 'div', sum_call_result_26155, n_26156)
        
        # Assigning a type to the variable 'af' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'af', result_div_26157)
        
        # Assigning a BinOp to a Name (line 267):
        # Getting the type of 'f' (line 267)
        f_26158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 16), 'f')
        # Getting the type of 'af' (line 267)
        af_26159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 18), 'af')
        # Applying the binary operator '-' (line 267)
        result_sub_26160 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 16), '-', f_26158, af_26159)
        
        # Assigning a type to the variable 'f' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'f', result_sub_26160)
        
        # Call to assert_almost_equal(...): (line 268)
        # Processing the call arguments (line 268)
        
        # Call to sum(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 'f' (line 268)
        f_26163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 36), 'f', False)
        # Processing the call keyword arguments (line 268)
        int_26164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 43), 'int')
        keyword_26165 = int_26164
        kwargs_26166 = {'axis': keyword_26165}
        # Getting the type of 'sum' (line 268)
        sum_26162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 32), 'sum', False)
        # Calling sum(args, kwargs) (line 268)
        sum_call_result_26167 = invoke(stypy.reporting.localization.Localization(__file__, 268, 32), sum_26162, *[f_26163], **kwargs_26166)
        
        float_26168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 46), 'float')
        # Processing the call keyword arguments (line 268)
        kwargs_26169 = {}
        # Getting the type of 'assert_almost_equal' (line 268)
        assert_almost_equal_26161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 268)
        assert_almost_equal_call_result_26170 = invoke(stypy.reporting.localization.Localization(__file__, 268, 12), assert_almost_equal_26161, *[sum_call_result_26167, float_26168], **kwargs_26169)
        
        
        # Call to assert_array_almost_equal(...): (line 269)
        # Processing the call arguments (line 269)
        
        # Call to ihilbert(...): (line 269)
        # Processing the call arguments (line 269)
        
        # Call to hilbert(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'f' (line 269)
        f_26174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 55), 'f', False)
        # Processing the call keyword arguments (line 269)
        kwargs_26175 = {}
        # Getting the type of 'hilbert' (line 269)
        hilbert_26173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 47), 'hilbert', False)
        # Calling hilbert(args, kwargs) (line 269)
        hilbert_call_result_26176 = invoke(stypy.reporting.localization.Localization(__file__, 269, 47), hilbert_26173, *[f_26174], **kwargs_26175)
        
        # Processing the call keyword arguments (line 269)
        kwargs_26177 = {}
        # Getting the type of 'ihilbert' (line 269)
        ihilbert_26172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 38), 'ihilbert', False)
        # Calling ihilbert(args, kwargs) (line 269)
        ihilbert_call_result_26178 = invoke(stypy.reporting.localization.Localization(__file__, 269, 38), ihilbert_26172, *[hilbert_call_result_26176], **kwargs_26177)
        
        # Getting the type of 'f' (line 269)
        f_26179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 59), 'f', False)
        # Processing the call keyword arguments (line 269)
        kwargs_26180 = {}
        # Getting the type of 'assert_array_almost_equal' (line 269)
        assert_array_almost_equal_26171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 269)
        assert_array_almost_equal_call_result_26181 = invoke(stypy.reporting.localization.Localization(__file__, 269, 12), assert_array_almost_equal_26171, *[ihilbert_call_result_26178, f_26179], **kwargs_26180)
        
        
        # Call to assert_array_almost_equal(...): (line 270)
        # Processing the call arguments (line 270)
        
        # Call to hilbert(...): (line 270)
        # Processing the call arguments (line 270)
        
        # Call to ihilbert(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'f' (line 270)
        f_26185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 55), 'f', False)
        # Processing the call keyword arguments (line 270)
        kwargs_26186 = {}
        # Getting the type of 'ihilbert' (line 270)
        ihilbert_26184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 46), 'ihilbert', False)
        # Calling ihilbert(args, kwargs) (line 270)
        ihilbert_call_result_26187 = invoke(stypy.reporting.localization.Localization(__file__, 270, 46), ihilbert_26184, *[f_26185], **kwargs_26186)
        
        # Processing the call keyword arguments (line 270)
        kwargs_26188 = {}
        # Getting the type of 'hilbert' (line 270)
        hilbert_26183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 38), 'hilbert', False)
        # Calling hilbert(args, kwargs) (line 270)
        hilbert_call_result_26189 = invoke(stypy.reporting.localization.Localization(__file__, 270, 38), hilbert_26183, *[ihilbert_call_result_26187], **kwargs_26188)
        
        # Getting the type of 'f' (line 270)
        f_26190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 59), 'f', False)
        # Processing the call keyword arguments (line 270)
        kwargs_26191 = {}
        # Getting the type of 'assert_array_almost_equal' (line 270)
        assert_array_almost_equal_26182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 270)
        assert_array_almost_equal_call_result_26192 = invoke(stypy.reporting.localization.Localization(__file__, 270, 12), assert_array_almost_equal_26182, *[hilbert_call_result_26189, f_26190], **kwargs_26191)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_random_odd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_random_odd' in the type store
        # Getting the type of 'stypy_return_type' (line 263)
        stypy_return_type_26193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26193)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_random_odd'
        return stypy_return_type_26193


    @norecursion
    def test_random_even(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_random_even'
        module_type_store = module_type_store.open_function_context('test_random_even', 272, 4, False)
        # Assigning a type to the variable 'self' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestHilbert.test_random_even.__dict__.__setitem__('stypy_localization', localization)
        TestHilbert.test_random_even.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestHilbert.test_random_even.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestHilbert.test_random_even.__dict__.__setitem__('stypy_function_name', 'TestHilbert.test_random_even')
        TestHilbert.test_random_even.__dict__.__setitem__('stypy_param_names_list', [])
        TestHilbert.test_random_even.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestHilbert.test_random_even.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestHilbert.test_random_even.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestHilbert.test_random_even.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestHilbert.test_random_even.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestHilbert.test_random_even.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHilbert.test_random_even', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_random_even', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_random_even(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 273)
        list_26194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 273)
        # Adding element type (line 273)
        int_26195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 17), list_26194, int_26195)
        # Adding element type (line 273)
        int_26196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 17), list_26194, int_26196)
        # Adding element type (line 273)
        int_26197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 273, 17), list_26194, int_26197)
        
        # Testing the type of a for loop iterable (line 273)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 273, 8), list_26194)
        # Getting the type of the for loop variable (line 273)
        for_loop_var_26198 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 273, 8), list_26194)
        # Assigning a type to the variable 'n' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'n', for_loop_var_26198)
        # SSA begins for a for statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 274):
        
        # Call to random(...): (line 274)
        # Processing the call arguments (line 274)
        
        # Obtaining an instance of the builtin type 'tuple' (line 274)
        tuple_26200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 274)
        # Adding element type (line 274)
        # Getting the type of 'n' (line 274)
        n_26201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 24), 'n', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 274, 24), tuple_26200, n_26201)
        
        # Processing the call keyword arguments (line 274)
        kwargs_26202 = {}
        # Getting the type of 'random' (line 274)
        random_26199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'random', False)
        # Calling random(args, kwargs) (line 274)
        random_call_result_26203 = invoke(stypy.reporting.localization.Localization(__file__, 274, 16), random_26199, *[tuple_26200], **kwargs_26202)
        
        # Assigning a type to the variable 'f' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'f', random_call_result_26203)
        
        # Assigning a BinOp to a Name (line 275):
        
        # Call to sum(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'f' (line 275)
        f_26205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 21), 'f', False)
        # Processing the call keyword arguments (line 275)
        int_26206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 28), 'int')
        keyword_26207 = int_26206
        kwargs_26208 = {'axis': keyword_26207}
        # Getting the type of 'sum' (line 275)
        sum_26204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 17), 'sum', False)
        # Calling sum(args, kwargs) (line 275)
        sum_call_result_26209 = invoke(stypy.reporting.localization.Localization(__file__, 275, 17), sum_26204, *[f_26205], **kwargs_26208)
        
        # Getting the type of 'n' (line 275)
        n_26210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 31), 'n')
        # Applying the binary operator 'div' (line 275)
        result_div_26211 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 17), 'div', sum_call_result_26209, n_26210)
        
        # Assigning a type to the variable 'af' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'af', result_div_26211)
        
        # Assigning a BinOp to a Name (line 276):
        # Getting the type of 'f' (line 276)
        f_26212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'f')
        # Getting the type of 'af' (line 276)
        af_26213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 18), 'af')
        # Applying the binary operator '-' (line 276)
        result_sub_26214 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 16), '-', f_26212, af_26213)
        
        # Assigning a type to the variable 'f' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'f', result_sub_26214)
        
        # Assigning a Call to a Name (line 278):
        
        # Call to diff(...): (line 278)
        # Processing the call arguments (line 278)
        
        # Call to diff(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'f' (line 278)
        f_26217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 26), 'f', False)
        int_26218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 28), 'int')
        # Processing the call keyword arguments (line 278)
        kwargs_26219 = {}
        # Getting the type of 'diff' (line 278)
        diff_26216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), 'diff', False)
        # Calling diff(args, kwargs) (line 278)
        diff_call_result_26220 = invoke(stypy.reporting.localization.Localization(__file__, 278, 21), diff_26216, *[f_26217, int_26218], **kwargs_26219)
        
        int_26221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 31), 'int')
        # Processing the call keyword arguments (line 278)
        kwargs_26222 = {}
        # Getting the type of 'diff' (line 278)
        diff_26215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'diff', False)
        # Calling diff(args, kwargs) (line 278)
        diff_call_result_26223 = invoke(stypy.reporting.localization.Localization(__file__, 278, 16), diff_26215, *[diff_call_result_26220, int_26221], **kwargs_26222)
        
        # Assigning a type to the variable 'f' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'f', diff_call_result_26223)
        
        # Call to assert_almost_equal(...): (line 279)
        # Processing the call arguments (line 279)
        
        # Call to sum(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'f' (line 279)
        f_26226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 36), 'f', False)
        # Processing the call keyword arguments (line 279)
        int_26227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 43), 'int')
        keyword_26228 = int_26227
        kwargs_26229 = {'axis': keyword_26228}
        # Getting the type of 'sum' (line 279)
        sum_26225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 32), 'sum', False)
        # Calling sum(args, kwargs) (line 279)
        sum_call_result_26230 = invoke(stypy.reporting.localization.Localization(__file__, 279, 32), sum_26225, *[f_26226], **kwargs_26229)
        
        float_26231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 46), 'float')
        # Processing the call keyword arguments (line 279)
        kwargs_26232 = {}
        # Getting the type of 'assert_almost_equal' (line 279)
        assert_almost_equal_26224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'assert_almost_equal', False)
        # Calling assert_almost_equal(args, kwargs) (line 279)
        assert_almost_equal_call_result_26233 = invoke(stypy.reporting.localization.Localization(__file__, 279, 12), assert_almost_equal_26224, *[sum_call_result_26230, float_26231], **kwargs_26232)
        
        
        # Call to assert_array_almost_equal(...): (line 280)
        # Processing the call arguments (line 280)
        
        # Call to direct_hilbert(...): (line 280)
        # Processing the call arguments (line 280)
        
        # Call to direct_ihilbert(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'f' (line 280)
        f_26237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 69), 'f', False)
        # Processing the call keyword arguments (line 280)
        kwargs_26238 = {}
        # Getting the type of 'direct_ihilbert' (line 280)
        direct_ihilbert_26236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 53), 'direct_ihilbert', False)
        # Calling direct_ihilbert(args, kwargs) (line 280)
        direct_ihilbert_call_result_26239 = invoke(stypy.reporting.localization.Localization(__file__, 280, 53), direct_ihilbert_26236, *[f_26237], **kwargs_26238)
        
        # Processing the call keyword arguments (line 280)
        kwargs_26240 = {}
        # Getting the type of 'direct_hilbert' (line 280)
        direct_hilbert_26235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 38), 'direct_hilbert', False)
        # Calling direct_hilbert(args, kwargs) (line 280)
        direct_hilbert_call_result_26241 = invoke(stypy.reporting.localization.Localization(__file__, 280, 38), direct_hilbert_26235, *[direct_ihilbert_call_result_26239], **kwargs_26240)
        
        # Getting the type of 'f' (line 280)
        f_26242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 73), 'f', False)
        # Processing the call keyword arguments (line 280)
        kwargs_26243 = {}
        # Getting the type of 'assert_array_almost_equal' (line 280)
        assert_array_almost_equal_26234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 280)
        assert_array_almost_equal_call_result_26244 = invoke(stypy.reporting.localization.Localization(__file__, 280, 12), assert_array_almost_equal_26234, *[direct_hilbert_call_result_26241, f_26242], **kwargs_26243)
        
        
        # Call to assert_array_almost_equal(...): (line 281)
        # Processing the call arguments (line 281)
        
        # Call to hilbert(...): (line 281)
        # Processing the call arguments (line 281)
        
        # Call to ihilbert(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'f' (line 281)
        f_26248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 55), 'f', False)
        # Processing the call keyword arguments (line 281)
        kwargs_26249 = {}
        # Getting the type of 'ihilbert' (line 281)
        ihilbert_26247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 46), 'ihilbert', False)
        # Calling ihilbert(args, kwargs) (line 281)
        ihilbert_call_result_26250 = invoke(stypy.reporting.localization.Localization(__file__, 281, 46), ihilbert_26247, *[f_26248], **kwargs_26249)
        
        # Processing the call keyword arguments (line 281)
        kwargs_26251 = {}
        # Getting the type of 'hilbert' (line 281)
        hilbert_26246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 38), 'hilbert', False)
        # Calling hilbert(args, kwargs) (line 281)
        hilbert_call_result_26252 = invoke(stypy.reporting.localization.Localization(__file__, 281, 38), hilbert_26246, *[ihilbert_call_result_26250], **kwargs_26251)
        
        # Getting the type of 'f' (line 281)
        f_26253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 59), 'f', False)
        # Processing the call keyword arguments (line 281)
        kwargs_26254 = {}
        # Getting the type of 'assert_array_almost_equal' (line 281)
        assert_array_almost_equal_26245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 281)
        assert_array_almost_equal_call_result_26255 = invoke(stypy.reporting.localization.Localization(__file__, 281, 12), assert_array_almost_equal_26245, *[hilbert_call_result_26252, f_26253], **kwargs_26254)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_random_even(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_random_even' in the type store
        # Getting the type of 'stypy_return_type' (line 272)
        stypy_return_type_26256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26256)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_random_even'
        return stypy_return_type_26256


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 242, 0, False)
        # Assigning a type to the variable 'self' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestHilbert.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestHilbert' (line 242)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 0), 'TestHilbert', TestHilbert)
# Declaration of the 'TestIHilbert' class

class TestIHilbert(object, ):

    @norecursion
    def test_definition(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_definition'
        module_type_store = module_type_store.open_function_context('test_definition', 286, 4, False)
        # Assigning a type to the variable 'self' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIHilbert.test_definition.__dict__.__setitem__('stypy_localization', localization)
        TestIHilbert.test_definition.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIHilbert.test_definition.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIHilbert.test_definition.__dict__.__setitem__('stypy_function_name', 'TestIHilbert.test_definition')
        TestIHilbert.test_definition.__dict__.__setitem__('stypy_param_names_list', [])
        TestIHilbert.test_definition.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIHilbert.test_definition.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIHilbert.test_definition.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIHilbert.test_definition.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIHilbert.test_definition.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIHilbert.test_definition.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIHilbert.test_definition', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_definition', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_definition(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 287)
        list_26257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 287)
        # Adding element type (line 287)
        int_26258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 17), list_26257, int_26258)
        # Adding element type (line 287)
        int_26259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 17), list_26257, int_26259)
        # Adding element type (line 287)
        int_26260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 17), list_26257, int_26260)
        # Adding element type (line 287)
        int_26261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 287, 17), list_26257, int_26261)
        
        # Testing the type of a for loop iterable (line 287)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 287, 8), list_26257)
        # Getting the type of the for loop variable (line 287)
        for_loop_var_26262 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 287, 8), list_26257)
        # Assigning a type to the variable 'n' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'n', for_loop_var_26262)
        # SSA begins for a for statement (line 287)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 288):
        
        # Call to arange(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'n' (line 288)
        n_26264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 23), 'n', False)
        # Processing the call keyword arguments (line 288)
        kwargs_26265 = {}
        # Getting the type of 'arange' (line 288)
        arange_26263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 16), 'arange', False)
        # Calling arange(args, kwargs) (line 288)
        arange_call_result_26266 = invoke(stypy.reporting.localization.Localization(__file__, 288, 16), arange_26263, *[n_26264], **kwargs_26265)
        
        int_26267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 26), 'int')
        # Applying the binary operator '*' (line 288)
        result_mul_26268 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 16), '*', arange_call_result_26266, int_26267)
        
        # Getting the type of 'pi' (line 288)
        pi_26269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 28), 'pi')
        # Applying the binary operator '*' (line 288)
        result_mul_26270 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 27), '*', result_mul_26268, pi_26269)
        
        # Getting the type of 'n' (line 288)
        n_26271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 31), 'n')
        # Applying the binary operator 'div' (line 288)
        result_div_26272 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 30), 'div', result_mul_26270, n_26271)
        
        # Assigning a type to the variable 'x' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'x', result_div_26272)
        
        # Assigning a Call to a Name (line 289):
        
        # Call to ihilbert(...): (line 289)
        # Processing the call arguments (line 289)
        
        # Call to sin(...): (line 289)
        # Processing the call arguments (line 289)
        # Getting the type of 'x' (line 289)
        x_26275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 29), 'x', False)
        # Processing the call keyword arguments (line 289)
        kwargs_26276 = {}
        # Getting the type of 'sin' (line 289)
        sin_26274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 25), 'sin', False)
        # Calling sin(args, kwargs) (line 289)
        sin_call_result_26277 = invoke(stypy.reporting.localization.Localization(__file__, 289, 25), sin_26274, *[x_26275], **kwargs_26276)
        
        # Processing the call keyword arguments (line 289)
        kwargs_26278 = {}
        # Getting the type of 'ihilbert' (line 289)
        ihilbert_26273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'ihilbert', False)
        # Calling ihilbert(args, kwargs) (line 289)
        ihilbert_call_result_26279 = invoke(stypy.reporting.localization.Localization(__file__, 289, 16), ihilbert_26273, *[sin_call_result_26277], **kwargs_26278)
        
        # Assigning a type to the variable 'y' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'y', ihilbert_call_result_26279)
        
        # Assigning a Call to a Name (line 290):
        
        # Call to direct_ihilbert(...): (line 290)
        # Processing the call arguments (line 290)
        
        # Call to sin(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'x' (line 290)
        x_26282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 37), 'x', False)
        # Processing the call keyword arguments (line 290)
        kwargs_26283 = {}
        # Getting the type of 'sin' (line 290)
        sin_26281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 33), 'sin', False)
        # Calling sin(args, kwargs) (line 290)
        sin_call_result_26284 = invoke(stypy.reporting.localization.Localization(__file__, 290, 33), sin_26281, *[x_26282], **kwargs_26283)
        
        # Processing the call keyword arguments (line 290)
        kwargs_26285 = {}
        # Getting the type of 'direct_ihilbert' (line 290)
        direct_ihilbert_26280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 17), 'direct_ihilbert', False)
        # Calling direct_ihilbert(args, kwargs) (line 290)
        direct_ihilbert_call_result_26286 = invoke(stypy.reporting.localization.Localization(__file__, 290, 17), direct_ihilbert_26280, *[sin_call_result_26284], **kwargs_26285)
        
        # Assigning a type to the variable 'y1' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'y1', direct_ihilbert_call_result_26286)
        
        # Call to assert_array_almost_equal(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'y' (line 291)
        y_26288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 38), 'y', False)
        # Getting the type of 'y1' (line 291)
        y1_26289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 40), 'y1', False)
        # Processing the call keyword arguments (line 291)
        kwargs_26290 = {}
        # Getting the type of 'assert_array_almost_equal' (line 291)
        assert_array_almost_equal_26287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 291)
        assert_array_almost_equal_call_result_26291 = invoke(stypy.reporting.localization.Localization(__file__, 291, 12), assert_array_almost_equal_26287, *[y_26288, y1_26289], **kwargs_26290)
        
        
        # Call to assert_array_almost_equal(...): (line 292)
        # Processing the call arguments (line 292)
        
        # Call to ihilbert(...): (line 292)
        # Processing the call arguments (line 292)
        
        # Call to sin(...): (line 292)
        # Processing the call arguments (line 292)
        int_26295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 51), 'int')
        # Getting the type of 'x' (line 292)
        x_26296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 53), 'x', False)
        # Applying the binary operator '*' (line 292)
        result_mul_26297 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 51), '*', int_26295, x_26296)
        
        # Processing the call keyword arguments (line 292)
        kwargs_26298 = {}
        # Getting the type of 'sin' (line 292)
        sin_26294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 47), 'sin', False)
        # Calling sin(args, kwargs) (line 292)
        sin_call_result_26299 = invoke(stypy.reporting.localization.Localization(__file__, 292, 47), sin_26294, *[result_mul_26297], **kwargs_26298)
        
        # Processing the call keyword arguments (line 292)
        kwargs_26300 = {}
        # Getting the type of 'ihilbert' (line 292)
        ihilbert_26293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 38), 'ihilbert', False)
        # Calling ihilbert(args, kwargs) (line 292)
        ihilbert_call_result_26301 = invoke(stypy.reporting.localization.Localization(__file__, 292, 38), ihilbert_26293, *[sin_call_result_26299], **kwargs_26300)
        
        
        # Call to direct_ihilbert(...): (line 293)
        # Processing the call arguments (line 293)
        
        # Call to sin(...): (line 293)
        # Processing the call arguments (line 293)
        int_26304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 58), 'int')
        # Getting the type of 'x' (line 293)
        x_26305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 60), 'x', False)
        # Applying the binary operator '*' (line 293)
        result_mul_26306 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 58), '*', int_26304, x_26305)
        
        # Processing the call keyword arguments (line 293)
        kwargs_26307 = {}
        # Getting the type of 'sin' (line 293)
        sin_26303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 54), 'sin', False)
        # Calling sin(args, kwargs) (line 293)
        sin_call_result_26308 = invoke(stypy.reporting.localization.Localization(__file__, 293, 54), sin_26303, *[result_mul_26306], **kwargs_26307)
        
        # Processing the call keyword arguments (line 293)
        kwargs_26309 = {}
        # Getting the type of 'direct_ihilbert' (line 293)
        direct_ihilbert_26302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 38), 'direct_ihilbert', False)
        # Calling direct_ihilbert(args, kwargs) (line 293)
        direct_ihilbert_call_result_26310 = invoke(stypy.reporting.localization.Localization(__file__, 293, 38), direct_ihilbert_26302, *[sin_call_result_26308], **kwargs_26309)
        
        # Processing the call keyword arguments (line 292)
        kwargs_26311 = {}
        # Getting the type of 'assert_array_almost_equal' (line 292)
        assert_array_almost_equal_26292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 292)
        assert_array_almost_equal_call_result_26312 = invoke(stypy.reporting.localization.Localization(__file__, 292, 12), assert_array_almost_equal_26292, *[ihilbert_call_result_26301, direct_ihilbert_call_result_26310], **kwargs_26311)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_definition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_definition' in the type store
        # Getting the type of 'stypy_return_type' (line 286)
        stypy_return_type_26313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26313)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_definition'
        return stypy_return_type_26313


    @norecursion
    def test_itilbert_relation(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_itilbert_relation'
        module_type_store = module_type_store.open_function_context('test_itilbert_relation', 295, 4, False)
        # Assigning a type to the variable 'self' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIHilbert.test_itilbert_relation.__dict__.__setitem__('stypy_localization', localization)
        TestIHilbert.test_itilbert_relation.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIHilbert.test_itilbert_relation.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIHilbert.test_itilbert_relation.__dict__.__setitem__('stypy_function_name', 'TestIHilbert.test_itilbert_relation')
        TestIHilbert.test_itilbert_relation.__dict__.__setitem__('stypy_param_names_list', [])
        TestIHilbert.test_itilbert_relation.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIHilbert.test_itilbert_relation.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIHilbert.test_itilbert_relation.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIHilbert.test_itilbert_relation.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIHilbert.test_itilbert_relation.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIHilbert.test_itilbert_relation.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIHilbert.test_itilbert_relation', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_itilbert_relation', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_itilbert_relation(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 296)
        list_26314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 296)
        # Adding element type (line 296)
        int_26315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 17), list_26314, int_26315)
        # Adding element type (line 296)
        int_26316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 17), list_26314, int_26316)
        # Adding element type (line 296)
        int_26317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 17), list_26314, int_26317)
        # Adding element type (line 296)
        int_26318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 17), list_26314, int_26318)
        
        # Testing the type of a for loop iterable (line 296)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 296, 8), list_26314)
        # Getting the type of the for loop variable (line 296)
        for_loop_var_26319 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 296, 8), list_26314)
        # Assigning a type to the variable 'n' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'n', for_loop_var_26319)
        # SSA begins for a for statement (line 296)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 297):
        
        # Call to arange(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'n' (line 297)
        n_26321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 23), 'n', False)
        # Processing the call keyword arguments (line 297)
        kwargs_26322 = {}
        # Getting the type of 'arange' (line 297)
        arange_26320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'arange', False)
        # Calling arange(args, kwargs) (line 297)
        arange_call_result_26323 = invoke(stypy.reporting.localization.Localization(__file__, 297, 16), arange_26320, *[n_26321], **kwargs_26322)
        
        int_26324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 26), 'int')
        # Applying the binary operator '*' (line 297)
        result_mul_26325 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 16), '*', arange_call_result_26323, int_26324)
        
        # Getting the type of 'pi' (line 297)
        pi_26326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 28), 'pi')
        # Applying the binary operator '*' (line 297)
        result_mul_26327 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 27), '*', result_mul_26325, pi_26326)
        
        # Getting the type of 'n' (line 297)
        n_26328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 31), 'n')
        # Applying the binary operator 'div' (line 297)
        result_div_26329 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 30), 'div', result_mul_26327, n_26328)
        
        # Assigning a type to the variable 'x' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'x', result_div_26329)
        
        # Assigning a BinOp to a Name (line 298):
        
        # Call to sin(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'x' (line 298)
        x_26331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 20), 'x', False)
        # Processing the call keyword arguments (line 298)
        kwargs_26332 = {}
        # Getting the type of 'sin' (line 298)
        sin_26330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'sin', False)
        # Calling sin(args, kwargs) (line 298)
        sin_call_result_26333 = invoke(stypy.reporting.localization.Localization(__file__, 298, 16), sin_26330, *[x_26331], **kwargs_26332)
        
        
        # Call to cos(...): (line 298)
        # Processing the call arguments (line 298)
        int_26335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 27), 'int')
        # Getting the type of 'x' (line 298)
        x_26336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 29), 'x', False)
        # Applying the binary operator '*' (line 298)
        result_mul_26337 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 27), '*', int_26335, x_26336)
        
        # Processing the call keyword arguments (line 298)
        kwargs_26338 = {}
        # Getting the type of 'cos' (line 298)
        cos_26334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 23), 'cos', False)
        # Calling cos(args, kwargs) (line 298)
        cos_call_result_26339 = invoke(stypy.reporting.localization.Localization(__file__, 298, 23), cos_26334, *[result_mul_26337], **kwargs_26338)
        
        
        # Call to sin(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'x' (line 298)
        x_26341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 36), 'x', False)
        # Processing the call keyword arguments (line 298)
        kwargs_26342 = {}
        # Getting the type of 'sin' (line 298)
        sin_26340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 32), 'sin', False)
        # Calling sin(args, kwargs) (line 298)
        sin_call_result_26343 = invoke(stypy.reporting.localization.Localization(__file__, 298, 32), sin_26340, *[x_26341], **kwargs_26342)
        
        # Applying the binary operator '*' (line 298)
        result_mul_26344 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 23), '*', cos_call_result_26339, sin_call_result_26343)
        
        # Applying the binary operator '+' (line 298)
        result_add_26345 = python_operator(stypy.reporting.localization.Localization(__file__, 298, 16), '+', sin_call_result_26333, result_mul_26344)
        
        # Assigning a type to the variable 'f' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 12), 'f', result_add_26345)
        
        # Assigning a Call to a Name (line 299):
        
        # Call to ihilbert(...): (line 299)
        # Processing the call arguments (line 299)
        # Getting the type of 'f' (line 299)
        f_26347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 25), 'f', False)
        # Processing the call keyword arguments (line 299)
        kwargs_26348 = {}
        # Getting the type of 'ihilbert' (line 299)
        ihilbert_26346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 'ihilbert', False)
        # Calling ihilbert(args, kwargs) (line 299)
        ihilbert_call_result_26349 = invoke(stypy.reporting.localization.Localization(__file__, 299, 16), ihilbert_26346, *[f_26347], **kwargs_26348)
        
        # Assigning a type to the variable 'y' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'y', ihilbert_call_result_26349)
        
        # Assigning a Call to a Name (line 300):
        
        # Call to direct_ihilbert(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'f' (line 300)
        f_26351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 33), 'f', False)
        # Processing the call keyword arguments (line 300)
        kwargs_26352 = {}
        # Getting the type of 'direct_ihilbert' (line 300)
        direct_ihilbert_26350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 17), 'direct_ihilbert', False)
        # Calling direct_ihilbert(args, kwargs) (line 300)
        direct_ihilbert_call_result_26353 = invoke(stypy.reporting.localization.Localization(__file__, 300, 17), direct_ihilbert_26350, *[f_26351], **kwargs_26352)
        
        # Assigning a type to the variable 'y1' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'y1', direct_ihilbert_call_result_26353)
        
        # Call to assert_array_almost_equal(...): (line 301)
        # Processing the call arguments (line 301)
        # Getting the type of 'y' (line 301)
        y_26355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 38), 'y', False)
        # Getting the type of 'y1' (line 301)
        y1_26356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 40), 'y1', False)
        # Processing the call keyword arguments (line 301)
        kwargs_26357 = {}
        # Getting the type of 'assert_array_almost_equal' (line 301)
        assert_array_almost_equal_26354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 301)
        assert_array_almost_equal_call_result_26358 = invoke(stypy.reporting.localization.Localization(__file__, 301, 12), assert_array_almost_equal_26354, *[y_26355, y1_26356], **kwargs_26357)
        
        
        # Assigning a Call to a Name (line 302):
        
        # Call to itilbert(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'f' (line 302)
        f_26360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 26), 'f', False)
        # Processing the call keyword arguments (line 302)
        int_26361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 30), 'int')
        keyword_26362 = int_26361
        kwargs_26363 = {'h': keyword_26362}
        # Getting the type of 'itilbert' (line 302)
        itilbert_26359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 17), 'itilbert', False)
        # Calling itilbert(args, kwargs) (line 302)
        itilbert_call_result_26364 = invoke(stypy.reporting.localization.Localization(__file__, 302, 17), itilbert_26359, *[f_26360], **kwargs_26363)
        
        # Assigning a type to the variable 'y2' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'y2', itilbert_call_result_26364)
        
        # Call to assert_array_almost_equal(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'y' (line 303)
        y_26366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 38), 'y', False)
        # Getting the type of 'y2' (line 303)
        y2_26367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 40), 'y2', False)
        # Processing the call keyword arguments (line 303)
        kwargs_26368 = {}
        # Getting the type of 'assert_array_almost_equal' (line 303)
        assert_array_almost_equal_26365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 303)
        assert_array_almost_equal_call_result_26369 = invoke(stypy.reporting.localization.Localization(__file__, 303, 12), assert_array_almost_equal_26365, *[y_26366, y2_26367], **kwargs_26368)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_itilbert_relation(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_itilbert_relation' in the type store
        # Getting the type of 'stypy_return_type' (line 295)
        stypy_return_type_26370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26370)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_itilbert_relation'
        return stypy_return_type_26370


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 284, 0, False)
        # Assigning a type to the variable 'self' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIHilbert.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIHilbert' (line 284)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 0), 'TestIHilbert', TestIHilbert)
# Declaration of the 'TestShift' class

class TestShift(object, ):

    @norecursion
    def test_definition(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_definition'
        module_type_store = module_type_store.open_function_context('test_definition', 308, 4, False)
        # Assigning a type to the variable 'self' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestShift.test_definition.__dict__.__setitem__('stypy_localization', localization)
        TestShift.test_definition.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestShift.test_definition.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestShift.test_definition.__dict__.__setitem__('stypy_function_name', 'TestShift.test_definition')
        TestShift.test_definition.__dict__.__setitem__('stypy_param_names_list', [])
        TestShift.test_definition.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestShift.test_definition.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestShift.test_definition.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestShift.test_definition.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestShift.test_definition.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestShift.test_definition.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestShift.test_definition', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_definition', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_definition(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 309)
        list_26371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 309)
        # Adding element type (line 309)
        int_26372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 17), list_26371, int_26372)
        # Adding element type (line 309)
        int_26373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 17), list_26371, int_26373)
        # Adding element type (line 309)
        int_26374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 17), list_26371, int_26374)
        # Adding element type (line 309)
        int_26375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 17), list_26371, int_26375)
        # Adding element type (line 309)
        int_26376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 17), list_26371, int_26376)
        # Adding element type (line 309)
        int_26377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 17), list_26371, int_26377)
        # Adding element type (line 309)
        int_26378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 17), list_26371, int_26378)
        
        # Testing the type of a for loop iterable (line 309)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 309, 8), list_26371)
        # Getting the type of the for loop variable (line 309)
        for_loop_var_26379 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 309, 8), list_26371)
        # Assigning a type to the variable 'n' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'n', for_loop_var_26379)
        # SSA begins for a for statement (line 309)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 310):
        
        # Call to arange(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'n' (line 310)
        n_26381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 23), 'n', False)
        # Processing the call keyword arguments (line 310)
        kwargs_26382 = {}
        # Getting the type of 'arange' (line 310)
        arange_26380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 16), 'arange', False)
        # Calling arange(args, kwargs) (line 310)
        arange_call_result_26383 = invoke(stypy.reporting.localization.Localization(__file__, 310, 16), arange_26380, *[n_26381], **kwargs_26382)
        
        int_26384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 26), 'int')
        # Applying the binary operator '*' (line 310)
        result_mul_26385 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 16), '*', arange_call_result_26383, int_26384)
        
        # Getting the type of 'pi' (line 310)
        pi_26386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 28), 'pi')
        # Applying the binary operator '*' (line 310)
        result_mul_26387 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 27), '*', result_mul_26385, pi_26386)
        
        # Getting the type of 'n' (line 310)
        n_26388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 31), 'n')
        # Applying the binary operator 'div' (line 310)
        result_div_26389 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 30), 'div', result_mul_26387, n_26388)
        
        # Assigning a type to the variable 'x' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 12), 'x', result_div_26389)
        
        
        # Obtaining an instance of the builtin type 'list' (line 311)
        list_26390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 311)
        # Adding element type (line 311)
        float_26391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 21), list_26390, float_26391)
        # Adding element type (line 311)
        int_26392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 21), list_26390, int_26392)
        
        # Testing the type of a for loop iterable (line 311)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 311, 12), list_26390)
        # Getting the type of the for loop variable (line 311)
        for_loop_var_26393 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 311, 12), list_26390)
        # Assigning a type to the variable 'a' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'a', for_loop_var_26393)
        # SSA begins for a for statement (line 311)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_array_almost_equal(...): (line 312)
        # Processing the call arguments (line 312)
        
        # Call to shift(...): (line 312)
        # Processing the call arguments (line 312)
        
        # Call to sin(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'x' (line 312)
        x_26397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 52), 'x', False)
        # Processing the call keyword arguments (line 312)
        kwargs_26398 = {}
        # Getting the type of 'sin' (line 312)
        sin_26396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 48), 'sin', False)
        # Calling sin(args, kwargs) (line 312)
        sin_call_result_26399 = invoke(stypy.reporting.localization.Localization(__file__, 312, 48), sin_26396, *[x_26397], **kwargs_26398)
        
        # Getting the type of 'a' (line 312)
        a_26400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 55), 'a', False)
        # Processing the call keyword arguments (line 312)
        kwargs_26401 = {}
        # Getting the type of 'shift' (line 312)
        shift_26395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 42), 'shift', False)
        # Calling shift(args, kwargs) (line 312)
        shift_call_result_26402 = invoke(stypy.reporting.localization.Localization(__file__, 312, 42), shift_26395, *[sin_call_result_26399, a_26400], **kwargs_26401)
        
        
        # Call to direct_shift(...): (line 312)
        # Processing the call arguments (line 312)
        
        # Call to sin(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'x' (line 312)
        x_26405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 75), 'x', False)
        # Processing the call keyword arguments (line 312)
        kwargs_26406 = {}
        # Getting the type of 'sin' (line 312)
        sin_26404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 71), 'sin', False)
        # Calling sin(args, kwargs) (line 312)
        sin_call_result_26407 = invoke(stypy.reporting.localization.Localization(__file__, 312, 71), sin_26404, *[x_26405], **kwargs_26406)
        
        # Getting the type of 'a' (line 312)
        a_26408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 78), 'a', False)
        # Processing the call keyword arguments (line 312)
        kwargs_26409 = {}
        # Getting the type of 'direct_shift' (line 312)
        direct_shift_26403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 58), 'direct_shift', False)
        # Calling direct_shift(args, kwargs) (line 312)
        direct_shift_call_result_26410 = invoke(stypy.reporting.localization.Localization(__file__, 312, 58), direct_shift_26403, *[sin_call_result_26407, a_26408], **kwargs_26409)
        
        # Processing the call keyword arguments (line 312)
        kwargs_26411 = {}
        # Getting the type of 'assert_array_almost_equal' (line 312)
        assert_array_almost_equal_26394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 312)
        assert_array_almost_equal_call_result_26412 = invoke(stypy.reporting.localization.Localization(__file__, 312, 16), assert_array_almost_equal_26394, *[shift_call_result_26402, direct_shift_call_result_26410], **kwargs_26411)
        
        
        # Call to assert_array_almost_equal(...): (line 313)
        # Processing the call arguments (line 313)
        
        # Call to shift(...): (line 313)
        # Processing the call arguments (line 313)
        
        # Call to sin(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 'x' (line 313)
        x_26416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 52), 'x', False)
        # Processing the call keyword arguments (line 313)
        kwargs_26417 = {}
        # Getting the type of 'sin' (line 313)
        sin_26415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 48), 'sin', False)
        # Calling sin(args, kwargs) (line 313)
        sin_call_result_26418 = invoke(stypy.reporting.localization.Localization(__file__, 313, 48), sin_26415, *[x_26416], **kwargs_26417)
        
        # Getting the type of 'a' (line 313)
        a_26419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 55), 'a', False)
        # Processing the call keyword arguments (line 313)
        kwargs_26420 = {}
        # Getting the type of 'shift' (line 313)
        shift_26414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 42), 'shift', False)
        # Calling shift(args, kwargs) (line 313)
        shift_call_result_26421 = invoke(stypy.reporting.localization.Localization(__file__, 313, 42), shift_26414, *[sin_call_result_26418, a_26419], **kwargs_26420)
        
        
        # Call to sin(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 'x' (line 313)
        x_26423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 62), 'x', False)
        # Getting the type of 'a' (line 313)
        a_26424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 64), 'a', False)
        # Applying the binary operator '+' (line 313)
        result_add_26425 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 62), '+', x_26423, a_26424)
        
        # Processing the call keyword arguments (line 313)
        kwargs_26426 = {}
        # Getting the type of 'sin' (line 313)
        sin_26422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 58), 'sin', False)
        # Calling sin(args, kwargs) (line 313)
        sin_call_result_26427 = invoke(stypy.reporting.localization.Localization(__file__, 313, 58), sin_26422, *[result_add_26425], **kwargs_26426)
        
        # Processing the call keyword arguments (line 313)
        kwargs_26428 = {}
        # Getting the type of 'assert_array_almost_equal' (line 313)
        assert_array_almost_equal_26413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 313)
        assert_array_almost_equal_call_result_26429 = invoke(stypy.reporting.localization.Localization(__file__, 313, 16), assert_array_almost_equal_26413, *[shift_call_result_26421, sin_call_result_26427], **kwargs_26428)
        
        
        # Call to assert_array_almost_equal(...): (line 314)
        # Processing the call arguments (line 314)
        
        # Call to shift(...): (line 314)
        # Processing the call arguments (line 314)
        
        # Call to cos(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'x' (line 314)
        x_26433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 52), 'x', False)
        # Processing the call keyword arguments (line 314)
        kwargs_26434 = {}
        # Getting the type of 'cos' (line 314)
        cos_26432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 48), 'cos', False)
        # Calling cos(args, kwargs) (line 314)
        cos_call_result_26435 = invoke(stypy.reporting.localization.Localization(__file__, 314, 48), cos_26432, *[x_26433], **kwargs_26434)
        
        # Getting the type of 'a' (line 314)
        a_26436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 55), 'a', False)
        # Processing the call keyword arguments (line 314)
        kwargs_26437 = {}
        # Getting the type of 'shift' (line 314)
        shift_26431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 42), 'shift', False)
        # Calling shift(args, kwargs) (line 314)
        shift_call_result_26438 = invoke(stypy.reporting.localization.Localization(__file__, 314, 42), shift_26431, *[cos_call_result_26435, a_26436], **kwargs_26437)
        
        
        # Call to cos(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'x' (line 314)
        x_26440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 62), 'x', False)
        # Getting the type of 'a' (line 314)
        a_26441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 64), 'a', False)
        # Applying the binary operator '+' (line 314)
        result_add_26442 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 62), '+', x_26440, a_26441)
        
        # Processing the call keyword arguments (line 314)
        kwargs_26443 = {}
        # Getting the type of 'cos' (line 314)
        cos_26439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 58), 'cos', False)
        # Calling cos(args, kwargs) (line 314)
        cos_call_result_26444 = invoke(stypy.reporting.localization.Localization(__file__, 314, 58), cos_26439, *[result_add_26442], **kwargs_26443)
        
        # Processing the call keyword arguments (line 314)
        kwargs_26445 = {}
        # Getting the type of 'assert_array_almost_equal' (line 314)
        assert_array_almost_equal_26430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 314)
        assert_array_almost_equal_call_result_26446 = invoke(stypy.reporting.localization.Localization(__file__, 314, 16), assert_array_almost_equal_26430, *[shift_call_result_26438, cos_call_result_26444], **kwargs_26445)
        
        
        # Call to assert_array_almost_equal(...): (line 315)
        # Processing the call arguments (line 315)
        
        # Call to shift(...): (line 315)
        # Processing the call arguments (line 315)
        
        # Call to cos(...): (line 315)
        # Processing the call arguments (line 315)
        int_26450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 52), 'int')
        # Getting the type of 'x' (line 315)
        x_26451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 54), 'x', False)
        # Applying the binary operator '*' (line 315)
        result_mul_26452 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 52), '*', int_26450, x_26451)
        
        # Processing the call keyword arguments (line 315)
        kwargs_26453 = {}
        # Getting the type of 'cos' (line 315)
        cos_26449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 48), 'cos', False)
        # Calling cos(args, kwargs) (line 315)
        cos_call_result_26454 = invoke(stypy.reporting.localization.Localization(__file__, 315, 48), cos_26449, *[result_mul_26452], **kwargs_26453)
        
        
        # Call to sin(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 'x' (line 315)
        x_26456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 61), 'x', False)
        # Processing the call keyword arguments (line 315)
        kwargs_26457 = {}
        # Getting the type of 'sin' (line 315)
        sin_26455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 57), 'sin', False)
        # Calling sin(args, kwargs) (line 315)
        sin_call_result_26458 = invoke(stypy.reporting.localization.Localization(__file__, 315, 57), sin_26455, *[x_26456], **kwargs_26457)
        
        # Applying the binary operator '+' (line 315)
        result_add_26459 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 48), '+', cos_call_result_26454, sin_call_result_26458)
        
        # Getting the type of 'a' (line 315)
        a_26460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 64), 'a', False)
        # Processing the call keyword arguments (line 315)
        kwargs_26461 = {}
        # Getting the type of 'shift' (line 315)
        shift_26448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 42), 'shift', False)
        # Calling shift(args, kwargs) (line 315)
        shift_call_result_26462 = invoke(stypy.reporting.localization.Localization(__file__, 315, 42), shift_26448, *[result_add_26459, a_26460], **kwargs_26461)
        
        
        # Call to cos(...): (line 316)
        # Processing the call arguments (line 316)
        int_26464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 46), 'int')
        # Getting the type of 'x' (line 316)
        x_26465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 49), 'x', False)
        # Getting the type of 'a' (line 316)
        a_26466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 51), 'a', False)
        # Applying the binary operator '+' (line 316)
        result_add_26467 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 49), '+', x_26465, a_26466)
        
        # Applying the binary operator '*' (line 316)
        result_mul_26468 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 46), '*', int_26464, result_add_26467)
        
        # Processing the call keyword arguments (line 316)
        kwargs_26469 = {}
        # Getting the type of 'cos' (line 316)
        cos_26463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 42), 'cos', False)
        # Calling cos(args, kwargs) (line 316)
        cos_call_result_26470 = invoke(stypy.reporting.localization.Localization(__file__, 316, 42), cos_26463, *[result_mul_26468], **kwargs_26469)
        
        
        # Call to sin(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'x' (line 316)
        x_26472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 59), 'x', False)
        # Getting the type of 'a' (line 316)
        a_26473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 61), 'a', False)
        # Applying the binary operator '+' (line 316)
        result_add_26474 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 59), '+', x_26472, a_26473)
        
        # Processing the call keyword arguments (line 316)
        kwargs_26475 = {}
        # Getting the type of 'sin' (line 316)
        sin_26471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 55), 'sin', False)
        # Calling sin(args, kwargs) (line 316)
        sin_call_result_26476 = invoke(stypy.reporting.localization.Localization(__file__, 316, 55), sin_26471, *[result_add_26474], **kwargs_26475)
        
        # Applying the binary operator '+' (line 316)
        result_add_26477 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 42), '+', cos_call_result_26470, sin_call_result_26476)
        
        # Processing the call keyword arguments (line 315)
        kwargs_26478 = {}
        # Getting the type of 'assert_array_almost_equal' (line 315)
        assert_array_almost_equal_26447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 315)
        assert_array_almost_equal_call_result_26479 = invoke(stypy.reporting.localization.Localization(__file__, 315, 16), assert_array_almost_equal_26447, *[shift_call_result_26462, result_add_26477], **kwargs_26478)
        
        
        # Call to assert_array_almost_equal(...): (line 317)
        # Processing the call arguments (line 317)
        
        # Call to shift(...): (line 317)
        # Processing the call arguments (line 317)
        
        # Call to exp(...): (line 317)
        # Processing the call arguments (line 317)
        
        # Call to sin(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'x' (line 317)
        x_26484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 56), 'x', False)
        # Processing the call keyword arguments (line 317)
        kwargs_26485 = {}
        # Getting the type of 'sin' (line 317)
        sin_26483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 52), 'sin', False)
        # Calling sin(args, kwargs) (line 317)
        sin_call_result_26486 = invoke(stypy.reporting.localization.Localization(__file__, 317, 52), sin_26483, *[x_26484], **kwargs_26485)
        
        # Processing the call keyword arguments (line 317)
        kwargs_26487 = {}
        # Getting the type of 'exp' (line 317)
        exp_26482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 48), 'exp', False)
        # Calling exp(args, kwargs) (line 317)
        exp_call_result_26488 = invoke(stypy.reporting.localization.Localization(__file__, 317, 48), exp_26482, *[sin_call_result_26486], **kwargs_26487)
        
        # Getting the type of 'a' (line 317)
        a_26489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 60), 'a', False)
        # Processing the call keyword arguments (line 317)
        kwargs_26490 = {}
        # Getting the type of 'shift' (line 317)
        shift_26481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 42), 'shift', False)
        # Calling shift(args, kwargs) (line 317)
        shift_call_result_26491 = invoke(stypy.reporting.localization.Localization(__file__, 317, 42), shift_26481, *[exp_call_result_26488, a_26489], **kwargs_26490)
        
        
        # Call to exp(...): (line 317)
        # Processing the call arguments (line 317)
        
        # Call to sin(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'x' (line 317)
        x_26494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 71), 'x', False)
        # Getting the type of 'a' (line 317)
        a_26495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 73), 'a', False)
        # Applying the binary operator '+' (line 317)
        result_add_26496 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 71), '+', x_26494, a_26495)
        
        # Processing the call keyword arguments (line 317)
        kwargs_26497 = {}
        # Getting the type of 'sin' (line 317)
        sin_26493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 67), 'sin', False)
        # Calling sin(args, kwargs) (line 317)
        sin_call_result_26498 = invoke(stypy.reporting.localization.Localization(__file__, 317, 67), sin_26493, *[result_add_26496], **kwargs_26497)
        
        # Processing the call keyword arguments (line 317)
        kwargs_26499 = {}
        # Getting the type of 'exp' (line 317)
        exp_26492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 63), 'exp', False)
        # Calling exp(args, kwargs) (line 317)
        exp_call_result_26500 = invoke(stypy.reporting.localization.Localization(__file__, 317, 63), exp_26492, *[sin_call_result_26498], **kwargs_26499)
        
        # Processing the call keyword arguments (line 317)
        kwargs_26501 = {}
        # Getting the type of 'assert_array_almost_equal' (line 317)
        assert_array_almost_equal_26480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 16), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 317)
        assert_array_almost_equal_call_result_26502 = invoke(stypy.reporting.localization.Localization(__file__, 317, 16), assert_array_almost_equal_26480, *[shift_call_result_26491, exp_call_result_26500], **kwargs_26501)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_array_almost_equal(...): (line 318)
        # Processing the call arguments (line 318)
        
        # Call to shift(...): (line 318)
        # Processing the call arguments (line 318)
        
        # Call to sin(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'x' (line 318)
        x_26506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 48), 'x', False)
        # Processing the call keyword arguments (line 318)
        kwargs_26507 = {}
        # Getting the type of 'sin' (line 318)
        sin_26505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 44), 'sin', False)
        # Calling sin(args, kwargs) (line 318)
        sin_call_result_26508 = invoke(stypy.reporting.localization.Localization(__file__, 318, 44), sin_26505, *[x_26506], **kwargs_26507)
        
        int_26509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 51), 'int')
        # Getting the type of 'pi' (line 318)
        pi_26510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 53), 'pi', False)
        # Applying the binary operator '*' (line 318)
        result_mul_26511 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 51), '*', int_26509, pi_26510)
        
        # Processing the call keyword arguments (line 318)
        kwargs_26512 = {}
        # Getting the type of 'shift' (line 318)
        shift_26504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 38), 'shift', False)
        # Calling shift(args, kwargs) (line 318)
        shift_call_result_26513 = invoke(stypy.reporting.localization.Localization(__file__, 318, 38), shift_26504, *[sin_call_result_26508, result_mul_26511], **kwargs_26512)
        
        
        # Call to sin(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'x' (line 318)
        x_26515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 61), 'x', False)
        # Processing the call keyword arguments (line 318)
        kwargs_26516 = {}
        # Getting the type of 'sin' (line 318)
        sin_26514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 57), 'sin', False)
        # Calling sin(args, kwargs) (line 318)
        sin_call_result_26517 = invoke(stypy.reporting.localization.Localization(__file__, 318, 57), sin_26514, *[x_26515], **kwargs_26516)
        
        # Processing the call keyword arguments (line 318)
        kwargs_26518 = {}
        # Getting the type of 'assert_array_almost_equal' (line 318)
        assert_array_almost_equal_26503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 318)
        assert_array_almost_equal_call_result_26519 = invoke(stypy.reporting.localization.Localization(__file__, 318, 12), assert_array_almost_equal_26503, *[shift_call_result_26513, sin_call_result_26517], **kwargs_26518)
        
        
        # Call to assert_array_almost_equal(...): (line 319)
        # Processing the call arguments (line 319)
        
        # Call to shift(...): (line 319)
        # Processing the call arguments (line 319)
        
        # Call to sin(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'x' (line 319)
        x_26523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 48), 'x', False)
        # Processing the call keyword arguments (line 319)
        kwargs_26524 = {}
        # Getting the type of 'sin' (line 319)
        sin_26522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 44), 'sin', False)
        # Calling sin(args, kwargs) (line 319)
        sin_call_result_26525 = invoke(stypy.reporting.localization.Localization(__file__, 319, 44), sin_26522, *[x_26523], **kwargs_26524)
        
        # Getting the type of 'pi' (line 319)
        pi_26526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 51), 'pi', False)
        # Processing the call keyword arguments (line 319)
        kwargs_26527 = {}
        # Getting the type of 'shift' (line 319)
        shift_26521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 38), 'shift', False)
        # Calling shift(args, kwargs) (line 319)
        shift_call_result_26528 = invoke(stypy.reporting.localization.Localization(__file__, 319, 38), shift_26521, *[sin_call_result_26525, pi_26526], **kwargs_26527)
        
        
        
        # Call to sin(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'x' (line 319)
        x_26530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 60), 'x', False)
        # Processing the call keyword arguments (line 319)
        kwargs_26531 = {}
        # Getting the type of 'sin' (line 319)
        sin_26529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 56), 'sin', False)
        # Calling sin(args, kwargs) (line 319)
        sin_call_result_26532 = invoke(stypy.reporting.localization.Localization(__file__, 319, 56), sin_26529, *[x_26530], **kwargs_26531)
        
        # Applying the 'usub' unary operator (line 319)
        result___neg___26533 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 55), 'usub', sin_call_result_26532)
        
        # Processing the call keyword arguments (line 319)
        kwargs_26534 = {}
        # Getting the type of 'assert_array_almost_equal' (line 319)
        assert_array_almost_equal_26520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 319)
        assert_array_almost_equal_call_result_26535 = invoke(stypy.reporting.localization.Localization(__file__, 319, 12), assert_array_almost_equal_26520, *[shift_call_result_26528, result___neg___26533], **kwargs_26534)
        
        
        # Call to assert_array_almost_equal(...): (line 320)
        # Processing the call arguments (line 320)
        
        # Call to shift(...): (line 320)
        # Processing the call arguments (line 320)
        
        # Call to sin(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'x' (line 320)
        x_26539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 48), 'x', False)
        # Processing the call keyword arguments (line 320)
        kwargs_26540 = {}
        # Getting the type of 'sin' (line 320)
        sin_26538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 44), 'sin', False)
        # Calling sin(args, kwargs) (line 320)
        sin_call_result_26541 = invoke(stypy.reporting.localization.Localization(__file__, 320, 44), sin_26538, *[x_26539], **kwargs_26540)
        
        # Getting the type of 'pi' (line 320)
        pi_26542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 51), 'pi', False)
        int_26543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 54), 'int')
        # Applying the binary operator 'div' (line 320)
        result_div_26544 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 51), 'div', pi_26542, int_26543)
        
        # Processing the call keyword arguments (line 320)
        kwargs_26545 = {}
        # Getting the type of 'shift' (line 320)
        shift_26537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 38), 'shift', False)
        # Calling shift(args, kwargs) (line 320)
        shift_call_result_26546 = invoke(stypy.reporting.localization.Localization(__file__, 320, 38), shift_26537, *[sin_call_result_26541, result_div_26544], **kwargs_26545)
        
        
        # Call to cos(...): (line 320)
        # Processing the call arguments (line 320)
        # Getting the type of 'x' (line 320)
        x_26548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 61), 'x', False)
        # Processing the call keyword arguments (line 320)
        kwargs_26549 = {}
        # Getting the type of 'cos' (line 320)
        cos_26547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 57), 'cos', False)
        # Calling cos(args, kwargs) (line 320)
        cos_call_result_26550 = invoke(stypy.reporting.localization.Localization(__file__, 320, 57), cos_26547, *[x_26548], **kwargs_26549)
        
        # Processing the call keyword arguments (line 320)
        kwargs_26551 = {}
        # Getting the type of 'assert_array_almost_equal' (line 320)
        assert_array_almost_equal_26536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'assert_array_almost_equal', False)
        # Calling assert_array_almost_equal(args, kwargs) (line 320)
        assert_array_almost_equal_call_result_26552 = invoke(stypy.reporting.localization.Localization(__file__, 320, 12), assert_array_almost_equal_26536, *[shift_call_result_26546, cos_call_result_26550], **kwargs_26551)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_definition(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_definition' in the type store
        # Getting the type of 'stypy_return_type' (line 308)
        stypy_return_type_26553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26553)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_definition'
        return stypy_return_type_26553


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 306, 0, False)
        # Assigning a type to the variable 'self' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestShift.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestShift' (line 306)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 0), 'TestShift', TestShift)
# Declaration of the 'TestOverwrite' class

class TestOverwrite(object, ):
    str_26554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 4), 'str', 'Check input overwrite behavior ')

    @norecursion
    def _check(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check'
        module_type_store = module_type_store.open_function_context('_check', 329, 4, False)
        # Assigning a type to the variable 'self' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite._check.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite._check.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite._check.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite._check.__dict__.__setitem__('stypy_function_name', 'TestOverwrite._check')
        TestOverwrite._check.__dict__.__setitem__('stypy_param_names_list', ['x', 'routine'])
        TestOverwrite._check.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        TestOverwrite._check.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        TestOverwrite._check.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite._check.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite._check.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite._check.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite._check', ['x', 'routine'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check', localization, ['x', 'routine'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check(...)' code ##################

        
        # Assigning a Call to a Name (line 330):
        
        # Call to copy(...): (line 330)
        # Processing the call keyword arguments (line 330)
        kwargs_26557 = {}
        # Getting the type of 'x' (line 330)
        x_26555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 13), 'x', False)
        # Obtaining the member 'copy' of a type (line 330)
        copy_26556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 13), x_26555, 'copy')
        # Calling copy(args, kwargs) (line 330)
        copy_call_result_26558 = invoke(stypy.reporting.localization.Localization(__file__, 330, 13), copy_26556, *[], **kwargs_26557)
        
        # Assigning a type to the variable 'x2' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'x2', copy_call_result_26558)
        
        # Call to routine(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 'x2' (line 331)
        x2_26560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 16), 'x2', False)
        # Getting the type of 'args' (line 331)
        args_26561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 21), 'args', False)
        # Processing the call keyword arguments (line 331)
        # Getting the type of 'kwargs' (line 331)
        kwargs_26562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 29), 'kwargs', False)
        kwargs_26563 = {'kwargs_26562': kwargs_26562}
        # Getting the type of 'routine' (line 331)
        routine_26559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'routine', False)
        # Calling routine(args, kwargs) (line 331)
        routine_call_result_26564 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), routine_26559, *[x2_26560, args_26561], **kwargs_26563)
        
        
        # Assigning a Attribute to a Name (line 332):
        # Getting the type of 'routine' (line 332)
        routine_26565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 14), 'routine')
        # Obtaining the member '__name__' of a type (line 332)
        name___26566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 14), routine_26565, '__name__')
        # Assigning a type to the variable 'sig' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'sig', name___26566)
        
        # Getting the type of 'args' (line 333)
        args_26567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 11), 'args')
        # Testing the type of an if condition (line 333)
        if_condition_26568 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 333, 8), args_26567)
        # Assigning a type to the variable 'if_condition_26568' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'if_condition_26568', if_condition_26568)
        # SSA begins for if statement (line 333)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'sig' (line 334)
        sig_26569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'sig')
        
        # Call to repr(...): (line 334)
        # Processing the call arguments (line 334)
        # Getting the type of 'args' (line 334)
        args_26571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 24), 'args', False)
        # Processing the call keyword arguments (line 334)
        kwargs_26572 = {}
        # Getting the type of 'repr' (line 334)
        repr_26570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'repr', False)
        # Calling repr(args, kwargs) (line 334)
        repr_call_result_26573 = invoke(stypy.reporting.localization.Localization(__file__, 334, 19), repr_26570, *[args_26571], **kwargs_26572)
        
        # Applying the binary operator '+=' (line 334)
        result_iadd_26574 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 12), '+=', sig_26569, repr_call_result_26573)
        # Assigning a type to the variable 'sig' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'sig', result_iadd_26574)
        
        # SSA join for if statement (line 333)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'kwargs' (line 335)
        kwargs_26575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 11), 'kwargs')
        # Testing the type of an if condition (line 335)
        if_condition_26576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 335, 8), kwargs_26575)
        # Assigning a type to the variable 'if_condition_26576' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'if_condition_26576', if_condition_26576)
        # SSA begins for if statement (line 335)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'sig' (line 336)
        sig_26577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'sig')
        
        # Call to repr(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'kwargs' (line 336)
        kwargs_26579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 24), 'kwargs', False)
        # Processing the call keyword arguments (line 336)
        kwargs_26580 = {}
        # Getting the type of 'repr' (line 336)
        repr_26578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 19), 'repr', False)
        # Calling repr(args, kwargs) (line 336)
        repr_call_result_26581 = invoke(stypy.reporting.localization.Localization(__file__, 336, 19), repr_26578, *[kwargs_26579], **kwargs_26580)
        
        # Applying the binary operator '+=' (line 336)
        result_iadd_26582 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 12), '+=', sig_26577, repr_call_result_26581)
        # Assigning a type to the variable 'sig' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'sig', result_iadd_26582)
        
        # SSA join for if statement (line 335)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to assert_equal(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 'x2' (line 337)
        x2_26584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 21), 'x2', False)
        # Getting the type of 'x' (line 337)
        x_26585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 25), 'x', False)
        # Processing the call keyword arguments (line 337)
        str_26586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 36), 'str', 'spurious overwrite in %s')
        # Getting the type of 'sig' (line 337)
        sig_26587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 65), 'sig', False)
        # Applying the binary operator '%' (line 337)
        result_mod_26588 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 36), '%', str_26586, sig_26587)
        
        keyword_26589 = result_mod_26588
        kwargs_26590 = {'err_msg': keyword_26589}
        # Getting the type of 'assert_equal' (line 337)
        assert_equal_26583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 337)
        assert_equal_call_result_26591 = invoke(stypy.reporting.localization.Localization(__file__, 337, 8), assert_equal_26583, *[x2_26584, x_26585], **kwargs_26590)
        
        
        # ################# End of '_check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check' in the type store
        # Getting the type of 'stypy_return_type' (line 329)
        stypy_return_type_26592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26592)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check'
        return stypy_return_type_26592


    @norecursion
    def _check_1d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_1d'
        module_type_store = module_type_store.open_function_context('_check_1d', 339, 4, False)
        # Assigning a type to the variable 'self' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_function_name', 'TestOverwrite._check_1d')
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_param_names_list', ['routine', 'dtype', 'shape'])
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite._check_1d.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite._check_1d', ['routine', 'dtype', 'shape'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_1d', localization, ['routine', 'dtype', 'shape'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_1d(...)' code ##################

        
        # Call to seed(...): (line 340)
        # Processing the call arguments (line 340)
        int_26596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 23), 'int')
        # Processing the call keyword arguments (line 340)
        kwargs_26597 = {}
        # Getting the type of 'np' (line 340)
        np_26593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'np', False)
        # Obtaining the member 'random' of a type (line 340)
        random_26594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), np_26593, 'random')
        # Obtaining the member 'seed' of a type (line 340)
        seed_26595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), random_26594, 'seed')
        # Calling seed(args, kwargs) (line 340)
        seed_call_result_26598 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), seed_26595, *[int_26596], **kwargs_26597)
        
        
        
        # Call to issubdtype(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'dtype' (line 341)
        dtype_26601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 25), 'dtype', False)
        # Getting the type of 'np' (line 341)
        np_26602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 32), 'np', False)
        # Obtaining the member 'complexfloating' of a type (line 341)
        complexfloating_26603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 32), np_26602, 'complexfloating')
        # Processing the call keyword arguments (line 341)
        kwargs_26604 = {}
        # Getting the type of 'np' (line 341)
        np_26599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 11), 'np', False)
        # Obtaining the member 'issubdtype' of a type (line 341)
        issubdtype_26600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 11), np_26599, 'issubdtype')
        # Calling issubdtype(args, kwargs) (line 341)
        issubdtype_call_result_26605 = invoke(stypy.reporting.localization.Localization(__file__, 341, 11), issubdtype_26600, *[dtype_26601, complexfloating_26603], **kwargs_26604)
        
        # Testing the type of an if condition (line 341)
        if_condition_26606 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 8), issubdtype_call_result_26605)
        # Assigning a type to the variable 'if_condition_26606' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'if_condition_26606', if_condition_26606)
        # SSA begins for if statement (line 341)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 342):
        
        # Call to randn(...): (line 342)
        # Getting the type of 'shape' (line 342)
        shape_26610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 36), 'shape', False)
        # Processing the call keyword arguments (line 342)
        kwargs_26611 = {}
        # Getting the type of 'np' (line 342)
        np_26607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 19), 'np', False)
        # Obtaining the member 'random' of a type (line 342)
        random_26608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 19), np_26607, 'random')
        # Obtaining the member 'randn' of a type (line 342)
        randn_26609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 19), random_26608, 'randn')
        # Calling randn(args, kwargs) (line 342)
        randn_call_result_26612 = invoke(stypy.reporting.localization.Localization(__file__, 342, 19), randn_26609, *[shape_26610], **kwargs_26611)
        
        complex_26613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 45), 'complex')
        
        # Call to randn(...): (line 342)
        # Getting the type of 'shape' (line 342)
        shape_26617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 65), 'shape', False)
        # Processing the call keyword arguments (line 342)
        kwargs_26618 = {}
        # Getting the type of 'np' (line 342)
        np_26614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 48), 'np', False)
        # Obtaining the member 'random' of a type (line 342)
        random_26615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 48), np_26614, 'random')
        # Obtaining the member 'randn' of a type (line 342)
        randn_26616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 48), random_26615, 'randn')
        # Calling randn(args, kwargs) (line 342)
        randn_call_result_26619 = invoke(stypy.reporting.localization.Localization(__file__, 342, 48), randn_26616, *[shape_26617], **kwargs_26618)
        
        # Applying the binary operator '*' (line 342)
        result_mul_26620 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 45), '*', complex_26613, randn_call_result_26619)
        
        # Applying the binary operator '+' (line 342)
        result_add_26621 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 19), '+', randn_call_result_26612, result_mul_26620)
        
        # Assigning a type to the variable 'data' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'data', result_add_26621)
        # SSA branch for the else part of an if statement (line 341)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 344):
        
        # Call to randn(...): (line 344)
        # Getting the type of 'shape' (line 344)
        shape_26625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 36), 'shape', False)
        # Processing the call keyword arguments (line 344)
        kwargs_26626 = {}
        # Getting the type of 'np' (line 344)
        np_26622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 19), 'np', False)
        # Obtaining the member 'random' of a type (line 344)
        random_26623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 19), np_26622, 'random')
        # Obtaining the member 'randn' of a type (line 344)
        randn_26624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 19), random_26623, 'randn')
        # Calling randn(args, kwargs) (line 344)
        randn_call_result_26627 = invoke(stypy.reporting.localization.Localization(__file__, 344, 19), randn_26624, *[shape_26625], **kwargs_26626)
        
        # Assigning a type to the variable 'data' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'data', randn_call_result_26627)
        # SSA join for if statement (line 341)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 345):
        
        # Call to astype(...): (line 345)
        # Processing the call arguments (line 345)
        # Getting the type of 'dtype' (line 345)
        dtype_26630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 27), 'dtype', False)
        # Processing the call keyword arguments (line 345)
        kwargs_26631 = {}
        # Getting the type of 'data' (line 345)
        data_26628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 15), 'data', False)
        # Obtaining the member 'astype' of a type (line 345)
        astype_26629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 15), data_26628, 'astype')
        # Calling astype(args, kwargs) (line 345)
        astype_call_result_26632 = invoke(stypy.reporting.localization.Localization(__file__, 345, 15), astype_26629, *[dtype_26630], **kwargs_26631)
        
        # Assigning a type to the variable 'data' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'data', astype_call_result_26632)
        
        # Call to _check(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'data' (line 346)
        data_26635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 20), 'data', False)
        # Getting the type of 'routine' (line 346)
        routine_26636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 26), 'routine', False)
        # Getting the type of 'args' (line 346)
        args_26637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 36), 'args', False)
        # Processing the call keyword arguments (line 346)
        # Getting the type of 'kwargs' (line 346)
        kwargs_26638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 44), 'kwargs', False)
        kwargs_26639 = {'kwargs_26638': kwargs_26638}
        # Getting the type of 'self' (line 346)
        self_26633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'self', False)
        # Obtaining the member '_check' of a type (line 346)
        _check_26634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 8), self_26633, '_check')
        # Calling _check(args, kwargs) (line 346)
        _check_call_result_26640 = invoke(stypy.reporting.localization.Localization(__file__, 346, 8), _check_26634, *[data_26635, routine_26636, args_26637], **kwargs_26639)
        
        
        # ################# End of '_check_1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_1d' in the type store
        # Getting the type of 'stypy_return_type' (line 339)
        stypy_return_type_26641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26641)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_1d'
        return stypy_return_type_26641


    @norecursion
    def test_diff(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_diff'
        module_type_store = module_type_store.open_function_context('test_diff', 348, 4, False)
        # Assigning a type to the variable 'self' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_diff.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_diff.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_diff.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_diff.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_diff')
        TestOverwrite.test_diff.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_diff.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_diff.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_diff.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_diff.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_diff.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_diff.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_diff', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_diff', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_diff(...)' code ##################

        
        # Getting the type of 'self' (line 349)
        self_26642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 21), 'self')
        # Obtaining the member 'dtypes' of a type (line 349)
        dtypes_26643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 21), self_26642, 'dtypes')
        # Testing the type of a for loop iterable (line 349)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 349, 8), dtypes_26643)
        # Getting the type of the for loop variable (line 349)
        for_loop_var_26644 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 349, 8), dtypes_26643)
        # Assigning a type to the variable 'dtype' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'dtype', for_loop_var_26644)
        # SSA begins for a for statement (line 349)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _check_1d(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'diff' (line 350)
        diff_26647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 27), 'diff', False)
        # Getting the type of 'dtype' (line 350)
        dtype_26648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 33), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 350)
        tuple_26649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 350)
        # Adding element type (line 350)
        int_26650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 350, 41), tuple_26649, int_26650)
        
        # Processing the call keyword arguments (line 350)
        kwargs_26651 = {}
        # Getting the type of 'self' (line 350)
        self_26645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 350)
        _check_1d_26646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 12), self_26645, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 350)
        _check_1d_call_result_26652 = invoke(stypy.reporting.localization.Localization(__file__, 350, 12), _check_1d_26646, *[diff_26647, dtype_26648, tuple_26649], **kwargs_26651)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_diff(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_diff' in the type store
        # Getting the type of 'stypy_return_type' (line 348)
        stypy_return_type_26653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26653)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_diff'
        return stypy_return_type_26653


    @norecursion
    def test_tilbert(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tilbert'
        module_type_store = module_type_store.open_function_context('test_tilbert', 352, 4, False)
        # Assigning a type to the variable 'self' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_tilbert.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_tilbert.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_tilbert.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_tilbert.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_tilbert')
        TestOverwrite.test_tilbert.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_tilbert.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_tilbert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_tilbert.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_tilbert.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_tilbert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_tilbert.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_tilbert', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tilbert', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tilbert(...)' code ##################

        
        # Getting the type of 'self' (line 353)
        self_26654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 21), 'self')
        # Obtaining the member 'dtypes' of a type (line 353)
        dtypes_26655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 21), self_26654, 'dtypes')
        # Testing the type of a for loop iterable (line 353)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 353, 8), dtypes_26655)
        # Getting the type of the for loop variable (line 353)
        for_loop_var_26656 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 353, 8), dtypes_26655)
        # Assigning a type to the variable 'dtype' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'dtype', for_loop_var_26656)
        # SSA begins for a for statement (line 353)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _check_1d(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'tilbert' (line 354)
        tilbert_26659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 27), 'tilbert', False)
        # Getting the type of 'dtype' (line 354)
        dtype_26660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 36), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 354)
        tuple_26661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 354)
        # Adding element type (line 354)
        int_26662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 44), tuple_26661, int_26662)
        
        float_26663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 50), 'float')
        # Processing the call keyword arguments (line 354)
        kwargs_26664 = {}
        # Getting the type of 'self' (line 354)
        self_26657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 354)
        _check_1d_26658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 12), self_26657, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 354)
        _check_1d_call_result_26665 = invoke(stypy.reporting.localization.Localization(__file__, 354, 12), _check_1d_26658, *[tilbert_26659, dtype_26660, tuple_26661, float_26663], **kwargs_26664)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_tilbert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tilbert' in the type store
        # Getting the type of 'stypy_return_type' (line 352)
        stypy_return_type_26666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26666)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tilbert'
        return stypy_return_type_26666


    @norecursion
    def test_itilbert(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_itilbert'
        module_type_store = module_type_store.open_function_context('test_itilbert', 356, 4, False)
        # Assigning a type to the variable 'self' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_itilbert.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_itilbert.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_itilbert.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_itilbert.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_itilbert')
        TestOverwrite.test_itilbert.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_itilbert.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_itilbert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_itilbert.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_itilbert.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_itilbert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_itilbert.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_itilbert', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_itilbert', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_itilbert(...)' code ##################

        
        # Getting the type of 'self' (line 357)
        self_26667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 21), 'self')
        # Obtaining the member 'dtypes' of a type (line 357)
        dtypes_26668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 21), self_26667, 'dtypes')
        # Testing the type of a for loop iterable (line 357)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 357, 8), dtypes_26668)
        # Getting the type of the for loop variable (line 357)
        for_loop_var_26669 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 357, 8), dtypes_26668)
        # Assigning a type to the variable 'dtype' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'dtype', for_loop_var_26669)
        # SSA begins for a for statement (line 357)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _check_1d(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'itilbert' (line 358)
        itilbert_26672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 27), 'itilbert', False)
        # Getting the type of 'dtype' (line 358)
        dtype_26673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 37), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 358)
        tuple_26674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 358)
        # Adding element type (line 358)
        int_26675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 45), tuple_26674, int_26675)
        
        float_26676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 51), 'float')
        # Processing the call keyword arguments (line 358)
        kwargs_26677 = {}
        # Getting the type of 'self' (line 358)
        self_26670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 358)
        _check_1d_26671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), self_26670, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 358)
        _check_1d_call_result_26678 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), _check_1d_26671, *[itilbert_26672, dtype_26673, tuple_26674, float_26676], **kwargs_26677)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_itilbert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_itilbert' in the type store
        # Getting the type of 'stypy_return_type' (line 356)
        stypy_return_type_26679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26679)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_itilbert'
        return stypy_return_type_26679


    @norecursion
    def test_hilbert(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_hilbert'
        module_type_store = module_type_store.open_function_context('test_hilbert', 360, 4, False)
        # Assigning a type to the variable 'self' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_hilbert.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_hilbert.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_hilbert.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_hilbert.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_hilbert')
        TestOverwrite.test_hilbert.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_hilbert.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_hilbert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_hilbert.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_hilbert.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_hilbert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_hilbert.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_hilbert', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_hilbert', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_hilbert(...)' code ##################

        
        # Getting the type of 'self' (line 361)
        self_26680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 21), 'self')
        # Obtaining the member 'dtypes' of a type (line 361)
        dtypes_26681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 21), self_26680, 'dtypes')
        # Testing the type of a for loop iterable (line 361)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 361, 8), dtypes_26681)
        # Getting the type of the for loop variable (line 361)
        for_loop_var_26682 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 361, 8), dtypes_26681)
        # Assigning a type to the variable 'dtype' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'dtype', for_loop_var_26682)
        # SSA begins for a for statement (line 361)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _check_1d(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'hilbert' (line 362)
        hilbert_26685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 27), 'hilbert', False)
        # Getting the type of 'dtype' (line 362)
        dtype_26686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 36), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 362)
        tuple_26687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 362)
        # Adding element type (line 362)
        int_26688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 44), tuple_26687, int_26688)
        
        # Processing the call keyword arguments (line 362)
        kwargs_26689 = {}
        # Getting the type of 'self' (line 362)
        self_26683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 362)
        _check_1d_26684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), self_26683, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 362)
        _check_1d_call_result_26690 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), _check_1d_26684, *[hilbert_26685, dtype_26686, tuple_26687], **kwargs_26689)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_hilbert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_hilbert' in the type store
        # Getting the type of 'stypy_return_type' (line 360)
        stypy_return_type_26691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26691)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_hilbert'
        return stypy_return_type_26691


    @norecursion
    def test_cs_diff(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cs_diff'
        module_type_store = module_type_store.open_function_context('test_cs_diff', 364, 4, False)
        # Assigning a type to the variable 'self' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_cs_diff.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_cs_diff.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_cs_diff.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_cs_diff.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_cs_diff')
        TestOverwrite.test_cs_diff.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_cs_diff.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_cs_diff.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_cs_diff.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_cs_diff.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_cs_diff.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_cs_diff.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_cs_diff', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cs_diff', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cs_diff(...)' code ##################

        
        # Getting the type of 'self' (line 365)
        self_26692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 21), 'self')
        # Obtaining the member 'dtypes' of a type (line 365)
        dtypes_26693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 21), self_26692, 'dtypes')
        # Testing the type of a for loop iterable (line 365)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 365, 8), dtypes_26693)
        # Getting the type of the for loop variable (line 365)
        for_loop_var_26694 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 365, 8), dtypes_26693)
        # Assigning a type to the variable 'dtype' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'dtype', for_loop_var_26694)
        # SSA begins for a for statement (line 365)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _check_1d(...): (line 366)
        # Processing the call arguments (line 366)
        # Getting the type of 'cs_diff' (line 366)
        cs_diff_26697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 27), 'cs_diff', False)
        # Getting the type of 'dtype' (line 366)
        dtype_26698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 36), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 366)
        tuple_26699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 366)
        # Adding element type (line 366)
        int_26700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 366, 44), tuple_26699, int_26700)
        
        float_26701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 50), 'float')
        float_26702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 55), 'float')
        # Processing the call keyword arguments (line 366)
        kwargs_26703 = {}
        # Getting the type of 'self' (line 366)
        self_26695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 366)
        _check_1d_26696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 12), self_26695, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 366)
        _check_1d_call_result_26704 = invoke(stypy.reporting.localization.Localization(__file__, 366, 12), _check_1d_26696, *[cs_diff_26697, dtype_26698, tuple_26699, float_26701, float_26702], **kwargs_26703)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_cs_diff(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cs_diff' in the type store
        # Getting the type of 'stypy_return_type' (line 364)
        stypy_return_type_26705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26705)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cs_diff'
        return stypy_return_type_26705


    @norecursion
    def test_sc_diff(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_sc_diff'
        module_type_store = module_type_store.open_function_context('test_sc_diff', 368, 4, False)
        # Assigning a type to the variable 'self' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_sc_diff.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_sc_diff.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_sc_diff.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_sc_diff.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_sc_diff')
        TestOverwrite.test_sc_diff.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_sc_diff.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_sc_diff.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_sc_diff.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_sc_diff.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_sc_diff.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_sc_diff.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_sc_diff', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_sc_diff', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_sc_diff(...)' code ##################

        
        # Getting the type of 'self' (line 369)
        self_26706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 21), 'self')
        # Obtaining the member 'dtypes' of a type (line 369)
        dtypes_26707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 21), self_26706, 'dtypes')
        # Testing the type of a for loop iterable (line 369)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 369, 8), dtypes_26707)
        # Getting the type of the for loop variable (line 369)
        for_loop_var_26708 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 369, 8), dtypes_26707)
        # Assigning a type to the variable 'dtype' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'dtype', for_loop_var_26708)
        # SSA begins for a for statement (line 369)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _check_1d(...): (line 370)
        # Processing the call arguments (line 370)
        # Getting the type of 'sc_diff' (line 370)
        sc_diff_26711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 27), 'sc_diff', False)
        # Getting the type of 'dtype' (line 370)
        dtype_26712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 36), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 370)
        tuple_26713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 370)
        # Adding element type (line 370)
        int_26714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 370, 44), tuple_26713, int_26714)
        
        float_26715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 50), 'float')
        float_26716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 55), 'float')
        # Processing the call keyword arguments (line 370)
        kwargs_26717 = {}
        # Getting the type of 'self' (line 370)
        self_26709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 370)
        _check_1d_26710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 12), self_26709, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 370)
        _check_1d_call_result_26718 = invoke(stypy.reporting.localization.Localization(__file__, 370, 12), _check_1d_26710, *[sc_diff_26711, dtype_26712, tuple_26713, float_26715, float_26716], **kwargs_26717)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_sc_diff(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_sc_diff' in the type store
        # Getting the type of 'stypy_return_type' (line 368)
        stypy_return_type_26719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26719)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_sc_diff'
        return stypy_return_type_26719


    @norecursion
    def test_ss_diff(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_ss_diff'
        module_type_store = module_type_store.open_function_context('test_ss_diff', 372, 4, False)
        # Assigning a type to the variable 'self' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_ss_diff.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_ss_diff.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_ss_diff.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_ss_diff.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_ss_diff')
        TestOverwrite.test_ss_diff.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_ss_diff.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_ss_diff.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_ss_diff.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_ss_diff.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_ss_diff.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_ss_diff.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_ss_diff', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_ss_diff', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_ss_diff(...)' code ##################

        
        # Getting the type of 'self' (line 373)
        self_26720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 21), 'self')
        # Obtaining the member 'dtypes' of a type (line 373)
        dtypes_26721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 21), self_26720, 'dtypes')
        # Testing the type of a for loop iterable (line 373)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 373, 8), dtypes_26721)
        # Getting the type of the for loop variable (line 373)
        for_loop_var_26722 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 373, 8), dtypes_26721)
        # Assigning a type to the variable 'dtype' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'dtype', for_loop_var_26722)
        # SSA begins for a for statement (line 373)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _check_1d(...): (line 374)
        # Processing the call arguments (line 374)
        # Getting the type of 'ss_diff' (line 374)
        ss_diff_26725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 27), 'ss_diff', False)
        # Getting the type of 'dtype' (line 374)
        dtype_26726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 36), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 374)
        tuple_26727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 374)
        # Adding element type (line 374)
        int_26728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 44), tuple_26727, int_26728)
        
        float_26729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 50), 'float')
        float_26730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 55), 'float')
        # Processing the call keyword arguments (line 374)
        kwargs_26731 = {}
        # Getting the type of 'self' (line 374)
        self_26723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 374)
        _check_1d_26724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 12), self_26723, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 374)
        _check_1d_call_result_26732 = invoke(stypy.reporting.localization.Localization(__file__, 374, 12), _check_1d_26724, *[ss_diff_26725, dtype_26726, tuple_26727, float_26729, float_26730], **kwargs_26731)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_ss_diff(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_ss_diff' in the type store
        # Getting the type of 'stypy_return_type' (line 372)
        stypy_return_type_26733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26733)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_ss_diff'
        return stypy_return_type_26733


    @norecursion
    def test_cc_diff(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_cc_diff'
        module_type_store = module_type_store.open_function_context('test_cc_diff', 376, 4, False)
        # Assigning a type to the variable 'self' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_cc_diff.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_cc_diff.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_cc_diff.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_cc_diff.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_cc_diff')
        TestOverwrite.test_cc_diff.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_cc_diff.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_cc_diff.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_cc_diff.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_cc_diff.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_cc_diff.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_cc_diff.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_cc_diff', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_cc_diff', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_cc_diff(...)' code ##################

        
        # Getting the type of 'self' (line 377)
        self_26734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 21), 'self')
        # Obtaining the member 'dtypes' of a type (line 377)
        dtypes_26735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 21), self_26734, 'dtypes')
        # Testing the type of a for loop iterable (line 377)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 377, 8), dtypes_26735)
        # Getting the type of the for loop variable (line 377)
        for_loop_var_26736 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 377, 8), dtypes_26735)
        # Assigning a type to the variable 'dtype' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'dtype', for_loop_var_26736)
        # SSA begins for a for statement (line 377)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _check_1d(...): (line 378)
        # Processing the call arguments (line 378)
        # Getting the type of 'cc_diff' (line 378)
        cc_diff_26739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 27), 'cc_diff', False)
        # Getting the type of 'dtype' (line 378)
        dtype_26740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 36), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 378)
        tuple_26741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 378)
        # Adding element type (line 378)
        int_26742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 44), tuple_26741, int_26742)
        
        float_26743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 50), 'float')
        float_26744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 55), 'float')
        # Processing the call keyword arguments (line 378)
        kwargs_26745 = {}
        # Getting the type of 'self' (line 378)
        self_26737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 378)
        _check_1d_26738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 12), self_26737, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 378)
        _check_1d_call_result_26746 = invoke(stypy.reporting.localization.Localization(__file__, 378, 12), _check_1d_26738, *[cc_diff_26739, dtype_26740, tuple_26741, float_26743, float_26744], **kwargs_26745)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_cc_diff(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_cc_diff' in the type store
        # Getting the type of 'stypy_return_type' (line 376)
        stypy_return_type_26747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26747)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_cc_diff'
        return stypy_return_type_26747


    @norecursion
    def test_shift(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_shift'
        module_type_store = module_type_store.open_function_context('test_shift', 380, 4, False)
        # Assigning a type to the variable 'self' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestOverwrite.test_shift.__dict__.__setitem__('stypy_localization', localization)
        TestOverwrite.test_shift.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestOverwrite.test_shift.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestOverwrite.test_shift.__dict__.__setitem__('stypy_function_name', 'TestOverwrite.test_shift')
        TestOverwrite.test_shift.__dict__.__setitem__('stypy_param_names_list', [])
        TestOverwrite.test_shift.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestOverwrite.test_shift.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestOverwrite.test_shift.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestOverwrite.test_shift.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestOverwrite.test_shift.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestOverwrite.test_shift.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.test_shift', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_shift', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_shift(...)' code ##################

        
        # Getting the type of 'self' (line 381)
        self_26748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 21), 'self')
        # Obtaining the member 'dtypes' of a type (line 381)
        dtypes_26749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 21), self_26748, 'dtypes')
        # Testing the type of a for loop iterable (line 381)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 381, 8), dtypes_26749)
        # Getting the type of the for loop variable (line 381)
        for_loop_var_26750 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 381, 8), dtypes_26749)
        # Assigning a type to the variable 'dtype' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'dtype', for_loop_var_26750)
        # SSA begins for a for statement (line 381)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _check_1d(...): (line 382)
        # Processing the call arguments (line 382)
        # Getting the type of 'shift' (line 382)
        shift_26753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 27), 'shift', False)
        # Getting the type of 'dtype' (line 382)
        dtype_26754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 34), 'dtype', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 382)
        tuple_26755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 42), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 382)
        # Adding element type (line 382)
        int_26756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 382, 42), tuple_26755, int_26756)
        
        float_26757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 48), 'float')
        # Processing the call keyword arguments (line 382)
        kwargs_26758 = {}
        # Getting the type of 'self' (line 382)
        self_26751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 12), 'self', False)
        # Obtaining the member '_check_1d' of a type (line 382)
        _check_1d_26752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 12), self_26751, '_check_1d')
        # Calling _check_1d(args, kwargs) (line 382)
        _check_1d_call_result_26759 = invoke(stypy.reporting.localization.Localization(__file__, 382, 12), _check_1d_26752, *[shift_26753, dtype_26754, tuple_26755, float_26757], **kwargs_26758)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_shift(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_shift' in the type store
        # Getting the type of 'stypy_return_type' (line 380)
        stypy_return_type_26760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26760)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_shift'
        return stypy_return_type_26760


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 323, 0, False)
        # Assigning a type to the variable 'self' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestOverwrite.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestOverwrite' (line 323)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 0), 'TestOverwrite', TestOverwrite)

# Assigning a List to a Name (line 326):

# Obtaining an instance of the builtin type 'list' (line 326)
list_26761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 326)
# Adding element type (line 326)
# Getting the type of 'np' (line 326)
np_26762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 19), 'np')
# Obtaining the member 'float32' of a type (line 326)
float32_26763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 19), np_26762, 'float32')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 18), list_26761, float32_26763)
# Adding element type (line 326)
# Getting the type of 'np' (line 326)
np_26764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 31), 'np')
# Obtaining the member 'float64' of a type (line 326)
float64_26765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 31), np_26764, 'float64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 18), list_26761, float64_26765)

# Getting the type of 'TestOverwrite'
TestOverwrite_26766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestOverwrite')
# Setting the type of the member 'real_dtypes' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestOverwrite_26766, 'real_dtypes', list_26761)

# Assigning a BinOp to a Name (line 327):
# Getting the type of 'TestOverwrite'
TestOverwrite_26767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestOverwrite')
# Obtaining the member 'real_dtypes' of a type
real_dtypes_26768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestOverwrite_26767, 'real_dtypes')

# Obtaining an instance of the builtin type 'list' (line 327)
list_26769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 327)
# Adding element type (line 327)
# Getting the type of 'np' (line 327)
np_26770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 28), 'np')
# Obtaining the member 'complex64' of a type (line 327)
complex64_26771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 28), np_26770, 'complex64')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 27), list_26769, complex64_26771)
# Adding element type (line 327)
# Getting the type of 'np' (line 327)
np_26772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 42), 'np')
# Obtaining the member 'complex128' of a type (line 327)
complex128_26773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 42), np_26772, 'complex128')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 27), list_26769, complex128_26773)

# Applying the binary operator '+' (line 327)
result_add_26774 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 13), '+', real_dtypes_26768, list_26769)

# Getting the type of 'TestOverwrite'
TestOverwrite_26775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestOverwrite')
# Setting the type of the member 'dtypes' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestOverwrite_26775, 'dtypes', result_add_26774)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
