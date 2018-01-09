
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import timeit
4: from functools import reduce
5: 
6: import numpy as np
7: from numpy import float_
8: import numpy.core.fromnumeric as fromnumeric
9: 
10: from numpy.testing.utils import build_err_msg
11: 
12: # Fixme: this does not look right.
13: np.seterr(all='ignore')
14: 
15: pi = np.pi
16: 
17: 
18: class ModuleTester(object):
19:     def __init__(self, module):
20:         self.module = module
21:         self.allequal = module.allequal
22:         self.arange = module.arange
23:         self.array = module.array
24:         self.concatenate = module.concatenate
25:         self.count = module.count
26:         self.equal = module.equal
27:         self.filled = module.filled
28:         self.getmask = module.getmask
29:         self.getmaskarray = module.getmaskarray
30:         self.id = id
31:         self.inner = module.inner
32:         self.make_mask = module.make_mask
33:         self.masked = module.masked
34:         self.masked_array = module.masked_array
35:         self.masked_values = module.masked_values
36:         self.mask_or = module.mask_or
37:         self.nomask = module.nomask
38:         self.ones = module.ones
39:         self.outer = module.outer
40:         self.repeat = module.repeat
41:         self.resize = module.resize
42:         self.sort = module.sort
43:         self.take = module.take
44:         self.transpose = module.transpose
45:         self.zeros = module.zeros
46:         self.MaskType = module.MaskType
47:         try:
48:             self.umath = module.umath
49:         except AttributeError:
50:             self.umath = module.core.umath
51:         self.testnames = []
52: 
53:     def assert_array_compare(self, comparison, x, y, err_msg='', header='',
54:                          fill_value=True):
55:         '''
56:         Assert that a comparison of two masked arrays is satisfied elementwise.
57: 
58:         '''
59:         xf = self.filled(x)
60:         yf = self.filled(y)
61:         m = self.mask_or(self.getmask(x), self.getmask(y))
62: 
63:         x = self.filled(self.masked_array(xf, mask=m), fill_value)
64:         y = self.filled(self.masked_array(yf, mask=m), fill_value)
65:         if (x.dtype.char != "O"):
66:             x = x.astype(float_)
67:             if isinstance(x, np.ndarray) and x.size > 1:
68:                 x[np.isnan(x)] = 0
69:             elif np.isnan(x):
70:                 x = 0
71:         if (y.dtype.char != "O"):
72:             y = y.astype(float_)
73:             if isinstance(y, np.ndarray) and y.size > 1:
74:                 y[np.isnan(y)] = 0
75:             elif np.isnan(y):
76:                 y = 0
77:         try:
78:             cond = (x.shape == () or y.shape == ()) or x.shape == y.shape
79:             if not cond:
80:                 msg = build_err_msg([x, y],
81:                                     err_msg
82:                                     + '\n(shapes %s, %s mismatch)' % (x.shape,
83:                                                                       y.shape),
84:                                     header=header,
85:                                     names=('x', 'y'))
86:                 assert cond, msg
87:             val = comparison(x, y)
88:             if m is not self.nomask and fill_value:
89:                 val = self.masked_array(val, mask=m)
90:             if isinstance(val, bool):
91:                 cond = val
92:                 reduced = [0]
93:             else:
94:                 reduced = val.ravel()
95:                 cond = reduced.all()
96:                 reduced = reduced.tolist()
97:             if not cond:
98:                 match = 100-100.0*reduced.count(1)/len(reduced)
99:                 msg = build_err_msg([x, y],
100:                                     err_msg
101:                                     + '\n(mismatch %s%%)' % (match,),
102:                                     header=header,
103:                                     names=('x', 'y'))
104:                 assert cond, msg
105:         except ValueError:
106:             msg = build_err_msg([x, y], err_msg, header=header, names=('x', 'y'))
107:             raise ValueError(msg)
108: 
109:     def assert_array_equal(self, x, y, err_msg=''):
110:         '''
111:         Checks the elementwise equality of two masked arrays.
112: 
113:         '''
114:         self.assert_array_compare(self.equal, x, y, err_msg=err_msg,
115:                                   header='Arrays are not equal')
116: 
117:     def test_0(self):
118:         '''
119:         Tests creation
120: 
121:         '''
122:         x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
123:         m = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
124:         xm = self.masked_array(x, mask=m)
125:         xm[0]
126: 
127:     def test_1(self):
128:         '''
129:         Tests creation
130: 
131:         '''
132:         x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
133:         y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
134:         m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
135:         m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
136:         xm = self.masked_array(x, mask=m1)
137:         ym = self.masked_array(y, mask=m2)
138:         xf = np.where(m1, 1.e+20, x)
139:         xm.set_fill_value(1.e+20)
140: 
141:         assert((xm-ym).filled(0).any())
142:         s = x.shape
143:         assert(xm.size == reduce(lambda x, y:x*y, s))
144:         assert(self.count(xm) == len(m1) - reduce(lambda x, y:x+y, m1))
145: 
146:         for s in [(4, 3), (6, 2)]:
147:             x.shape = s
148:             y.shape = s
149:             xm.shape = s
150:             ym.shape = s
151:             xf.shape = s
152:             assert(self.count(xm) == len(m1) - reduce(lambda x, y:x+y, m1))
153: 
154:     def test_2(self):
155:         '''
156:         Tests conversions and indexing.
157: 
158:         '''
159:         x1 = np.array([1, 2, 4, 3])
160:         x2 = self.array(x1, mask=[1, 0, 0, 0])
161:         x3 = self.array(x1, mask=[0, 1, 0, 1])
162:         x4 = self.array(x1)
163:         # test conversion to strings, no errors
164:         str(x2)
165:         repr(x2)
166:         # tests of indexing
167:         assert type(x2[1]) is type(x1[1])
168:         assert x1[1] == x2[1]
169:         x1[2] = 9
170:         x2[2] = 9
171:         self.assert_array_equal(x1, x2)
172:         x1[1:3] = 99
173:         x2[1:3] = 99
174:         x2[1] = self.masked
175:         x2[1:3] = self.masked
176:         x2[:] = x1
177:         x2[1] = self.masked
178:         x3[:] = self.masked_array([1, 2, 3, 4], [0, 1, 1, 0])
179:         x4[:] = self.masked_array([1, 2, 3, 4], [0, 1, 1, 0])
180:         x1 = np.arange(5)*1.0
181:         x2 = self.masked_values(x1, 3.0)
182:         x1 = self.array([1, 'hello', 2, 3], object)
183:         x2 = np.array([1, 'hello', 2, 3], object)
184:         # check that no error occurs.
185:         x1[1]
186:         x2[1]
187:         assert x1[1:1].shape == (0,)
188:         # Tests copy-size
189:         n = [0, 0, 1, 0, 0]
190:         m = self.make_mask(n)
191:         m2 = self.make_mask(m)
192:         assert(m is m2)
193:         m3 = self.make_mask(m, copy=1)
194:         assert(m is not m3)
195: 
196:     def test_3(self):
197:         '''
198:         Tests resize/repeat
199: 
200:         '''
201:         x4 = self.arange(4)
202:         x4[2] = self.masked
203:         y4 = self.resize(x4, (8,))
204:         assert self.allequal(self.concatenate([x4, x4]), y4)
205:         assert self.allequal(self.getmask(y4), [0, 0, 1, 0, 0, 0, 1, 0])
206:         y5 = self.repeat(x4, (2, 2, 2, 2), axis=0)
207:         self.assert_array_equal(y5, [0, 0, 1, 1, 2, 2, 3, 3])
208:         y6 = self.repeat(x4, 2, axis=0)
209:         assert self.allequal(y5, y6)
210:         y7 = x4.repeat((2, 2, 2, 2), axis=0)
211:         assert self.allequal(y5, y7)
212:         y8 = x4.repeat(2, 0)
213:         assert self.allequal(y5, y8)
214: 
215:     def test_4(self):
216:         '''
217:         Test of take, transpose, inner, outer products.
218: 
219:         '''
220:         x = self.arange(24)
221:         y = np.arange(24)
222:         x[5:6] = self.masked
223:         x = x.reshape(2, 3, 4)
224:         y = y.reshape(2, 3, 4)
225:         assert self.allequal(np.transpose(y, (2, 0, 1)), self.transpose(x, (2, 0, 1)))
226:         assert self.allequal(np.take(y, (2, 0, 1), 1), self.take(x, (2, 0, 1), 1))
227:         assert self.allequal(np.inner(self.filled(x, 0), self.filled(y, 0)),
228:                             self.inner(x, y))
229:         assert self.allequal(np.outer(self.filled(x, 0), self.filled(y, 0)),
230:                             self.outer(x, y))
231:         y = self.array(['abc', 1, 'def', 2, 3], object)
232:         y[2] = self.masked
233:         t = self.take(y, [0, 3, 4])
234:         assert t[0] == 'abc'
235:         assert t[1] == 2
236:         assert t[2] == 3
237: 
238:     def test_5(self):
239:         '''
240:         Tests inplace w/ scalar
241: 
242:         '''
243:         x = self.arange(10)
244:         y = self.arange(10)
245:         xm = self.arange(10)
246:         xm[2] = self.masked
247:         x += 1
248:         assert self.allequal(x, y+1)
249:         xm += 1
250:         assert self.allequal(xm, y+1)
251: 
252:         x = self.arange(10)
253:         xm = self.arange(10)
254:         xm[2] = self.masked
255:         x -= 1
256:         assert self.allequal(x, y-1)
257:         xm -= 1
258:         assert self.allequal(xm, y-1)
259: 
260:         x = self.arange(10)*1.0
261:         xm = self.arange(10)*1.0
262:         xm[2] = self.masked
263:         x *= 2.0
264:         assert self.allequal(x, y*2)
265:         xm *= 2.0
266:         assert self.allequal(xm, y*2)
267: 
268:         x = self.arange(10)*2
269:         xm = self.arange(10)*2
270:         xm[2] = self.masked
271:         x /= 2
272:         assert self.allequal(x, y)
273:         xm /= 2
274:         assert self.allequal(xm, y)
275: 
276:         x = self.arange(10)*1.0
277:         xm = self.arange(10)*1.0
278:         xm[2] = self.masked
279:         x /= 2.0
280:         assert self.allequal(x, y/2.0)
281:         xm /= self.arange(10)
282:         self.assert_array_equal(xm, self.ones((10,)))
283: 
284:         x = self.arange(10).astype(float_)
285:         xm = self.arange(10)
286:         xm[2] = self.masked
287:         x += 1.
288:         assert self.allequal(x, y + 1.)
289: 
290:     def test_6(self):
291:         '''
292:         Tests inplace w/ array
293: 
294:         '''
295:         x = self.arange(10, dtype=float_)
296:         y = self.arange(10)
297:         xm = self.arange(10, dtype=float_)
298:         xm[2] = self.masked
299:         m = xm.mask
300:         a = self.arange(10, dtype=float_)
301:         a[-1] = self.masked
302:         x += a
303:         xm += a
304:         assert self.allequal(x, y+a)
305:         assert self.allequal(xm, y+a)
306:         assert self.allequal(xm.mask, self.mask_or(m, a.mask))
307: 
308:         x = self.arange(10, dtype=float_)
309:         xm = self.arange(10, dtype=float_)
310:         xm[2] = self.masked
311:         m = xm.mask
312:         a = self.arange(10, dtype=float_)
313:         a[-1] = self.masked
314:         x -= a
315:         xm -= a
316:         assert self.allequal(x, y-a)
317:         assert self.allequal(xm, y-a)
318:         assert self.allequal(xm.mask, self.mask_or(m, a.mask))
319: 
320:         x = self.arange(10, dtype=float_)
321:         xm = self.arange(10, dtype=float_)
322:         xm[2] = self.masked
323:         m = xm.mask
324:         a = self.arange(10, dtype=float_)
325:         a[-1] = self.masked
326:         x *= a
327:         xm *= a
328:         assert self.allequal(x, y*a)
329:         assert self.allequal(xm, y*a)
330:         assert self.allequal(xm.mask, self.mask_or(m, a.mask))
331: 
332:         x = self.arange(10, dtype=float_)
333:         xm = self.arange(10, dtype=float_)
334:         xm[2] = self.masked
335:         m = xm.mask
336:         a = self.arange(10, dtype=float_)
337:         a[-1] = self.masked
338:         x /= a
339:         xm /= a
340: 
341:     def test_7(self):
342:         "Tests ufunc"
343:         d = (self.array([1.0, 0, -1, pi/2]*2, mask=[0, 1]+[0]*6),
344:              self.array([1.0, 0, -1, pi/2]*2, mask=[1, 0]+[0]*6),)
345:         for f in ['sqrt', 'log', 'log10', 'exp', 'conjugate',
346: #                  'sin', 'cos', 'tan',
347: #                  'arcsin', 'arccos', 'arctan',
348: #                  'sinh', 'cosh', 'tanh',
349: #                  'arcsinh',
350: #                  'arccosh',
351: #                  'arctanh',
352: #                  'absolute', 'fabs', 'negative',
353: #                  # 'nonzero', 'around',
354: #                  'floor', 'ceil',
355: #                  # 'sometrue', 'alltrue',
356: #                  'logical_not',
357: #                  'add', 'subtract', 'multiply',
358: #                  'divide', 'true_divide', 'floor_divide',
359: #                  'remainder', 'fmod', 'hypot', 'arctan2',
360: #                  'equal', 'not_equal', 'less_equal', 'greater_equal',
361: #                  'less', 'greater',
362: #                  'logical_and', 'logical_or', 'logical_xor',
363:                   ]:
364:             try:
365:                 uf = getattr(self.umath, f)
366:             except AttributeError:
367:                 uf = getattr(fromnumeric, f)
368:             mf = getattr(self.module, f)
369:             args = d[:uf.nin]
370:             ur = uf(*args)
371:             mr = mf(*args)
372:             self.assert_array_equal(ur.filled(0), mr.filled(0), f)
373:             self.assert_array_equal(ur._mask, mr._mask)
374: 
375:     def test_99(self):
376:         # test average
377:         ott = self.array([0., 1., 2., 3.], mask=[1, 0, 0, 0])
378:         self.assert_array_equal(2.0, self.average(ott, axis=0))
379:         self.assert_array_equal(2.0, self.average(ott, weights=[1., 1., 2., 1.]))
380:         result, wts = self.average(ott, weights=[1., 1., 2., 1.], returned=1)
381:         self.assert_array_equal(2.0, result)
382:         assert(wts == 4.0)
383:         ott[:] = self.masked
384:         assert(self.average(ott, axis=0) is self.masked)
385:         ott = self.array([0., 1., 2., 3.], mask=[1, 0, 0, 0])
386:         ott = ott.reshape(2, 2)
387:         ott[:, 1] = self.masked
388:         self.assert_array_equal(self.average(ott, axis=0), [2.0, 0.0])
389:         assert(self.average(ott, axis=1)[0] is self.masked)
390:         self.assert_array_equal([2., 0.], self.average(ott, axis=0))
391:         result, wts = self.average(ott, axis=0, returned=1)
392:         self.assert_array_equal(wts, [1., 0.])
393:         w1 = [0, 1, 1, 1, 1, 0]
394:         w2 = [[0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1]]
395:         x = self.arange(6)
396:         self.assert_array_equal(self.average(x, axis=0), 2.5)
397:         self.assert_array_equal(self.average(x, axis=0, weights=w1), 2.5)
398:         y = self.array([self.arange(6), 2.0*self.arange(6)])
399:         self.assert_array_equal(self.average(y, None), np.add.reduce(np.arange(6))*3./12.)
400:         self.assert_array_equal(self.average(y, axis=0), np.arange(6) * 3./2.)
401:         self.assert_array_equal(self.average(y, axis=1), [self.average(x, axis=0), self.average(x, axis=0) * 2.0])
402:         self.assert_array_equal(self.average(y, None, weights=w2), 20./6.)
403:         self.assert_array_equal(self.average(y, axis=0, weights=w2), [0., 1., 2., 3., 4., 10.])
404:         self.assert_array_equal(self.average(y, axis=1), [self.average(x, axis=0), self.average(x, axis=0) * 2.0])
405:         m1 = self.zeros(6)
406:         m2 = [0, 0, 1, 1, 0, 0]
407:         m3 = [[0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0]]
408:         m4 = self.ones(6)
409:         m5 = [0, 1, 1, 1, 1, 1]
410:         self.assert_array_equal(self.average(self.masked_array(x, m1), axis=0), 2.5)
411:         self.assert_array_equal(self.average(self.masked_array(x, m2), axis=0), 2.5)
412:         self.assert_array_equal(self.average(self.masked_array(x, m5), axis=0), 0.0)
413:         self.assert_array_equal(self.count(self.average(self.masked_array(x, m4), axis=0)), 0)
414:         z = self.masked_array(y, m3)
415:         self.assert_array_equal(self.average(z, None), 20./6.)
416:         self.assert_array_equal(self.average(z, axis=0), [0., 1., 99., 99., 4.0, 7.5])
417:         self.assert_array_equal(self.average(z, axis=1), [2.5, 5.0])
418:         self.assert_array_equal(self.average(z, axis=0, weights=w2), [0., 1., 99., 99., 4.0, 10.0])
419: 
420:     def test_A(self):
421:         x = self.arange(24)
422:         x[5:6] = self.masked
423:         x = x.reshape(2, 3, 4)
424: 
425: 
426: if __name__ == '__main__':
427:     setup_base = ("from __main__ import ModuleTester \n"
428:                   "import numpy\n"
429:                   "tester = ModuleTester(module)\n")
430:     setup_cur = "import numpy.ma.core as module\n" + setup_base
431:     (nrepeat, nloop) = (10, 10)
432: 
433:     if 1:
434:         for i in range(1, 8):
435:             func = 'tester.test_%i()' % i
436:             cur = timeit.Timer(func, setup_cur).repeat(nrepeat, nloop*10)
437:             cur = np.sort(cur)
438:             print("#%i" % i + 50*'.')
439:             print(eval("ModuleTester.test_%i.__doc__" % i))
440:             print("core_current : %.3f - %.3f" % (cur[0], cur[1]))
441: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import timeit' statement (line 3)
import timeit

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'timeit', timeit, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from functools import reduce' statement (line 4)
from functools import reduce

import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'functools', None, module_type_store, ['reduce'], [reduce])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_158057 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_158057) is not StypyTypeError):

    if (import_158057 != 'pyd_module'):
        __import__(import_158057)
        sys_modules_158058 = sys.modules[import_158057]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_158058.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_158057)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy import float_' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_158059 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_158059) is not StypyTypeError):

    if (import_158059 != 'pyd_module'):
        __import__(import_158059)
        sys_modules_158060 = sys.modules[import_158059]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', sys_modules_158060.module_type_store, module_type_store, ['float_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_158060, sys_modules_158060.module_type_store, module_type_store)
    else:
        from numpy import float_

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', None, module_type_store, ['float_'], [float_])

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_158059)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy.core.fromnumeric' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_158061 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core.fromnumeric')

if (type(import_158061) is not StypyTypeError):

    if (import_158061 != 'pyd_module'):
        __import__(import_158061)
        sys_modules_158062 = sys.modules[import_158061]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'fromnumeric', sys_modules_158062.module_type_store, module_type_store)
    else:
        import numpy.core.fromnumeric as fromnumeric

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'fromnumeric', numpy.core.fromnumeric, module_type_store)

else:
    # Assigning a type to the variable 'numpy.core.fromnumeric' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core.fromnumeric', import_158061)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy.testing.utils import build_err_msg' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/ma/')
import_158063 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing.utils')

if (type(import_158063) is not StypyTypeError):

    if (import_158063 != 'pyd_module'):
        __import__(import_158063)
        sys_modules_158064 = sys.modules[import_158063]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing.utils', sys_modules_158064.module_type_store, module_type_store, ['build_err_msg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_158064, sys_modules_158064.module_type_store, module_type_store)
    else:
        from numpy.testing.utils import build_err_msg

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing.utils', None, module_type_store, ['build_err_msg'], [build_err_msg])

else:
    # Assigning a type to the variable 'numpy.testing.utils' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.testing.utils', import_158063)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/ma/')


# Call to seterr(...): (line 13)
# Processing the call keyword arguments (line 13)
str_158067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 14), 'str', 'ignore')
keyword_158068 = str_158067
kwargs_158069 = {'all': keyword_158068}
# Getting the type of 'np' (line 13)
np_158065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'np', False)
# Obtaining the member 'seterr' of a type (line 13)
seterr_158066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 0), np_158065, 'seterr')
# Calling seterr(args, kwargs) (line 13)
seterr_call_result_158070 = invoke(stypy.reporting.localization.Localization(__file__, 13, 0), seterr_158066, *[], **kwargs_158069)


# Assigning a Attribute to a Name (line 15):

# Assigning a Attribute to a Name (line 15):
# Getting the type of 'np' (line 15)
np_158071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'np')
# Obtaining the member 'pi' of a type (line 15)
pi_158072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), np_158071, 'pi')
# Assigning a type to the variable 'pi' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'pi', pi_158072)
# Declaration of the 'ModuleTester' class

class ModuleTester(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ModuleTester.__init__', ['module'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['module'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 20):
        
        # Assigning a Name to a Attribute (line 20):
        # Getting the type of 'module' (line 20)
        module_158073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'module')
        # Getting the type of 'self' (line 20)
        self_158074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self')
        # Setting the type of the member 'module' of a type (line 20)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), self_158074, 'module', module_158073)
        
        # Assigning a Attribute to a Attribute (line 21):
        
        # Assigning a Attribute to a Attribute (line 21):
        # Getting the type of 'module' (line 21)
        module_158075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 24), 'module')
        # Obtaining the member 'allequal' of a type (line 21)
        allequal_158076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 24), module_158075, 'allequal')
        # Getting the type of 'self' (line 21)
        self_158077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self')
        # Setting the type of the member 'allequal' of a type (line 21)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), self_158077, 'allequal', allequal_158076)
        
        # Assigning a Attribute to a Attribute (line 22):
        
        # Assigning a Attribute to a Attribute (line 22):
        # Getting the type of 'module' (line 22)
        module_158078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 22), 'module')
        # Obtaining the member 'arange' of a type (line 22)
        arange_158079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 22), module_158078, 'arange')
        # Getting the type of 'self' (line 22)
        self_158080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Setting the type of the member 'arange' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_158080, 'arange', arange_158079)
        
        # Assigning a Attribute to a Attribute (line 23):
        
        # Assigning a Attribute to a Attribute (line 23):
        # Getting the type of 'module' (line 23)
        module_158081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 21), 'module')
        # Obtaining the member 'array' of a type (line 23)
        array_158082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 21), module_158081, 'array')
        # Getting the type of 'self' (line 23)
        self_158083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self')
        # Setting the type of the member 'array' of a type (line 23)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_158083, 'array', array_158082)
        
        # Assigning a Attribute to a Attribute (line 24):
        
        # Assigning a Attribute to a Attribute (line 24):
        # Getting the type of 'module' (line 24)
        module_158084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 27), 'module')
        # Obtaining the member 'concatenate' of a type (line 24)
        concatenate_158085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 27), module_158084, 'concatenate')
        # Getting the type of 'self' (line 24)
        self_158086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self')
        # Setting the type of the member 'concatenate' of a type (line 24)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_158086, 'concatenate', concatenate_158085)
        
        # Assigning a Attribute to a Attribute (line 25):
        
        # Assigning a Attribute to a Attribute (line 25):
        # Getting the type of 'module' (line 25)
        module_158087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 21), 'module')
        # Obtaining the member 'count' of a type (line 25)
        count_158088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 21), module_158087, 'count')
        # Getting the type of 'self' (line 25)
        self_158089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member 'count' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_158089, 'count', count_158088)
        
        # Assigning a Attribute to a Attribute (line 26):
        
        # Assigning a Attribute to a Attribute (line 26):
        # Getting the type of 'module' (line 26)
        module_158090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 21), 'module')
        # Obtaining the member 'equal' of a type (line 26)
        equal_158091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 21), module_158090, 'equal')
        # Getting the type of 'self' (line 26)
        self_158092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        # Setting the type of the member 'equal' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_158092, 'equal', equal_158091)
        
        # Assigning a Attribute to a Attribute (line 27):
        
        # Assigning a Attribute to a Attribute (line 27):
        # Getting the type of 'module' (line 27)
        module_158093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 22), 'module')
        # Obtaining the member 'filled' of a type (line 27)
        filled_158094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 22), module_158093, 'filled')
        # Getting the type of 'self' (line 27)
        self_158095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self')
        # Setting the type of the member 'filled' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_158095, 'filled', filled_158094)
        
        # Assigning a Attribute to a Attribute (line 28):
        
        # Assigning a Attribute to a Attribute (line 28):
        # Getting the type of 'module' (line 28)
        module_158096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 'module')
        # Obtaining the member 'getmask' of a type (line 28)
        getmask_158097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 23), module_158096, 'getmask')
        # Getting the type of 'self' (line 28)
        self_158098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member 'getmask' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_158098, 'getmask', getmask_158097)
        
        # Assigning a Attribute to a Attribute (line 29):
        
        # Assigning a Attribute to a Attribute (line 29):
        # Getting the type of 'module' (line 29)
        module_158099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 28), 'module')
        # Obtaining the member 'getmaskarray' of a type (line 29)
        getmaskarray_158100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 28), module_158099, 'getmaskarray')
        # Getting the type of 'self' (line 29)
        self_158101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'getmaskarray' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_158101, 'getmaskarray', getmaskarray_158100)
        
        # Assigning a Name to a Attribute (line 30):
        
        # Assigning a Name to a Attribute (line 30):
        # Getting the type of 'id' (line 30)
        id_158102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 18), 'id')
        # Getting the type of 'self' (line 30)
        self_158103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'id' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_158103, 'id', id_158102)
        
        # Assigning a Attribute to a Attribute (line 31):
        
        # Assigning a Attribute to a Attribute (line 31):
        # Getting the type of 'module' (line 31)
        module_158104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 21), 'module')
        # Obtaining the member 'inner' of a type (line 31)
        inner_158105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 21), module_158104, 'inner')
        # Getting the type of 'self' (line 31)
        self_158106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'inner' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_158106, 'inner', inner_158105)
        
        # Assigning a Attribute to a Attribute (line 32):
        
        # Assigning a Attribute to a Attribute (line 32):
        # Getting the type of 'module' (line 32)
        module_158107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 25), 'module')
        # Obtaining the member 'make_mask' of a type (line 32)
        make_mask_158108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 25), module_158107, 'make_mask')
        # Getting the type of 'self' (line 32)
        self_158109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member 'make_mask' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_158109, 'make_mask', make_mask_158108)
        
        # Assigning a Attribute to a Attribute (line 33):
        
        # Assigning a Attribute to a Attribute (line 33):
        # Getting the type of 'module' (line 33)
        module_158110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 22), 'module')
        # Obtaining the member 'masked' of a type (line 33)
        masked_158111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 22), module_158110, 'masked')
        # Getting the type of 'self' (line 33)
        self_158112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member 'masked' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_158112, 'masked', masked_158111)
        
        # Assigning a Attribute to a Attribute (line 34):
        
        # Assigning a Attribute to a Attribute (line 34):
        # Getting the type of 'module' (line 34)
        module_158113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 28), 'module')
        # Obtaining the member 'masked_array' of a type (line 34)
        masked_array_158114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 28), module_158113, 'masked_array')
        # Getting the type of 'self' (line 34)
        self_158115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'masked_array' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_158115, 'masked_array', masked_array_158114)
        
        # Assigning a Attribute to a Attribute (line 35):
        
        # Assigning a Attribute to a Attribute (line 35):
        # Getting the type of 'module' (line 35)
        module_158116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 29), 'module')
        # Obtaining the member 'masked_values' of a type (line 35)
        masked_values_158117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 29), module_158116, 'masked_values')
        # Getting the type of 'self' (line 35)
        self_158118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self')
        # Setting the type of the member 'masked_values' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_158118, 'masked_values', masked_values_158117)
        
        # Assigning a Attribute to a Attribute (line 36):
        
        # Assigning a Attribute to a Attribute (line 36):
        # Getting the type of 'module' (line 36)
        module_158119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'module')
        # Obtaining the member 'mask_or' of a type (line 36)
        mask_or_158120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 23), module_158119, 'mask_or')
        # Getting the type of 'self' (line 36)
        self_158121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self')
        # Setting the type of the member 'mask_or' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_158121, 'mask_or', mask_or_158120)
        
        # Assigning a Attribute to a Attribute (line 37):
        
        # Assigning a Attribute to a Attribute (line 37):
        # Getting the type of 'module' (line 37)
        module_158122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 22), 'module')
        # Obtaining the member 'nomask' of a type (line 37)
        nomask_158123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 22), module_158122, 'nomask')
        # Getting the type of 'self' (line 37)
        self_158124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self')
        # Setting the type of the member 'nomask' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_158124, 'nomask', nomask_158123)
        
        # Assigning a Attribute to a Attribute (line 38):
        
        # Assigning a Attribute to a Attribute (line 38):
        # Getting the type of 'module' (line 38)
        module_158125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 20), 'module')
        # Obtaining the member 'ones' of a type (line 38)
        ones_158126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 20), module_158125, 'ones')
        # Getting the type of 'self' (line 38)
        self_158127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member 'ones' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_158127, 'ones', ones_158126)
        
        # Assigning a Attribute to a Attribute (line 39):
        
        # Assigning a Attribute to a Attribute (line 39):
        # Getting the type of 'module' (line 39)
        module_158128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 21), 'module')
        # Obtaining the member 'outer' of a type (line 39)
        outer_158129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 21), module_158128, 'outer')
        # Getting the type of 'self' (line 39)
        self_158130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self')
        # Setting the type of the member 'outer' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_158130, 'outer', outer_158129)
        
        # Assigning a Attribute to a Attribute (line 40):
        
        # Assigning a Attribute to a Attribute (line 40):
        # Getting the type of 'module' (line 40)
        module_158131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'module')
        # Obtaining the member 'repeat' of a type (line 40)
        repeat_158132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 22), module_158131, 'repeat')
        # Getting the type of 'self' (line 40)
        self_158133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self')
        # Setting the type of the member 'repeat' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_158133, 'repeat', repeat_158132)
        
        # Assigning a Attribute to a Attribute (line 41):
        
        # Assigning a Attribute to a Attribute (line 41):
        # Getting the type of 'module' (line 41)
        module_158134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'module')
        # Obtaining the member 'resize' of a type (line 41)
        resize_158135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 22), module_158134, 'resize')
        # Getting the type of 'self' (line 41)
        self_158136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Setting the type of the member 'resize' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_158136, 'resize', resize_158135)
        
        # Assigning a Attribute to a Attribute (line 42):
        
        # Assigning a Attribute to a Attribute (line 42):
        # Getting the type of 'module' (line 42)
        module_158137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'module')
        # Obtaining the member 'sort' of a type (line 42)
        sort_158138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 20), module_158137, 'sort')
        # Getting the type of 'self' (line 42)
        self_158139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self')
        # Setting the type of the member 'sort' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_158139, 'sort', sort_158138)
        
        # Assigning a Attribute to a Attribute (line 43):
        
        # Assigning a Attribute to a Attribute (line 43):
        # Getting the type of 'module' (line 43)
        module_158140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'module')
        # Obtaining the member 'take' of a type (line 43)
        take_158141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 20), module_158140, 'take')
        # Getting the type of 'self' (line 43)
        self_158142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member 'take' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_158142, 'take', take_158141)
        
        # Assigning a Attribute to a Attribute (line 44):
        
        # Assigning a Attribute to a Attribute (line 44):
        # Getting the type of 'module' (line 44)
        module_158143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 25), 'module')
        # Obtaining the member 'transpose' of a type (line 44)
        transpose_158144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 25), module_158143, 'transpose')
        # Getting the type of 'self' (line 44)
        self_158145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'self')
        # Setting the type of the member 'transpose' of a type (line 44)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), self_158145, 'transpose', transpose_158144)
        
        # Assigning a Attribute to a Attribute (line 45):
        
        # Assigning a Attribute to a Attribute (line 45):
        # Getting the type of 'module' (line 45)
        module_158146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 21), 'module')
        # Obtaining the member 'zeros' of a type (line 45)
        zeros_158147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 21), module_158146, 'zeros')
        # Getting the type of 'self' (line 45)
        self_158148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self')
        # Setting the type of the member 'zeros' of a type (line 45)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_158148, 'zeros', zeros_158147)
        
        # Assigning a Attribute to a Attribute (line 46):
        
        # Assigning a Attribute to a Attribute (line 46):
        # Getting the type of 'module' (line 46)
        module_158149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 24), 'module')
        # Obtaining the member 'MaskType' of a type (line 46)
        MaskType_158150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 24), module_158149, 'MaskType')
        # Getting the type of 'self' (line 46)
        self_158151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self')
        # Setting the type of the member 'MaskType' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_158151, 'MaskType', MaskType_158150)
        
        
        # SSA begins for try-except statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Attribute to a Attribute (line 48):
        
        # Assigning a Attribute to a Attribute (line 48):
        # Getting the type of 'module' (line 48)
        module_158152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'module')
        # Obtaining the member 'umath' of a type (line 48)
        umath_158153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 25), module_158152, 'umath')
        # Getting the type of 'self' (line 48)
        self_158154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'self')
        # Setting the type of the member 'umath' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), self_158154, 'umath', umath_158153)
        # SSA branch for the except part of a try statement (line 47)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 47)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Attribute to a Attribute (line 50):
        
        # Assigning a Attribute to a Attribute (line 50):
        # Getting the type of 'module' (line 50)
        module_158155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 25), 'module')
        # Obtaining the member 'core' of a type (line 50)
        core_158156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 25), module_158155, 'core')
        # Obtaining the member 'umath' of a type (line 50)
        umath_158157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 25), core_158156, 'umath')
        # Getting the type of 'self' (line 50)
        self_158158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'self')
        # Setting the type of the member 'umath' of a type (line 50)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), self_158158, 'umath', umath_158157)
        # SSA join for try-except statement (line 47)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Attribute (line 51):
        
        # Assigning a List to a Attribute (line 51):
        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_158159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        
        # Getting the type of 'self' (line 51)
        self_158160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self')
        # Setting the type of the member 'testnames' of a type (line 51)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_158160, 'testnames', list_158159)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def assert_array_compare(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_158161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 61), 'str', '')
        str_158162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 72), 'str', '')
        # Getting the type of 'True' (line 54)
        True_158163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 36), 'True')
        defaults = [str_158161, str_158162, True_158163]
        # Create a new context for function 'assert_array_compare'
        module_type_store = module_type_store.open_function_context('assert_array_compare', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ModuleTester.assert_array_compare.__dict__.__setitem__('stypy_localization', localization)
        ModuleTester.assert_array_compare.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ModuleTester.assert_array_compare.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleTester.assert_array_compare.__dict__.__setitem__('stypy_function_name', 'ModuleTester.assert_array_compare')
        ModuleTester.assert_array_compare.__dict__.__setitem__('stypy_param_names_list', ['comparison', 'x', 'y', 'err_msg', 'header', 'fill_value'])
        ModuleTester.assert_array_compare.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleTester.assert_array_compare.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleTester.assert_array_compare.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleTester.assert_array_compare.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleTester.assert_array_compare.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleTester.assert_array_compare.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ModuleTester.assert_array_compare', ['comparison', 'x', 'y', 'err_msg', 'header', 'fill_value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assert_array_compare', localization, ['comparison', 'x', 'y', 'err_msg', 'header', 'fill_value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assert_array_compare(...)' code ##################

        str_158164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, (-1)), 'str', '\n        Assert that a comparison of two masked arrays is satisfied elementwise.\n\n        ')
        
        # Assigning a Call to a Name (line 59):
        
        # Assigning a Call to a Name (line 59):
        
        # Call to filled(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'x' (line 59)
        x_158167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'x', False)
        # Processing the call keyword arguments (line 59)
        kwargs_158168 = {}
        # Getting the type of 'self' (line 59)
        self_158165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 13), 'self', False)
        # Obtaining the member 'filled' of a type (line 59)
        filled_158166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 13), self_158165, 'filled')
        # Calling filled(args, kwargs) (line 59)
        filled_call_result_158169 = invoke(stypy.reporting.localization.Localization(__file__, 59, 13), filled_158166, *[x_158167], **kwargs_158168)
        
        # Assigning a type to the variable 'xf' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'xf', filled_call_result_158169)
        
        # Assigning a Call to a Name (line 60):
        
        # Assigning a Call to a Name (line 60):
        
        # Call to filled(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'y' (line 60)
        y_158172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'y', False)
        # Processing the call keyword arguments (line 60)
        kwargs_158173 = {}
        # Getting the type of 'self' (line 60)
        self_158170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 13), 'self', False)
        # Obtaining the member 'filled' of a type (line 60)
        filled_158171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 13), self_158170, 'filled')
        # Calling filled(args, kwargs) (line 60)
        filled_call_result_158174 = invoke(stypy.reporting.localization.Localization(__file__, 60, 13), filled_158171, *[y_158172], **kwargs_158173)
        
        # Assigning a type to the variable 'yf' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'yf', filled_call_result_158174)
        
        # Assigning a Call to a Name (line 61):
        
        # Assigning a Call to a Name (line 61):
        
        # Call to mask_or(...): (line 61)
        # Processing the call arguments (line 61)
        
        # Call to getmask(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'x' (line 61)
        x_158179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 38), 'x', False)
        # Processing the call keyword arguments (line 61)
        kwargs_158180 = {}
        # Getting the type of 'self' (line 61)
        self_158177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'self', False)
        # Obtaining the member 'getmask' of a type (line 61)
        getmask_158178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 25), self_158177, 'getmask')
        # Calling getmask(args, kwargs) (line 61)
        getmask_call_result_158181 = invoke(stypy.reporting.localization.Localization(__file__, 61, 25), getmask_158178, *[x_158179], **kwargs_158180)
        
        
        # Call to getmask(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'y' (line 61)
        y_158184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 55), 'y', False)
        # Processing the call keyword arguments (line 61)
        kwargs_158185 = {}
        # Getting the type of 'self' (line 61)
        self_158182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 42), 'self', False)
        # Obtaining the member 'getmask' of a type (line 61)
        getmask_158183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 42), self_158182, 'getmask')
        # Calling getmask(args, kwargs) (line 61)
        getmask_call_result_158186 = invoke(stypy.reporting.localization.Localization(__file__, 61, 42), getmask_158183, *[y_158184], **kwargs_158185)
        
        # Processing the call keyword arguments (line 61)
        kwargs_158187 = {}
        # Getting the type of 'self' (line 61)
        self_158175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'self', False)
        # Obtaining the member 'mask_or' of a type (line 61)
        mask_or_158176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), self_158175, 'mask_or')
        # Calling mask_or(args, kwargs) (line 61)
        mask_or_call_result_158188 = invoke(stypy.reporting.localization.Localization(__file__, 61, 12), mask_or_158176, *[getmask_call_result_158181, getmask_call_result_158186], **kwargs_158187)
        
        # Assigning a type to the variable 'm' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'm', mask_or_call_result_158188)
        
        # Assigning a Call to a Name (line 63):
        
        # Assigning a Call to a Name (line 63):
        
        # Call to filled(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Call to masked_array(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'xf' (line 63)
        xf_158193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 42), 'xf', False)
        # Processing the call keyword arguments (line 63)
        # Getting the type of 'm' (line 63)
        m_158194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 51), 'm', False)
        keyword_158195 = m_158194
        kwargs_158196 = {'mask': keyword_158195}
        # Getting the type of 'self' (line 63)
        self_158191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'self', False)
        # Obtaining the member 'masked_array' of a type (line 63)
        masked_array_158192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 24), self_158191, 'masked_array')
        # Calling masked_array(args, kwargs) (line 63)
        masked_array_call_result_158197 = invoke(stypy.reporting.localization.Localization(__file__, 63, 24), masked_array_158192, *[xf_158193], **kwargs_158196)
        
        # Getting the type of 'fill_value' (line 63)
        fill_value_158198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 55), 'fill_value', False)
        # Processing the call keyword arguments (line 63)
        kwargs_158199 = {}
        # Getting the type of 'self' (line 63)
        self_158189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'self', False)
        # Obtaining the member 'filled' of a type (line 63)
        filled_158190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), self_158189, 'filled')
        # Calling filled(args, kwargs) (line 63)
        filled_call_result_158200 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), filled_158190, *[masked_array_call_result_158197, fill_value_158198], **kwargs_158199)
        
        # Assigning a type to the variable 'x' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'x', filled_call_result_158200)
        
        # Assigning a Call to a Name (line 64):
        
        # Assigning a Call to a Name (line 64):
        
        # Call to filled(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to masked_array(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'yf' (line 64)
        yf_158205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 42), 'yf', False)
        # Processing the call keyword arguments (line 64)
        # Getting the type of 'm' (line 64)
        m_158206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 51), 'm', False)
        keyword_158207 = m_158206
        kwargs_158208 = {'mask': keyword_158207}
        # Getting the type of 'self' (line 64)
        self_158203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'self', False)
        # Obtaining the member 'masked_array' of a type (line 64)
        masked_array_158204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 24), self_158203, 'masked_array')
        # Calling masked_array(args, kwargs) (line 64)
        masked_array_call_result_158209 = invoke(stypy.reporting.localization.Localization(__file__, 64, 24), masked_array_158204, *[yf_158205], **kwargs_158208)
        
        # Getting the type of 'fill_value' (line 64)
        fill_value_158210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 55), 'fill_value', False)
        # Processing the call keyword arguments (line 64)
        kwargs_158211 = {}
        # Getting the type of 'self' (line 64)
        self_158201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'self', False)
        # Obtaining the member 'filled' of a type (line 64)
        filled_158202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), self_158201, 'filled')
        # Calling filled(args, kwargs) (line 64)
        filled_call_result_158212 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), filled_158202, *[masked_array_call_result_158209, fill_value_158210], **kwargs_158211)
        
        # Assigning a type to the variable 'y' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'y', filled_call_result_158212)
        
        
        # Getting the type of 'x' (line 65)
        x_158213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'x')
        # Obtaining the member 'dtype' of a type (line 65)
        dtype_158214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), x_158213, 'dtype')
        # Obtaining the member 'char' of a type (line 65)
        char_158215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), dtype_158214, 'char')
        str_158216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 28), 'str', 'O')
        # Applying the binary operator '!=' (line 65)
        result_ne_158217 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 12), '!=', char_158215, str_158216)
        
        # Testing the type of an if condition (line 65)
        if_condition_158218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 8), result_ne_158217)
        # Assigning a type to the variable 'if_condition_158218' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'if_condition_158218', if_condition_158218)
        # SSA begins for if statement (line 65)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 66):
        
        # Assigning a Call to a Name (line 66):
        
        # Call to astype(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'float_' (line 66)
        float__158221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'float_', False)
        # Processing the call keyword arguments (line 66)
        kwargs_158222 = {}
        # Getting the type of 'x' (line 66)
        x_158219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'x', False)
        # Obtaining the member 'astype' of a type (line 66)
        astype_158220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 16), x_158219, 'astype')
        # Calling astype(args, kwargs) (line 66)
        astype_call_result_158223 = invoke(stypy.reporting.localization.Localization(__file__, 66, 16), astype_158220, *[float__158221], **kwargs_158222)
        
        # Assigning a type to the variable 'x' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'x', astype_call_result_158223)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'x' (line 67)
        x_158225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 26), 'x', False)
        # Getting the type of 'np' (line 67)
        np_158226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 29), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 67)
        ndarray_158227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 29), np_158226, 'ndarray')
        # Processing the call keyword arguments (line 67)
        kwargs_158228 = {}
        # Getting the type of 'isinstance' (line 67)
        isinstance_158224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 67)
        isinstance_call_result_158229 = invoke(stypy.reporting.localization.Localization(__file__, 67, 15), isinstance_158224, *[x_158225, ndarray_158227], **kwargs_158228)
        
        
        # Getting the type of 'x' (line 67)
        x_158230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 45), 'x')
        # Obtaining the member 'size' of a type (line 67)
        size_158231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 45), x_158230, 'size')
        int_158232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 54), 'int')
        # Applying the binary operator '>' (line 67)
        result_gt_158233 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 45), '>', size_158231, int_158232)
        
        # Applying the binary operator 'and' (line 67)
        result_and_keyword_158234 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 15), 'and', isinstance_call_result_158229, result_gt_158233)
        
        # Testing the type of an if condition (line 67)
        if_condition_158235 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 12), result_and_keyword_158234)
        # Assigning a type to the variable 'if_condition_158235' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'if_condition_158235', if_condition_158235)
        # SSA begins for if statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Subscript (line 68):
        
        # Assigning a Num to a Subscript (line 68):
        int_158236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 33), 'int')
        # Getting the type of 'x' (line 68)
        x_158237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'x')
        
        # Call to isnan(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'x' (line 68)
        x_158240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 27), 'x', False)
        # Processing the call keyword arguments (line 68)
        kwargs_158241 = {}
        # Getting the type of 'np' (line 68)
        np_158238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 18), 'np', False)
        # Obtaining the member 'isnan' of a type (line 68)
        isnan_158239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 18), np_158238, 'isnan')
        # Calling isnan(args, kwargs) (line 68)
        isnan_call_result_158242 = invoke(stypy.reporting.localization.Localization(__file__, 68, 18), isnan_158239, *[x_158240], **kwargs_158241)
        
        # Storing an element on a container (line 68)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 16), x_158237, (isnan_call_result_158242, int_158236))
        # SSA branch for the else part of an if statement (line 67)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isnan(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'x' (line 69)
        x_158245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'x', False)
        # Processing the call keyword arguments (line 69)
        kwargs_158246 = {}
        # Getting the type of 'np' (line 69)
        np_158243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 'np', False)
        # Obtaining the member 'isnan' of a type (line 69)
        isnan_158244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 17), np_158243, 'isnan')
        # Calling isnan(args, kwargs) (line 69)
        isnan_call_result_158247 = invoke(stypy.reporting.localization.Localization(__file__, 69, 17), isnan_158244, *[x_158245], **kwargs_158246)
        
        # Testing the type of an if condition (line 69)
        if_condition_158248 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 17), isnan_call_result_158247)
        # Assigning a type to the variable 'if_condition_158248' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 'if_condition_158248', if_condition_158248)
        # SSA begins for if statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 70):
        
        # Assigning a Num to a Name (line 70):
        int_158249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 20), 'int')
        # Assigning a type to the variable 'x' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'x', int_158249)
        # SSA join for if statement (line 69)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 67)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 65)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'y' (line 71)
        y_158250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'y')
        # Obtaining the member 'dtype' of a type (line 71)
        dtype_158251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), y_158250, 'dtype')
        # Obtaining the member 'char' of a type (line 71)
        char_158252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), dtype_158251, 'char')
        str_158253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 28), 'str', 'O')
        # Applying the binary operator '!=' (line 71)
        result_ne_158254 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 12), '!=', char_158252, str_158253)
        
        # Testing the type of an if condition (line 71)
        if_condition_158255 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 8), result_ne_158254)
        # Assigning a type to the variable 'if_condition_158255' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'if_condition_158255', if_condition_158255)
        # SSA begins for if statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 72):
        
        # Assigning a Call to a Name (line 72):
        
        # Call to astype(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'float_' (line 72)
        float__158258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'float_', False)
        # Processing the call keyword arguments (line 72)
        kwargs_158259 = {}
        # Getting the type of 'y' (line 72)
        y_158256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'y', False)
        # Obtaining the member 'astype' of a type (line 72)
        astype_158257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 16), y_158256, 'astype')
        # Calling astype(args, kwargs) (line 72)
        astype_call_result_158260 = invoke(stypy.reporting.localization.Localization(__file__, 72, 16), astype_158257, *[float__158258], **kwargs_158259)
        
        # Assigning a type to the variable 'y' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'y', astype_call_result_158260)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'y' (line 73)
        y_158262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 26), 'y', False)
        # Getting the type of 'np' (line 73)
        np_158263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 29), 'np', False)
        # Obtaining the member 'ndarray' of a type (line 73)
        ndarray_158264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 29), np_158263, 'ndarray')
        # Processing the call keyword arguments (line 73)
        kwargs_158265 = {}
        # Getting the type of 'isinstance' (line 73)
        isinstance_158261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 73)
        isinstance_call_result_158266 = invoke(stypy.reporting.localization.Localization(__file__, 73, 15), isinstance_158261, *[y_158262, ndarray_158264], **kwargs_158265)
        
        
        # Getting the type of 'y' (line 73)
        y_158267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 45), 'y')
        # Obtaining the member 'size' of a type (line 73)
        size_158268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 45), y_158267, 'size')
        int_158269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 54), 'int')
        # Applying the binary operator '>' (line 73)
        result_gt_158270 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 45), '>', size_158268, int_158269)
        
        # Applying the binary operator 'and' (line 73)
        result_and_keyword_158271 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 15), 'and', isinstance_call_result_158266, result_gt_158270)
        
        # Testing the type of an if condition (line 73)
        if_condition_158272 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 12), result_and_keyword_158271)
        # Assigning a type to the variable 'if_condition_158272' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'if_condition_158272', if_condition_158272)
        # SSA begins for if statement (line 73)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Subscript (line 74):
        
        # Assigning a Num to a Subscript (line 74):
        int_158273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 33), 'int')
        # Getting the type of 'y' (line 74)
        y_158274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'y')
        
        # Call to isnan(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'y' (line 74)
        y_158277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 27), 'y', False)
        # Processing the call keyword arguments (line 74)
        kwargs_158278 = {}
        # Getting the type of 'np' (line 74)
        np_158275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'np', False)
        # Obtaining the member 'isnan' of a type (line 74)
        isnan_158276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 18), np_158275, 'isnan')
        # Calling isnan(args, kwargs) (line 74)
        isnan_call_result_158279 = invoke(stypy.reporting.localization.Localization(__file__, 74, 18), isnan_158276, *[y_158277], **kwargs_158278)
        
        # Storing an element on a container (line 74)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 16), y_158274, (isnan_call_result_158279, int_158273))
        # SSA branch for the else part of an if statement (line 73)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isnan(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'y' (line 75)
        y_158282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'y', False)
        # Processing the call keyword arguments (line 75)
        kwargs_158283 = {}
        # Getting the type of 'np' (line 75)
        np_158280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'np', False)
        # Obtaining the member 'isnan' of a type (line 75)
        isnan_158281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 17), np_158280, 'isnan')
        # Calling isnan(args, kwargs) (line 75)
        isnan_call_result_158284 = invoke(stypy.reporting.localization.Localization(__file__, 75, 17), isnan_158281, *[y_158282], **kwargs_158283)
        
        # Testing the type of an if condition (line 75)
        if_condition_158285 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 17), isnan_call_result_158284)
        # Assigning a type to the variable 'if_condition_158285' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 17), 'if_condition_158285', if_condition_158285)
        # SSA begins for if statement (line 75)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 76):
        
        # Assigning a Num to a Name (line 76):
        int_158286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 20), 'int')
        # Assigning a type to the variable 'y' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'y', int_158286)
        # SSA join for if statement (line 75)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 73)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 71)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 77)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a BoolOp to a Name (line 78):
        
        # Assigning a BoolOp to a Name (line 78):
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        
        # Getting the type of 'x' (line 78)
        x_158287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'x')
        # Obtaining the member 'shape' of a type (line 78)
        shape_158288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 20), x_158287, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 78)
        tuple_158289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 78)
        
        # Applying the binary operator '==' (line 78)
        result_eq_158290 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 20), '==', shape_158288, tuple_158289)
        
        
        # Getting the type of 'y' (line 78)
        y_158291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 37), 'y')
        # Obtaining the member 'shape' of a type (line 78)
        shape_158292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 37), y_158291, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 78)
        tuple_158293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 48), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 78)
        
        # Applying the binary operator '==' (line 78)
        result_eq_158294 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 37), '==', shape_158292, tuple_158293)
        
        # Applying the binary operator 'or' (line 78)
        result_or_keyword_158295 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 20), 'or', result_eq_158290, result_eq_158294)
        
        
        # Getting the type of 'x' (line 78)
        x_158296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 55), 'x')
        # Obtaining the member 'shape' of a type (line 78)
        shape_158297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 55), x_158296, 'shape')
        # Getting the type of 'y' (line 78)
        y_158298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 66), 'y')
        # Obtaining the member 'shape' of a type (line 78)
        shape_158299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 66), y_158298, 'shape')
        # Applying the binary operator '==' (line 78)
        result_eq_158300 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 55), '==', shape_158297, shape_158299)
        
        # Applying the binary operator 'or' (line 78)
        result_or_keyword_158301 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 19), 'or', result_or_keyword_158295, result_eq_158300)
        
        # Assigning a type to the variable 'cond' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'cond', result_or_keyword_158301)
        
        
        # Getting the type of 'cond' (line 79)
        cond_158302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 'cond')
        # Applying the 'not' unary operator (line 79)
        result_not__158303 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 15), 'not', cond_158302)
        
        # Testing the type of an if condition (line 79)
        if_condition_158304 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 12), result_not__158303)
        # Assigning a type to the variable 'if_condition_158304' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'if_condition_158304', if_condition_158304)
        # SSA begins for if statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 80):
        
        # Assigning a Call to a Name (line 80):
        
        # Call to build_err_msg(...): (line 80)
        # Processing the call arguments (line 80)
        
        # Obtaining an instance of the builtin type 'list' (line 80)
        list_158306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 80)
        # Adding element type (line 80)
        # Getting the type of 'x' (line 80)
        x_158307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 37), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 36), list_158306, x_158307)
        # Adding element type (line 80)
        # Getting the type of 'y' (line 80)
        y_158308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 40), 'y', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 36), list_158306, y_158308)
        
        # Getting the type of 'err_msg' (line 81)
        err_msg_158309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 36), 'err_msg', False)
        str_158310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 38), 'str', '\n(shapes %s, %s mismatch)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 82)
        tuple_158311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 70), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 82)
        # Adding element type (line 82)
        # Getting the type of 'x' (line 82)
        x_158312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 70), 'x', False)
        # Obtaining the member 'shape' of a type (line 82)
        shape_158313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 70), x_158312, 'shape')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 70), tuple_158311, shape_158313)
        # Adding element type (line 82)
        # Getting the type of 'y' (line 83)
        y_158314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 70), 'y', False)
        # Obtaining the member 'shape' of a type (line 83)
        shape_158315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 70), y_158314, 'shape')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 70), tuple_158311, shape_158315)
        
        # Applying the binary operator '%' (line 82)
        result_mod_158316 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 38), '%', str_158310, tuple_158311)
        
        # Applying the binary operator '+' (line 81)
        result_add_158317 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 36), '+', err_msg_158309, result_mod_158316)
        
        # Processing the call keyword arguments (line 80)
        # Getting the type of 'header' (line 84)
        header_158318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 43), 'header', False)
        keyword_158319 = header_158318
        
        # Obtaining an instance of the builtin type 'tuple' (line 85)
        tuple_158320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 85)
        # Adding element type (line 85)
        str_158321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 43), 'str', 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 43), tuple_158320, str_158321)
        # Adding element type (line 85)
        str_158322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 48), 'str', 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 43), tuple_158320, str_158322)
        
        keyword_158323 = tuple_158320
        kwargs_158324 = {'header': keyword_158319, 'names': keyword_158323}
        # Getting the type of 'build_err_msg' (line 80)
        build_err_msg_158305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 22), 'build_err_msg', False)
        # Calling build_err_msg(args, kwargs) (line 80)
        build_err_msg_call_result_158325 = invoke(stypy.reporting.localization.Localization(__file__, 80, 22), build_err_msg_158305, *[list_158306, result_add_158317], **kwargs_158324)
        
        # Assigning a type to the variable 'msg' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'msg', build_err_msg_call_result_158325)
        # Evaluating assert statement condition
        # Getting the type of 'cond' (line 86)
        cond_158326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 23), 'cond')
        # SSA join for if statement (line 79)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 87):
        
        # Assigning a Call to a Name (line 87):
        
        # Call to comparison(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'x' (line 87)
        x_158328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 29), 'x', False)
        # Getting the type of 'y' (line 87)
        y_158329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 32), 'y', False)
        # Processing the call keyword arguments (line 87)
        kwargs_158330 = {}
        # Getting the type of 'comparison' (line 87)
        comparison_158327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 18), 'comparison', False)
        # Calling comparison(args, kwargs) (line 87)
        comparison_call_result_158331 = invoke(stypy.reporting.localization.Localization(__file__, 87, 18), comparison_158327, *[x_158328, y_158329], **kwargs_158330)
        
        # Assigning a type to the variable 'val' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'val', comparison_call_result_158331)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'm' (line 88)
        m_158332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'm')
        # Getting the type of 'self' (line 88)
        self_158333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'self')
        # Obtaining the member 'nomask' of a type (line 88)
        nomask_158334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 24), self_158333, 'nomask')
        # Applying the binary operator 'isnot' (line 88)
        result_is_not_158335 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 15), 'isnot', m_158332, nomask_158334)
        
        # Getting the type of 'fill_value' (line 88)
        fill_value_158336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 40), 'fill_value')
        # Applying the binary operator 'and' (line 88)
        result_and_keyword_158337 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 15), 'and', result_is_not_158335, fill_value_158336)
        
        # Testing the type of an if condition (line 88)
        if_condition_158338 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 12), result_and_keyword_158337)
        # Assigning a type to the variable 'if_condition_158338' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'if_condition_158338', if_condition_158338)
        # SSA begins for if statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 89):
        
        # Assigning a Call to a Name (line 89):
        
        # Call to masked_array(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'val' (line 89)
        val_158341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 40), 'val', False)
        # Processing the call keyword arguments (line 89)
        # Getting the type of 'm' (line 89)
        m_158342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 50), 'm', False)
        keyword_158343 = m_158342
        kwargs_158344 = {'mask': keyword_158343}
        # Getting the type of 'self' (line 89)
        self_158339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'self', False)
        # Obtaining the member 'masked_array' of a type (line 89)
        masked_array_158340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 22), self_158339, 'masked_array')
        # Calling masked_array(args, kwargs) (line 89)
        masked_array_call_result_158345 = invoke(stypy.reporting.localization.Localization(__file__, 89, 22), masked_array_158340, *[val_158341], **kwargs_158344)
        
        # Assigning a type to the variable 'val' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'val', masked_array_call_result_158345)
        # SSA join for if statement (line 88)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 90)
        # Getting the type of 'bool' (line 90)
        bool_158346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 31), 'bool')
        # Getting the type of 'val' (line 90)
        val_158347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 26), 'val')
        
        (may_be_158348, more_types_in_union_158349) = may_be_subtype(bool_158346, val_158347)

        if may_be_158348:

            if more_types_in_union_158349:
                # Runtime conditional SSA (line 90)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'val' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'val', remove_not_subtype_from_union(val_158347, bool))
            
            # Assigning a Name to a Name (line 91):
            
            # Assigning a Name to a Name (line 91):
            # Getting the type of 'val' (line 91)
            val_158350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 23), 'val')
            # Assigning a type to the variable 'cond' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'cond', val_158350)
            
            # Assigning a List to a Name (line 92):
            
            # Assigning a List to a Name (line 92):
            
            # Obtaining an instance of the builtin type 'list' (line 92)
            list_158351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 26), 'list')
            # Adding type elements to the builtin type 'list' instance (line 92)
            # Adding element type (line 92)
            int_158352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 27), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 26), list_158351, int_158352)
            
            # Assigning a type to the variable 'reduced' (line 92)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'reduced', list_158351)

            if more_types_in_union_158349:
                # Runtime conditional SSA for else branch (line 90)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_158348) or more_types_in_union_158349):
            # Assigning a type to the variable 'val' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'val', remove_subtype_from_union(val_158347, bool))
            
            # Assigning a Call to a Name (line 94):
            
            # Assigning a Call to a Name (line 94):
            
            # Call to ravel(...): (line 94)
            # Processing the call keyword arguments (line 94)
            kwargs_158355 = {}
            # Getting the type of 'val' (line 94)
            val_158353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 26), 'val', False)
            # Obtaining the member 'ravel' of a type (line 94)
            ravel_158354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 26), val_158353, 'ravel')
            # Calling ravel(args, kwargs) (line 94)
            ravel_call_result_158356 = invoke(stypy.reporting.localization.Localization(__file__, 94, 26), ravel_158354, *[], **kwargs_158355)
            
            # Assigning a type to the variable 'reduced' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'reduced', ravel_call_result_158356)
            
            # Assigning a Call to a Name (line 95):
            
            # Assigning a Call to a Name (line 95):
            
            # Call to all(...): (line 95)
            # Processing the call keyword arguments (line 95)
            kwargs_158359 = {}
            # Getting the type of 'reduced' (line 95)
            reduced_158357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 23), 'reduced', False)
            # Obtaining the member 'all' of a type (line 95)
            all_158358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 23), reduced_158357, 'all')
            # Calling all(args, kwargs) (line 95)
            all_call_result_158360 = invoke(stypy.reporting.localization.Localization(__file__, 95, 23), all_158358, *[], **kwargs_158359)
            
            # Assigning a type to the variable 'cond' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'cond', all_call_result_158360)
            
            # Assigning a Call to a Name (line 96):
            
            # Assigning a Call to a Name (line 96):
            
            # Call to tolist(...): (line 96)
            # Processing the call keyword arguments (line 96)
            kwargs_158363 = {}
            # Getting the type of 'reduced' (line 96)
            reduced_158361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'reduced', False)
            # Obtaining the member 'tolist' of a type (line 96)
            tolist_158362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 26), reduced_158361, 'tolist')
            # Calling tolist(args, kwargs) (line 96)
            tolist_call_result_158364 = invoke(stypy.reporting.localization.Localization(__file__, 96, 26), tolist_158362, *[], **kwargs_158363)
            
            # Assigning a type to the variable 'reduced' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'reduced', tolist_call_result_158364)

            if (may_be_158348 and more_types_in_union_158349):
                # SSA join for if statement (line 90)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'cond' (line 97)
        cond_158365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 19), 'cond')
        # Applying the 'not' unary operator (line 97)
        result_not__158366 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 15), 'not', cond_158365)
        
        # Testing the type of an if condition (line 97)
        if_condition_158367 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 12), result_not__158366)
        # Assigning a type to the variable 'if_condition_158367' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'if_condition_158367', if_condition_158367)
        # SSA begins for if statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 98):
        
        # Assigning a BinOp to a Name (line 98):
        int_158368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 24), 'int')
        float_158369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 28), 'float')
        
        # Call to count(...): (line 98)
        # Processing the call arguments (line 98)
        int_158372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 48), 'int')
        # Processing the call keyword arguments (line 98)
        kwargs_158373 = {}
        # Getting the type of 'reduced' (line 98)
        reduced_158370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 34), 'reduced', False)
        # Obtaining the member 'count' of a type (line 98)
        count_158371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 34), reduced_158370, 'count')
        # Calling count(args, kwargs) (line 98)
        count_call_result_158374 = invoke(stypy.reporting.localization.Localization(__file__, 98, 34), count_158371, *[int_158372], **kwargs_158373)
        
        # Applying the binary operator '*' (line 98)
        result_mul_158375 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 28), '*', float_158369, count_call_result_158374)
        
        
        # Call to len(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'reduced' (line 98)
        reduced_158377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 55), 'reduced', False)
        # Processing the call keyword arguments (line 98)
        kwargs_158378 = {}
        # Getting the type of 'len' (line 98)
        len_158376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 51), 'len', False)
        # Calling len(args, kwargs) (line 98)
        len_call_result_158379 = invoke(stypy.reporting.localization.Localization(__file__, 98, 51), len_158376, *[reduced_158377], **kwargs_158378)
        
        # Applying the binary operator 'div' (line 98)
        result_div_158380 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 50), 'div', result_mul_158375, len_call_result_158379)
        
        # Applying the binary operator '-' (line 98)
        result_sub_158381 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 24), '-', int_158368, result_div_158380)
        
        # Assigning a type to the variable 'match' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'match', result_sub_158381)
        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to build_err_msg(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Obtaining an instance of the builtin type 'list' (line 99)
        list_158383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 99)
        # Adding element type (line 99)
        # Getting the type of 'x' (line 99)
        x_158384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 37), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 36), list_158383, x_158384)
        # Adding element type (line 99)
        # Getting the type of 'y' (line 99)
        y_158385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 40), 'y', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 36), list_158383, y_158385)
        
        # Getting the type of 'err_msg' (line 100)
        err_msg_158386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 36), 'err_msg', False)
        str_158387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 38), 'str', '\n(mismatch %s%%)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 101)
        tuple_158388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 61), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 101)
        # Adding element type (line 101)
        # Getting the type of 'match' (line 101)
        match_158389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 61), 'match', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 61), tuple_158388, match_158389)
        
        # Applying the binary operator '%' (line 101)
        result_mod_158390 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 38), '%', str_158387, tuple_158388)
        
        # Applying the binary operator '+' (line 100)
        result_add_158391 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 36), '+', err_msg_158386, result_mod_158390)
        
        # Processing the call keyword arguments (line 99)
        # Getting the type of 'header' (line 102)
        header_158392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 43), 'header', False)
        keyword_158393 = header_158392
        
        # Obtaining an instance of the builtin type 'tuple' (line 103)
        tuple_158394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 103)
        # Adding element type (line 103)
        str_158395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 43), 'str', 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 43), tuple_158394, str_158395)
        # Adding element type (line 103)
        str_158396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 48), 'str', 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 43), tuple_158394, str_158396)
        
        keyword_158397 = tuple_158394
        kwargs_158398 = {'header': keyword_158393, 'names': keyword_158397}
        # Getting the type of 'build_err_msg' (line 99)
        build_err_msg_158382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'build_err_msg', False)
        # Calling build_err_msg(args, kwargs) (line 99)
        build_err_msg_call_result_158399 = invoke(stypy.reporting.localization.Localization(__file__, 99, 22), build_err_msg_158382, *[list_158383, result_add_158391], **kwargs_158398)
        
        # Assigning a type to the variable 'msg' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'msg', build_err_msg_call_result_158399)
        # Evaluating assert statement condition
        # Getting the type of 'cond' (line 104)
        cond_158400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 23), 'cond')
        # SSA join for if statement (line 97)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 77)
        # SSA branch for the except 'ValueError' branch of a try statement (line 77)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 106):
        
        # Assigning a Call to a Name (line 106):
        
        # Call to build_err_msg(...): (line 106)
        # Processing the call arguments (line 106)
        
        # Obtaining an instance of the builtin type 'list' (line 106)
        list_158402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 106)
        # Adding element type (line 106)
        # Getting the type of 'x' (line 106)
        x_158403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'x', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 32), list_158402, x_158403)
        # Adding element type (line 106)
        # Getting the type of 'y' (line 106)
        y_158404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 36), 'y', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 32), list_158402, y_158404)
        
        # Getting the type of 'err_msg' (line 106)
        err_msg_158405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), 'err_msg', False)
        # Processing the call keyword arguments (line 106)
        # Getting the type of 'header' (line 106)
        header_158406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 56), 'header', False)
        keyword_158407 = header_158406
        
        # Obtaining an instance of the builtin type 'tuple' (line 106)
        tuple_158408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 71), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 106)
        # Adding element type (line 106)
        str_158409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 71), 'str', 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 71), tuple_158408, str_158409)
        # Adding element type (line 106)
        str_158410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 76), 'str', 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 71), tuple_158408, str_158410)
        
        keyword_158411 = tuple_158408
        kwargs_158412 = {'header': keyword_158407, 'names': keyword_158411}
        # Getting the type of 'build_err_msg' (line 106)
        build_err_msg_158401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'build_err_msg', False)
        # Calling build_err_msg(args, kwargs) (line 106)
        build_err_msg_call_result_158413 = invoke(stypy.reporting.localization.Localization(__file__, 106, 18), build_err_msg_158401, *[list_158402, err_msg_158405], **kwargs_158412)
        
        # Assigning a type to the variable 'msg' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'msg', build_err_msg_call_result_158413)
        
        # Call to ValueError(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'msg' (line 107)
        msg_158415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 29), 'msg', False)
        # Processing the call keyword arguments (line 107)
        kwargs_158416 = {}
        # Getting the type of 'ValueError' (line 107)
        ValueError_158414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 107)
        ValueError_call_result_158417 = invoke(stypy.reporting.localization.Localization(__file__, 107, 18), ValueError_158414, *[msg_158415], **kwargs_158416)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 107, 12), ValueError_call_result_158417, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 77)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'assert_array_compare(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assert_array_compare' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_158418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_158418)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assert_array_compare'
        return stypy_return_type_158418


    @norecursion
    def assert_array_equal(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_158419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 47), 'str', '')
        defaults = [str_158419]
        # Create a new context for function 'assert_array_equal'
        module_type_store = module_type_store.open_function_context('assert_array_equal', 109, 4, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ModuleTester.assert_array_equal.__dict__.__setitem__('stypy_localization', localization)
        ModuleTester.assert_array_equal.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ModuleTester.assert_array_equal.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleTester.assert_array_equal.__dict__.__setitem__('stypy_function_name', 'ModuleTester.assert_array_equal')
        ModuleTester.assert_array_equal.__dict__.__setitem__('stypy_param_names_list', ['x', 'y', 'err_msg'])
        ModuleTester.assert_array_equal.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleTester.assert_array_equal.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleTester.assert_array_equal.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleTester.assert_array_equal.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleTester.assert_array_equal.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleTester.assert_array_equal.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ModuleTester.assert_array_equal', ['x', 'y', 'err_msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assert_array_equal', localization, ['x', 'y', 'err_msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assert_array_equal(...)' code ##################

        str_158420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, (-1)), 'str', '\n        Checks the elementwise equality of two masked arrays.\n\n        ')
        
        # Call to assert_array_compare(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'self' (line 114)
        self_158423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 34), 'self', False)
        # Obtaining the member 'equal' of a type (line 114)
        equal_158424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 34), self_158423, 'equal')
        # Getting the type of 'x' (line 114)
        x_158425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 46), 'x', False)
        # Getting the type of 'y' (line 114)
        y_158426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 49), 'y', False)
        # Processing the call keyword arguments (line 114)
        # Getting the type of 'err_msg' (line 114)
        err_msg_158427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 60), 'err_msg', False)
        keyword_158428 = err_msg_158427
        str_158429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 41), 'str', 'Arrays are not equal')
        keyword_158430 = str_158429
        kwargs_158431 = {'header': keyword_158430, 'err_msg': keyword_158428}
        # Getting the type of 'self' (line 114)
        self_158421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self', False)
        # Obtaining the member 'assert_array_compare' of a type (line 114)
        assert_array_compare_158422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_158421, 'assert_array_compare')
        # Calling assert_array_compare(args, kwargs) (line 114)
        assert_array_compare_call_result_158432 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), assert_array_compare_158422, *[equal_158424, x_158425, y_158426], **kwargs_158431)
        
        
        # ################# End of 'assert_array_equal(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assert_array_equal' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_158433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_158433)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assert_array_equal'
        return stypy_return_type_158433


    @norecursion
    def test_0(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_0'
        module_type_store = module_type_store.open_function_context('test_0', 117, 4, False)
        # Assigning a type to the variable 'self' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ModuleTester.test_0.__dict__.__setitem__('stypy_localization', localization)
        ModuleTester.test_0.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ModuleTester.test_0.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleTester.test_0.__dict__.__setitem__('stypy_function_name', 'ModuleTester.test_0')
        ModuleTester.test_0.__dict__.__setitem__('stypy_param_names_list', [])
        ModuleTester.test_0.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleTester.test_0.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleTester.test_0.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleTester.test_0.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleTester.test_0.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleTester.test_0.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ModuleTester.test_0', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_0', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_0(...)' code ##################

        str_158434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, (-1)), 'str', '\n        Tests creation\n\n        ')
        
        # Assigning a Call to a Name (line 122):
        
        # Assigning a Call to a Name (line 122):
        
        # Call to array(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Obtaining an instance of the builtin type 'list' (line 122)
        list_158437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 122)
        # Adding element type (line 122)
        float_158438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_158437, float_158438)
        # Adding element type (line 122)
        float_158439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_158437, float_158439)
        # Adding element type (line 122)
        float_158440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_158437, float_158440)
        # Adding element type (line 122)
        float_158441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_158437, float_158441)
        # Adding element type (line 122)
        # Getting the type of 'pi' (line 122)
        pi_158442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 39), 'pi', False)
        float_158443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 42), 'float')
        # Applying the binary operator 'div' (line 122)
        result_div_158444 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 39), 'div', pi_158442, float_158443)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_158437, result_div_158444)
        # Adding element type (line 122)
        float_158445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_158437, float_158445)
        # Adding element type (line 122)
        float_158446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_158437, float_158446)
        # Adding element type (line 122)
        float_158447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_158437, float_158447)
        # Adding element type (line 122)
        float_158448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_158437, float_158448)
        # Adding element type (line 122)
        float_158449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 66), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_158437, float_158449)
        # Adding element type (line 122)
        float_158450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_158437, float_158450)
        # Adding element type (line 122)
        float_158451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 74), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 21), list_158437, float_158451)
        
        # Processing the call keyword arguments (line 122)
        kwargs_158452 = {}
        # Getting the type of 'np' (line 122)
        np_158435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 122)
        array_158436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 12), np_158435, 'array')
        # Calling array(args, kwargs) (line 122)
        array_call_result_158453 = invoke(stypy.reporting.localization.Localization(__file__, 122, 12), array_158436, *[list_158437], **kwargs_158452)
        
        # Assigning a type to the variable 'x' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'x', array_call_result_158453)
        
        # Assigning a List to a Name (line 123):
        
        # Assigning a List to a Name (line 123):
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_158454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        # Adding element type (line 123)
        int_158455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 12), list_158454, int_158455)
        # Adding element type (line 123)
        int_158456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 12), list_158454, int_158456)
        # Adding element type (line 123)
        int_158457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 12), list_158454, int_158457)
        # Adding element type (line 123)
        int_158458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 12), list_158454, int_158458)
        # Adding element type (line 123)
        int_158459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 12), list_158454, int_158459)
        # Adding element type (line 123)
        int_158460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 28), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 12), list_158454, int_158460)
        # Adding element type (line 123)
        int_158461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 12), list_158454, int_158461)
        # Adding element type (line 123)
        int_158462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 12), list_158454, int_158462)
        # Adding element type (line 123)
        int_158463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 12), list_158454, int_158463)
        # Adding element type (line 123)
        int_158464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 12), list_158454, int_158464)
        # Adding element type (line 123)
        int_158465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 12), list_158454, int_158465)
        # Adding element type (line 123)
        int_158466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 12), list_158454, int_158466)
        
        # Assigning a type to the variable 'm' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'm', list_158454)
        
        # Assigning a Call to a Name (line 124):
        
        # Assigning a Call to a Name (line 124):
        
        # Call to masked_array(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'x' (line 124)
        x_158469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 31), 'x', False)
        # Processing the call keyword arguments (line 124)
        # Getting the type of 'm' (line 124)
        m_158470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 39), 'm', False)
        keyword_158471 = m_158470
        kwargs_158472 = {'mask': keyword_158471}
        # Getting the type of 'self' (line 124)
        self_158467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 13), 'self', False)
        # Obtaining the member 'masked_array' of a type (line 124)
        masked_array_158468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 13), self_158467, 'masked_array')
        # Calling masked_array(args, kwargs) (line 124)
        masked_array_call_result_158473 = invoke(stypy.reporting.localization.Localization(__file__, 124, 13), masked_array_158468, *[x_158469], **kwargs_158472)
        
        # Assigning a type to the variable 'xm' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'xm', masked_array_call_result_158473)
        
        # Obtaining the type of the subscript
        int_158474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 11), 'int')
        # Getting the type of 'xm' (line 125)
        xm_158475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'xm')
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___158476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), xm_158475, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_158477 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), getitem___158476, int_158474)
        
        
        # ################# End of 'test_0(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_0' in the type store
        # Getting the type of 'stypy_return_type' (line 117)
        stypy_return_type_158478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_158478)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_0'
        return stypy_return_type_158478


    @norecursion
    def test_1(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_1'
        module_type_store = module_type_store.open_function_context('test_1', 127, 4, False)
        # Assigning a type to the variable 'self' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ModuleTester.test_1.__dict__.__setitem__('stypy_localization', localization)
        ModuleTester.test_1.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ModuleTester.test_1.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleTester.test_1.__dict__.__setitem__('stypy_function_name', 'ModuleTester.test_1')
        ModuleTester.test_1.__dict__.__setitem__('stypy_param_names_list', [])
        ModuleTester.test_1.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleTester.test_1.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleTester.test_1.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleTester.test_1.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleTester.test_1.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleTester.test_1.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ModuleTester.test_1', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_1', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_1(...)' code ##################

        str_158479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, (-1)), 'str', '\n        Tests creation\n\n        ')
        
        # Assigning a Call to a Name (line 132):
        
        # Assigning a Call to a Name (line 132):
        
        # Call to array(...): (line 132)
        # Processing the call arguments (line 132)
        
        # Obtaining an instance of the builtin type 'list' (line 132)
        list_158482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 132)
        # Adding element type (line 132)
        float_158483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_158482, float_158483)
        # Adding element type (line 132)
        float_158484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_158482, float_158484)
        # Adding element type (line 132)
        float_158485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_158482, float_158485)
        # Adding element type (line 132)
        float_158486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_158482, float_158486)
        # Adding element type (line 132)
        # Getting the type of 'pi' (line 132)
        pi_158487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 39), 'pi', False)
        float_158488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 42), 'float')
        # Applying the binary operator 'div' (line 132)
        result_div_158489 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 39), 'div', pi_158487, float_158488)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_158482, result_div_158489)
        # Adding element type (line 132)
        float_158490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 47), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_158482, float_158490)
        # Adding element type (line 132)
        float_158491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 51), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_158482, float_158491)
        # Adding element type (line 132)
        float_158492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 55), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_158482, float_158492)
        # Adding element type (line 132)
        float_158493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_158482, float_158493)
        # Adding element type (line 132)
        float_158494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 66), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_158482, float_158494)
        # Adding element type (line 132)
        float_158495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_158482, float_158495)
        # Adding element type (line 132)
        float_158496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 74), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 21), list_158482, float_158496)
        
        # Processing the call keyword arguments (line 132)
        kwargs_158497 = {}
        # Getting the type of 'np' (line 132)
        np_158480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 132)
        array_158481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), np_158480, 'array')
        # Calling array(args, kwargs) (line 132)
        array_call_result_158498 = invoke(stypy.reporting.localization.Localization(__file__, 132, 12), array_158481, *[list_158482], **kwargs_158497)
        
        # Assigning a type to the variable 'x' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'x', array_call_result_158498)
        
        # Assigning a Call to a Name (line 133):
        
        # Assigning a Call to a Name (line 133):
        
        # Call to array(...): (line 133)
        # Processing the call arguments (line 133)
        
        # Obtaining an instance of the builtin type 'list' (line 133)
        list_158501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 133)
        # Adding element type (line 133)
        float_158502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 22), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 21), list_158501, float_158502)
        # Adding element type (line 133)
        float_158503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 21), list_158501, float_158503)
        # Adding element type (line 133)
        float_158504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 21), list_158501, float_158504)
        # Adding element type (line 133)
        float_158505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 21), list_158501, float_158505)
        # Adding element type (line 133)
        float_158506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 21), list_158501, float_158506)
        # Adding element type (line 133)
        float_158507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 43), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 21), list_158501, float_158507)
        # Adding element type (line 133)
        float_158508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 48), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 21), list_158501, float_158508)
        # Adding element type (line 133)
        float_158509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 52), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 21), list_158501, float_158509)
        # Adding element type (line 133)
        float_158510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 21), list_158501, float_158510)
        # Adding element type (line 133)
        float_158511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 63), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 21), list_158501, float_158511)
        # Adding element type (line 133)
        float_158512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 67), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 21), list_158501, float_158512)
        # Adding element type (line 133)
        float_158513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 71), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 21), list_158501, float_158513)
        
        # Processing the call keyword arguments (line 133)
        kwargs_158514 = {}
        # Getting the type of 'np' (line 133)
        np_158499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'np', False)
        # Obtaining the member 'array' of a type (line 133)
        array_158500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 12), np_158499, 'array')
        # Calling array(args, kwargs) (line 133)
        array_call_result_158515 = invoke(stypy.reporting.localization.Localization(__file__, 133, 12), array_158500, *[list_158501], **kwargs_158514)
        
        # Assigning a type to the variable 'y' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'y', array_call_result_158515)
        
        # Assigning a List to a Name (line 134):
        
        # Assigning a List to a Name (line 134):
        
        # Obtaining an instance of the builtin type 'list' (line 134)
        list_158516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 134)
        # Adding element type (line 134)
        int_158517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 13), list_158516, int_158517)
        # Adding element type (line 134)
        int_158518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 13), list_158516, int_158518)
        # Adding element type (line 134)
        int_158519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 13), list_158516, int_158519)
        # Adding element type (line 134)
        int_158520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 13), list_158516, int_158520)
        # Adding element type (line 134)
        int_158521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 13), list_158516, int_158521)
        # Adding element type (line 134)
        int_158522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 13), list_158516, int_158522)
        # Adding element type (line 134)
        int_158523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 13), list_158516, int_158523)
        # Adding element type (line 134)
        int_158524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 13), list_158516, int_158524)
        # Adding element type (line 134)
        int_158525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 13), list_158516, int_158525)
        # Adding element type (line 134)
        int_158526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 13), list_158516, int_158526)
        # Adding element type (line 134)
        int_158527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 13), list_158516, int_158527)
        # Adding element type (line 134)
        int_158528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 13), list_158516, int_158528)
        
        # Assigning a type to the variable 'm1' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'm1', list_158516)
        
        # Assigning a List to a Name (line 135):
        
        # Assigning a List to a Name (line 135):
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_158529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        # Adding element type (line 135)
        int_158530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 13), list_158529, int_158530)
        # Adding element type (line 135)
        int_158531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 13), list_158529, int_158531)
        # Adding element type (line 135)
        int_158532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 13), list_158529, int_158532)
        # Adding element type (line 135)
        int_158533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 13), list_158529, int_158533)
        # Adding element type (line 135)
        int_158534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 13), list_158529, int_158534)
        # Adding element type (line 135)
        int_158535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 13), list_158529, int_158535)
        # Adding element type (line 135)
        int_158536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 13), list_158529, int_158536)
        # Adding element type (line 135)
        int_158537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 13), list_158529, int_158537)
        # Adding element type (line 135)
        int_158538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 13), list_158529, int_158538)
        # Adding element type (line 135)
        int_158539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 13), list_158529, int_158539)
        # Adding element type (line 135)
        int_158540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 13), list_158529, int_158540)
        # Adding element type (line 135)
        int_158541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 13), list_158529, int_158541)
        
        # Assigning a type to the variable 'm2' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'm2', list_158529)
        
        # Assigning a Call to a Name (line 136):
        
        # Assigning a Call to a Name (line 136):
        
        # Call to masked_array(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'x' (line 136)
        x_158544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 31), 'x', False)
        # Processing the call keyword arguments (line 136)
        # Getting the type of 'm1' (line 136)
        m1_158545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 39), 'm1', False)
        keyword_158546 = m1_158545
        kwargs_158547 = {'mask': keyword_158546}
        # Getting the type of 'self' (line 136)
        self_158542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 13), 'self', False)
        # Obtaining the member 'masked_array' of a type (line 136)
        masked_array_158543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 13), self_158542, 'masked_array')
        # Calling masked_array(args, kwargs) (line 136)
        masked_array_call_result_158548 = invoke(stypy.reporting.localization.Localization(__file__, 136, 13), masked_array_158543, *[x_158544], **kwargs_158547)
        
        # Assigning a type to the variable 'xm' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'xm', masked_array_call_result_158548)
        
        # Assigning a Call to a Name (line 137):
        
        # Assigning a Call to a Name (line 137):
        
        # Call to masked_array(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'y' (line 137)
        y_158551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 31), 'y', False)
        # Processing the call keyword arguments (line 137)
        # Getting the type of 'm2' (line 137)
        m2_158552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 39), 'm2', False)
        keyword_158553 = m2_158552
        kwargs_158554 = {'mask': keyword_158553}
        # Getting the type of 'self' (line 137)
        self_158549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 13), 'self', False)
        # Obtaining the member 'masked_array' of a type (line 137)
        masked_array_158550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 13), self_158549, 'masked_array')
        # Calling masked_array(args, kwargs) (line 137)
        masked_array_call_result_158555 = invoke(stypy.reporting.localization.Localization(__file__, 137, 13), masked_array_158550, *[y_158551], **kwargs_158554)
        
        # Assigning a type to the variable 'ym' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'ym', masked_array_call_result_158555)
        
        # Assigning a Call to a Name (line 138):
        
        # Assigning a Call to a Name (line 138):
        
        # Call to where(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'm1' (line 138)
        m1_158558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 22), 'm1', False)
        float_158559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 26), 'float')
        # Getting the type of 'x' (line 138)
        x_158560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 34), 'x', False)
        # Processing the call keyword arguments (line 138)
        kwargs_158561 = {}
        # Getting the type of 'np' (line 138)
        np_158556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 13), 'np', False)
        # Obtaining the member 'where' of a type (line 138)
        where_158557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 13), np_158556, 'where')
        # Calling where(args, kwargs) (line 138)
        where_call_result_158562 = invoke(stypy.reporting.localization.Localization(__file__, 138, 13), where_158557, *[m1_158558, float_158559, x_158560], **kwargs_158561)
        
        # Assigning a type to the variable 'xf' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'xf', where_call_result_158562)
        
        # Call to set_fill_value(...): (line 139)
        # Processing the call arguments (line 139)
        float_158565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 26), 'float')
        # Processing the call keyword arguments (line 139)
        kwargs_158566 = {}
        # Getting the type of 'xm' (line 139)
        xm_158563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'xm', False)
        # Obtaining the member 'set_fill_value' of a type (line 139)
        set_fill_value_158564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), xm_158563, 'set_fill_value')
        # Calling set_fill_value(args, kwargs) (line 139)
        set_fill_value_call_result_158567 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), set_fill_value_158564, *[float_158565], **kwargs_158566)
        
        # Evaluating assert statement condition
        
        # Call to any(...): (line 141)
        # Processing the call keyword arguments (line 141)
        kwargs_158576 = {}
        
        # Call to filled(...): (line 141)
        # Processing the call arguments (line 141)
        int_158572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 30), 'int')
        # Processing the call keyword arguments (line 141)
        kwargs_158573 = {}
        # Getting the type of 'xm' (line 141)
        xm_158568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'xm', False)
        # Getting the type of 'ym' (line 141)
        ym_158569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 19), 'ym', False)
        # Applying the binary operator '-' (line 141)
        result_sub_158570 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 16), '-', xm_158568, ym_158569)
        
        # Obtaining the member 'filled' of a type (line 141)
        filled_158571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 16), result_sub_158570, 'filled')
        # Calling filled(args, kwargs) (line 141)
        filled_call_result_158574 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), filled_158571, *[int_158572], **kwargs_158573)
        
        # Obtaining the member 'any' of a type (line 141)
        any_158575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 16), filled_call_result_158574, 'any')
        # Calling any(args, kwargs) (line 141)
        any_call_result_158577 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), any_158575, *[], **kwargs_158576)
        
        
        # Assigning a Attribute to a Name (line 142):
        
        # Assigning a Attribute to a Name (line 142):
        # Getting the type of 'x' (line 142)
        x_158578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'x')
        # Obtaining the member 'shape' of a type (line 142)
        shape_158579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 12), x_158578, 'shape')
        # Assigning a type to the variable 's' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 's', shape_158579)
        # Evaluating assert statement condition
        
        # Getting the type of 'xm' (line 143)
        xm_158580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'xm')
        # Obtaining the member 'size' of a type (line 143)
        size_158581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 15), xm_158580, 'size')
        
        # Call to reduce(...): (line 143)
        # Processing the call arguments (line 143)

        @norecursion
        def _stypy_temp_lambda_41(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_41'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_41', 143, 33, True)
            # Passed parameters checking function
            _stypy_temp_lambda_41.stypy_localization = localization
            _stypy_temp_lambda_41.stypy_type_of_self = None
            _stypy_temp_lambda_41.stypy_type_store = module_type_store
            _stypy_temp_lambda_41.stypy_function_name = '_stypy_temp_lambda_41'
            _stypy_temp_lambda_41.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_41.stypy_varargs_param_name = None
            _stypy_temp_lambda_41.stypy_kwargs_param_name = None
            _stypy_temp_lambda_41.stypy_call_defaults = defaults
            _stypy_temp_lambda_41.stypy_call_varargs = varargs
            _stypy_temp_lambda_41.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_41', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_41', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 143)
            x_158583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 45), 'x', False)
            # Getting the type of 'y' (line 143)
            y_158584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 47), 'y', False)
            # Applying the binary operator '*' (line 143)
            result_mul_158585 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 45), '*', x_158583, y_158584)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 143)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 33), 'stypy_return_type', result_mul_158585)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_41' in the type store
            # Getting the type of 'stypy_return_type' (line 143)
            stypy_return_type_158586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 33), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_158586)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_41'
            return stypy_return_type_158586

        # Assigning a type to the variable '_stypy_temp_lambda_41' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 33), '_stypy_temp_lambda_41', _stypy_temp_lambda_41)
        # Getting the type of '_stypy_temp_lambda_41' (line 143)
        _stypy_temp_lambda_41_158587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 33), '_stypy_temp_lambda_41')
        # Getting the type of 's' (line 143)
        s_158588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 50), 's', False)
        # Processing the call keyword arguments (line 143)
        kwargs_158589 = {}
        # Getting the type of 'reduce' (line 143)
        reduce_158582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 26), 'reduce', False)
        # Calling reduce(args, kwargs) (line 143)
        reduce_call_result_158590 = invoke(stypy.reporting.localization.Localization(__file__, 143, 26), reduce_158582, *[_stypy_temp_lambda_41_158587, s_158588], **kwargs_158589)
        
        # Applying the binary operator '==' (line 143)
        result_eq_158591 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 15), '==', size_158581, reduce_call_result_158590)
        
        # Evaluating assert statement condition
        
        
        # Call to count(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'xm' (line 144)
        xm_158594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 26), 'xm', False)
        # Processing the call keyword arguments (line 144)
        kwargs_158595 = {}
        # Getting the type of 'self' (line 144)
        self_158592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'self', False)
        # Obtaining the member 'count' of a type (line 144)
        count_158593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 15), self_158592, 'count')
        # Calling count(args, kwargs) (line 144)
        count_call_result_158596 = invoke(stypy.reporting.localization.Localization(__file__, 144, 15), count_158593, *[xm_158594], **kwargs_158595)
        
        
        # Call to len(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'm1' (line 144)
        m1_158598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 37), 'm1', False)
        # Processing the call keyword arguments (line 144)
        kwargs_158599 = {}
        # Getting the type of 'len' (line 144)
        len_158597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 33), 'len', False)
        # Calling len(args, kwargs) (line 144)
        len_call_result_158600 = invoke(stypy.reporting.localization.Localization(__file__, 144, 33), len_158597, *[m1_158598], **kwargs_158599)
        
        
        # Call to reduce(...): (line 144)
        # Processing the call arguments (line 144)

        @norecursion
        def _stypy_temp_lambda_42(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_42'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_42', 144, 50, True)
            # Passed parameters checking function
            _stypy_temp_lambda_42.stypy_localization = localization
            _stypy_temp_lambda_42.stypy_type_of_self = None
            _stypy_temp_lambda_42.stypy_type_store = module_type_store
            _stypy_temp_lambda_42.stypy_function_name = '_stypy_temp_lambda_42'
            _stypy_temp_lambda_42.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_42.stypy_varargs_param_name = None
            _stypy_temp_lambda_42.stypy_kwargs_param_name = None
            _stypy_temp_lambda_42.stypy_call_defaults = defaults
            _stypy_temp_lambda_42.stypy_call_varargs = varargs
            _stypy_temp_lambda_42.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_42', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_42', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 144)
            x_158602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 62), 'x', False)
            # Getting the type of 'y' (line 144)
            y_158603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 64), 'y', False)
            # Applying the binary operator '+' (line 144)
            result_add_158604 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 62), '+', x_158602, y_158603)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 50), 'stypy_return_type', result_add_158604)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_42' in the type store
            # Getting the type of 'stypy_return_type' (line 144)
            stypy_return_type_158605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 50), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_158605)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_42'
            return stypy_return_type_158605

        # Assigning a type to the variable '_stypy_temp_lambda_42' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 50), '_stypy_temp_lambda_42', _stypy_temp_lambda_42)
        # Getting the type of '_stypy_temp_lambda_42' (line 144)
        _stypy_temp_lambda_42_158606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 50), '_stypy_temp_lambda_42')
        # Getting the type of 'm1' (line 144)
        m1_158607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 67), 'm1', False)
        # Processing the call keyword arguments (line 144)
        kwargs_158608 = {}
        # Getting the type of 'reduce' (line 144)
        reduce_158601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 43), 'reduce', False)
        # Calling reduce(args, kwargs) (line 144)
        reduce_call_result_158609 = invoke(stypy.reporting.localization.Localization(__file__, 144, 43), reduce_158601, *[_stypy_temp_lambda_42_158606, m1_158607], **kwargs_158608)
        
        # Applying the binary operator '-' (line 144)
        result_sub_158610 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 33), '-', len_call_result_158600, reduce_call_result_158609)
        
        # Applying the binary operator '==' (line 144)
        result_eq_158611 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 15), '==', count_call_result_158596, result_sub_158610)
        
        
        
        # Obtaining an instance of the builtin type 'list' (line 146)
        list_158612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 146)
        # Adding element type (line 146)
        
        # Obtaining an instance of the builtin type 'tuple' (line 146)
        tuple_158613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 19), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 146)
        # Adding element type (line 146)
        int_158614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 19), tuple_158613, int_158614)
        # Adding element type (line 146)
        int_158615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 19), tuple_158613, int_158615)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 17), list_158612, tuple_158613)
        # Adding element type (line 146)
        
        # Obtaining an instance of the builtin type 'tuple' (line 146)
        tuple_158616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 146)
        # Adding element type (line 146)
        int_158617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 27), tuple_158616, int_158617)
        # Adding element type (line 146)
        int_158618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 27), tuple_158616, int_158618)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 17), list_158612, tuple_158616)
        
        # Testing the type of a for loop iterable (line 146)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 146, 8), list_158612)
        # Getting the type of the for loop variable (line 146)
        for_loop_var_158619 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 146, 8), list_158612)
        # Assigning a type to the variable 's' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 's', for_loop_var_158619)
        # SSA begins for a for statement (line 146)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Attribute (line 147):
        
        # Assigning a Name to a Attribute (line 147):
        # Getting the type of 's' (line 147)
        s_158620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 22), 's')
        # Getting the type of 'x' (line 147)
        x_158621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'x')
        # Setting the type of the member 'shape' of a type (line 147)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), x_158621, 'shape', s_158620)
        
        # Assigning a Name to a Attribute (line 148):
        
        # Assigning a Name to a Attribute (line 148):
        # Getting the type of 's' (line 148)
        s_158622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 22), 's')
        # Getting the type of 'y' (line 148)
        y_158623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'y')
        # Setting the type of the member 'shape' of a type (line 148)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), y_158623, 'shape', s_158622)
        
        # Assigning a Name to a Attribute (line 149):
        
        # Assigning a Name to a Attribute (line 149):
        # Getting the type of 's' (line 149)
        s_158624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 23), 's')
        # Getting the type of 'xm' (line 149)
        xm_158625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'xm')
        # Setting the type of the member 'shape' of a type (line 149)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), xm_158625, 'shape', s_158624)
        
        # Assigning a Name to a Attribute (line 150):
        
        # Assigning a Name to a Attribute (line 150):
        # Getting the type of 's' (line 150)
        s_158626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 23), 's')
        # Getting the type of 'ym' (line 150)
        ym_158627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'ym')
        # Setting the type of the member 'shape' of a type (line 150)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 12), ym_158627, 'shape', s_158626)
        
        # Assigning a Name to a Attribute (line 151):
        
        # Assigning a Name to a Attribute (line 151):
        # Getting the type of 's' (line 151)
        s_158628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), 's')
        # Getting the type of 'xf' (line 151)
        xf_158629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'xf')
        # Setting the type of the member 'shape' of a type (line 151)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), xf_158629, 'shape', s_158628)
        # Evaluating assert statement condition
        
        
        # Call to count(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'xm' (line 152)
        xm_158632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 30), 'xm', False)
        # Processing the call keyword arguments (line 152)
        kwargs_158633 = {}
        # Getting the type of 'self' (line 152)
        self_158630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 19), 'self', False)
        # Obtaining the member 'count' of a type (line 152)
        count_158631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 19), self_158630, 'count')
        # Calling count(args, kwargs) (line 152)
        count_call_result_158634 = invoke(stypy.reporting.localization.Localization(__file__, 152, 19), count_158631, *[xm_158632], **kwargs_158633)
        
        
        # Call to len(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'm1' (line 152)
        m1_158636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 41), 'm1', False)
        # Processing the call keyword arguments (line 152)
        kwargs_158637 = {}
        # Getting the type of 'len' (line 152)
        len_158635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 37), 'len', False)
        # Calling len(args, kwargs) (line 152)
        len_call_result_158638 = invoke(stypy.reporting.localization.Localization(__file__, 152, 37), len_158635, *[m1_158636], **kwargs_158637)
        
        
        # Call to reduce(...): (line 152)
        # Processing the call arguments (line 152)

        @norecursion
        def _stypy_temp_lambda_43(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_43'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_43', 152, 54, True)
            # Passed parameters checking function
            _stypy_temp_lambda_43.stypy_localization = localization
            _stypy_temp_lambda_43.stypy_type_of_self = None
            _stypy_temp_lambda_43.stypy_type_store = module_type_store
            _stypy_temp_lambda_43.stypy_function_name = '_stypy_temp_lambda_43'
            _stypy_temp_lambda_43.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_43.stypy_varargs_param_name = None
            _stypy_temp_lambda_43.stypy_kwargs_param_name = None
            _stypy_temp_lambda_43.stypy_call_defaults = defaults
            _stypy_temp_lambda_43.stypy_call_varargs = varargs
            _stypy_temp_lambda_43.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_43', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_43', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 152)
            x_158640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 66), 'x', False)
            # Getting the type of 'y' (line 152)
            y_158641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 68), 'y', False)
            # Applying the binary operator '+' (line 152)
            result_add_158642 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 66), '+', x_158640, y_158641)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 54), 'stypy_return_type', result_add_158642)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_43' in the type store
            # Getting the type of 'stypy_return_type' (line 152)
            stypy_return_type_158643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 54), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_158643)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_43'
            return stypy_return_type_158643

        # Assigning a type to the variable '_stypy_temp_lambda_43' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 54), '_stypy_temp_lambda_43', _stypy_temp_lambda_43)
        # Getting the type of '_stypy_temp_lambda_43' (line 152)
        _stypy_temp_lambda_43_158644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 54), '_stypy_temp_lambda_43')
        # Getting the type of 'm1' (line 152)
        m1_158645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 71), 'm1', False)
        # Processing the call keyword arguments (line 152)
        kwargs_158646 = {}
        # Getting the type of 'reduce' (line 152)
        reduce_158639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 47), 'reduce', False)
        # Calling reduce(args, kwargs) (line 152)
        reduce_call_result_158647 = invoke(stypy.reporting.localization.Localization(__file__, 152, 47), reduce_158639, *[_stypy_temp_lambda_43_158644, m1_158645], **kwargs_158646)
        
        # Applying the binary operator '-' (line 152)
        result_sub_158648 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 37), '-', len_call_result_158638, reduce_call_result_158647)
        
        # Applying the binary operator '==' (line 152)
        result_eq_158649 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 19), '==', count_call_result_158634, result_sub_158648)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_1(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_1' in the type store
        # Getting the type of 'stypy_return_type' (line 127)
        stypy_return_type_158650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_158650)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_1'
        return stypy_return_type_158650


    @norecursion
    def test_2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_2'
        module_type_store = module_type_store.open_function_context('test_2', 154, 4, False)
        # Assigning a type to the variable 'self' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ModuleTester.test_2.__dict__.__setitem__('stypy_localization', localization)
        ModuleTester.test_2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ModuleTester.test_2.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleTester.test_2.__dict__.__setitem__('stypy_function_name', 'ModuleTester.test_2')
        ModuleTester.test_2.__dict__.__setitem__('stypy_param_names_list', [])
        ModuleTester.test_2.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleTester.test_2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleTester.test_2.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleTester.test_2.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleTester.test_2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleTester.test_2.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ModuleTester.test_2', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_2', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_2(...)' code ##################

        str_158651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, (-1)), 'str', '\n        Tests conversions and indexing.\n\n        ')
        
        # Assigning a Call to a Name (line 159):
        
        # Assigning a Call to a Name (line 159):
        
        # Call to array(...): (line 159)
        # Processing the call arguments (line 159)
        
        # Obtaining an instance of the builtin type 'list' (line 159)
        list_158654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 159)
        # Adding element type (line 159)
        int_158655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 22), list_158654, int_158655)
        # Adding element type (line 159)
        int_158656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 22), list_158654, int_158656)
        # Adding element type (line 159)
        int_158657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 22), list_158654, int_158657)
        # Adding element type (line 159)
        int_158658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 22), list_158654, int_158658)
        
        # Processing the call keyword arguments (line 159)
        kwargs_158659 = {}
        # Getting the type of 'np' (line 159)
        np_158652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 159)
        array_158653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 13), np_158652, 'array')
        # Calling array(args, kwargs) (line 159)
        array_call_result_158660 = invoke(stypy.reporting.localization.Localization(__file__, 159, 13), array_158653, *[list_158654], **kwargs_158659)
        
        # Assigning a type to the variable 'x1' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'x1', array_call_result_158660)
        
        # Assigning a Call to a Name (line 160):
        
        # Assigning a Call to a Name (line 160):
        
        # Call to array(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'x1' (line 160)
        x1_158663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'x1', False)
        # Processing the call keyword arguments (line 160)
        
        # Obtaining an instance of the builtin type 'list' (line 160)
        list_158664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 160)
        # Adding element type (line 160)
        int_158665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 33), list_158664, int_158665)
        # Adding element type (line 160)
        int_158666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 33), list_158664, int_158666)
        # Adding element type (line 160)
        int_158667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 33), list_158664, int_158667)
        # Adding element type (line 160)
        int_158668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 33), list_158664, int_158668)
        
        keyword_158669 = list_158664
        kwargs_158670 = {'mask': keyword_158669}
        # Getting the type of 'self' (line 160)
        self_158661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 13), 'self', False)
        # Obtaining the member 'array' of a type (line 160)
        array_158662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 13), self_158661, 'array')
        # Calling array(args, kwargs) (line 160)
        array_call_result_158671 = invoke(stypy.reporting.localization.Localization(__file__, 160, 13), array_158662, *[x1_158663], **kwargs_158670)
        
        # Assigning a type to the variable 'x2' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'x2', array_call_result_158671)
        
        # Assigning a Call to a Name (line 161):
        
        # Assigning a Call to a Name (line 161):
        
        # Call to array(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'x1' (line 161)
        x1_158674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'x1', False)
        # Processing the call keyword arguments (line 161)
        
        # Obtaining an instance of the builtin type 'list' (line 161)
        list_158675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 161)
        # Adding element type (line 161)
        int_158676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 34), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 33), list_158675, int_158676)
        # Adding element type (line 161)
        int_158677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 33), list_158675, int_158677)
        # Adding element type (line 161)
        int_158678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 33), list_158675, int_158678)
        # Adding element type (line 161)
        int_158679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 33), list_158675, int_158679)
        
        keyword_158680 = list_158675
        kwargs_158681 = {'mask': keyword_158680}
        # Getting the type of 'self' (line 161)
        self_158672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 13), 'self', False)
        # Obtaining the member 'array' of a type (line 161)
        array_158673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 13), self_158672, 'array')
        # Calling array(args, kwargs) (line 161)
        array_call_result_158682 = invoke(stypy.reporting.localization.Localization(__file__, 161, 13), array_158673, *[x1_158674], **kwargs_158681)
        
        # Assigning a type to the variable 'x3' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'x3', array_call_result_158682)
        
        # Assigning a Call to a Name (line 162):
        
        # Assigning a Call to a Name (line 162):
        
        # Call to array(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'x1' (line 162)
        x1_158685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 24), 'x1', False)
        # Processing the call keyword arguments (line 162)
        kwargs_158686 = {}
        # Getting the type of 'self' (line 162)
        self_158683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 13), 'self', False)
        # Obtaining the member 'array' of a type (line 162)
        array_158684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 13), self_158683, 'array')
        # Calling array(args, kwargs) (line 162)
        array_call_result_158687 = invoke(stypy.reporting.localization.Localization(__file__, 162, 13), array_158684, *[x1_158685], **kwargs_158686)
        
        # Assigning a type to the variable 'x4' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'x4', array_call_result_158687)
        
        # Call to str(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'x2' (line 164)
        x2_158689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'x2', False)
        # Processing the call keyword arguments (line 164)
        kwargs_158690 = {}
        # Getting the type of 'str' (line 164)
        str_158688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'str', False)
        # Calling str(args, kwargs) (line 164)
        str_call_result_158691 = invoke(stypy.reporting.localization.Localization(__file__, 164, 8), str_158688, *[x2_158689], **kwargs_158690)
        
        
        # Call to repr(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'x2' (line 165)
        x2_158693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 13), 'x2', False)
        # Processing the call keyword arguments (line 165)
        kwargs_158694 = {}
        # Getting the type of 'repr' (line 165)
        repr_158692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'repr', False)
        # Calling repr(args, kwargs) (line 165)
        repr_call_result_158695 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), repr_158692, *[x2_158693], **kwargs_158694)
        
        # Evaluating assert statement condition
        
        
        # Call to type(...): (line 167)
        # Processing the call arguments (line 167)
        
        # Obtaining the type of the subscript
        int_158697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 23), 'int')
        # Getting the type of 'x2' (line 167)
        x2_158698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 20), 'x2', False)
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___158699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 20), x2_158698, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_158700 = invoke(stypy.reporting.localization.Localization(__file__, 167, 20), getitem___158699, int_158697)
        
        # Processing the call keyword arguments (line 167)
        kwargs_158701 = {}
        # Getting the type of 'type' (line 167)
        type_158696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 15), 'type', False)
        # Calling type(args, kwargs) (line 167)
        type_call_result_158702 = invoke(stypy.reporting.localization.Localization(__file__, 167, 15), type_158696, *[subscript_call_result_158700], **kwargs_158701)
        
        
        # Call to type(...): (line 167)
        # Processing the call arguments (line 167)
        
        # Obtaining the type of the subscript
        int_158704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 38), 'int')
        # Getting the type of 'x1' (line 167)
        x1_158705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 35), 'x1', False)
        # Obtaining the member '__getitem__' of a type (line 167)
        getitem___158706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 35), x1_158705, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 167)
        subscript_call_result_158707 = invoke(stypy.reporting.localization.Localization(__file__, 167, 35), getitem___158706, int_158704)
        
        # Processing the call keyword arguments (line 167)
        kwargs_158708 = {}
        # Getting the type of 'type' (line 167)
        type_158703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'type', False)
        # Calling type(args, kwargs) (line 167)
        type_call_result_158709 = invoke(stypy.reporting.localization.Localization(__file__, 167, 30), type_158703, *[subscript_call_result_158707], **kwargs_158708)
        
        # Applying the binary operator 'is' (line 167)
        result_is__158710 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 15), 'is', type_call_result_158702, type_call_result_158709)
        
        # Evaluating assert statement condition
        
        
        # Obtaining the type of the subscript
        int_158711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 18), 'int')
        # Getting the type of 'x1' (line 168)
        x1_158712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'x1')
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___158713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 15), x1_158712, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_158714 = invoke(stypy.reporting.localization.Localization(__file__, 168, 15), getitem___158713, int_158711)
        
        
        # Obtaining the type of the subscript
        int_158715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 27), 'int')
        # Getting the type of 'x2' (line 168)
        x2_158716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 24), 'x2')
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___158717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 24), x2_158716, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_158718 = invoke(stypy.reporting.localization.Localization(__file__, 168, 24), getitem___158717, int_158715)
        
        # Applying the binary operator '==' (line 168)
        result_eq_158719 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 15), '==', subscript_call_result_158714, subscript_call_result_158718)
        
        
        # Assigning a Num to a Subscript (line 169):
        
        # Assigning a Num to a Subscript (line 169):
        int_158720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 16), 'int')
        # Getting the type of 'x1' (line 169)
        x1_158721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'x1')
        int_158722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 11), 'int')
        # Storing an element on a container (line 169)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 8), x1_158721, (int_158722, int_158720))
        
        # Assigning a Num to a Subscript (line 170):
        
        # Assigning a Num to a Subscript (line 170):
        int_158723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 16), 'int')
        # Getting the type of 'x2' (line 170)
        x2_158724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'x2')
        int_158725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 11), 'int')
        # Storing an element on a container (line 170)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), x2_158724, (int_158725, int_158723))
        
        # Call to assert_array_equal(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'x1' (line 171)
        x1_158728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 32), 'x1', False)
        # Getting the type of 'x2' (line 171)
        x2_158729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 36), 'x2', False)
        # Processing the call keyword arguments (line 171)
        kwargs_158730 = {}
        # Getting the type of 'self' (line 171)
        self_158726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 171)
        assert_array_equal_158727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), self_158726, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 171)
        assert_array_equal_call_result_158731 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), assert_array_equal_158727, *[x1_158728, x2_158729], **kwargs_158730)
        
        
        # Assigning a Num to a Subscript (line 172):
        
        # Assigning a Num to a Subscript (line 172):
        int_158732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 18), 'int')
        # Getting the type of 'x1' (line 172)
        x1_158733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'x1')
        int_158734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 11), 'int')
        int_158735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 13), 'int')
        slice_158736 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 172, 8), int_158734, int_158735, None)
        # Storing an element on a container (line 172)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 8), x1_158733, (slice_158736, int_158732))
        
        # Assigning a Num to a Subscript (line 173):
        
        # Assigning a Num to a Subscript (line 173):
        int_158737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 18), 'int')
        # Getting the type of 'x2' (line 173)
        x2_158738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'x2')
        int_158739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 11), 'int')
        int_158740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 13), 'int')
        slice_158741 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 173, 8), int_158739, int_158740, None)
        # Storing an element on a container (line 173)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 8), x2_158738, (slice_158741, int_158737))
        
        # Assigning a Attribute to a Subscript (line 174):
        
        # Assigning a Attribute to a Subscript (line 174):
        # Getting the type of 'self' (line 174)
        self_158742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'self')
        # Obtaining the member 'masked' of a type (line 174)
        masked_158743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 16), self_158742, 'masked')
        # Getting the type of 'x2' (line 174)
        x2_158744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'x2')
        int_158745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 11), 'int')
        # Storing an element on a container (line 174)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 8), x2_158744, (int_158745, masked_158743))
        
        # Assigning a Attribute to a Subscript (line 175):
        
        # Assigning a Attribute to a Subscript (line 175):
        # Getting the type of 'self' (line 175)
        self_158746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 18), 'self')
        # Obtaining the member 'masked' of a type (line 175)
        masked_158747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 18), self_158746, 'masked')
        # Getting the type of 'x2' (line 175)
        x2_158748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'x2')
        int_158749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 11), 'int')
        int_158750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 13), 'int')
        slice_158751 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 175, 8), int_158749, int_158750, None)
        # Storing an element on a container (line 175)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 8), x2_158748, (slice_158751, masked_158747))
        
        # Assigning a Name to a Subscript (line 176):
        
        # Assigning a Name to a Subscript (line 176):
        # Getting the type of 'x1' (line 176)
        x1_158752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'x1')
        # Getting the type of 'x2' (line 176)
        x2_158753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'x2')
        slice_158754 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 176, 8), None, None, None)
        # Storing an element on a container (line 176)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 8), x2_158753, (slice_158754, x1_158752))
        
        # Assigning a Attribute to a Subscript (line 177):
        
        # Assigning a Attribute to a Subscript (line 177):
        # Getting the type of 'self' (line 177)
        self_158755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'self')
        # Obtaining the member 'masked' of a type (line 177)
        masked_158756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 16), self_158755, 'masked')
        # Getting the type of 'x2' (line 177)
        x2_158757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'x2')
        int_158758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 11), 'int')
        # Storing an element on a container (line 177)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 8), x2_158757, (int_158758, masked_158756))
        
        # Assigning a Call to a Subscript (line 178):
        
        # Assigning a Call to a Subscript (line 178):
        
        # Call to masked_array(...): (line 178)
        # Processing the call arguments (line 178)
        
        # Obtaining an instance of the builtin type 'list' (line 178)
        list_158761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 178)
        # Adding element type (line 178)
        int_158762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 34), list_158761, int_158762)
        # Adding element type (line 178)
        int_158763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 34), list_158761, int_158763)
        # Adding element type (line 178)
        int_158764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 34), list_158761, int_158764)
        # Adding element type (line 178)
        int_158765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 34), list_158761, int_158765)
        
        
        # Obtaining an instance of the builtin type 'list' (line 178)
        list_158766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 178)
        # Adding element type (line 178)
        int_158767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 48), list_158766, int_158767)
        # Adding element type (line 178)
        int_158768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 48), list_158766, int_158768)
        # Adding element type (line 178)
        int_158769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 48), list_158766, int_158769)
        # Adding element type (line 178)
        int_158770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 48), list_158766, int_158770)
        
        # Processing the call keyword arguments (line 178)
        kwargs_158771 = {}
        # Getting the type of 'self' (line 178)
        self_158759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'self', False)
        # Obtaining the member 'masked_array' of a type (line 178)
        masked_array_158760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 16), self_158759, 'masked_array')
        # Calling masked_array(args, kwargs) (line 178)
        masked_array_call_result_158772 = invoke(stypy.reporting.localization.Localization(__file__, 178, 16), masked_array_158760, *[list_158761, list_158766], **kwargs_158771)
        
        # Getting the type of 'x3' (line 178)
        x3_158773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'x3')
        slice_158774 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 178, 8), None, None, None)
        # Storing an element on a container (line 178)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 8), x3_158773, (slice_158774, masked_array_call_result_158772))
        
        # Assigning a Call to a Subscript (line 179):
        
        # Assigning a Call to a Subscript (line 179):
        
        # Call to masked_array(...): (line 179)
        # Processing the call arguments (line 179)
        
        # Obtaining an instance of the builtin type 'list' (line 179)
        list_158777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 179)
        # Adding element type (line 179)
        int_158778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 34), list_158777, int_158778)
        # Adding element type (line 179)
        int_158779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 34), list_158777, int_158779)
        # Adding element type (line 179)
        int_158780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 34), list_158777, int_158780)
        # Adding element type (line 179)
        int_158781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 34), list_158777, int_158781)
        
        
        # Obtaining an instance of the builtin type 'list' (line 179)
        list_158782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 179)
        # Adding element type (line 179)
        int_158783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 48), list_158782, int_158783)
        # Adding element type (line 179)
        int_158784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 48), list_158782, int_158784)
        # Adding element type (line 179)
        int_158785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 48), list_158782, int_158785)
        # Adding element type (line 179)
        int_158786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 48), list_158782, int_158786)
        
        # Processing the call keyword arguments (line 179)
        kwargs_158787 = {}
        # Getting the type of 'self' (line 179)
        self_158775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'self', False)
        # Obtaining the member 'masked_array' of a type (line 179)
        masked_array_158776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 16), self_158775, 'masked_array')
        # Calling masked_array(args, kwargs) (line 179)
        masked_array_call_result_158788 = invoke(stypy.reporting.localization.Localization(__file__, 179, 16), masked_array_158776, *[list_158777, list_158782], **kwargs_158787)
        
        # Getting the type of 'x4' (line 179)
        x4_158789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'x4')
        slice_158790 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 179, 8), None, None, None)
        # Storing an element on a container (line 179)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 8), x4_158789, (slice_158790, masked_array_call_result_158788))
        
        # Assigning a BinOp to a Name (line 180):
        
        # Assigning a BinOp to a Name (line 180):
        
        # Call to arange(...): (line 180)
        # Processing the call arguments (line 180)
        int_158793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 23), 'int')
        # Processing the call keyword arguments (line 180)
        kwargs_158794 = {}
        # Getting the type of 'np' (line 180)
        np_158791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 13), 'np', False)
        # Obtaining the member 'arange' of a type (line 180)
        arange_158792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 13), np_158791, 'arange')
        # Calling arange(args, kwargs) (line 180)
        arange_call_result_158795 = invoke(stypy.reporting.localization.Localization(__file__, 180, 13), arange_158792, *[int_158793], **kwargs_158794)
        
        float_158796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 26), 'float')
        # Applying the binary operator '*' (line 180)
        result_mul_158797 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 13), '*', arange_call_result_158795, float_158796)
        
        # Assigning a type to the variable 'x1' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'x1', result_mul_158797)
        
        # Assigning a Call to a Name (line 181):
        
        # Assigning a Call to a Name (line 181):
        
        # Call to masked_values(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'x1' (line 181)
        x1_158800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 32), 'x1', False)
        float_158801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 36), 'float')
        # Processing the call keyword arguments (line 181)
        kwargs_158802 = {}
        # Getting the type of 'self' (line 181)
        self_158798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 13), 'self', False)
        # Obtaining the member 'masked_values' of a type (line 181)
        masked_values_158799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 13), self_158798, 'masked_values')
        # Calling masked_values(args, kwargs) (line 181)
        masked_values_call_result_158803 = invoke(stypy.reporting.localization.Localization(__file__, 181, 13), masked_values_158799, *[x1_158800, float_158801], **kwargs_158802)
        
        # Assigning a type to the variable 'x2' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'x2', masked_values_call_result_158803)
        
        # Assigning a Call to a Name (line 182):
        
        # Assigning a Call to a Name (line 182):
        
        # Call to array(...): (line 182)
        # Processing the call arguments (line 182)
        
        # Obtaining an instance of the builtin type 'list' (line 182)
        list_158806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 182)
        # Adding element type (line 182)
        int_158807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 24), list_158806, int_158807)
        # Adding element type (line 182)
        str_158808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 28), 'str', 'hello')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 24), list_158806, str_158808)
        # Adding element type (line 182)
        int_158809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 24), list_158806, int_158809)
        # Adding element type (line 182)
        int_158810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 24), list_158806, int_158810)
        
        # Getting the type of 'object' (line 182)
        object_158811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 44), 'object', False)
        # Processing the call keyword arguments (line 182)
        kwargs_158812 = {}
        # Getting the type of 'self' (line 182)
        self_158804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 13), 'self', False)
        # Obtaining the member 'array' of a type (line 182)
        array_158805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 13), self_158804, 'array')
        # Calling array(args, kwargs) (line 182)
        array_call_result_158813 = invoke(stypy.reporting.localization.Localization(__file__, 182, 13), array_158805, *[list_158806, object_158811], **kwargs_158812)
        
        # Assigning a type to the variable 'x1' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'x1', array_call_result_158813)
        
        # Assigning a Call to a Name (line 183):
        
        # Assigning a Call to a Name (line 183):
        
        # Call to array(...): (line 183)
        # Processing the call arguments (line 183)
        
        # Obtaining an instance of the builtin type 'list' (line 183)
        list_158816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 183)
        # Adding element type (line 183)
        int_158817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 22), list_158816, int_158817)
        # Adding element type (line 183)
        str_158818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 26), 'str', 'hello')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 22), list_158816, str_158818)
        # Adding element type (line 183)
        int_158819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 22), list_158816, int_158819)
        # Adding element type (line 183)
        int_158820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 22), list_158816, int_158820)
        
        # Getting the type of 'object' (line 183)
        object_158821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 42), 'object', False)
        # Processing the call keyword arguments (line 183)
        kwargs_158822 = {}
        # Getting the type of 'np' (line 183)
        np_158814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 13), 'np', False)
        # Obtaining the member 'array' of a type (line 183)
        array_158815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 13), np_158814, 'array')
        # Calling array(args, kwargs) (line 183)
        array_call_result_158823 = invoke(stypy.reporting.localization.Localization(__file__, 183, 13), array_158815, *[list_158816, object_158821], **kwargs_158822)
        
        # Assigning a type to the variable 'x2' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'x2', array_call_result_158823)
        
        # Obtaining the type of the subscript
        int_158824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 11), 'int')
        # Getting the type of 'x1' (line 185)
        x1_158825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'x1')
        # Obtaining the member '__getitem__' of a type (line 185)
        getitem___158826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), x1_158825, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 185)
        subscript_call_result_158827 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), getitem___158826, int_158824)
        
        
        # Obtaining the type of the subscript
        int_158828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 11), 'int')
        # Getting the type of 'x2' (line 186)
        x2_158829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'x2')
        # Obtaining the member '__getitem__' of a type (line 186)
        getitem___158830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), x2_158829, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 186)
        subscript_call_result_158831 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), getitem___158830, int_158828)
        
        # Evaluating assert statement condition
        
        
        # Obtaining the type of the subscript
        int_158832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 18), 'int')
        int_158833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 20), 'int')
        slice_158834 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 187, 15), int_158832, int_158833, None)
        # Getting the type of 'x1' (line 187)
        x1_158835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 15), 'x1')
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___158836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 15), x1_158835, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_158837 = invoke(stypy.reporting.localization.Localization(__file__, 187, 15), getitem___158836, slice_158834)
        
        # Obtaining the member 'shape' of a type (line 187)
        shape_158838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 15), subscript_call_result_158837, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 187)
        tuple_158839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 187)
        # Adding element type (line 187)
        int_158840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 187, 33), tuple_158839, int_158840)
        
        # Applying the binary operator '==' (line 187)
        result_eq_158841 = python_operator(stypy.reporting.localization.Localization(__file__, 187, 15), '==', shape_158838, tuple_158839)
        
        
        # Assigning a List to a Name (line 189):
        
        # Assigning a List to a Name (line 189):
        
        # Obtaining an instance of the builtin type 'list' (line 189)
        list_158842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 189)
        # Adding element type (line 189)
        int_158843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 13), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 12), list_158842, int_158843)
        # Adding element type (line 189)
        int_158844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 16), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 12), list_158842, int_158844)
        # Adding element type (line 189)
        int_158845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 19), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 12), list_158842, int_158845)
        # Adding element type (line 189)
        int_158846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 12), list_158842, int_158846)
        # Adding element type (line 189)
        int_158847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 25), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 12), list_158842, int_158847)
        
        # Assigning a type to the variable 'n' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'n', list_158842)
        
        # Assigning a Call to a Name (line 190):
        
        # Assigning a Call to a Name (line 190):
        
        # Call to make_mask(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'n' (line 190)
        n_158850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 27), 'n', False)
        # Processing the call keyword arguments (line 190)
        kwargs_158851 = {}
        # Getting the type of 'self' (line 190)
        self_158848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'self', False)
        # Obtaining the member 'make_mask' of a type (line 190)
        make_mask_158849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 12), self_158848, 'make_mask')
        # Calling make_mask(args, kwargs) (line 190)
        make_mask_call_result_158852 = invoke(stypy.reporting.localization.Localization(__file__, 190, 12), make_mask_158849, *[n_158850], **kwargs_158851)
        
        # Assigning a type to the variable 'm' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'm', make_mask_call_result_158852)
        
        # Assigning a Call to a Name (line 191):
        
        # Assigning a Call to a Name (line 191):
        
        # Call to make_mask(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'm' (line 191)
        m_158855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 28), 'm', False)
        # Processing the call keyword arguments (line 191)
        kwargs_158856 = {}
        # Getting the type of 'self' (line 191)
        self_158853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 13), 'self', False)
        # Obtaining the member 'make_mask' of a type (line 191)
        make_mask_158854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 13), self_158853, 'make_mask')
        # Calling make_mask(args, kwargs) (line 191)
        make_mask_call_result_158857 = invoke(stypy.reporting.localization.Localization(__file__, 191, 13), make_mask_158854, *[m_158855], **kwargs_158856)
        
        # Assigning a type to the variable 'm2' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'm2', make_mask_call_result_158857)
        # Evaluating assert statement condition
        
        # Getting the type of 'm' (line 192)
        m_158858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), 'm')
        # Getting the type of 'm2' (line 192)
        m2_158859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'm2')
        # Applying the binary operator 'is' (line 192)
        result_is__158860 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 15), 'is', m_158858, m2_158859)
        
        
        # Assigning a Call to a Name (line 193):
        
        # Assigning a Call to a Name (line 193):
        
        # Call to make_mask(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'm' (line 193)
        m_158863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 28), 'm', False)
        # Processing the call keyword arguments (line 193)
        int_158864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 36), 'int')
        keyword_158865 = int_158864
        kwargs_158866 = {'copy': keyword_158865}
        # Getting the type of 'self' (line 193)
        self_158861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 13), 'self', False)
        # Obtaining the member 'make_mask' of a type (line 193)
        make_mask_158862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 13), self_158861, 'make_mask')
        # Calling make_mask(args, kwargs) (line 193)
        make_mask_call_result_158867 = invoke(stypy.reporting.localization.Localization(__file__, 193, 13), make_mask_158862, *[m_158863], **kwargs_158866)
        
        # Assigning a type to the variable 'm3' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'm3', make_mask_call_result_158867)
        # Evaluating assert statement condition
        
        # Getting the type of 'm' (line 194)
        m_158868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 'm')
        # Getting the type of 'm3' (line 194)
        m3_158869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 24), 'm3')
        # Applying the binary operator 'isnot' (line 194)
        result_is_not_158870 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 15), 'isnot', m_158868, m3_158869)
        
        
        # ################# End of 'test_2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_2' in the type store
        # Getting the type of 'stypy_return_type' (line 154)
        stypy_return_type_158871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_158871)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_2'
        return stypy_return_type_158871


    @norecursion
    def test_3(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_3'
        module_type_store = module_type_store.open_function_context('test_3', 196, 4, False)
        # Assigning a type to the variable 'self' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ModuleTester.test_3.__dict__.__setitem__('stypy_localization', localization)
        ModuleTester.test_3.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ModuleTester.test_3.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleTester.test_3.__dict__.__setitem__('stypy_function_name', 'ModuleTester.test_3')
        ModuleTester.test_3.__dict__.__setitem__('stypy_param_names_list', [])
        ModuleTester.test_3.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleTester.test_3.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleTester.test_3.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleTester.test_3.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleTester.test_3.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleTester.test_3.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ModuleTester.test_3', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_3', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_3(...)' code ##################

        str_158872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, (-1)), 'str', '\n        Tests resize/repeat\n\n        ')
        
        # Assigning a Call to a Name (line 201):
        
        # Assigning a Call to a Name (line 201):
        
        # Call to arange(...): (line 201)
        # Processing the call arguments (line 201)
        int_158875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 25), 'int')
        # Processing the call keyword arguments (line 201)
        kwargs_158876 = {}
        # Getting the type of 'self' (line 201)
        self_158873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 13), 'self', False)
        # Obtaining the member 'arange' of a type (line 201)
        arange_158874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 13), self_158873, 'arange')
        # Calling arange(args, kwargs) (line 201)
        arange_call_result_158877 = invoke(stypy.reporting.localization.Localization(__file__, 201, 13), arange_158874, *[int_158875], **kwargs_158876)
        
        # Assigning a type to the variable 'x4' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'x4', arange_call_result_158877)
        
        # Assigning a Attribute to a Subscript (line 202):
        
        # Assigning a Attribute to a Subscript (line 202):
        # Getting the type of 'self' (line 202)
        self_158878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'self')
        # Obtaining the member 'masked' of a type (line 202)
        masked_158879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 16), self_158878, 'masked')
        # Getting the type of 'x4' (line 202)
        x4_158880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'x4')
        int_158881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 11), 'int')
        # Storing an element on a container (line 202)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 8), x4_158880, (int_158881, masked_158879))
        
        # Assigning a Call to a Name (line 203):
        
        # Assigning a Call to a Name (line 203):
        
        # Call to resize(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'x4' (line 203)
        x4_158884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 25), 'x4', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 203)
        tuple_158885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 203)
        # Adding element type (line 203)
        int_158886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 30), tuple_158885, int_158886)
        
        # Processing the call keyword arguments (line 203)
        kwargs_158887 = {}
        # Getting the type of 'self' (line 203)
        self_158882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 13), 'self', False)
        # Obtaining the member 'resize' of a type (line 203)
        resize_158883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 13), self_158882, 'resize')
        # Calling resize(args, kwargs) (line 203)
        resize_call_result_158888 = invoke(stypy.reporting.localization.Localization(__file__, 203, 13), resize_158883, *[x4_158884, tuple_158885], **kwargs_158887)
        
        # Assigning a type to the variable 'y4' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'y4', resize_call_result_158888)
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 204)
        # Processing the call arguments (line 204)
        
        # Call to concatenate(...): (line 204)
        # Processing the call arguments (line 204)
        
        # Obtaining an instance of the builtin type 'list' (line 204)
        list_158893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 204)
        # Adding element type (line 204)
        # Getting the type of 'x4' (line 204)
        x4_158894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 47), 'x4', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 46), list_158893, x4_158894)
        # Adding element type (line 204)
        # Getting the type of 'x4' (line 204)
        x4_158895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 51), 'x4', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 46), list_158893, x4_158895)
        
        # Processing the call keyword arguments (line 204)
        kwargs_158896 = {}
        # Getting the type of 'self' (line 204)
        self_158891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 29), 'self', False)
        # Obtaining the member 'concatenate' of a type (line 204)
        concatenate_158892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 29), self_158891, 'concatenate')
        # Calling concatenate(args, kwargs) (line 204)
        concatenate_call_result_158897 = invoke(stypy.reporting.localization.Localization(__file__, 204, 29), concatenate_158892, *[list_158893], **kwargs_158896)
        
        # Getting the type of 'y4' (line 204)
        y4_158898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 57), 'y4', False)
        # Processing the call keyword arguments (line 204)
        kwargs_158899 = {}
        # Getting the type of 'self' (line 204)
        self_158889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 204)
        allequal_158890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 15), self_158889, 'allequal')
        # Calling allequal(args, kwargs) (line 204)
        allequal_call_result_158900 = invoke(stypy.reporting.localization.Localization(__file__, 204, 15), allequal_158890, *[concatenate_call_result_158897, y4_158898], **kwargs_158899)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 205)
        # Processing the call arguments (line 205)
        
        # Call to getmask(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'y4' (line 205)
        y4_158905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 42), 'y4', False)
        # Processing the call keyword arguments (line 205)
        kwargs_158906 = {}
        # Getting the type of 'self' (line 205)
        self_158903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 29), 'self', False)
        # Obtaining the member 'getmask' of a type (line 205)
        getmask_158904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 29), self_158903, 'getmask')
        # Calling getmask(args, kwargs) (line 205)
        getmask_call_result_158907 = invoke(stypy.reporting.localization.Localization(__file__, 205, 29), getmask_158904, *[y4_158905], **kwargs_158906)
        
        
        # Obtaining an instance of the builtin type 'list' (line 205)
        list_158908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 205)
        # Adding element type (line 205)
        int_158909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 47), list_158908, int_158909)
        # Adding element type (line 205)
        int_158910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 47), list_158908, int_158910)
        # Adding element type (line 205)
        int_158911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 47), list_158908, int_158911)
        # Adding element type (line 205)
        int_158912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 47), list_158908, int_158912)
        # Adding element type (line 205)
        int_158913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 47), list_158908, int_158913)
        # Adding element type (line 205)
        int_158914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 63), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 47), list_158908, int_158914)
        # Adding element type (line 205)
        int_158915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 47), list_158908, int_158915)
        # Adding element type (line 205)
        int_158916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 69), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 47), list_158908, int_158916)
        
        # Processing the call keyword arguments (line 205)
        kwargs_158917 = {}
        # Getting the type of 'self' (line 205)
        self_158901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 205)
        allequal_158902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 15), self_158901, 'allequal')
        # Calling allequal(args, kwargs) (line 205)
        allequal_call_result_158918 = invoke(stypy.reporting.localization.Localization(__file__, 205, 15), allequal_158902, *[getmask_call_result_158907, list_158908], **kwargs_158917)
        
        
        # Assigning a Call to a Name (line 206):
        
        # Assigning a Call to a Name (line 206):
        
        # Call to repeat(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'x4' (line 206)
        x4_158921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 25), 'x4', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 206)
        tuple_158922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 206)
        # Adding element type (line 206)
        int_158923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 30), tuple_158922, int_158923)
        # Adding element type (line 206)
        int_158924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 30), tuple_158922, int_158924)
        # Adding element type (line 206)
        int_158925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 36), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 30), tuple_158922, int_158925)
        # Adding element type (line 206)
        int_158926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 30), tuple_158922, int_158926)
        
        # Processing the call keyword arguments (line 206)
        int_158927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 48), 'int')
        keyword_158928 = int_158927
        kwargs_158929 = {'axis': keyword_158928}
        # Getting the type of 'self' (line 206)
        self_158919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 13), 'self', False)
        # Obtaining the member 'repeat' of a type (line 206)
        repeat_158920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 13), self_158919, 'repeat')
        # Calling repeat(args, kwargs) (line 206)
        repeat_call_result_158930 = invoke(stypy.reporting.localization.Localization(__file__, 206, 13), repeat_158920, *[x4_158921, tuple_158922], **kwargs_158929)
        
        # Assigning a type to the variable 'y5' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'y5', repeat_call_result_158930)
        
        # Call to assert_array_equal(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'y5' (line 207)
        y5_158933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 32), 'y5', False)
        
        # Obtaining an instance of the builtin type 'list' (line 207)
        list_158934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 207)
        # Adding element type (line 207)
        int_158935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 37), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 36), list_158934, int_158935)
        # Adding element type (line 207)
        int_158936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 40), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 36), list_158934, int_158936)
        # Adding element type (line 207)
        int_158937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 36), list_158934, int_158937)
        # Adding element type (line 207)
        int_158938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 36), list_158934, int_158938)
        # Adding element type (line 207)
        int_158939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 36), list_158934, int_158939)
        # Adding element type (line 207)
        int_158940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 36), list_158934, int_158940)
        # Adding element type (line 207)
        int_158941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 36), list_158934, int_158941)
        # Adding element type (line 207)
        int_158942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 36), list_158934, int_158942)
        
        # Processing the call keyword arguments (line 207)
        kwargs_158943 = {}
        # Getting the type of 'self' (line 207)
        self_158931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 207)
        assert_array_equal_158932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), self_158931, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 207)
        assert_array_equal_call_result_158944 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), assert_array_equal_158932, *[y5_158933, list_158934], **kwargs_158943)
        
        
        # Assigning a Call to a Name (line 208):
        
        # Assigning a Call to a Name (line 208):
        
        # Call to repeat(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'x4' (line 208)
        x4_158947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 25), 'x4', False)
        int_158948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 29), 'int')
        # Processing the call keyword arguments (line 208)
        int_158949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 37), 'int')
        keyword_158950 = int_158949
        kwargs_158951 = {'axis': keyword_158950}
        # Getting the type of 'self' (line 208)
        self_158945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 13), 'self', False)
        # Obtaining the member 'repeat' of a type (line 208)
        repeat_158946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 13), self_158945, 'repeat')
        # Calling repeat(args, kwargs) (line 208)
        repeat_call_result_158952 = invoke(stypy.reporting.localization.Localization(__file__, 208, 13), repeat_158946, *[x4_158947, int_158948], **kwargs_158951)
        
        # Assigning a type to the variable 'y6' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'y6', repeat_call_result_158952)
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'y5' (line 209)
        y5_158955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 29), 'y5', False)
        # Getting the type of 'y6' (line 209)
        y6_158956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 33), 'y6', False)
        # Processing the call keyword arguments (line 209)
        kwargs_158957 = {}
        # Getting the type of 'self' (line 209)
        self_158953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 209)
        allequal_158954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 15), self_158953, 'allequal')
        # Calling allequal(args, kwargs) (line 209)
        allequal_call_result_158958 = invoke(stypy.reporting.localization.Localization(__file__, 209, 15), allequal_158954, *[y5_158955, y6_158956], **kwargs_158957)
        
        
        # Assigning a Call to a Name (line 210):
        
        # Assigning a Call to a Name (line 210):
        
        # Call to repeat(...): (line 210)
        # Processing the call arguments (line 210)
        
        # Obtaining an instance of the builtin type 'tuple' (line 210)
        tuple_158961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 210)
        # Adding element type (line 210)
        int_158962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 24), tuple_158961, int_158962)
        # Adding element type (line 210)
        int_158963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 24), tuple_158961, int_158963)
        # Adding element type (line 210)
        int_158964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 24), tuple_158961, int_158964)
        # Adding element type (line 210)
        int_158965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 24), tuple_158961, int_158965)
        
        # Processing the call keyword arguments (line 210)
        int_158966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 42), 'int')
        keyword_158967 = int_158966
        kwargs_158968 = {'axis': keyword_158967}
        # Getting the type of 'x4' (line 210)
        x4_158959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 13), 'x4', False)
        # Obtaining the member 'repeat' of a type (line 210)
        repeat_158960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 13), x4_158959, 'repeat')
        # Calling repeat(args, kwargs) (line 210)
        repeat_call_result_158969 = invoke(stypy.reporting.localization.Localization(__file__, 210, 13), repeat_158960, *[tuple_158961], **kwargs_158968)
        
        # Assigning a type to the variable 'y7' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'y7', repeat_call_result_158969)
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'y5' (line 211)
        y5_158972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 29), 'y5', False)
        # Getting the type of 'y7' (line 211)
        y7_158973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 33), 'y7', False)
        # Processing the call keyword arguments (line 211)
        kwargs_158974 = {}
        # Getting the type of 'self' (line 211)
        self_158970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 211)
        allequal_158971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 15), self_158970, 'allequal')
        # Calling allequal(args, kwargs) (line 211)
        allequal_call_result_158975 = invoke(stypy.reporting.localization.Localization(__file__, 211, 15), allequal_158971, *[y5_158972, y7_158973], **kwargs_158974)
        
        
        # Assigning a Call to a Name (line 212):
        
        # Assigning a Call to a Name (line 212):
        
        # Call to repeat(...): (line 212)
        # Processing the call arguments (line 212)
        int_158978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 23), 'int')
        int_158979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 26), 'int')
        # Processing the call keyword arguments (line 212)
        kwargs_158980 = {}
        # Getting the type of 'x4' (line 212)
        x4_158976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 13), 'x4', False)
        # Obtaining the member 'repeat' of a type (line 212)
        repeat_158977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 13), x4_158976, 'repeat')
        # Calling repeat(args, kwargs) (line 212)
        repeat_call_result_158981 = invoke(stypy.reporting.localization.Localization(__file__, 212, 13), repeat_158977, *[int_158978, int_158979], **kwargs_158980)
        
        # Assigning a type to the variable 'y8' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'y8', repeat_call_result_158981)
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 213)
        # Processing the call arguments (line 213)
        # Getting the type of 'y5' (line 213)
        y5_158984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 29), 'y5', False)
        # Getting the type of 'y8' (line 213)
        y8_158985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 33), 'y8', False)
        # Processing the call keyword arguments (line 213)
        kwargs_158986 = {}
        # Getting the type of 'self' (line 213)
        self_158982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 213)
        allequal_158983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 15), self_158982, 'allequal')
        # Calling allequal(args, kwargs) (line 213)
        allequal_call_result_158987 = invoke(stypy.reporting.localization.Localization(__file__, 213, 15), allequal_158983, *[y5_158984, y8_158985], **kwargs_158986)
        
        
        # ################# End of 'test_3(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_3' in the type store
        # Getting the type of 'stypy_return_type' (line 196)
        stypy_return_type_158988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_158988)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_3'
        return stypy_return_type_158988


    @norecursion
    def test_4(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_4'
        module_type_store = module_type_store.open_function_context('test_4', 215, 4, False)
        # Assigning a type to the variable 'self' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ModuleTester.test_4.__dict__.__setitem__('stypy_localization', localization)
        ModuleTester.test_4.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ModuleTester.test_4.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleTester.test_4.__dict__.__setitem__('stypy_function_name', 'ModuleTester.test_4')
        ModuleTester.test_4.__dict__.__setitem__('stypy_param_names_list', [])
        ModuleTester.test_4.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleTester.test_4.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleTester.test_4.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleTester.test_4.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleTester.test_4.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleTester.test_4.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ModuleTester.test_4', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_4', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_4(...)' code ##################

        str_158989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, (-1)), 'str', '\n        Test of take, transpose, inner, outer products.\n\n        ')
        
        # Assigning a Call to a Name (line 220):
        
        # Assigning a Call to a Name (line 220):
        
        # Call to arange(...): (line 220)
        # Processing the call arguments (line 220)
        int_158992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 24), 'int')
        # Processing the call keyword arguments (line 220)
        kwargs_158993 = {}
        # Getting the type of 'self' (line 220)
        self_158990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 220)
        arange_158991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), self_158990, 'arange')
        # Calling arange(args, kwargs) (line 220)
        arange_call_result_158994 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), arange_158991, *[int_158992], **kwargs_158993)
        
        # Assigning a type to the variable 'x' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'x', arange_call_result_158994)
        
        # Assigning a Call to a Name (line 221):
        
        # Assigning a Call to a Name (line 221):
        
        # Call to arange(...): (line 221)
        # Processing the call arguments (line 221)
        int_158997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 22), 'int')
        # Processing the call keyword arguments (line 221)
        kwargs_158998 = {}
        # Getting the type of 'np' (line 221)
        np_158995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'np', False)
        # Obtaining the member 'arange' of a type (line 221)
        arange_158996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 12), np_158995, 'arange')
        # Calling arange(args, kwargs) (line 221)
        arange_call_result_158999 = invoke(stypy.reporting.localization.Localization(__file__, 221, 12), arange_158996, *[int_158997], **kwargs_158998)
        
        # Assigning a type to the variable 'y' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'y', arange_call_result_158999)
        
        # Assigning a Attribute to a Subscript (line 222):
        
        # Assigning a Attribute to a Subscript (line 222):
        # Getting the type of 'self' (line 222)
        self_159000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 17), 'self')
        # Obtaining the member 'masked' of a type (line 222)
        masked_159001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 17), self_159000, 'masked')
        # Getting the type of 'x' (line 222)
        x_159002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'x')
        int_159003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 10), 'int')
        int_159004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 12), 'int')
        slice_159005 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 222, 8), int_159003, int_159004, None)
        # Storing an element on a container (line 222)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 8), x_159002, (slice_159005, masked_159001))
        
        # Assigning a Call to a Name (line 223):
        
        # Assigning a Call to a Name (line 223):
        
        # Call to reshape(...): (line 223)
        # Processing the call arguments (line 223)
        int_159008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 22), 'int')
        int_159009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 25), 'int')
        int_159010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 28), 'int')
        # Processing the call keyword arguments (line 223)
        kwargs_159011 = {}
        # Getting the type of 'x' (line 223)
        x_159006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'x', False)
        # Obtaining the member 'reshape' of a type (line 223)
        reshape_159007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 12), x_159006, 'reshape')
        # Calling reshape(args, kwargs) (line 223)
        reshape_call_result_159012 = invoke(stypy.reporting.localization.Localization(__file__, 223, 12), reshape_159007, *[int_159008, int_159009, int_159010], **kwargs_159011)
        
        # Assigning a type to the variable 'x' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'x', reshape_call_result_159012)
        
        # Assigning a Call to a Name (line 224):
        
        # Assigning a Call to a Name (line 224):
        
        # Call to reshape(...): (line 224)
        # Processing the call arguments (line 224)
        int_159015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 22), 'int')
        int_159016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 25), 'int')
        int_159017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 28), 'int')
        # Processing the call keyword arguments (line 224)
        kwargs_159018 = {}
        # Getting the type of 'y' (line 224)
        y_159013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'y', False)
        # Obtaining the member 'reshape' of a type (line 224)
        reshape_159014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 12), y_159013, 'reshape')
        # Calling reshape(args, kwargs) (line 224)
        reshape_call_result_159019 = invoke(stypy.reporting.localization.Localization(__file__, 224, 12), reshape_159014, *[int_159015, int_159016, int_159017], **kwargs_159018)
        
        # Assigning a type to the variable 'y' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'y', reshape_call_result_159019)
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 225)
        # Processing the call arguments (line 225)
        
        # Call to transpose(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'y' (line 225)
        y_159024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 42), 'y', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 225)
        tuple_159025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 46), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 225)
        # Adding element type (line 225)
        int_159026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 46), tuple_159025, int_159026)
        # Adding element type (line 225)
        int_159027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 46), tuple_159025, int_159027)
        # Adding element type (line 225)
        int_159028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 46), tuple_159025, int_159028)
        
        # Processing the call keyword arguments (line 225)
        kwargs_159029 = {}
        # Getting the type of 'np' (line 225)
        np_159022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 29), 'np', False)
        # Obtaining the member 'transpose' of a type (line 225)
        transpose_159023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 29), np_159022, 'transpose')
        # Calling transpose(args, kwargs) (line 225)
        transpose_call_result_159030 = invoke(stypy.reporting.localization.Localization(__file__, 225, 29), transpose_159023, *[y_159024, tuple_159025], **kwargs_159029)
        
        
        # Call to transpose(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'x' (line 225)
        x_159033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 72), 'x', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 225)
        tuple_159034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 76), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 225)
        # Adding element type (line 225)
        int_159035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 76), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 76), tuple_159034, int_159035)
        # Adding element type (line 225)
        int_159036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 79), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 76), tuple_159034, int_159036)
        # Adding element type (line 225)
        int_159037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 82), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 76), tuple_159034, int_159037)
        
        # Processing the call keyword arguments (line 225)
        kwargs_159038 = {}
        # Getting the type of 'self' (line 225)
        self_159031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 57), 'self', False)
        # Obtaining the member 'transpose' of a type (line 225)
        transpose_159032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 57), self_159031, 'transpose')
        # Calling transpose(args, kwargs) (line 225)
        transpose_call_result_159039 = invoke(stypy.reporting.localization.Localization(__file__, 225, 57), transpose_159032, *[x_159033, tuple_159034], **kwargs_159038)
        
        # Processing the call keyword arguments (line 225)
        kwargs_159040 = {}
        # Getting the type of 'self' (line 225)
        self_159020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 225)
        allequal_159021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 15), self_159020, 'allequal')
        # Calling allequal(args, kwargs) (line 225)
        allequal_call_result_159041 = invoke(stypy.reporting.localization.Localization(__file__, 225, 15), allequal_159021, *[transpose_call_result_159030, transpose_call_result_159039], **kwargs_159040)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Call to take(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'y' (line 226)
        y_159046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 37), 'y', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 226)
        tuple_159047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 226)
        # Adding element type (line 226)
        int_159048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 41), tuple_159047, int_159048)
        # Adding element type (line 226)
        int_159049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 41), tuple_159047, int_159049)
        # Adding element type (line 226)
        int_159050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 41), tuple_159047, int_159050)
        
        int_159051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 51), 'int')
        # Processing the call keyword arguments (line 226)
        kwargs_159052 = {}
        # Getting the type of 'np' (line 226)
        np_159044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 29), 'np', False)
        # Obtaining the member 'take' of a type (line 226)
        take_159045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 29), np_159044, 'take')
        # Calling take(args, kwargs) (line 226)
        take_call_result_159053 = invoke(stypy.reporting.localization.Localization(__file__, 226, 29), take_159045, *[y_159046, tuple_159047, int_159051], **kwargs_159052)
        
        
        # Call to take(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'x' (line 226)
        x_159056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 65), 'x', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 226)
        tuple_159057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 69), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 226)
        # Adding element type (line 226)
        int_159058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 69), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 69), tuple_159057, int_159058)
        # Adding element type (line 226)
        int_159059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 72), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 69), tuple_159057, int_159059)
        # Adding element type (line 226)
        int_159060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 75), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 69), tuple_159057, int_159060)
        
        int_159061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 79), 'int')
        # Processing the call keyword arguments (line 226)
        kwargs_159062 = {}
        # Getting the type of 'self' (line 226)
        self_159054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 55), 'self', False)
        # Obtaining the member 'take' of a type (line 226)
        take_159055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 55), self_159054, 'take')
        # Calling take(args, kwargs) (line 226)
        take_call_result_159063 = invoke(stypy.reporting.localization.Localization(__file__, 226, 55), take_159055, *[x_159056, tuple_159057, int_159061], **kwargs_159062)
        
        # Processing the call keyword arguments (line 226)
        kwargs_159064 = {}
        # Getting the type of 'self' (line 226)
        self_159042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 226)
        allequal_159043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 15), self_159042, 'allequal')
        # Calling allequal(args, kwargs) (line 226)
        allequal_call_result_159065 = invoke(stypy.reporting.localization.Localization(__file__, 226, 15), allequal_159043, *[take_call_result_159053, take_call_result_159063], **kwargs_159064)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 227)
        # Processing the call arguments (line 227)
        
        # Call to inner(...): (line 227)
        # Processing the call arguments (line 227)
        
        # Call to filled(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'x' (line 227)
        x_159072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 50), 'x', False)
        int_159073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 53), 'int')
        # Processing the call keyword arguments (line 227)
        kwargs_159074 = {}
        # Getting the type of 'self' (line 227)
        self_159070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 38), 'self', False)
        # Obtaining the member 'filled' of a type (line 227)
        filled_159071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 38), self_159070, 'filled')
        # Calling filled(args, kwargs) (line 227)
        filled_call_result_159075 = invoke(stypy.reporting.localization.Localization(__file__, 227, 38), filled_159071, *[x_159072, int_159073], **kwargs_159074)
        
        
        # Call to filled(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'y' (line 227)
        y_159078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 69), 'y', False)
        int_159079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 72), 'int')
        # Processing the call keyword arguments (line 227)
        kwargs_159080 = {}
        # Getting the type of 'self' (line 227)
        self_159076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 57), 'self', False)
        # Obtaining the member 'filled' of a type (line 227)
        filled_159077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 57), self_159076, 'filled')
        # Calling filled(args, kwargs) (line 227)
        filled_call_result_159081 = invoke(stypy.reporting.localization.Localization(__file__, 227, 57), filled_159077, *[y_159078, int_159079], **kwargs_159080)
        
        # Processing the call keyword arguments (line 227)
        kwargs_159082 = {}
        # Getting the type of 'np' (line 227)
        np_159068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 29), 'np', False)
        # Obtaining the member 'inner' of a type (line 227)
        inner_159069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 29), np_159068, 'inner')
        # Calling inner(args, kwargs) (line 227)
        inner_call_result_159083 = invoke(stypy.reporting.localization.Localization(__file__, 227, 29), inner_159069, *[filled_call_result_159075, filled_call_result_159081], **kwargs_159082)
        
        
        # Call to inner(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'x' (line 228)
        x_159086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 39), 'x', False)
        # Getting the type of 'y' (line 228)
        y_159087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 42), 'y', False)
        # Processing the call keyword arguments (line 228)
        kwargs_159088 = {}
        # Getting the type of 'self' (line 228)
        self_159084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 28), 'self', False)
        # Obtaining the member 'inner' of a type (line 228)
        inner_159085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 28), self_159084, 'inner')
        # Calling inner(args, kwargs) (line 228)
        inner_call_result_159089 = invoke(stypy.reporting.localization.Localization(__file__, 228, 28), inner_159085, *[x_159086, y_159087], **kwargs_159088)
        
        # Processing the call keyword arguments (line 227)
        kwargs_159090 = {}
        # Getting the type of 'self' (line 227)
        self_159066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 227)
        allequal_159067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 15), self_159066, 'allequal')
        # Calling allequal(args, kwargs) (line 227)
        allequal_call_result_159091 = invoke(stypy.reporting.localization.Localization(__file__, 227, 15), allequal_159067, *[inner_call_result_159083, inner_call_result_159089], **kwargs_159090)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 229)
        # Processing the call arguments (line 229)
        
        # Call to outer(...): (line 229)
        # Processing the call arguments (line 229)
        
        # Call to filled(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'x' (line 229)
        x_159098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 50), 'x', False)
        int_159099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 53), 'int')
        # Processing the call keyword arguments (line 229)
        kwargs_159100 = {}
        # Getting the type of 'self' (line 229)
        self_159096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 38), 'self', False)
        # Obtaining the member 'filled' of a type (line 229)
        filled_159097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 38), self_159096, 'filled')
        # Calling filled(args, kwargs) (line 229)
        filled_call_result_159101 = invoke(stypy.reporting.localization.Localization(__file__, 229, 38), filled_159097, *[x_159098, int_159099], **kwargs_159100)
        
        
        # Call to filled(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'y' (line 229)
        y_159104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 69), 'y', False)
        int_159105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 72), 'int')
        # Processing the call keyword arguments (line 229)
        kwargs_159106 = {}
        # Getting the type of 'self' (line 229)
        self_159102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 57), 'self', False)
        # Obtaining the member 'filled' of a type (line 229)
        filled_159103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 57), self_159102, 'filled')
        # Calling filled(args, kwargs) (line 229)
        filled_call_result_159107 = invoke(stypy.reporting.localization.Localization(__file__, 229, 57), filled_159103, *[y_159104, int_159105], **kwargs_159106)
        
        # Processing the call keyword arguments (line 229)
        kwargs_159108 = {}
        # Getting the type of 'np' (line 229)
        np_159094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 29), 'np', False)
        # Obtaining the member 'outer' of a type (line 229)
        outer_159095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 29), np_159094, 'outer')
        # Calling outer(args, kwargs) (line 229)
        outer_call_result_159109 = invoke(stypy.reporting.localization.Localization(__file__, 229, 29), outer_159095, *[filled_call_result_159101, filled_call_result_159107], **kwargs_159108)
        
        
        # Call to outer(...): (line 230)
        # Processing the call arguments (line 230)
        # Getting the type of 'x' (line 230)
        x_159112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 39), 'x', False)
        # Getting the type of 'y' (line 230)
        y_159113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 42), 'y', False)
        # Processing the call keyword arguments (line 230)
        kwargs_159114 = {}
        # Getting the type of 'self' (line 230)
        self_159110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 28), 'self', False)
        # Obtaining the member 'outer' of a type (line 230)
        outer_159111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 28), self_159110, 'outer')
        # Calling outer(args, kwargs) (line 230)
        outer_call_result_159115 = invoke(stypy.reporting.localization.Localization(__file__, 230, 28), outer_159111, *[x_159112, y_159113], **kwargs_159114)
        
        # Processing the call keyword arguments (line 229)
        kwargs_159116 = {}
        # Getting the type of 'self' (line 229)
        self_159092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 229)
        allequal_159093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 15), self_159092, 'allequal')
        # Calling allequal(args, kwargs) (line 229)
        allequal_call_result_159117 = invoke(stypy.reporting.localization.Localization(__file__, 229, 15), allequal_159093, *[outer_call_result_159109, outer_call_result_159115], **kwargs_159116)
        
        
        # Assigning a Call to a Name (line 231):
        
        # Assigning a Call to a Name (line 231):
        
        # Call to array(...): (line 231)
        # Processing the call arguments (line 231)
        
        # Obtaining an instance of the builtin type 'list' (line 231)
        list_159120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 231)
        # Adding element type (line 231)
        str_159121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 24), 'str', 'abc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 23), list_159120, str_159121)
        # Adding element type (line 231)
        int_159122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 31), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 23), list_159120, int_159122)
        # Adding element type (line 231)
        str_159123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 34), 'str', 'def')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 23), list_159120, str_159123)
        # Adding element type (line 231)
        int_159124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 23), list_159120, int_159124)
        # Adding element type (line 231)
        int_159125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 23), list_159120, int_159125)
        
        # Getting the type of 'object' (line 231)
        object_159126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 48), 'object', False)
        # Processing the call keyword arguments (line 231)
        kwargs_159127 = {}
        # Getting the type of 'self' (line 231)
        self_159118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'self', False)
        # Obtaining the member 'array' of a type (line 231)
        array_159119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), self_159118, 'array')
        # Calling array(args, kwargs) (line 231)
        array_call_result_159128 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), array_159119, *[list_159120, object_159126], **kwargs_159127)
        
        # Assigning a type to the variable 'y' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'y', array_call_result_159128)
        
        # Assigning a Attribute to a Subscript (line 232):
        
        # Assigning a Attribute to a Subscript (line 232):
        # Getting the type of 'self' (line 232)
        self_159129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 'self')
        # Obtaining the member 'masked' of a type (line 232)
        masked_159130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 15), self_159129, 'masked')
        # Getting the type of 'y' (line 232)
        y_159131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'y')
        int_159132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 10), 'int')
        # Storing an element on a container (line 232)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 232, 8), y_159131, (int_159132, masked_159130))
        
        # Assigning a Call to a Name (line 233):
        
        # Assigning a Call to a Name (line 233):
        
        # Call to take(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'y' (line 233)
        y_159135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 22), 'y', False)
        
        # Obtaining an instance of the builtin type 'list' (line 233)
        list_159136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 233)
        # Adding element type (line 233)
        int_159137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 25), list_159136, int_159137)
        # Adding element type (line 233)
        int_159138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 25), list_159136, int_159138)
        # Adding element type (line 233)
        int_159139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 32), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 25), list_159136, int_159139)
        
        # Processing the call keyword arguments (line 233)
        kwargs_159140 = {}
        # Getting the type of 'self' (line 233)
        self_159133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'self', False)
        # Obtaining the member 'take' of a type (line 233)
        take_159134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 12), self_159133, 'take')
        # Calling take(args, kwargs) (line 233)
        take_call_result_159141 = invoke(stypy.reporting.localization.Localization(__file__, 233, 12), take_159134, *[y_159135, list_159136], **kwargs_159140)
        
        # Assigning a type to the variable 't' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 't', take_call_result_159141)
        # Evaluating assert statement condition
        
        
        # Obtaining the type of the subscript
        int_159142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 17), 'int')
        # Getting the type of 't' (line 234)
        t_159143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 15), 't')
        # Obtaining the member '__getitem__' of a type (line 234)
        getitem___159144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 15), t_159143, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 234)
        subscript_call_result_159145 = invoke(stypy.reporting.localization.Localization(__file__, 234, 15), getitem___159144, int_159142)
        
        str_159146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 23), 'str', 'abc')
        # Applying the binary operator '==' (line 234)
        result_eq_159147 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 15), '==', subscript_call_result_159145, str_159146)
        
        # Evaluating assert statement condition
        
        
        # Obtaining the type of the subscript
        int_159148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 17), 'int')
        # Getting the type of 't' (line 235)
        t_159149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 15), 't')
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___159150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 15), t_159149, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_159151 = invoke(stypy.reporting.localization.Localization(__file__, 235, 15), getitem___159150, int_159148)
        
        int_159152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 23), 'int')
        # Applying the binary operator '==' (line 235)
        result_eq_159153 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 15), '==', subscript_call_result_159151, int_159152)
        
        # Evaluating assert statement condition
        
        
        # Obtaining the type of the subscript
        int_159154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 17), 'int')
        # Getting the type of 't' (line 236)
        t_159155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 15), 't')
        # Obtaining the member '__getitem__' of a type (line 236)
        getitem___159156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 15), t_159155, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 236)
        subscript_call_result_159157 = invoke(stypy.reporting.localization.Localization(__file__, 236, 15), getitem___159156, int_159154)
        
        int_159158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 23), 'int')
        # Applying the binary operator '==' (line 236)
        result_eq_159159 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 15), '==', subscript_call_result_159157, int_159158)
        
        
        # ################# End of 'test_4(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_4' in the type store
        # Getting the type of 'stypy_return_type' (line 215)
        stypy_return_type_159160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_159160)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_4'
        return stypy_return_type_159160


    @norecursion
    def test_5(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_5'
        module_type_store = module_type_store.open_function_context('test_5', 238, 4, False)
        # Assigning a type to the variable 'self' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ModuleTester.test_5.__dict__.__setitem__('stypy_localization', localization)
        ModuleTester.test_5.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ModuleTester.test_5.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleTester.test_5.__dict__.__setitem__('stypy_function_name', 'ModuleTester.test_5')
        ModuleTester.test_5.__dict__.__setitem__('stypy_param_names_list', [])
        ModuleTester.test_5.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleTester.test_5.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleTester.test_5.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleTester.test_5.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleTester.test_5.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleTester.test_5.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ModuleTester.test_5', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_5', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_5(...)' code ##################

        str_159161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, (-1)), 'str', '\n        Tests inplace w/ scalar\n\n        ')
        
        # Assigning a Call to a Name (line 243):
        
        # Assigning a Call to a Name (line 243):
        
        # Call to arange(...): (line 243)
        # Processing the call arguments (line 243)
        int_159164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 24), 'int')
        # Processing the call keyword arguments (line 243)
        kwargs_159165 = {}
        # Getting the type of 'self' (line 243)
        self_159162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 243)
        arange_159163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), self_159162, 'arange')
        # Calling arange(args, kwargs) (line 243)
        arange_call_result_159166 = invoke(stypy.reporting.localization.Localization(__file__, 243, 12), arange_159163, *[int_159164], **kwargs_159165)
        
        # Assigning a type to the variable 'x' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'x', arange_call_result_159166)
        
        # Assigning a Call to a Name (line 244):
        
        # Assigning a Call to a Name (line 244):
        
        # Call to arange(...): (line 244)
        # Processing the call arguments (line 244)
        int_159169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 24), 'int')
        # Processing the call keyword arguments (line 244)
        kwargs_159170 = {}
        # Getting the type of 'self' (line 244)
        self_159167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 244)
        arange_159168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), self_159167, 'arange')
        # Calling arange(args, kwargs) (line 244)
        arange_call_result_159171 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), arange_159168, *[int_159169], **kwargs_159170)
        
        # Assigning a type to the variable 'y' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'y', arange_call_result_159171)
        
        # Assigning a Call to a Name (line 245):
        
        # Assigning a Call to a Name (line 245):
        
        # Call to arange(...): (line 245)
        # Processing the call arguments (line 245)
        int_159174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 25), 'int')
        # Processing the call keyword arguments (line 245)
        kwargs_159175 = {}
        # Getting the type of 'self' (line 245)
        self_159172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 13), 'self', False)
        # Obtaining the member 'arange' of a type (line 245)
        arange_159173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 13), self_159172, 'arange')
        # Calling arange(args, kwargs) (line 245)
        arange_call_result_159176 = invoke(stypy.reporting.localization.Localization(__file__, 245, 13), arange_159173, *[int_159174], **kwargs_159175)
        
        # Assigning a type to the variable 'xm' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'xm', arange_call_result_159176)
        
        # Assigning a Attribute to a Subscript (line 246):
        
        # Assigning a Attribute to a Subscript (line 246):
        # Getting the type of 'self' (line 246)
        self_159177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'self')
        # Obtaining the member 'masked' of a type (line 246)
        masked_159178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 16), self_159177, 'masked')
        # Getting the type of 'xm' (line 246)
        xm_159179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'xm')
        int_159180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 11), 'int')
        # Storing an element on a container (line 246)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 8), xm_159179, (int_159180, masked_159178))
        
        # Getting the type of 'x' (line 247)
        x_159181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'x')
        int_159182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 13), 'int')
        # Applying the binary operator '+=' (line 247)
        result_iadd_159183 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 8), '+=', x_159181, int_159182)
        # Assigning a type to the variable 'x' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'x', result_iadd_159183)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'x' (line 248)
        x_159186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 29), 'x', False)
        # Getting the type of 'y' (line 248)
        y_159187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 32), 'y', False)
        int_159188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 34), 'int')
        # Applying the binary operator '+' (line 248)
        result_add_159189 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 32), '+', y_159187, int_159188)
        
        # Processing the call keyword arguments (line 248)
        kwargs_159190 = {}
        # Getting the type of 'self' (line 248)
        self_159184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 248)
        allequal_159185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 15), self_159184, 'allequal')
        # Calling allequal(args, kwargs) (line 248)
        allequal_call_result_159191 = invoke(stypy.reporting.localization.Localization(__file__, 248, 15), allequal_159185, *[x_159186, result_add_159189], **kwargs_159190)
        
        
        # Getting the type of 'xm' (line 249)
        xm_159192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'xm')
        int_159193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 14), 'int')
        # Applying the binary operator '+=' (line 249)
        result_iadd_159194 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 8), '+=', xm_159192, int_159193)
        # Assigning a type to the variable 'xm' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'xm', result_iadd_159194)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'xm' (line 250)
        xm_159197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 29), 'xm', False)
        # Getting the type of 'y' (line 250)
        y_159198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 33), 'y', False)
        int_159199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 35), 'int')
        # Applying the binary operator '+' (line 250)
        result_add_159200 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 33), '+', y_159198, int_159199)
        
        # Processing the call keyword arguments (line 250)
        kwargs_159201 = {}
        # Getting the type of 'self' (line 250)
        self_159195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 250)
        allequal_159196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 15), self_159195, 'allequal')
        # Calling allequal(args, kwargs) (line 250)
        allequal_call_result_159202 = invoke(stypy.reporting.localization.Localization(__file__, 250, 15), allequal_159196, *[xm_159197, result_add_159200], **kwargs_159201)
        
        
        # Assigning a Call to a Name (line 252):
        
        # Assigning a Call to a Name (line 252):
        
        # Call to arange(...): (line 252)
        # Processing the call arguments (line 252)
        int_159205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 24), 'int')
        # Processing the call keyword arguments (line 252)
        kwargs_159206 = {}
        # Getting the type of 'self' (line 252)
        self_159203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 252)
        arange_159204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 12), self_159203, 'arange')
        # Calling arange(args, kwargs) (line 252)
        arange_call_result_159207 = invoke(stypy.reporting.localization.Localization(__file__, 252, 12), arange_159204, *[int_159205], **kwargs_159206)
        
        # Assigning a type to the variable 'x' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'x', arange_call_result_159207)
        
        # Assigning a Call to a Name (line 253):
        
        # Assigning a Call to a Name (line 253):
        
        # Call to arange(...): (line 253)
        # Processing the call arguments (line 253)
        int_159210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 25), 'int')
        # Processing the call keyword arguments (line 253)
        kwargs_159211 = {}
        # Getting the type of 'self' (line 253)
        self_159208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 13), 'self', False)
        # Obtaining the member 'arange' of a type (line 253)
        arange_159209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 13), self_159208, 'arange')
        # Calling arange(args, kwargs) (line 253)
        arange_call_result_159212 = invoke(stypy.reporting.localization.Localization(__file__, 253, 13), arange_159209, *[int_159210], **kwargs_159211)
        
        # Assigning a type to the variable 'xm' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'xm', arange_call_result_159212)
        
        # Assigning a Attribute to a Subscript (line 254):
        
        # Assigning a Attribute to a Subscript (line 254):
        # Getting the type of 'self' (line 254)
        self_159213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'self')
        # Obtaining the member 'masked' of a type (line 254)
        masked_159214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 16), self_159213, 'masked')
        # Getting the type of 'xm' (line 254)
        xm_159215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'xm')
        int_159216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 11), 'int')
        # Storing an element on a container (line 254)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 8), xm_159215, (int_159216, masked_159214))
        
        # Getting the type of 'x' (line 255)
        x_159217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'x')
        int_159218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 13), 'int')
        # Applying the binary operator '-=' (line 255)
        result_isub_159219 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 8), '-=', x_159217, int_159218)
        # Assigning a type to the variable 'x' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'x', result_isub_159219)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 256)
        # Processing the call arguments (line 256)
        # Getting the type of 'x' (line 256)
        x_159222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 29), 'x', False)
        # Getting the type of 'y' (line 256)
        y_159223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 32), 'y', False)
        int_159224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 34), 'int')
        # Applying the binary operator '-' (line 256)
        result_sub_159225 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 32), '-', y_159223, int_159224)
        
        # Processing the call keyword arguments (line 256)
        kwargs_159226 = {}
        # Getting the type of 'self' (line 256)
        self_159220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 256)
        allequal_159221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 15), self_159220, 'allequal')
        # Calling allequal(args, kwargs) (line 256)
        allequal_call_result_159227 = invoke(stypy.reporting.localization.Localization(__file__, 256, 15), allequal_159221, *[x_159222, result_sub_159225], **kwargs_159226)
        
        
        # Getting the type of 'xm' (line 257)
        xm_159228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'xm')
        int_159229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 14), 'int')
        # Applying the binary operator '-=' (line 257)
        result_isub_159230 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 8), '-=', xm_159228, int_159229)
        # Assigning a type to the variable 'xm' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'xm', result_isub_159230)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'xm' (line 258)
        xm_159233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 29), 'xm', False)
        # Getting the type of 'y' (line 258)
        y_159234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 33), 'y', False)
        int_159235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 35), 'int')
        # Applying the binary operator '-' (line 258)
        result_sub_159236 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 33), '-', y_159234, int_159235)
        
        # Processing the call keyword arguments (line 258)
        kwargs_159237 = {}
        # Getting the type of 'self' (line 258)
        self_159231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 258)
        allequal_159232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 15), self_159231, 'allequal')
        # Calling allequal(args, kwargs) (line 258)
        allequal_call_result_159238 = invoke(stypy.reporting.localization.Localization(__file__, 258, 15), allequal_159232, *[xm_159233, result_sub_159236], **kwargs_159237)
        
        
        # Assigning a BinOp to a Name (line 260):
        
        # Assigning a BinOp to a Name (line 260):
        
        # Call to arange(...): (line 260)
        # Processing the call arguments (line 260)
        int_159241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 24), 'int')
        # Processing the call keyword arguments (line 260)
        kwargs_159242 = {}
        # Getting the type of 'self' (line 260)
        self_159239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 260)
        arange_159240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 12), self_159239, 'arange')
        # Calling arange(args, kwargs) (line 260)
        arange_call_result_159243 = invoke(stypy.reporting.localization.Localization(__file__, 260, 12), arange_159240, *[int_159241], **kwargs_159242)
        
        float_159244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 28), 'float')
        # Applying the binary operator '*' (line 260)
        result_mul_159245 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 12), '*', arange_call_result_159243, float_159244)
        
        # Assigning a type to the variable 'x' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'x', result_mul_159245)
        
        # Assigning a BinOp to a Name (line 261):
        
        # Assigning a BinOp to a Name (line 261):
        
        # Call to arange(...): (line 261)
        # Processing the call arguments (line 261)
        int_159248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 25), 'int')
        # Processing the call keyword arguments (line 261)
        kwargs_159249 = {}
        # Getting the type of 'self' (line 261)
        self_159246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 13), 'self', False)
        # Obtaining the member 'arange' of a type (line 261)
        arange_159247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 13), self_159246, 'arange')
        # Calling arange(args, kwargs) (line 261)
        arange_call_result_159250 = invoke(stypy.reporting.localization.Localization(__file__, 261, 13), arange_159247, *[int_159248], **kwargs_159249)
        
        float_159251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 29), 'float')
        # Applying the binary operator '*' (line 261)
        result_mul_159252 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 13), '*', arange_call_result_159250, float_159251)
        
        # Assigning a type to the variable 'xm' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'xm', result_mul_159252)
        
        # Assigning a Attribute to a Subscript (line 262):
        
        # Assigning a Attribute to a Subscript (line 262):
        # Getting the type of 'self' (line 262)
        self_159253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 16), 'self')
        # Obtaining the member 'masked' of a type (line 262)
        masked_159254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 16), self_159253, 'masked')
        # Getting the type of 'xm' (line 262)
        xm_159255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'xm')
        int_159256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 11), 'int')
        # Storing an element on a container (line 262)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 8), xm_159255, (int_159256, masked_159254))
        
        # Getting the type of 'x' (line 263)
        x_159257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'x')
        float_159258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 13), 'float')
        # Applying the binary operator '*=' (line 263)
        result_imul_159259 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 8), '*=', x_159257, float_159258)
        # Assigning a type to the variable 'x' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'x', result_imul_159259)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'x' (line 264)
        x_159262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 29), 'x', False)
        # Getting the type of 'y' (line 264)
        y_159263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 32), 'y', False)
        int_159264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 34), 'int')
        # Applying the binary operator '*' (line 264)
        result_mul_159265 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 32), '*', y_159263, int_159264)
        
        # Processing the call keyword arguments (line 264)
        kwargs_159266 = {}
        # Getting the type of 'self' (line 264)
        self_159260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 264)
        allequal_159261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 15), self_159260, 'allequal')
        # Calling allequal(args, kwargs) (line 264)
        allequal_call_result_159267 = invoke(stypy.reporting.localization.Localization(__file__, 264, 15), allequal_159261, *[x_159262, result_mul_159265], **kwargs_159266)
        
        
        # Getting the type of 'xm' (line 265)
        xm_159268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'xm')
        float_159269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 14), 'float')
        # Applying the binary operator '*=' (line 265)
        result_imul_159270 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 8), '*=', xm_159268, float_159269)
        # Assigning a type to the variable 'xm' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'xm', result_imul_159270)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'xm' (line 266)
        xm_159273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 29), 'xm', False)
        # Getting the type of 'y' (line 266)
        y_159274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 33), 'y', False)
        int_159275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 35), 'int')
        # Applying the binary operator '*' (line 266)
        result_mul_159276 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 33), '*', y_159274, int_159275)
        
        # Processing the call keyword arguments (line 266)
        kwargs_159277 = {}
        # Getting the type of 'self' (line 266)
        self_159271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 266)
        allequal_159272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 15), self_159271, 'allequal')
        # Calling allequal(args, kwargs) (line 266)
        allequal_call_result_159278 = invoke(stypy.reporting.localization.Localization(__file__, 266, 15), allequal_159272, *[xm_159273, result_mul_159276], **kwargs_159277)
        
        
        # Assigning a BinOp to a Name (line 268):
        
        # Assigning a BinOp to a Name (line 268):
        
        # Call to arange(...): (line 268)
        # Processing the call arguments (line 268)
        int_159281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 24), 'int')
        # Processing the call keyword arguments (line 268)
        kwargs_159282 = {}
        # Getting the type of 'self' (line 268)
        self_159279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 268)
        arange_159280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 12), self_159279, 'arange')
        # Calling arange(args, kwargs) (line 268)
        arange_call_result_159283 = invoke(stypy.reporting.localization.Localization(__file__, 268, 12), arange_159280, *[int_159281], **kwargs_159282)
        
        int_159284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 28), 'int')
        # Applying the binary operator '*' (line 268)
        result_mul_159285 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 12), '*', arange_call_result_159283, int_159284)
        
        # Assigning a type to the variable 'x' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'x', result_mul_159285)
        
        # Assigning a BinOp to a Name (line 269):
        
        # Assigning a BinOp to a Name (line 269):
        
        # Call to arange(...): (line 269)
        # Processing the call arguments (line 269)
        int_159288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 25), 'int')
        # Processing the call keyword arguments (line 269)
        kwargs_159289 = {}
        # Getting the type of 'self' (line 269)
        self_159286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 13), 'self', False)
        # Obtaining the member 'arange' of a type (line 269)
        arange_159287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 13), self_159286, 'arange')
        # Calling arange(args, kwargs) (line 269)
        arange_call_result_159290 = invoke(stypy.reporting.localization.Localization(__file__, 269, 13), arange_159287, *[int_159288], **kwargs_159289)
        
        int_159291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 29), 'int')
        # Applying the binary operator '*' (line 269)
        result_mul_159292 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 13), '*', arange_call_result_159290, int_159291)
        
        # Assigning a type to the variable 'xm' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'xm', result_mul_159292)
        
        # Assigning a Attribute to a Subscript (line 270):
        
        # Assigning a Attribute to a Subscript (line 270):
        # Getting the type of 'self' (line 270)
        self_159293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'self')
        # Obtaining the member 'masked' of a type (line 270)
        masked_159294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 16), self_159293, 'masked')
        # Getting the type of 'xm' (line 270)
        xm_159295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'xm')
        int_159296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 11), 'int')
        # Storing an element on a container (line 270)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 8), xm_159295, (int_159296, masked_159294))
        
        # Getting the type of 'x' (line 271)
        x_159297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'x')
        int_159298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 13), 'int')
        # Applying the binary operator 'div=' (line 271)
        result_div_159299 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 8), 'div=', x_159297, int_159298)
        # Assigning a type to the variable 'x' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'x', result_div_159299)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'x' (line 272)
        x_159302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 29), 'x', False)
        # Getting the type of 'y' (line 272)
        y_159303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 32), 'y', False)
        # Processing the call keyword arguments (line 272)
        kwargs_159304 = {}
        # Getting the type of 'self' (line 272)
        self_159300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 272)
        allequal_159301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 15), self_159300, 'allequal')
        # Calling allequal(args, kwargs) (line 272)
        allequal_call_result_159305 = invoke(stypy.reporting.localization.Localization(__file__, 272, 15), allequal_159301, *[x_159302, y_159303], **kwargs_159304)
        
        
        # Getting the type of 'xm' (line 273)
        xm_159306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'xm')
        int_159307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 14), 'int')
        # Applying the binary operator 'div=' (line 273)
        result_div_159308 = python_operator(stypy.reporting.localization.Localization(__file__, 273, 8), 'div=', xm_159306, int_159307)
        # Assigning a type to the variable 'xm' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'xm', result_div_159308)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 'xm' (line 274)
        xm_159311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 29), 'xm', False)
        # Getting the type of 'y' (line 274)
        y_159312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 33), 'y', False)
        # Processing the call keyword arguments (line 274)
        kwargs_159313 = {}
        # Getting the type of 'self' (line 274)
        self_159309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 274)
        allequal_159310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 15), self_159309, 'allequal')
        # Calling allequal(args, kwargs) (line 274)
        allequal_call_result_159314 = invoke(stypy.reporting.localization.Localization(__file__, 274, 15), allequal_159310, *[xm_159311, y_159312], **kwargs_159313)
        
        
        # Assigning a BinOp to a Name (line 276):
        
        # Assigning a BinOp to a Name (line 276):
        
        # Call to arange(...): (line 276)
        # Processing the call arguments (line 276)
        int_159317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 24), 'int')
        # Processing the call keyword arguments (line 276)
        kwargs_159318 = {}
        # Getting the type of 'self' (line 276)
        self_159315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 276)
        arange_159316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 12), self_159315, 'arange')
        # Calling arange(args, kwargs) (line 276)
        arange_call_result_159319 = invoke(stypy.reporting.localization.Localization(__file__, 276, 12), arange_159316, *[int_159317], **kwargs_159318)
        
        float_159320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 28), 'float')
        # Applying the binary operator '*' (line 276)
        result_mul_159321 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 12), '*', arange_call_result_159319, float_159320)
        
        # Assigning a type to the variable 'x' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'x', result_mul_159321)
        
        # Assigning a BinOp to a Name (line 277):
        
        # Assigning a BinOp to a Name (line 277):
        
        # Call to arange(...): (line 277)
        # Processing the call arguments (line 277)
        int_159324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 25), 'int')
        # Processing the call keyword arguments (line 277)
        kwargs_159325 = {}
        # Getting the type of 'self' (line 277)
        self_159322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 13), 'self', False)
        # Obtaining the member 'arange' of a type (line 277)
        arange_159323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 13), self_159322, 'arange')
        # Calling arange(args, kwargs) (line 277)
        arange_call_result_159326 = invoke(stypy.reporting.localization.Localization(__file__, 277, 13), arange_159323, *[int_159324], **kwargs_159325)
        
        float_159327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 29), 'float')
        # Applying the binary operator '*' (line 277)
        result_mul_159328 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 13), '*', arange_call_result_159326, float_159327)
        
        # Assigning a type to the variable 'xm' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'xm', result_mul_159328)
        
        # Assigning a Attribute to a Subscript (line 278):
        
        # Assigning a Attribute to a Subscript (line 278):
        # Getting the type of 'self' (line 278)
        self_159329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'self')
        # Obtaining the member 'masked' of a type (line 278)
        masked_159330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 16), self_159329, 'masked')
        # Getting the type of 'xm' (line 278)
        xm_159331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'xm')
        int_159332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 11), 'int')
        # Storing an element on a container (line 278)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 8), xm_159331, (int_159332, masked_159330))
        
        # Getting the type of 'x' (line 279)
        x_159333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'x')
        float_159334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 13), 'float')
        # Applying the binary operator 'div=' (line 279)
        result_div_159335 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 8), 'div=', x_159333, float_159334)
        # Assigning a type to the variable 'x' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'x', result_div_159335)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 280)
        # Processing the call arguments (line 280)
        # Getting the type of 'x' (line 280)
        x_159338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 29), 'x', False)
        # Getting the type of 'y' (line 280)
        y_159339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 32), 'y', False)
        float_159340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 34), 'float')
        # Applying the binary operator 'div' (line 280)
        result_div_159341 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 32), 'div', y_159339, float_159340)
        
        # Processing the call keyword arguments (line 280)
        kwargs_159342 = {}
        # Getting the type of 'self' (line 280)
        self_159336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 280)
        allequal_159337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 15), self_159336, 'allequal')
        # Calling allequal(args, kwargs) (line 280)
        allequal_call_result_159343 = invoke(stypy.reporting.localization.Localization(__file__, 280, 15), allequal_159337, *[x_159338, result_div_159341], **kwargs_159342)
        
        
        # Getting the type of 'xm' (line 281)
        xm_159344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'xm')
        
        # Call to arange(...): (line 281)
        # Processing the call arguments (line 281)
        int_159347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 26), 'int')
        # Processing the call keyword arguments (line 281)
        kwargs_159348 = {}
        # Getting the type of 'self' (line 281)
        self_159345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 14), 'self', False)
        # Obtaining the member 'arange' of a type (line 281)
        arange_159346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 14), self_159345, 'arange')
        # Calling arange(args, kwargs) (line 281)
        arange_call_result_159349 = invoke(stypy.reporting.localization.Localization(__file__, 281, 14), arange_159346, *[int_159347], **kwargs_159348)
        
        # Applying the binary operator 'div=' (line 281)
        result_div_159350 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 8), 'div=', xm_159344, arange_call_result_159349)
        # Assigning a type to the variable 'xm' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'xm', result_div_159350)
        
        
        # Call to assert_array_equal(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'xm' (line 282)
        xm_159353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 32), 'xm', False)
        
        # Call to ones(...): (line 282)
        # Processing the call arguments (line 282)
        
        # Obtaining an instance of the builtin type 'tuple' (line 282)
        tuple_159356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 282)
        # Adding element type (line 282)
        int_159357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 47), tuple_159356, int_159357)
        
        # Processing the call keyword arguments (line 282)
        kwargs_159358 = {}
        # Getting the type of 'self' (line 282)
        self_159354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 36), 'self', False)
        # Obtaining the member 'ones' of a type (line 282)
        ones_159355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 36), self_159354, 'ones')
        # Calling ones(args, kwargs) (line 282)
        ones_call_result_159359 = invoke(stypy.reporting.localization.Localization(__file__, 282, 36), ones_159355, *[tuple_159356], **kwargs_159358)
        
        # Processing the call keyword arguments (line 282)
        kwargs_159360 = {}
        # Getting the type of 'self' (line 282)
        self_159351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 282)
        assert_array_equal_159352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), self_159351, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 282)
        assert_array_equal_call_result_159361 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), assert_array_equal_159352, *[xm_159353, ones_call_result_159359], **kwargs_159360)
        
        
        # Assigning a Call to a Name (line 284):
        
        # Assigning a Call to a Name (line 284):
        
        # Call to astype(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'float_' (line 284)
        float__159368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 35), 'float_', False)
        # Processing the call keyword arguments (line 284)
        kwargs_159369 = {}
        
        # Call to arange(...): (line 284)
        # Processing the call arguments (line 284)
        int_159364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 24), 'int')
        # Processing the call keyword arguments (line 284)
        kwargs_159365 = {}
        # Getting the type of 'self' (line 284)
        self_159362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 284)
        arange_159363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 12), self_159362, 'arange')
        # Calling arange(args, kwargs) (line 284)
        arange_call_result_159366 = invoke(stypy.reporting.localization.Localization(__file__, 284, 12), arange_159363, *[int_159364], **kwargs_159365)
        
        # Obtaining the member 'astype' of a type (line 284)
        astype_159367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 12), arange_call_result_159366, 'astype')
        # Calling astype(args, kwargs) (line 284)
        astype_call_result_159370 = invoke(stypy.reporting.localization.Localization(__file__, 284, 12), astype_159367, *[float__159368], **kwargs_159369)
        
        # Assigning a type to the variable 'x' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'x', astype_call_result_159370)
        
        # Assigning a Call to a Name (line 285):
        
        # Assigning a Call to a Name (line 285):
        
        # Call to arange(...): (line 285)
        # Processing the call arguments (line 285)
        int_159373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 25), 'int')
        # Processing the call keyword arguments (line 285)
        kwargs_159374 = {}
        # Getting the type of 'self' (line 285)
        self_159371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 13), 'self', False)
        # Obtaining the member 'arange' of a type (line 285)
        arange_159372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 13), self_159371, 'arange')
        # Calling arange(args, kwargs) (line 285)
        arange_call_result_159375 = invoke(stypy.reporting.localization.Localization(__file__, 285, 13), arange_159372, *[int_159373], **kwargs_159374)
        
        # Assigning a type to the variable 'xm' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'xm', arange_call_result_159375)
        
        # Assigning a Attribute to a Subscript (line 286):
        
        # Assigning a Attribute to a Subscript (line 286):
        # Getting the type of 'self' (line 286)
        self_159376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'self')
        # Obtaining the member 'masked' of a type (line 286)
        masked_159377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 16), self_159376, 'masked')
        # Getting the type of 'xm' (line 286)
        xm_159378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'xm')
        int_159379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 11), 'int')
        # Storing an element on a container (line 286)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 286, 8), xm_159378, (int_159379, masked_159377))
        
        # Getting the type of 'x' (line 287)
        x_159380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'x')
        float_159381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 13), 'float')
        # Applying the binary operator '+=' (line 287)
        result_iadd_159382 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 8), '+=', x_159380, float_159381)
        # Assigning a type to the variable 'x' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'x', result_iadd_159382)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 'x' (line 288)
        x_159385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 29), 'x', False)
        # Getting the type of 'y' (line 288)
        y_159386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 32), 'y', False)
        float_159387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 36), 'float')
        # Applying the binary operator '+' (line 288)
        result_add_159388 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 32), '+', y_159386, float_159387)
        
        # Processing the call keyword arguments (line 288)
        kwargs_159389 = {}
        # Getting the type of 'self' (line 288)
        self_159383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 288)
        allequal_159384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 15), self_159383, 'allequal')
        # Calling allequal(args, kwargs) (line 288)
        allequal_call_result_159390 = invoke(stypy.reporting.localization.Localization(__file__, 288, 15), allequal_159384, *[x_159385, result_add_159388], **kwargs_159389)
        
        
        # ################# End of 'test_5(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_5' in the type store
        # Getting the type of 'stypy_return_type' (line 238)
        stypy_return_type_159391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_159391)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_5'
        return stypy_return_type_159391


    @norecursion
    def test_6(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_6'
        module_type_store = module_type_store.open_function_context('test_6', 290, 4, False)
        # Assigning a type to the variable 'self' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ModuleTester.test_6.__dict__.__setitem__('stypy_localization', localization)
        ModuleTester.test_6.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ModuleTester.test_6.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleTester.test_6.__dict__.__setitem__('stypy_function_name', 'ModuleTester.test_6')
        ModuleTester.test_6.__dict__.__setitem__('stypy_param_names_list', [])
        ModuleTester.test_6.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleTester.test_6.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleTester.test_6.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleTester.test_6.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleTester.test_6.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleTester.test_6.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ModuleTester.test_6', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_6', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_6(...)' code ##################

        str_159392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, (-1)), 'str', '\n        Tests inplace w/ array\n\n        ')
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to arange(...): (line 295)
        # Processing the call arguments (line 295)
        int_159395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 24), 'int')
        # Processing the call keyword arguments (line 295)
        # Getting the type of 'float_' (line 295)
        float__159396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 34), 'float_', False)
        keyword_159397 = float__159396
        kwargs_159398 = {'dtype': keyword_159397}
        # Getting the type of 'self' (line 295)
        self_159393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 295)
        arange_159394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 12), self_159393, 'arange')
        # Calling arange(args, kwargs) (line 295)
        arange_call_result_159399 = invoke(stypy.reporting.localization.Localization(__file__, 295, 12), arange_159394, *[int_159395], **kwargs_159398)
        
        # Assigning a type to the variable 'x' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'x', arange_call_result_159399)
        
        # Assigning a Call to a Name (line 296):
        
        # Assigning a Call to a Name (line 296):
        
        # Call to arange(...): (line 296)
        # Processing the call arguments (line 296)
        int_159402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 24), 'int')
        # Processing the call keyword arguments (line 296)
        kwargs_159403 = {}
        # Getting the type of 'self' (line 296)
        self_159400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 296)
        arange_159401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 12), self_159400, 'arange')
        # Calling arange(args, kwargs) (line 296)
        arange_call_result_159404 = invoke(stypy.reporting.localization.Localization(__file__, 296, 12), arange_159401, *[int_159402], **kwargs_159403)
        
        # Assigning a type to the variable 'y' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'y', arange_call_result_159404)
        
        # Assigning a Call to a Name (line 297):
        
        # Assigning a Call to a Name (line 297):
        
        # Call to arange(...): (line 297)
        # Processing the call arguments (line 297)
        int_159407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 25), 'int')
        # Processing the call keyword arguments (line 297)
        # Getting the type of 'float_' (line 297)
        float__159408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 35), 'float_', False)
        keyword_159409 = float__159408
        kwargs_159410 = {'dtype': keyword_159409}
        # Getting the type of 'self' (line 297)
        self_159405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 13), 'self', False)
        # Obtaining the member 'arange' of a type (line 297)
        arange_159406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 13), self_159405, 'arange')
        # Calling arange(args, kwargs) (line 297)
        arange_call_result_159411 = invoke(stypy.reporting.localization.Localization(__file__, 297, 13), arange_159406, *[int_159407], **kwargs_159410)
        
        # Assigning a type to the variable 'xm' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'xm', arange_call_result_159411)
        
        # Assigning a Attribute to a Subscript (line 298):
        
        # Assigning a Attribute to a Subscript (line 298):
        # Getting the type of 'self' (line 298)
        self_159412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'self')
        # Obtaining the member 'masked' of a type (line 298)
        masked_159413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 16), self_159412, 'masked')
        # Getting the type of 'xm' (line 298)
        xm_159414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'xm')
        int_159415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 11), 'int')
        # Storing an element on a container (line 298)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 8), xm_159414, (int_159415, masked_159413))
        
        # Assigning a Attribute to a Name (line 299):
        
        # Assigning a Attribute to a Name (line 299):
        # Getting the type of 'xm' (line 299)
        xm_159416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'xm')
        # Obtaining the member 'mask' of a type (line 299)
        mask_159417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 12), xm_159416, 'mask')
        # Assigning a type to the variable 'm' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'm', mask_159417)
        
        # Assigning a Call to a Name (line 300):
        
        # Assigning a Call to a Name (line 300):
        
        # Call to arange(...): (line 300)
        # Processing the call arguments (line 300)
        int_159420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 24), 'int')
        # Processing the call keyword arguments (line 300)
        # Getting the type of 'float_' (line 300)
        float__159421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 34), 'float_', False)
        keyword_159422 = float__159421
        kwargs_159423 = {'dtype': keyword_159422}
        # Getting the type of 'self' (line 300)
        self_159418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 300)
        arange_159419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 12), self_159418, 'arange')
        # Calling arange(args, kwargs) (line 300)
        arange_call_result_159424 = invoke(stypy.reporting.localization.Localization(__file__, 300, 12), arange_159419, *[int_159420], **kwargs_159423)
        
        # Assigning a type to the variable 'a' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'a', arange_call_result_159424)
        
        # Assigning a Attribute to a Subscript (line 301):
        
        # Assigning a Attribute to a Subscript (line 301):
        # Getting the type of 'self' (line 301)
        self_159425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'self')
        # Obtaining the member 'masked' of a type (line 301)
        masked_159426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 16), self_159425, 'masked')
        # Getting the type of 'a' (line 301)
        a_159427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'a')
        int_159428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 10), 'int')
        # Storing an element on a container (line 301)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 8), a_159427, (int_159428, masked_159426))
        
        # Getting the type of 'x' (line 302)
        x_159429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'x')
        # Getting the type of 'a' (line 302)
        a_159430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 13), 'a')
        # Applying the binary operator '+=' (line 302)
        result_iadd_159431 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 8), '+=', x_159429, a_159430)
        # Assigning a type to the variable 'x' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'x', result_iadd_159431)
        
        
        # Getting the type of 'xm' (line 303)
        xm_159432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'xm')
        # Getting the type of 'a' (line 303)
        a_159433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 14), 'a')
        # Applying the binary operator '+=' (line 303)
        result_iadd_159434 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 8), '+=', xm_159432, a_159433)
        # Assigning a type to the variable 'xm' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'xm', result_iadd_159434)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'x' (line 304)
        x_159437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 29), 'x', False)
        # Getting the type of 'y' (line 304)
        y_159438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 32), 'y', False)
        # Getting the type of 'a' (line 304)
        a_159439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 34), 'a', False)
        # Applying the binary operator '+' (line 304)
        result_add_159440 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 32), '+', y_159438, a_159439)
        
        # Processing the call keyword arguments (line 304)
        kwargs_159441 = {}
        # Getting the type of 'self' (line 304)
        self_159435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 304)
        allequal_159436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 15), self_159435, 'allequal')
        # Calling allequal(args, kwargs) (line 304)
        allequal_call_result_159442 = invoke(stypy.reporting.localization.Localization(__file__, 304, 15), allequal_159436, *[x_159437, result_add_159440], **kwargs_159441)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 'xm' (line 305)
        xm_159445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 29), 'xm', False)
        # Getting the type of 'y' (line 305)
        y_159446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 33), 'y', False)
        # Getting the type of 'a' (line 305)
        a_159447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 35), 'a', False)
        # Applying the binary operator '+' (line 305)
        result_add_159448 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 33), '+', y_159446, a_159447)
        
        # Processing the call keyword arguments (line 305)
        kwargs_159449 = {}
        # Getting the type of 'self' (line 305)
        self_159443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 305)
        allequal_159444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 15), self_159443, 'allequal')
        # Calling allequal(args, kwargs) (line 305)
        allequal_call_result_159450 = invoke(stypy.reporting.localization.Localization(__file__, 305, 15), allequal_159444, *[xm_159445, result_add_159448], **kwargs_159449)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'xm' (line 306)
        xm_159453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 29), 'xm', False)
        # Obtaining the member 'mask' of a type (line 306)
        mask_159454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 29), xm_159453, 'mask')
        
        # Call to mask_or(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'm' (line 306)
        m_159457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 51), 'm', False)
        # Getting the type of 'a' (line 306)
        a_159458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 54), 'a', False)
        # Obtaining the member 'mask' of a type (line 306)
        mask_159459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 54), a_159458, 'mask')
        # Processing the call keyword arguments (line 306)
        kwargs_159460 = {}
        # Getting the type of 'self' (line 306)
        self_159455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 38), 'self', False)
        # Obtaining the member 'mask_or' of a type (line 306)
        mask_or_159456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 38), self_159455, 'mask_or')
        # Calling mask_or(args, kwargs) (line 306)
        mask_or_call_result_159461 = invoke(stypy.reporting.localization.Localization(__file__, 306, 38), mask_or_159456, *[m_159457, mask_159459], **kwargs_159460)
        
        # Processing the call keyword arguments (line 306)
        kwargs_159462 = {}
        # Getting the type of 'self' (line 306)
        self_159451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 306)
        allequal_159452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 15), self_159451, 'allequal')
        # Calling allequal(args, kwargs) (line 306)
        allequal_call_result_159463 = invoke(stypy.reporting.localization.Localization(__file__, 306, 15), allequal_159452, *[mask_159454, mask_or_call_result_159461], **kwargs_159462)
        
        
        # Assigning a Call to a Name (line 308):
        
        # Assigning a Call to a Name (line 308):
        
        # Call to arange(...): (line 308)
        # Processing the call arguments (line 308)
        int_159466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 24), 'int')
        # Processing the call keyword arguments (line 308)
        # Getting the type of 'float_' (line 308)
        float__159467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 34), 'float_', False)
        keyword_159468 = float__159467
        kwargs_159469 = {'dtype': keyword_159468}
        # Getting the type of 'self' (line 308)
        self_159464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 308)
        arange_159465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 12), self_159464, 'arange')
        # Calling arange(args, kwargs) (line 308)
        arange_call_result_159470 = invoke(stypy.reporting.localization.Localization(__file__, 308, 12), arange_159465, *[int_159466], **kwargs_159469)
        
        # Assigning a type to the variable 'x' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'x', arange_call_result_159470)
        
        # Assigning a Call to a Name (line 309):
        
        # Assigning a Call to a Name (line 309):
        
        # Call to arange(...): (line 309)
        # Processing the call arguments (line 309)
        int_159473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 25), 'int')
        # Processing the call keyword arguments (line 309)
        # Getting the type of 'float_' (line 309)
        float__159474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 35), 'float_', False)
        keyword_159475 = float__159474
        kwargs_159476 = {'dtype': keyword_159475}
        # Getting the type of 'self' (line 309)
        self_159471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 13), 'self', False)
        # Obtaining the member 'arange' of a type (line 309)
        arange_159472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 13), self_159471, 'arange')
        # Calling arange(args, kwargs) (line 309)
        arange_call_result_159477 = invoke(stypy.reporting.localization.Localization(__file__, 309, 13), arange_159472, *[int_159473], **kwargs_159476)
        
        # Assigning a type to the variable 'xm' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'xm', arange_call_result_159477)
        
        # Assigning a Attribute to a Subscript (line 310):
        
        # Assigning a Attribute to a Subscript (line 310):
        # Getting the type of 'self' (line 310)
        self_159478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 16), 'self')
        # Obtaining the member 'masked' of a type (line 310)
        masked_159479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 16), self_159478, 'masked')
        # Getting the type of 'xm' (line 310)
        xm_159480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'xm')
        int_159481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 11), 'int')
        # Storing an element on a container (line 310)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 310, 8), xm_159480, (int_159481, masked_159479))
        
        # Assigning a Attribute to a Name (line 311):
        
        # Assigning a Attribute to a Name (line 311):
        # Getting the type of 'xm' (line 311)
        xm_159482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'xm')
        # Obtaining the member 'mask' of a type (line 311)
        mask_159483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), xm_159482, 'mask')
        # Assigning a type to the variable 'm' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'm', mask_159483)
        
        # Assigning a Call to a Name (line 312):
        
        # Assigning a Call to a Name (line 312):
        
        # Call to arange(...): (line 312)
        # Processing the call arguments (line 312)
        int_159486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 24), 'int')
        # Processing the call keyword arguments (line 312)
        # Getting the type of 'float_' (line 312)
        float__159487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 34), 'float_', False)
        keyword_159488 = float__159487
        kwargs_159489 = {'dtype': keyword_159488}
        # Getting the type of 'self' (line 312)
        self_159484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 312)
        arange_159485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 12), self_159484, 'arange')
        # Calling arange(args, kwargs) (line 312)
        arange_call_result_159490 = invoke(stypy.reporting.localization.Localization(__file__, 312, 12), arange_159485, *[int_159486], **kwargs_159489)
        
        # Assigning a type to the variable 'a' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'a', arange_call_result_159490)
        
        # Assigning a Attribute to a Subscript (line 313):
        
        # Assigning a Attribute to a Subscript (line 313):
        # Getting the type of 'self' (line 313)
        self_159491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 16), 'self')
        # Obtaining the member 'masked' of a type (line 313)
        masked_159492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 16), self_159491, 'masked')
        # Getting the type of 'a' (line 313)
        a_159493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'a')
        int_159494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 10), 'int')
        # Storing an element on a container (line 313)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 8), a_159493, (int_159494, masked_159492))
        
        # Getting the type of 'x' (line 314)
        x_159495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'x')
        # Getting the type of 'a' (line 314)
        a_159496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 13), 'a')
        # Applying the binary operator '-=' (line 314)
        result_isub_159497 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 8), '-=', x_159495, a_159496)
        # Assigning a type to the variable 'x' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'x', result_isub_159497)
        
        
        # Getting the type of 'xm' (line 315)
        xm_159498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'xm')
        # Getting the type of 'a' (line 315)
        a_159499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 14), 'a')
        # Applying the binary operator '-=' (line 315)
        result_isub_159500 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 8), '-=', xm_159498, a_159499)
        # Assigning a type to the variable 'xm' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'xm', result_isub_159500)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'x' (line 316)
        x_159503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 29), 'x', False)
        # Getting the type of 'y' (line 316)
        y_159504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 32), 'y', False)
        # Getting the type of 'a' (line 316)
        a_159505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 34), 'a', False)
        # Applying the binary operator '-' (line 316)
        result_sub_159506 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 32), '-', y_159504, a_159505)
        
        # Processing the call keyword arguments (line 316)
        kwargs_159507 = {}
        # Getting the type of 'self' (line 316)
        self_159501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 316)
        allequal_159502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 15), self_159501, 'allequal')
        # Calling allequal(args, kwargs) (line 316)
        allequal_call_result_159508 = invoke(stypy.reporting.localization.Localization(__file__, 316, 15), allequal_159502, *[x_159503, result_sub_159506], **kwargs_159507)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'xm' (line 317)
        xm_159511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 29), 'xm', False)
        # Getting the type of 'y' (line 317)
        y_159512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 33), 'y', False)
        # Getting the type of 'a' (line 317)
        a_159513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 35), 'a', False)
        # Applying the binary operator '-' (line 317)
        result_sub_159514 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 33), '-', y_159512, a_159513)
        
        # Processing the call keyword arguments (line 317)
        kwargs_159515 = {}
        # Getting the type of 'self' (line 317)
        self_159509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 317)
        allequal_159510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 15), self_159509, 'allequal')
        # Calling allequal(args, kwargs) (line 317)
        allequal_call_result_159516 = invoke(stypy.reporting.localization.Localization(__file__, 317, 15), allequal_159510, *[xm_159511, result_sub_159514], **kwargs_159515)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'xm' (line 318)
        xm_159519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 29), 'xm', False)
        # Obtaining the member 'mask' of a type (line 318)
        mask_159520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 29), xm_159519, 'mask')
        
        # Call to mask_or(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'm' (line 318)
        m_159523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 51), 'm', False)
        # Getting the type of 'a' (line 318)
        a_159524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 54), 'a', False)
        # Obtaining the member 'mask' of a type (line 318)
        mask_159525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 54), a_159524, 'mask')
        # Processing the call keyword arguments (line 318)
        kwargs_159526 = {}
        # Getting the type of 'self' (line 318)
        self_159521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 38), 'self', False)
        # Obtaining the member 'mask_or' of a type (line 318)
        mask_or_159522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 38), self_159521, 'mask_or')
        # Calling mask_or(args, kwargs) (line 318)
        mask_or_call_result_159527 = invoke(stypy.reporting.localization.Localization(__file__, 318, 38), mask_or_159522, *[m_159523, mask_159525], **kwargs_159526)
        
        # Processing the call keyword arguments (line 318)
        kwargs_159528 = {}
        # Getting the type of 'self' (line 318)
        self_159517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 318)
        allequal_159518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 15), self_159517, 'allequal')
        # Calling allequal(args, kwargs) (line 318)
        allequal_call_result_159529 = invoke(stypy.reporting.localization.Localization(__file__, 318, 15), allequal_159518, *[mask_159520, mask_or_call_result_159527], **kwargs_159528)
        
        
        # Assigning a Call to a Name (line 320):
        
        # Assigning a Call to a Name (line 320):
        
        # Call to arange(...): (line 320)
        # Processing the call arguments (line 320)
        int_159532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 24), 'int')
        # Processing the call keyword arguments (line 320)
        # Getting the type of 'float_' (line 320)
        float__159533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 34), 'float_', False)
        keyword_159534 = float__159533
        kwargs_159535 = {'dtype': keyword_159534}
        # Getting the type of 'self' (line 320)
        self_159530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 320)
        arange_159531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 12), self_159530, 'arange')
        # Calling arange(args, kwargs) (line 320)
        arange_call_result_159536 = invoke(stypy.reporting.localization.Localization(__file__, 320, 12), arange_159531, *[int_159532], **kwargs_159535)
        
        # Assigning a type to the variable 'x' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'x', arange_call_result_159536)
        
        # Assigning a Call to a Name (line 321):
        
        # Assigning a Call to a Name (line 321):
        
        # Call to arange(...): (line 321)
        # Processing the call arguments (line 321)
        int_159539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 25), 'int')
        # Processing the call keyword arguments (line 321)
        # Getting the type of 'float_' (line 321)
        float__159540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 35), 'float_', False)
        keyword_159541 = float__159540
        kwargs_159542 = {'dtype': keyword_159541}
        # Getting the type of 'self' (line 321)
        self_159537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 13), 'self', False)
        # Obtaining the member 'arange' of a type (line 321)
        arange_159538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 13), self_159537, 'arange')
        # Calling arange(args, kwargs) (line 321)
        arange_call_result_159543 = invoke(stypy.reporting.localization.Localization(__file__, 321, 13), arange_159538, *[int_159539], **kwargs_159542)
        
        # Assigning a type to the variable 'xm' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'xm', arange_call_result_159543)
        
        # Assigning a Attribute to a Subscript (line 322):
        
        # Assigning a Attribute to a Subscript (line 322):
        # Getting the type of 'self' (line 322)
        self_159544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 16), 'self')
        # Obtaining the member 'masked' of a type (line 322)
        masked_159545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 16), self_159544, 'masked')
        # Getting the type of 'xm' (line 322)
        xm_159546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'xm')
        int_159547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 11), 'int')
        # Storing an element on a container (line 322)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 8), xm_159546, (int_159547, masked_159545))
        
        # Assigning a Attribute to a Name (line 323):
        
        # Assigning a Attribute to a Name (line 323):
        # Getting the type of 'xm' (line 323)
        xm_159548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'xm')
        # Obtaining the member 'mask' of a type (line 323)
        mask_159549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 12), xm_159548, 'mask')
        # Assigning a type to the variable 'm' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'm', mask_159549)
        
        # Assigning a Call to a Name (line 324):
        
        # Assigning a Call to a Name (line 324):
        
        # Call to arange(...): (line 324)
        # Processing the call arguments (line 324)
        int_159552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 24), 'int')
        # Processing the call keyword arguments (line 324)
        # Getting the type of 'float_' (line 324)
        float__159553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 34), 'float_', False)
        keyword_159554 = float__159553
        kwargs_159555 = {'dtype': keyword_159554}
        # Getting the type of 'self' (line 324)
        self_159550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 324)
        arange_159551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 12), self_159550, 'arange')
        # Calling arange(args, kwargs) (line 324)
        arange_call_result_159556 = invoke(stypy.reporting.localization.Localization(__file__, 324, 12), arange_159551, *[int_159552], **kwargs_159555)
        
        # Assigning a type to the variable 'a' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'a', arange_call_result_159556)
        
        # Assigning a Attribute to a Subscript (line 325):
        
        # Assigning a Attribute to a Subscript (line 325):
        # Getting the type of 'self' (line 325)
        self_159557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 16), 'self')
        # Obtaining the member 'masked' of a type (line 325)
        masked_159558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 16), self_159557, 'masked')
        # Getting the type of 'a' (line 325)
        a_159559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'a')
        int_159560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 10), 'int')
        # Storing an element on a container (line 325)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 8), a_159559, (int_159560, masked_159558))
        
        # Getting the type of 'x' (line 326)
        x_159561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'x')
        # Getting the type of 'a' (line 326)
        a_159562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 13), 'a')
        # Applying the binary operator '*=' (line 326)
        result_imul_159563 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 8), '*=', x_159561, a_159562)
        # Assigning a type to the variable 'x' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'x', result_imul_159563)
        
        
        # Getting the type of 'xm' (line 327)
        xm_159564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'xm')
        # Getting the type of 'a' (line 327)
        a_159565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 14), 'a')
        # Applying the binary operator '*=' (line 327)
        result_imul_159566 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 8), '*=', xm_159564, a_159565)
        # Assigning a type to the variable 'xm' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'xm', result_imul_159566)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 328)
        # Processing the call arguments (line 328)
        # Getting the type of 'x' (line 328)
        x_159569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 29), 'x', False)
        # Getting the type of 'y' (line 328)
        y_159570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 32), 'y', False)
        # Getting the type of 'a' (line 328)
        a_159571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 34), 'a', False)
        # Applying the binary operator '*' (line 328)
        result_mul_159572 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 32), '*', y_159570, a_159571)
        
        # Processing the call keyword arguments (line 328)
        kwargs_159573 = {}
        # Getting the type of 'self' (line 328)
        self_159567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 328)
        allequal_159568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 15), self_159567, 'allequal')
        # Calling allequal(args, kwargs) (line 328)
        allequal_call_result_159574 = invoke(stypy.reporting.localization.Localization(__file__, 328, 15), allequal_159568, *[x_159569, result_mul_159572], **kwargs_159573)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 329)
        # Processing the call arguments (line 329)
        # Getting the type of 'xm' (line 329)
        xm_159577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 29), 'xm', False)
        # Getting the type of 'y' (line 329)
        y_159578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 33), 'y', False)
        # Getting the type of 'a' (line 329)
        a_159579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 35), 'a', False)
        # Applying the binary operator '*' (line 329)
        result_mul_159580 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 33), '*', y_159578, a_159579)
        
        # Processing the call keyword arguments (line 329)
        kwargs_159581 = {}
        # Getting the type of 'self' (line 329)
        self_159575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 329)
        allequal_159576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 15), self_159575, 'allequal')
        # Calling allequal(args, kwargs) (line 329)
        allequal_call_result_159582 = invoke(stypy.reporting.localization.Localization(__file__, 329, 15), allequal_159576, *[xm_159577, result_mul_159580], **kwargs_159581)
        
        # Evaluating assert statement condition
        
        # Call to allequal(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 'xm' (line 330)
        xm_159585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 29), 'xm', False)
        # Obtaining the member 'mask' of a type (line 330)
        mask_159586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 29), xm_159585, 'mask')
        
        # Call to mask_or(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 'm' (line 330)
        m_159589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 51), 'm', False)
        # Getting the type of 'a' (line 330)
        a_159590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 54), 'a', False)
        # Obtaining the member 'mask' of a type (line 330)
        mask_159591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 54), a_159590, 'mask')
        # Processing the call keyword arguments (line 330)
        kwargs_159592 = {}
        # Getting the type of 'self' (line 330)
        self_159587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 38), 'self', False)
        # Obtaining the member 'mask_or' of a type (line 330)
        mask_or_159588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 38), self_159587, 'mask_or')
        # Calling mask_or(args, kwargs) (line 330)
        mask_or_call_result_159593 = invoke(stypy.reporting.localization.Localization(__file__, 330, 38), mask_or_159588, *[m_159589, mask_159591], **kwargs_159592)
        
        # Processing the call keyword arguments (line 330)
        kwargs_159594 = {}
        # Getting the type of 'self' (line 330)
        self_159583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 15), 'self', False)
        # Obtaining the member 'allequal' of a type (line 330)
        allequal_159584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 15), self_159583, 'allequal')
        # Calling allequal(args, kwargs) (line 330)
        allequal_call_result_159595 = invoke(stypy.reporting.localization.Localization(__file__, 330, 15), allequal_159584, *[mask_159586, mask_or_call_result_159593], **kwargs_159594)
        
        
        # Assigning a Call to a Name (line 332):
        
        # Assigning a Call to a Name (line 332):
        
        # Call to arange(...): (line 332)
        # Processing the call arguments (line 332)
        int_159598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 24), 'int')
        # Processing the call keyword arguments (line 332)
        # Getting the type of 'float_' (line 332)
        float__159599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 34), 'float_', False)
        keyword_159600 = float__159599
        kwargs_159601 = {'dtype': keyword_159600}
        # Getting the type of 'self' (line 332)
        self_159596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 332)
        arange_159597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 12), self_159596, 'arange')
        # Calling arange(args, kwargs) (line 332)
        arange_call_result_159602 = invoke(stypy.reporting.localization.Localization(__file__, 332, 12), arange_159597, *[int_159598], **kwargs_159601)
        
        # Assigning a type to the variable 'x' (line 332)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'x', arange_call_result_159602)
        
        # Assigning a Call to a Name (line 333):
        
        # Assigning a Call to a Name (line 333):
        
        # Call to arange(...): (line 333)
        # Processing the call arguments (line 333)
        int_159605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 25), 'int')
        # Processing the call keyword arguments (line 333)
        # Getting the type of 'float_' (line 333)
        float__159606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 35), 'float_', False)
        keyword_159607 = float__159606
        kwargs_159608 = {'dtype': keyword_159607}
        # Getting the type of 'self' (line 333)
        self_159603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 13), 'self', False)
        # Obtaining the member 'arange' of a type (line 333)
        arange_159604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 13), self_159603, 'arange')
        # Calling arange(args, kwargs) (line 333)
        arange_call_result_159609 = invoke(stypy.reporting.localization.Localization(__file__, 333, 13), arange_159604, *[int_159605], **kwargs_159608)
        
        # Assigning a type to the variable 'xm' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'xm', arange_call_result_159609)
        
        # Assigning a Attribute to a Subscript (line 334):
        
        # Assigning a Attribute to a Subscript (line 334):
        # Getting the type of 'self' (line 334)
        self_159610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'self')
        # Obtaining the member 'masked' of a type (line 334)
        masked_159611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 16), self_159610, 'masked')
        # Getting the type of 'xm' (line 334)
        xm_159612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'xm')
        int_159613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 11), 'int')
        # Storing an element on a container (line 334)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 334, 8), xm_159612, (int_159613, masked_159611))
        
        # Assigning a Attribute to a Name (line 335):
        
        # Assigning a Attribute to a Name (line 335):
        # Getting the type of 'xm' (line 335)
        xm_159614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'xm')
        # Obtaining the member 'mask' of a type (line 335)
        mask_159615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 12), xm_159614, 'mask')
        # Assigning a type to the variable 'm' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'm', mask_159615)
        
        # Assigning a Call to a Name (line 336):
        
        # Assigning a Call to a Name (line 336):
        
        # Call to arange(...): (line 336)
        # Processing the call arguments (line 336)
        int_159618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 24), 'int')
        # Processing the call keyword arguments (line 336)
        # Getting the type of 'float_' (line 336)
        float__159619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 34), 'float_', False)
        keyword_159620 = float__159619
        kwargs_159621 = {'dtype': keyword_159620}
        # Getting the type of 'self' (line 336)
        self_159616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 336)
        arange_159617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 12), self_159616, 'arange')
        # Calling arange(args, kwargs) (line 336)
        arange_call_result_159622 = invoke(stypy.reporting.localization.Localization(__file__, 336, 12), arange_159617, *[int_159618], **kwargs_159621)
        
        # Assigning a type to the variable 'a' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'a', arange_call_result_159622)
        
        # Assigning a Attribute to a Subscript (line 337):
        
        # Assigning a Attribute to a Subscript (line 337):
        # Getting the type of 'self' (line 337)
        self_159623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 16), 'self')
        # Obtaining the member 'masked' of a type (line 337)
        masked_159624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 16), self_159623, 'masked')
        # Getting the type of 'a' (line 337)
        a_159625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'a')
        int_159626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 10), 'int')
        # Storing an element on a container (line 337)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 8), a_159625, (int_159626, masked_159624))
        
        # Getting the type of 'x' (line 338)
        x_159627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'x')
        # Getting the type of 'a' (line 338)
        a_159628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 13), 'a')
        # Applying the binary operator 'div=' (line 338)
        result_div_159629 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 8), 'div=', x_159627, a_159628)
        # Assigning a type to the variable 'x' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'x', result_div_159629)
        
        
        # Getting the type of 'xm' (line 339)
        xm_159630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'xm')
        # Getting the type of 'a' (line 339)
        a_159631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 14), 'a')
        # Applying the binary operator 'div=' (line 339)
        result_div_159632 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 8), 'div=', xm_159630, a_159631)
        # Assigning a type to the variable 'xm' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'xm', result_div_159632)
        
        
        # ################# End of 'test_6(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_6' in the type store
        # Getting the type of 'stypy_return_type' (line 290)
        stypy_return_type_159633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_159633)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_6'
        return stypy_return_type_159633


    @norecursion
    def test_7(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_7'
        module_type_store = module_type_store.open_function_context('test_7', 341, 4, False)
        # Assigning a type to the variable 'self' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ModuleTester.test_7.__dict__.__setitem__('stypy_localization', localization)
        ModuleTester.test_7.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ModuleTester.test_7.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleTester.test_7.__dict__.__setitem__('stypy_function_name', 'ModuleTester.test_7')
        ModuleTester.test_7.__dict__.__setitem__('stypy_param_names_list', [])
        ModuleTester.test_7.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleTester.test_7.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleTester.test_7.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleTester.test_7.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleTester.test_7.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleTester.test_7.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ModuleTester.test_7', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_7', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_7(...)' code ##################

        str_159634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 8), 'str', 'Tests ufunc')
        
        # Assigning a Tuple to a Name (line 343):
        
        # Assigning a Tuple to a Name (line 343):
        
        # Obtaining an instance of the builtin type 'tuple' (line 343)
        tuple_159635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 343)
        # Adding element type (line 343)
        
        # Call to array(...): (line 343)
        # Processing the call arguments (line 343)
        
        # Obtaining an instance of the builtin type 'list' (line 343)
        list_159638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 343)
        # Adding element type (line 343)
        float_159639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 24), list_159638, float_159639)
        # Adding element type (line 343)
        int_159640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 24), list_159638, int_159640)
        # Adding element type (line 343)
        int_159641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 24), list_159638, int_159641)
        # Adding element type (line 343)
        # Getting the type of 'pi' (line 343)
        pi_159642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 37), 'pi', False)
        int_159643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 40), 'int')
        # Applying the binary operator 'div' (line 343)
        result_div_159644 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 37), 'div', pi_159642, int_159643)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 24), list_159638, result_div_159644)
        
        int_159645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 43), 'int')
        # Applying the binary operator '*' (line 343)
        result_mul_159646 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 24), '*', list_159638, int_159645)
        
        # Processing the call keyword arguments (line 343)
        
        # Obtaining an instance of the builtin type 'list' (line 343)
        list_159647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 343)
        # Adding element type (line 343)
        int_159648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 51), list_159647, int_159648)
        # Adding element type (line 343)
        int_159649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 51), list_159647, int_159649)
        
        
        # Obtaining an instance of the builtin type 'list' (line 343)
        list_159650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 343)
        # Adding element type (line 343)
        int_159651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 58), list_159650, int_159651)
        
        int_159652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 62), 'int')
        # Applying the binary operator '*' (line 343)
        result_mul_159653 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 58), '*', list_159650, int_159652)
        
        # Applying the binary operator '+' (line 343)
        result_add_159654 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 51), '+', list_159647, result_mul_159653)
        
        keyword_159655 = result_add_159654
        kwargs_159656 = {'mask': keyword_159655}
        # Getting the type of 'self' (line 343)
        self_159636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 13), 'self', False)
        # Obtaining the member 'array' of a type (line 343)
        array_159637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 13), self_159636, 'array')
        # Calling array(args, kwargs) (line 343)
        array_call_result_159657 = invoke(stypy.reporting.localization.Localization(__file__, 343, 13), array_159637, *[result_mul_159646], **kwargs_159656)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 13), tuple_159635, array_call_result_159657)
        # Adding element type (line 343)
        
        # Call to array(...): (line 344)
        # Processing the call arguments (line 344)
        
        # Obtaining an instance of the builtin type 'list' (line 344)
        list_159660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 344)
        # Adding element type (line 344)
        float_159661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 24), list_159660, float_159661)
        # Adding element type (line 344)
        int_159662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 24), list_159660, int_159662)
        # Adding element type (line 344)
        int_159663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 33), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 24), list_159660, int_159663)
        # Adding element type (line 344)
        # Getting the type of 'pi' (line 344)
        pi_159664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 37), 'pi', False)
        int_159665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 40), 'int')
        # Applying the binary operator 'div' (line 344)
        result_div_159666 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 37), 'div', pi_159664, int_159665)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 24), list_159660, result_div_159666)
        
        int_159667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 43), 'int')
        # Applying the binary operator '*' (line 344)
        result_mul_159668 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 24), '*', list_159660, int_159667)
        
        # Processing the call keyword arguments (line 344)
        
        # Obtaining an instance of the builtin type 'list' (line 344)
        list_159669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 344)
        # Adding element type (line 344)
        int_159670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 51), list_159669, int_159670)
        # Adding element type (line 344)
        int_159671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 51), list_159669, int_159671)
        
        
        # Obtaining an instance of the builtin type 'list' (line 344)
        list_159672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 344)
        # Adding element type (line 344)
        int_159673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 344, 58), list_159672, int_159673)
        
        int_159674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 62), 'int')
        # Applying the binary operator '*' (line 344)
        result_mul_159675 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 58), '*', list_159672, int_159674)
        
        # Applying the binary operator '+' (line 344)
        result_add_159676 = python_operator(stypy.reporting.localization.Localization(__file__, 344, 51), '+', list_159669, result_mul_159675)
        
        keyword_159677 = result_add_159676
        kwargs_159678 = {'mask': keyword_159677}
        # Getting the type of 'self' (line 344)
        self_159658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 13), 'self', False)
        # Obtaining the member 'array' of a type (line 344)
        array_159659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 13), self_159658, 'array')
        # Calling array(args, kwargs) (line 344)
        array_call_result_159679 = invoke(stypy.reporting.localization.Localization(__file__, 344, 13), array_159659, *[result_mul_159668], **kwargs_159678)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 13), tuple_159635, array_call_result_159679)
        
        # Assigning a type to the variable 'd' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'd', tuple_159635)
        
        
        # Obtaining an instance of the builtin type 'list' (line 345)
        list_159680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 345)
        # Adding element type (line 345)
        str_159681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 18), 'str', 'sqrt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 17), list_159680, str_159681)
        # Adding element type (line 345)
        str_159682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 26), 'str', 'log')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 17), list_159680, str_159682)
        # Adding element type (line 345)
        str_159683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 33), 'str', 'log10')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 17), list_159680, str_159683)
        # Adding element type (line 345)
        str_159684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 42), 'str', 'exp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 17), list_159680, str_159684)
        # Adding element type (line 345)
        str_159685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 49), 'str', 'conjugate')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 345, 17), list_159680, str_159685)
        
        # Testing the type of a for loop iterable (line 345)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 345, 8), list_159680)
        # Getting the type of the for loop variable (line 345)
        for_loop_var_159686 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 345, 8), list_159680)
        # Assigning a type to the variable 'f' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'f', for_loop_var_159686)
        # SSA begins for a for statement (line 345)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 364)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 365):
        
        # Assigning a Call to a Name (line 365):
        
        # Call to getattr(...): (line 365)
        # Processing the call arguments (line 365)
        # Getting the type of 'self' (line 365)
        self_159688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 29), 'self', False)
        # Obtaining the member 'umath' of a type (line 365)
        umath_159689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 29), self_159688, 'umath')
        # Getting the type of 'f' (line 365)
        f_159690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 41), 'f', False)
        # Processing the call keyword arguments (line 365)
        kwargs_159691 = {}
        # Getting the type of 'getattr' (line 365)
        getattr_159687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 21), 'getattr', False)
        # Calling getattr(args, kwargs) (line 365)
        getattr_call_result_159692 = invoke(stypy.reporting.localization.Localization(__file__, 365, 21), getattr_159687, *[umath_159689, f_159690], **kwargs_159691)
        
        # Assigning a type to the variable 'uf' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 16), 'uf', getattr_call_result_159692)
        # SSA branch for the except part of a try statement (line 364)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 364)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 367):
        
        # Assigning a Call to a Name (line 367):
        
        # Call to getattr(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'fromnumeric' (line 367)
        fromnumeric_159694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 29), 'fromnumeric', False)
        # Getting the type of 'f' (line 367)
        f_159695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 42), 'f', False)
        # Processing the call keyword arguments (line 367)
        kwargs_159696 = {}
        # Getting the type of 'getattr' (line 367)
        getattr_159693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 21), 'getattr', False)
        # Calling getattr(args, kwargs) (line 367)
        getattr_call_result_159697 = invoke(stypy.reporting.localization.Localization(__file__, 367, 21), getattr_159693, *[fromnumeric_159694, f_159695], **kwargs_159696)
        
        # Assigning a type to the variable 'uf' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'uf', getattr_call_result_159697)
        # SSA join for try-except statement (line 364)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 368):
        
        # Assigning a Call to a Name (line 368):
        
        # Call to getattr(...): (line 368)
        # Processing the call arguments (line 368)
        # Getting the type of 'self' (line 368)
        self_159699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 25), 'self', False)
        # Obtaining the member 'module' of a type (line 368)
        module_159700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 25), self_159699, 'module')
        # Getting the type of 'f' (line 368)
        f_159701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 38), 'f', False)
        # Processing the call keyword arguments (line 368)
        kwargs_159702 = {}
        # Getting the type of 'getattr' (line 368)
        getattr_159698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 17), 'getattr', False)
        # Calling getattr(args, kwargs) (line 368)
        getattr_call_result_159703 = invoke(stypy.reporting.localization.Localization(__file__, 368, 17), getattr_159698, *[module_159700, f_159701], **kwargs_159702)
        
        # Assigning a type to the variable 'mf' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 12), 'mf', getattr_call_result_159703)
        
        # Assigning a Subscript to a Name (line 369):
        
        # Assigning a Subscript to a Name (line 369):
        
        # Obtaining the type of the subscript
        # Getting the type of 'uf' (line 369)
        uf_159704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 22), 'uf')
        # Obtaining the member 'nin' of a type (line 369)
        nin_159705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 22), uf_159704, 'nin')
        slice_159706 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 369, 19), None, nin_159705, None)
        # Getting the type of 'd' (line 369)
        d_159707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 19), 'd')
        # Obtaining the member '__getitem__' of a type (line 369)
        getitem___159708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 19), d_159707, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 369)
        subscript_call_result_159709 = invoke(stypy.reporting.localization.Localization(__file__, 369, 19), getitem___159708, slice_159706)
        
        # Assigning a type to the variable 'args' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 12), 'args', subscript_call_result_159709)
        
        # Assigning a Call to a Name (line 370):
        
        # Assigning a Call to a Name (line 370):
        
        # Call to uf(...): (line 370)
        # Getting the type of 'args' (line 370)
        args_159711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 21), 'args', False)
        # Processing the call keyword arguments (line 370)
        kwargs_159712 = {}
        # Getting the type of 'uf' (line 370)
        uf_159710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 17), 'uf', False)
        # Calling uf(args, kwargs) (line 370)
        uf_call_result_159713 = invoke(stypy.reporting.localization.Localization(__file__, 370, 17), uf_159710, *[args_159711], **kwargs_159712)
        
        # Assigning a type to the variable 'ur' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'ur', uf_call_result_159713)
        
        # Assigning a Call to a Name (line 371):
        
        # Assigning a Call to a Name (line 371):
        
        # Call to mf(...): (line 371)
        # Getting the type of 'args' (line 371)
        args_159715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 21), 'args', False)
        # Processing the call keyword arguments (line 371)
        kwargs_159716 = {}
        # Getting the type of 'mf' (line 371)
        mf_159714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 17), 'mf', False)
        # Calling mf(args, kwargs) (line 371)
        mf_call_result_159717 = invoke(stypy.reporting.localization.Localization(__file__, 371, 17), mf_159714, *[args_159715], **kwargs_159716)
        
        # Assigning a type to the variable 'mr' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'mr', mf_call_result_159717)
        
        # Call to assert_array_equal(...): (line 372)
        # Processing the call arguments (line 372)
        
        # Call to filled(...): (line 372)
        # Processing the call arguments (line 372)
        int_159722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 46), 'int')
        # Processing the call keyword arguments (line 372)
        kwargs_159723 = {}
        # Getting the type of 'ur' (line 372)
        ur_159720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 36), 'ur', False)
        # Obtaining the member 'filled' of a type (line 372)
        filled_159721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 36), ur_159720, 'filled')
        # Calling filled(args, kwargs) (line 372)
        filled_call_result_159724 = invoke(stypy.reporting.localization.Localization(__file__, 372, 36), filled_159721, *[int_159722], **kwargs_159723)
        
        
        # Call to filled(...): (line 372)
        # Processing the call arguments (line 372)
        int_159727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 60), 'int')
        # Processing the call keyword arguments (line 372)
        kwargs_159728 = {}
        # Getting the type of 'mr' (line 372)
        mr_159725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 50), 'mr', False)
        # Obtaining the member 'filled' of a type (line 372)
        filled_159726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 50), mr_159725, 'filled')
        # Calling filled(args, kwargs) (line 372)
        filled_call_result_159729 = invoke(stypy.reporting.localization.Localization(__file__, 372, 50), filled_159726, *[int_159727], **kwargs_159728)
        
        # Getting the type of 'f' (line 372)
        f_159730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 64), 'f', False)
        # Processing the call keyword arguments (line 372)
        kwargs_159731 = {}
        # Getting the type of 'self' (line 372)
        self_159718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 372)
        assert_array_equal_159719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 12), self_159718, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 372)
        assert_array_equal_call_result_159732 = invoke(stypy.reporting.localization.Localization(__file__, 372, 12), assert_array_equal_159719, *[filled_call_result_159724, filled_call_result_159729, f_159730], **kwargs_159731)
        
        
        # Call to assert_array_equal(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'ur' (line 373)
        ur_159735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 36), 'ur', False)
        # Obtaining the member '_mask' of a type (line 373)
        _mask_159736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 36), ur_159735, '_mask')
        # Getting the type of 'mr' (line 373)
        mr_159737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 46), 'mr', False)
        # Obtaining the member '_mask' of a type (line 373)
        _mask_159738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 46), mr_159737, '_mask')
        # Processing the call keyword arguments (line 373)
        kwargs_159739 = {}
        # Getting the type of 'self' (line 373)
        self_159733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 373)
        assert_array_equal_159734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 12), self_159733, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 373)
        assert_array_equal_call_result_159740 = invoke(stypy.reporting.localization.Localization(__file__, 373, 12), assert_array_equal_159734, *[_mask_159736, _mask_159738], **kwargs_159739)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_7(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_7' in the type store
        # Getting the type of 'stypy_return_type' (line 341)
        stypy_return_type_159741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_159741)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_7'
        return stypy_return_type_159741


    @norecursion
    def test_99(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_99'
        module_type_store = module_type_store.open_function_context('test_99', 375, 4, False)
        # Assigning a type to the variable 'self' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ModuleTester.test_99.__dict__.__setitem__('stypy_localization', localization)
        ModuleTester.test_99.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ModuleTester.test_99.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleTester.test_99.__dict__.__setitem__('stypy_function_name', 'ModuleTester.test_99')
        ModuleTester.test_99.__dict__.__setitem__('stypy_param_names_list', [])
        ModuleTester.test_99.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleTester.test_99.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleTester.test_99.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleTester.test_99.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleTester.test_99.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleTester.test_99.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ModuleTester.test_99', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_99', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_99(...)' code ##################

        
        # Assigning a Call to a Name (line 377):
        
        # Assigning a Call to a Name (line 377):
        
        # Call to array(...): (line 377)
        # Processing the call arguments (line 377)
        
        # Obtaining an instance of the builtin type 'list' (line 377)
        list_159744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 377)
        # Adding element type (line 377)
        float_159745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 25), list_159744, float_159745)
        # Adding element type (line 377)
        float_159746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 25), list_159744, float_159746)
        # Adding element type (line 377)
        float_159747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 25), list_159744, float_159747)
        # Adding element type (line 377)
        float_159748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 25), list_159744, float_159748)
        
        # Processing the call keyword arguments (line 377)
        
        # Obtaining an instance of the builtin type 'list' (line 377)
        list_159749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 377)
        # Adding element type (line 377)
        int_159750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 48), list_159749, int_159750)
        # Adding element type (line 377)
        int_159751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 48), list_159749, int_159751)
        # Adding element type (line 377)
        int_159752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 48), list_159749, int_159752)
        # Adding element type (line 377)
        int_159753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 48), list_159749, int_159753)
        
        keyword_159754 = list_159749
        kwargs_159755 = {'mask': keyword_159754}
        # Getting the type of 'self' (line 377)
        self_159742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 14), 'self', False)
        # Obtaining the member 'array' of a type (line 377)
        array_159743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 14), self_159742, 'array')
        # Calling array(args, kwargs) (line 377)
        array_call_result_159756 = invoke(stypy.reporting.localization.Localization(__file__, 377, 14), array_159743, *[list_159744], **kwargs_159755)
        
        # Assigning a type to the variable 'ott' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'ott', array_call_result_159756)
        
        # Call to assert_array_equal(...): (line 378)
        # Processing the call arguments (line 378)
        float_159759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 32), 'float')
        
        # Call to average(...): (line 378)
        # Processing the call arguments (line 378)
        # Getting the type of 'ott' (line 378)
        ott_159762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 50), 'ott', False)
        # Processing the call keyword arguments (line 378)
        int_159763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 60), 'int')
        keyword_159764 = int_159763
        kwargs_159765 = {'axis': keyword_159764}
        # Getting the type of 'self' (line 378)
        self_159760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 37), 'self', False)
        # Obtaining the member 'average' of a type (line 378)
        average_159761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 37), self_159760, 'average')
        # Calling average(args, kwargs) (line 378)
        average_call_result_159766 = invoke(stypy.reporting.localization.Localization(__file__, 378, 37), average_159761, *[ott_159762], **kwargs_159765)
        
        # Processing the call keyword arguments (line 378)
        kwargs_159767 = {}
        # Getting the type of 'self' (line 378)
        self_159757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 378)
        assert_array_equal_159758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), self_159757, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 378)
        assert_array_equal_call_result_159768 = invoke(stypy.reporting.localization.Localization(__file__, 378, 8), assert_array_equal_159758, *[float_159759, average_call_result_159766], **kwargs_159767)
        
        
        # Call to assert_array_equal(...): (line 379)
        # Processing the call arguments (line 379)
        float_159771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 32), 'float')
        
        # Call to average(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'ott' (line 379)
        ott_159774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 50), 'ott', False)
        # Processing the call keyword arguments (line 379)
        
        # Obtaining an instance of the builtin type 'list' (line 379)
        list_159775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 63), 'list')
        # Adding type elements to the builtin type 'list' instance (line 379)
        # Adding element type (line 379)
        float_159776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 63), list_159775, float_159776)
        # Adding element type (line 379)
        float_159777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 68), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 63), list_159775, float_159777)
        # Adding element type (line 379)
        float_159778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 72), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 63), list_159775, float_159778)
        # Adding element type (line 379)
        float_159779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 76), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 379, 63), list_159775, float_159779)
        
        keyword_159780 = list_159775
        kwargs_159781 = {'weights': keyword_159780}
        # Getting the type of 'self' (line 379)
        self_159772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 37), 'self', False)
        # Obtaining the member 'average' of a type (line 379)
        average_159773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 37), self_159772, 'average')
        # Calling average(args, kwargs) (line 379)
        average_call_result_159782 = invoke(stypy.reporting.localization.Localization(__file__, 379, 37), average_159773, *[ott_159774], **kwargs_159781)
        
        # Processing the call keyword arguments (line 379)
        kwargs_159783 = {}
        # Getting the type of 'self' (line 379)
        self_159769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 379)
        assert_array_equal_159770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 8), self_159769, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 379)
        assert_array_equal_call_result_159784 = invoke(stypy.reporting.localization.Localization(__file__, 379, 8), assert_array_equal_159770, *[float_159771, average_call_result_159782], **kwargs_159783)
        
        
        # Assigning a Call to a Tuple (line 380):
        
        # Assigning a Call to a Name:
        
        # Call to average(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'ott' (line 380)
        ott_159787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 35), 'ott', False)
        # Processing the call keyword arguments (line 380)
        
        # Obtaining an instance of the builtin type 'list' (line 380)
        list_159788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 380)
        # Adding element type (line 380)
        float_159789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 48), list_159788, float_159789)
        # Adding element type (line 380)
        float_159790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 48), list_159788, float_159790)
        # Adding element type (line 380)
        float_159791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 48), list_159788, float_159791)
        # Adding element type (line 380)
        float_159792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 48), list_159788, float_159792)
        
        keyword_159793 = list_159788
        int_159794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 75), 'int')
        keyword_159795 = int_159794
        kwargs_159796 = {'weights': keyword_159793, 'returned': keyword_159795}
        # Getting the type of 'self' (line 380)
        self_159785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 22), 'self', False)
        # Obtaining the member 'average' of a type (line 380)
        average_159786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 22), self_159785, 'average')
        # Calling average(args, kwargs) (line 380)
        average_call_result_159797 = invoke(stypy.reporting.localization.Localization(__file__, 380, 22), average_159786, *[ott_159787], **kwargs_159796)
        
        # Assigning a type to the variable 'call_assignment_158049' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'call_assignment_158049', average_call_result_159797)
        
        # Assigning a Call to a Name (line 380):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_159800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 8), 'int')
        # Processing the call keyword arguments
        kwargs_159801 = {}
        # Getting the type of 'call_assignment_158049' (line 380)
        call_assignment_158049_159798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'call_assignment_158049', False)
        # Obtaining the member '__getitem__' of a type (line 380)
        getitem___159799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 8), call_assignment_158049_159798, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_159802 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___159799, *[int_159800], **kwargs_159801)
        
        # Assigning a type to the variable 'call_assignment_158050' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'call_assignment_158050', getitem___call_result_159802)
        
        # Assigning a Name to a Name (line 380):
        # Getting the type of 'call_assignment_158050' (line 380)
        call_assignment_158050_159803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'call_assignment_158050')
        # Assigning a type to the variable 'result' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'result', call_assignment_158050_159803)
        
        # Assigning a Call to a Name (line 380):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_159806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 8), 'int')
        # Processing the call keyword arguments
        kwargs_159807 = {}
        # Getting the type of 'call_assignment_158049' (line 380)
        call_assignment_158049_159804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'call_assignment_158049', False)
        # Obtaining the member '__getitem__' of a type (line 380)
        getitem___159805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 8), call_assignment_158049_159804, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_159808 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___159805, *[int_159806], **kwargs_159807)
        
        # Assigning a type to the variable 'call_assignment_158051' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'call_assignment_158051', getitem___call_result_159808)
        
        # Assigning a Name to a Name (line 380):
        # Getting the type of 'call_assignment_158051' (line 380)
        call_assignment_158051_159809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'call_assignment_158051')
        # Assigning a type to the variable 'wts' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 16), 'wts', call_assignment_158051_159809)
        
        # Call to assert_array_equal(...): (line 381)
        # Processing the call arguments (line 381)
        float_159812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 32), 'float')
        # Getting the type of 'result' (line 381)
        result_159813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 37), 'result', False)
        # Processing the call keyword arguments (line 381)
        kwargs_159814 = {}
        # Getting the type of 'self' (line 381)
        self_159810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 381)
        assert_array_equal_159811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 8), self_159810, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 381)
        assert_array_equal_call_result_159815 = invoke(stypy.reporting.localization.Localization(__file__, 381, 8), assert_array_equal_159811, *[float_159812, result_159813], **kwargs_159814)
        
        # Evaluating assert statement condition
        
        # Getting the type of 'wts' (line 382)
        wts_159816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 15), 'wts')
        float_159817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 22), 'float')
        # Applying the binary operator '==' (line 382)
        result_eq_159818 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 15), '==', wts_159816, float_159817)
        
        
        # Assigning a Attribute to a Subscript (line 383):
        
        # Assigning a Attribute to a Subscript (line 383):
        # Getting the type of 'self' (line 383)
        self_159819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 17), 'self')
        # Obtaining the member 'masked' of a type (line 383)
        masked_159820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 17), self_159819, 'masked')
        # Getting the type of 'ott' (line 383)
        ott_159821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'ott')
        slice_159822 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 383, 8), None, None, None)
        # Storing an element on a container (line 383)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 8), ott_159821, (slice_159822, masked_159820))
        # Evaluating assert statement condition
        
        
        # Call to average(...): (line 384)
        # Processing the call arguments (line 384)
        # Getting the type of 'ott' (line 384)
        ott_159825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 28), 'ott', False)
        # Processing the call keyword arguments (line 384)
        int_159826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 38), 'int')
        keyword_159827 = int_159826
        kwargs_159828 = {'axis': keyword_159827}
        # Getting the type of 'self' (line 384)
        self_159823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 15), 'self', False)
        # Obtaining the member 'average' of a type (line 384)
        average_159824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 15), self_159823, 'average')
        # Calling average(args, kwargs) (line 384)
        average_call_result_159829 = invoke(stypy.reporting.localization.Localization(__file__, 384, 15), average_159824, *[ott_159825], **kwargs_159828)
        
        # Getting the type of 'self' (line 384)
        self_159830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 44), 'self')
        # Obtaining the member 'masked' of a type (line 384)
        masked_159831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 44), self_159830, 'masked')
        # Applying the binary operator 'is' (line 384)
        result_is__159832 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 15), 'is', average_call_result_159829, masked_159831)
        
        
        # Assigning a Call to a Name (line 385):
        
        # Assigning a Call to a Name (line 385):
        
        # Call to array(...): (line 385)
        # Processing the call arguments (line 385)
        
        # Obtaining an instance of the builtin type 'list' (line 385)
        list_159835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 385)
        # Adding element type (line 385)
        float_159836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 26), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 25), list_159835, float_159836)
        # Adding element type (line 385)
        float_159837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 30), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 25), list_159835, float_159837)
        # Adding element type (line 385)
        float_159838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 34), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 25), list_159835, float_159838)
        # Adding element type (line 385)
        float_159839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 25), list_159835, float_159839)
        
        # Processing the call keyword arguments (line 385)
        
        # Obtaining an instance of the builtin type 'list' (line 385)
        list_159840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 385)
        # Adding element type (line 385)
        int_159841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 48), list_159840, int_159841)
        # Adding element type (line 385)
        int_159842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 52), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 48), list_159840, int_159842)
        # Adding element type (line 385)
        int_159843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 55), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 48), list_159840, int_159843)
        # Adding element type (line 385)
        int_159844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 385, 48), list_159840, int_159844)
        
        keyword_159845 = list_159840
        kwargs_159846 = {'mask': keyword_159845}
        # Getting the type of 'self' (line 385)
        self_159833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 14), 'self', False)
        # Obtaining the member 'array' of a type (line 385)
        array_159834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 14), self_159833, 'array')
        # Calling array(args, kwargs) (line 385)
        array_call_result_159847 = invoke(stypy.reporting.localization.Localization(__file__, 385, 14), array_159834, *[list_159835], **kwargs_159846)
        
        # Assigning a type to the variable 'ott' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'ott', array_call_result_159847)
        
        # Assigning a Call to a Name (line 386):
        
        # Assigning a Call to a Name (line 386):
        
        # Call to reshape(...): (line 386)
        # Processing the call arguments (line 386)
        int_159850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 26), 'int')
        int_159851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 29), 'int')
        # Processing the call keyword arguments (line 386)
        kwargs_159852 = {}
        # Getting the type of 'ott' (line 386)
        ott_159848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 14), 'ott', False)
        # Obtaining the member 'reshape' of a type (line 386)
        reshape_159849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 14), ott_159848, 'reshape')
        # Calling reshape(args, kwargs) (line 386)
        reshape_call_result_159853 = invoke(stypy.reporting.localization.Localization(__file__, 386, 14), reshape_159849, *[int_159850, int_159851], **kwargs_159852)
        
        # Assigning a type to the variable 'ott' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'ott', reshape_call_result_159853)
        
        # Assigning a Attribute to a Subscript (line 387):
        
        # Assigning a Attribute to a Subscript (line 387):
        # Getting the type of 'self' (line 387)
        self_159854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 20), 'self')
        # Obtaining the member 'masked' of a type (line 387)
        masked_159855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 20), self_159854, 'masked')
        # Getting the type of 'ott' (line 387)
        ott_159856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'ott')
        slice_159857 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 387, 8), None, None, None)
        int_159858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 15), 'int')
        # Storing an element on a container (line 387)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 8), ott_159856, ((slice_159857, int_159858), masked_159855))
        
        # Call to assert_array_equal(...): (line 388)
        # Processing the call arguments (line 388)
        
        # Call to average(...): (line 388)
        # Processing the call arguments (line 388)
        # Getting the type of 'ott' (line 388)
        ott_159863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 45), 'ott', False)
        # Processing the call keyword arguments (line 388)
        int_159864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 55), 'int')
        keyword_159865 = int_159864
        kwargs_159866 = {'axis': keyword_159865}
        # Getting the type of 'self' (line 388)
        self_159861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 32), 'self', False)
        # Obtaining the member 'average' of a type (line 388)
        average_159862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 32), self_159861, 'average')
        # Calling average(args, kwargs) (line 388)
        average_call_result_159867 = invoke(stypy.reporting.localization.Localization(__file__, 388, 32), average_159862, *[ott_159863], **kwargs_159866)
        
        
        # Obtaining an instance of the builtin type 'list' (line 388)
        list_159868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 59), 'list')
        # Adding type elements to the builtin type 'list' instance (line 388)
        # Adding element type (line 388)
        float_159869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 59), list_159868, float_159869)
        # Adding element type (line 388)
        float_159870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 59), list_159868, float_159870)
        
        # Processing the call keyword arguments (line 388)
        kwargs_159871 = {}
        # Getting the type of 'self' (line 388)
        self_159859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 388)
        assert_array_equal_159860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 8), self_159859, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 388)
        assert_array_equal_call_result_159872 = invoke(stypy.reporting.localization.Localization(__file__, 388, 8), assert_array_equal_159860, *[average_call_result_159867, list_159868], **kwargs_159871)
        
        # Evaluating assert statement condition
        
        
        # Obtaining the type of the subscript
        int_159873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 41), 'int')
        
        # Call to average(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'ott' (line 389)
        ott_159876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 28), 'ott', False)
        # Processing the call keyword arguments (line 389)
        int_159877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 38), 'int')
        keyword_159878 = int_159877
        kwargs_159879 = {'axis': keyword_159878}
        # Getting the type of 'self' (line 389)
        self_159874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 15), 'self', False)
        # Obtaining the member 'average' of a type (line 389)
        average_159875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 15), self_159874, 'average')
        # Calling average(args, kwargs) (line 389)
        average_call_result_159880 = invoke(stypy.reporting.localization.Localization(__file__, 389, 15), average_159875, *[ott_159876], **kwargs_159879)
        
        # Obtaining the member '__getitem__' of a type (line 389)
        getitem___159881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 15), average_call_result_159880, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 389)
        subscript_call_result_159882 = invoke(stypy.reporting.localization.Localization(__file__, 389, 15), getitem___159881, int_159873)
        
        # Getting the type of 'self' (line 389)
        self_159883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 47), 'self')
        # Obtaining the member 'masked' of a type (line 389)
        masked_159884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 47), self_159883, 'masked')
        # Applying the binary operator 'is' (line 389)
        result_is__159885 = python_operator(stypy.reporting.localization.Localization(__file__, 389, 15), 'is', subscript_call_result_159882, masked_159884)
        
        
        # Call to assert_array_equal(...): (line 390)
        # Processing the call arguments (line 390)
        
        # Obtaining an instance of the builtin type 'list' (line 390)
        list_159888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 390)
        # Adding element type (line 390)
        float_159889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 32), list_159888, float_159889)
        # Adding element type (line 390)
        float_159890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 37), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 390, 32), list_159888, float_159890)
        
        
        # Call to average(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'ott' (line 390)
        ott_159893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 55), 'ott', False)
        # Processing the call keyword arguments (line 390)
        int_159894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 65), 'int')
        keyword_159895 = int_159894
        kwargs_159896 = {'axis': keyword_159895}
        # Getting the type of 'self' (line 390)
        self_159891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 42), 'self', False)
        # Obtaining the member 'average' of a type (line 390)
        average_159892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 42), self_159891, 'average')
        # Calling average(args, kwargs) (line 390)
        average_call_result_159897 = invoke(stypy.reporting.localization.Localization(__file__, 390, 42), average_159892, *[ott_159893], **kwargs_159896)
        
        # Processing the call keyword arguments (line 390)
        kwargs_159898 = {}
        # Getting the type of 'self' (line 390)
        self_159886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 390)
        assert_array_equal_159887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 8), self_159886, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 390)
        assert_array_equal_call_result_159899 = invoke(stypy.reporting.localization.Localization(__file__, 390, 8), assert_array_equal_159887, *[list_159888, average_call_result_159897], **kwargs_159898)
        
        
        # Assigning a Call to a Tuple (line 391):
        
        # Assigning a Call to a Name:
        
        # Call to average(...): (line 391)
        # Processing the call arguments (line 391)
        # Getting the type of 'ott' (line 391)
        ott_159902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 35), 'ott', False)
        # Processing the call keyword arguments (line 391)
        int_159903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 45), 'int')
        keyword_159904 = int_159903
        int_159905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 57), 'int')
        keyword_159906 = int_159905
        kwargs_159907 = {'returned': keyword_159906, 'axis': keyword_159904}
        # Getting the type of 'self' (line 391)
        self_159900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 22), 'self', False)
        # Obtaining the member 'average' of a type (line 391)
        average_159901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 22), self_159900, 'average')
        # Calling average(args, kwargs) (line 391)
        average_call_result_159908 = invoke(stypy.reporting.localization.Localization(__file__, 391, 22), average_159901, *[ott_159902], **kwargs_159907)
        
        # Assigning a type to the variable 'call_assignment_158052' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'call_assignment_158052', average_call_result_159908)
        
        # Assigning a Call to a Name (line 391):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_159911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 8), 'int')
        # Processing the call keyword arguments
        kwargs_159912 = {}
        # Getting the type of 'call_assignment_158052' (line 391)
        call_assignment_158052_159909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'call_assignment_158052', False)
        # Obtaining the member '__getitem__' of a type (line 391)
        getitem___159910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 8), call_assignment_158052_159909, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_159913 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___159910, *[int_159911], **kwargs_159912)
        
        # Assigning a type to the variable 'call_assignment_158053' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'call_assignment_158053', getitem___call_result_159913)
        
        # Assigning a Name to a Name (line 391):
        # Getting the type of 'call_assignment_158053' (line 391)
        call_assignment_158053_159914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'call_assignment_158053')
        # Assigning a type to the variable 'result' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'result', call_assignment_158053_159914)
        
        # Assigning a Call to a Name (line 391):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_159917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 8), 'int')
        # Processing the call keyword arguments
        kwargs_159918 = {}
        # Getting the type of 'call_assignment_158052' (line 391)
        call_assignment_158052_159915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'call_assignment_158052', False)
        # Obtaining the member '__getitem__' of a type (line 391)
        getitem___159916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 8), call_assignment_158052_159915, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_159919 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___159916, *[int_159917], **kwargs_159918)
        
        # Assigning a type to the variable 'call_assignment_158054' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'call_assignment_158054', getitem___call_result_159919)
        
        # Assigning a Name to a Name (line 391):
        # Getting the type of 'call_assignment_158054' (line 391)
        call_assignment_158054_159920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'call_assignment_158054')
        # Assigning a type to the variable 'wts' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 16), 'wts', call_assignment_158054_159920)
        
        # Call to assert_array_equal(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'wts' (line 392)
        wts_159923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 32), 'wts', False)
        
        # Obtaining an instance of the builtin type 'list' (line 392)
        list_159924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 392)
        # Adding element type (line 392)
        float_159925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 38), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 37), list_159924, float_159925)
        # Adding element type (line 392)
        float_159926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 42), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 37), list_159924, float_159926)
        
        # Processing the call keyword arguments (line 392)
        kwargs_159927 = {}
        # Getting the type of 'self' (line 392)
        self_159921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 392)
        assert_array_equal_159922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 8), self_159921, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 392)
        assert_array_equal_call_result_159928 = invoke(stypy.reporting.localization.Localization(__file__, 392, 8), assert_array_equal_159922, *[wts_159923, list_159924], **kwargs_159927)
        
        
        # Assigning a List to a Name (line 393):
        
        # Assigning a List to a Name (line 393):
        
        # Obtaining an instance of the builtin type 'list' (line 393)
        list_159929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 393)
        # Adding element type (line 393)
        int_159930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 13), list_159929, int_159930)
        # Adding element type (line 393)
        int_159931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 13), list_159929, int_159931)
        # Adding element type (line 393)
        int_159932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 13), list_159929, int_159932)
        # Adding element type (line 393)
        int_159933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 13), list_159929, int_159933)
        # Adding element type (line 393)
        int_159934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 13), list_159929, int_159934)
        # Adding element type (line 393)
        int_159935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 393, 13), list_159929, int_159935)
        
        # Assigning a type to the variable 'w1' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'w1', list_159929)
        
        # Assigning a List to a Name (line 394):
        
        # Assigning a List to a Name (line 394):
        
        # Obtaining an instance of the builtin type 'list' (line 394)
        list_159936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 394)
        # Adding element type (line 394)
        
        # Obtaining an instance of the builtin type 'list' (line 394)
        list_159937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 394)
        # Adding element type (line 394)
        int_159938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 14), list_159937, int_159938)
        # Adding element type (line 394)
        int_159939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 14), list_159937, int_159939)
        # Adding element type (line 394)
        int_159940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 14), list_159937, int_159940)
        # Adding element type (line 394)
        int_159941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 14), list_159937, int_159941)
        # Adding element type (line 394)
        int_159942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 14), list_159937, int_159942)
        # Adding element type (line 394)
        int_159943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 14), list_159937, int_159943)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 13), list_159936, list_159937)
        # Adding element type (line 394)
        
        # Obtaining an instance of the builtin type 'list' (line 394)
        list_159944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 394)
        # Adding element type (line 394)
        int_159945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 34), list_159944, int_159945)
        # Adding element type (line 394)
        int_159946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 34), list_159944, int_159946)
        # Adding element type (line 394)
        int_159947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 34), list_159944, int_159947)
        # Adding element type (line 394)
        int_159948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 34), list_159944, int_159948)
        # Adding element type (line 394)
        int_159949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 34), list_159944, int_159949)
        # Adding element type (line 394)
        int_159950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 34), list_159944, int_159950)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 394, 13), list_159936, list_159944)
        
        # Assigning a type to the variable 'w2' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 8), 'w2', list_159936)
        
        # Assigning a Call to a Name (line 395):
        
        # Assigning a Call to a Name (line 395):
        
        # Call to arange(...): (line 395)
        # Processing the call arguments (line 395)
        int_159953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 24), 'int')
        # Processing the call keyword arguments (line 395)
        kwargs_159954 = {}
        # Getting the type of 'self' (line 395)
        self_159951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 395)
        arange_159952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 12), self_159951, 'arange')
        # Calling arange(args, kwargs) (line 395)
        arange_call_result_159955 = invoke(stypy.reporting.localization.Localization(__file__, 395, 12), arange_159952, *[int_159953], **kwargs_159954)
        
        # Assigning a type to the variable 'x' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'x', arange_call_result_159955)
        
        # Call to assert_array_equal(...): (line 396)
        # Processing the call arguments (line 396)
        
        # Call to average(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'x' (line 396)
        x_159960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 45), 'x', False)
        # Processing the call keyword arguments (line 396)
        int_159961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 53), 'int')
        keyword_159962 = int_159961
        kwargs_159963 = {'axis': keyword_159962}
        # Getting the type of 'self' (line 396)
        self_159958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 32), 'self', False)
        # Obtaining the member 'average' of a type (line 396)
        average_159959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 32), self_159958, 'average')
        # Calling average(args, kwargs) (line 396)
        average_call_result_159964 = invoke(stypy.reporting.localization.Localization(__file__, 396, 32), average_159959, *[x_159960], **kwargs_159963)
        
        float_159965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 57), 'float')
        # Processing the call keyword arguments (line 396)
        kwargs_159966 = {}
        # Getting the type of 'self' (line 396)
        self_159956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 396)
        assert_array_equal_159957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 8), self_159956, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 396)
        assert_array_equal_call_result_159967 = invoke(stypy.reporting.localization.Localization(__file__, 396, 8), assert_array_equal_159957, *[average_call_result_159964, float_159965], **kwargs_159966)
        
        
        # Call to assert_array_equal(...): (line 397)
        # Processing the call arguments (line 397)
        
        # Call to average(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'x' (line 397)
        x_159972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 45), 'x', False)
        # Processing the call keyword arguments (line 397)
        int_159973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 53), 'int')
        keyword_159974 = int_159973
        # Getting the type of 'w1' (line 397)
        w1_159975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 64), 'w1', False)
        keyword_159976 = w1_159975
        kwargs_159977 = {'weights': keyword_159976, 'axis': keyword_159974}
        # Getting the type of 'self' (line 397)
        self_159970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 32), 'self', False)
        # Obtaining the member 'average' of a type (line 397)
        average_159971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 32), self_159970, 'average')
        # Calling average(args, kwargs) (line 397)
        average_call_result_159978 = invoke(stypy.reporting.localization.Localization(__file__, 397, 32), average_159971, *[x_159972], **kwargs_159977)
        
        float_159979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 69), 'float')
        # Processing the call keyword arguments (line 397)
        kwargs_159980 = {}
        # Getting the type of 'self' (line 397)
        self_159968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 397)
        assert_array_equal_159969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), self_159968, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 397)
        assert_array_equal_call_result_159981 = invoke(stypy.reporting.localization.Localization(__file__, 397, 8), assert_array_equal_159969, *[average_call_result_159978, float_159979], **kwargs_159980)
        
        
        # Assigning a Call to a Name (line 398):
        
        # Assigning a Call to a Name (line 398):
        
        # Call to array(...): (line 398)
        # Processing the call arguments (line 398)
        
        # Obtaining an instance of the builtin type 'list' (line 398)
        list_159984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 398)
        # Adding element type (line 398)
        
        # Call to arange(...): (line 398)
        # Processing the call arguments (line 398)
        int_159987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 36), 'int')
        # Processing the call keyword arguments (line 398)
        kwargs_159988 = {}
        # Getting the type of 'self' (line 398)
        self_159985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 24), 'self', False)
        # Obtaining the member 'arange' of a type (line 398)
        arange_159986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 24), self_159985, 'arange')
        # Calling arange(args, kwargs) (line 398)
        arange_call_result_159989 = invoke(stypy.reporting.localization.Localization(__file__, 398, 24), arange_159986, *[int_159987], **kwargs_159988)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 23), list_159984, arange_call_result_159989)
        # Adding element type (line 398)
        float_159990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 40), 'float')
        
        # Call to arange(...): (line 398)
        # Processing the call arguments (line 398)
        int_159993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 56), 'int')
        # Processing the call keyword arguments (line 398)
        kwargs_159994 = {}
        # Getting the type of 'self' (line 398)
        self_159991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 44), 'self', False)
        # Obtaining the member 'arange' of a type (line 398)
        arange_159992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 44), self_159991, 'arange')
        # Calling arange(args, kwargs) (line 398)
        arange_call_result_159995 = invoke(stypy.reporting.localization.Localization(__file__, 398, 44), arange_159992, *[int_159993], **kwargs_159994)
        
        # Applying the binary operator '*' (line 398)
        result_mul_159996 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 40), '*', float_159990, arange_call_result_159995)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 23), list_159984, result_mul_159996)
        
        # Processing the call keyword arguments (line 398)
        kwargs_159997 = {}
        # Getting the type of 'self' (line 398)
        self_159982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'self', False)
        # Obtaining the member 'array' of a type (line 398)
        array_159983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 12), self_159982, 'array')
        # Calling array(args, kwargs) (line 398)
        array_call_result_159998 = invoke(stypy.reporting.localization.Localization(__file__, 398, 12), array_159983, *[list_159984], **kwargs_159997)
        
        # Assigning a type to the variable 'y' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'y', array_call_result_159998)
        
        # Call to assert_array_equal(...): (line 399)
        # Processing the call arguments (line 399)
        
        # Call to average(...): (line 399)
        # Processing the call arguments (line 399)
        # Getting the type of 'y' (line 399)
        y_160003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 45), 'y', False)
        # Getting the type of 'None' (line 399)
        None_160004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 48), 'None', False)
        # Processing the call keyword arguments (line 399)
        kwargs_160005 = {}
        # Getting the type of 'self' (line 399)
        self_160001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 32), 'self', False)
        # Obtaining the member 'average' of a type (line 399)
        average_160002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 32), self_160001, 'average')
        # Calling average(args, kwargs) (line 399)
        average_call_result_160006 = invoke(stypy.reporting.localization.Localization(__file__, 399, 32), average_160002, *[y_160003, None_160004], **kwargs_160005)
        
        
        # Call to reduce(...): (line 399)
        # Processing the call arguments (line 399)
        
        # Call to arange(...): (line 399)
        # Processing the call arguments (line 399)
        int_160012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 79), 'int')
        # Processing the call keyword arguments (line 399)
        kwargs_160013 = {}
        # Getting the type of 'np' (line 399)
        np_160010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 69), 'np', False)
        # Obtaining the member 'arange' of a type (line 399)
        arange_160011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 69), np_160010, 'arange')
        # Calling arange(args, kwargs) (line 399)
        arange_call_result_160014 = invoke(stypy.reporting.localization.Localization(__file__, 399, 69), arange_160011, *[int_160012], **kwargs_160013)
        
        # Processing the call keyword arguments (line 399)
        kwargs_160015 = {}
        # Getting the type of 'np' (line 399)
        np_160007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 55), 'np', False)
        # Obtaining the member 'add' of a type (line 399)
        add_160008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 55), np_160007, 'add')
        # Obtaining the member 'reduce' of a type (line 399)
        reduce_160009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 55), add_160008, 'reduce')
        # Calling reduce(args, kwargs) (line 399)
        reduce_call_result_160016 = invoke(stypy.reporting.localization.Localization(__file__, 399, 55), reduce_160009, *[arange_call_result_160014], **kwargs_160015)
        
        float_160017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 83), 'float')
        # Applying the binary operator '*' (line 399)
        result_mul_160018 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 55), '*', reduce_call_result_160016, float_160017)
        
        float_160019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 86), 'float')
        # Applying the binary operator 'div' (line 399)
        result_div_160020 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 85), 'div', result_mul_160018, float_160019)
        
        # Processing the call keyword arguments (line 399)
        kwargs_160021 = {}
        # Getting the type of 'self' (line 399)
        self_159999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 399)
        assert_array_equal_160000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), self_159999, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 399)
        assert_array_equal_call_result_160022 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), assert_array_equal_160000, *[average_call_result_160006, result_div_160020], **kwargs_160021)
        
        
        # Call to assert_array_equal(...): (line 400)
        # Processing the call arguments (line 400)
        
        # Call to average(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'y' (line 400)
        y_160027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 45), 'y', False)
        # Processing the call keyword arguments (line 400)
        int_160028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 53), 'int')
        keyword_160029 = int_160028
        kwargs_160030 = {'axis': keyword_160029}
        # Getting the type of 'self' (line 400)
        self_160025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 32), 'self', False)
        # Obtaining the member 'average' of a type (line 400)
        average_160026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 32), self_160025, 'average')
        # Calling average(args, kwargs) (line 400)
        average_call_result_160031 = invoke(stypy.reporting.localization.Localization(__file__, 400, 32), average_160026, *[y_160027], **kwargs_160030)
        
        
        # Call to arange(...): (line 400)
        # Processing the call arguments (line 400)
        int_160034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 67), 'int')
        # Processing the call keyword arguments (line 400)
        kwargs_160035 = {}
        # Getting the type of 'np' (line 400)
        np_160032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 57), 'np', False)
        # Obtaining the member 'arange' of a type (line 400)
        arange_160033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 57), np_160032, 'arange')
        # Calling arange(args, kwargs) (line 400)
        arange_call_result_160036 = invoke(stypy.reporting.localization.Localization(__file__, 400, 57), arange_160033, *[int_160034], **kwargs_160035)
        
        float_160037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 72), 'float')
        # Applying the binary operator '*' (line 400)
        result_mul_160038 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 57), '*', arange_call_result_160036, float_160037)
        
        float_160039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 75), 'float')
        # Applying the binary operator 'div' (line 400)
        result_div_160040 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 74), 'div', result_mul_160038, float_160039)
        
        # Processing the call keyword arguments (line 400)
        kwargs_160041 = {}
        # Getting the type of 'self' (line 400)
        self_160023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 400)
        assert_array_equal_160024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 8), self_160023, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 400)
        assert_array_equal_call_result_160042 = invoke(stypy.reporting.localization.Localization(__file__, 400, 8), assert_array_equal_160024, *[average_call_result_160031, result_div_160040], **kwargs_160041)
        
        
        # Call to assert_array_equal(...): (line 401)
        # Processing the call arguments (line 401)
        
        # Call to average(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'y' (line 401)
        y_160047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 45), 'y', False)
        # Processing the call keyword arguments (line 401)
        int_160048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 53), 'int')
        keyword_160049 = int_160048
        kwargs_160050 = {'axis': keyword_160049}
        # Getting the type of 'self' (line 401)
        self_160045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 32), 'self', False)
        # Obtaining the member 'average' of a type (line 401)
        average_160046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 32), self_160045, 'average')
        # Calling average(args, kwargs) (line 401)
        average_call_result_160051 = invoke(stypy.reporting.localization.Localization(__file__, 401, 32), average_160046, *[y_160047], **kwargs_160050)
        
        
        # Obtaining an instance of the builtin type 'list' (line 401)
        list_160052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 401)
        # Adding element type (line 401)
        
        # Call to average(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'x' (line 401)
        x_160055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 71), 'x', False)
        # Processing the call keyword arguments (line 401)
        int_160056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 79), 'int')
        keyword_160057 = int_160056
        kwargs_160058 = {'axis': keyword_160057}
        # Getting the type of 'self' (line 401)
        self_160053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 58), 'self', False)
        # Obtaining the member 'average' of a type (line 401)
        average_160054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 58), self_160053, 'average')
        # Calling average(args, kwargs) (line 401)
        average_call_result_160059 = invoke(stypy.reporting.localization.Localization(__file__, 401, 58), average_160054, *[x_160055], **kwargs_160058)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 57), list_160052, average_call_result_160059)
        # Adding element type (line 401)
        
        # Call to average(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'x' (line 401)
        x_160062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 96), 'x', False)
        # Processing the call keyword arguments (line 401)
        int_160063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 104), 'int')
        keyword_160064 = int_160063
        kwargs_160065 = {'axis': keyword_160064}
        # Getting the type of 'self' (line 401)
        self_160060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 83), 'self', False)
        # Obtaining the member 'average' of a type (line 401)
        average_160061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 83), self_160060, 'average')
        # Calling average(args, kwargs) (line 401)
        average_call_result_160066 = invoke(stypy.reporting.localization.Localization(__file__, 401, 83), average_160061, *[x_160062], **kwargs_160065)
        
        float_160067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 109), 'float')
        # Applying the binary operator '*' (line 401)
        result_mul_160068 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 83), '*', average_call_result_160066, float_160067)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 401, 57), list_160052, result_mul_160068)
        
        # Processing the call keyword arguments (line 401)
        kwargs_160069 = {}
        # Getting the type of 'self' (line 401)
        self_160043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 401)
        assert_array_equal_160044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 8), self_160043, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 401)
        assert_array_equal_call_result_160070 = invoke(stypy.reporting.localization.Localization(__file__, 401, 8), assert_array_equal_160044, *[average_call_result_160051, list_160052], **kwargs_160069)
        
        
        # Call to assert_array_equal(...): (line 402)
        # Processing the call arguments (line 402)
        
        # Call to average(...): (line 402)
        # Processing the call arguments (line 402)
        # Getting the type of 'y' (line 402)
        y_160075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 45), 'y', False)
        # Getting the type of 'None' (line 402)
        None_160076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 48), 'None', False)
        # Processing the call keyword arguments (line 402)
        # Getting the type of 'w2' (line 402)
        w2_160077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 62), 'w2', False)
        keyword_160078 = w2_160077
        kwargs_160079 = {'weights': keyword_160078}
        # Getting the type of 'self' (line 402)
        self_160073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 32), 'self', False)
        # Obtaining the member 'average' of a type (line 402)
        average_160074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 32), self_160073, 'average')
        # Calling average(args, kwargs) (line 402)
        average_call_result_160080 = invoke(stypy.reporting.localization.Localization(__file__, 402, 32), average_160074, *[y_160075, None_160076], **kwargs_160079)
        
        float_160081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 67), 'float')
        float_160082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 71), 'float')
        # Applying the binary operator 'div' (line 402)
        result_div_160083 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 67), 'div', float_160081, float_160082)
        
        # Processing the call keyword arguments (line 402)
        kwargs_160084 = {}
        # Getting the type of 'self' (line 402)
        self_160071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 402)
        assert_array_equal_160072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 8), self_160071, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 402)
        assert_array_equal_call_result_160085 = invoke(stypy.reporting.localization.Localization(__file__, 402, 8), assert_array_equal_160072, *[average_call_result_160080, result_div_160083], **kwargs_160084)
        
        
        # Call to assert_array_equal(...): (line 403)
        # Processing the call arguments (line 403)
        
        # Call to average(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 'y' (line 403)
        y_160090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 45), 'y', False)
        # Processing the call keyword arguments (line 403)
        int_160091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 53), 'int')
        keyword_160092 = int_160091
        # Getting the type of 'w2' (line 403)
        w2_160093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 64), 'w2', False)
        keyword_160094 = w2_160093
        kwargs_160095 = {'weights': keyword_160094, 'axis': keyword_160092}
        # Getting the type of 'self' (line 403)
        self_160088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 32), 'self', False)
        # Obtaining the member 'average' of a type (line 403)
        average_160089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 32), self_160088, 'average')
        # Calling average(args, kwargs) (line 403)
        average_call_result_160096 = invoke(stypy.reporting.localization.Localization(__file__, 403, 32), average_160089, *[y_160090], **kwargs_160095)
        
        
        # Obtaining an instance of the builtin type 'list' (line 403)
        list_160097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 69), 'list')
        # Adding type elements to the builtin type 'list' instance (line 403)
        # Adding element type (line 403)
        float_160098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 69), list_160097, float_160098)
        # Adding element type (line 403)
        float_160099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 74), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 69), list_160097, float_160099)
        # Adding element type (line 403)
        float_160100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 78), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 69), list_160097, float_160100)
        # Adding element type (line 403)
        float_160101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 82), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 69), list_160097, float_160101)
        # Adding element type (line 403)
        float_160102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 86), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 69), list_160097, float_160102)
        # Adding element type (line 403)
        float_160103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 90), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 403, 69), list_160097, float_160103)
        
        # Processing the call keyword arguments (line 403)
        kwargs_160104 = {}
        # Getting the type of 'self' (line 403)
        self_160086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 403)
        assert_array_equal_160087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 8), self_160086, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 403)
        assert_array_equal_call_result_160105 = invoke(stypy.reporting.localization.Localization(__file__, 403, 8), assert_array_equal_160087, *[average_call_result_160096, list_160097], **kwargs_160104)
        
        
        # Call to assert_array_equal(...): (line 404)
        # Processing the call arguments (line 404)
        
        # Call to average(...): (line 404)
        # Processing the call arguments (line 404)
        # Getting the type of 'y' (line 404)
        y_160110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 45), 'y', False)
        # Processing the call keyword arguments (line 404)
        int_160111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 53), 'int')
        keyword_160112 = int_160111
        kwargs_160113 = {'axis': keyword_160112}
        # Getting the type of 'self' (line 404)
        self_160108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 32), 'self', False)
        # Obtaining the member 'average' of a type (line 404)
        average_160109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 32), self_160108, 'average')
        # Calling average(args, kwargs) (line 404)
        average_call_result_160114 = invoke(stypy.reporting.localization.Localization(__file__, 404, 32), average_160109, *[y_160110], **kwargs_160113)
        
        
        # Obtaining an instance of the builtin type 'list' (line 404)
        list_160115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 404)
        # Adding element type (line 404)
        
        # Call to average(...): (line 404)
        # Processing the call arguments (line 404)
        # Getting the type of 'x' (line 404)
        x_160118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 71), 'x', False)
        # Processing the call keyword arguments (line 404)
        int_160119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 79), 'int')
        keyword_160120 = int_160119
        kwargs_160121 = {'axis': keyword_160120}
        # Getting the type of 'self' (line 404)
        self_160116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 58), 'self', False)
        # Obtaining the member 'average' of a type (line 404)
        average_160117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 58), self_160116, 'average')
        # Calling average(args, kwargs) (line 404)
        average_call_result_160122 = invoke(stypy.reporting.localization.Localization(__file__, 404, 58), average_160117, *[x_160118], **kwargs_160121)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 57), list_160115, average_call_result_160122)
        # Adding element type (line 404)
        
        # Call to average(...): (line 404)
        # Processing the call arguments (line 404)
        # Getting the type of 'x' (line 404)
        x_160125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 96), 'x', False)
        # Processing the call keyword arguments (line 404)
        int_160126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 104), 'int')
        keyword_160127 = int_160126
        kwargs_160128 = {'axis': keyword_160127}
        # Getting the type of 'self' (line 404)
        self_160123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 83), 'self', False)
        # Obtaining the member 'average' of a type (line 404)
        average_160124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 83), self_160123, 'average')
        # Calling average(args, kwargs) (line 404)
        average_call_result_160129 = invoke(stypy.reporting.localization.Localization(__file__, 404, 83), average_160124, *[x_160125], **kwargs_160128)
        
        float_160130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 109), 'float')
        # Applying the binary operator '*' (line 404)
        result_mul_160131 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 83), '*', average_call_result_160129, float_160130)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 57), list_160115, result_mul_160131)
        
        # Processing the call keyword arguments (line 404)
        kwargs_160132 = {}
        # Getting the type of 'self' (line 404)
        self_160106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 404)
        assert_array_equal_160107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 8), self_160106, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 404)
        assert_array_equal_call_result_160133 = invoke(stypy.reporting.localization.Localization(__file__, 404, 8), assert_array_equal_160107, *[average_call_result_160114, list_160115], **kwargs_160132)
        
        
        # Assigning a Call to a Name (line 405):
        
        # Assigning a Call to a Name (line 405):
        
        # Call to zeros(...): (line 405)
        # Processing the call arguments (line 405)
        int_160136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 24), 'int')
        # Processing the call keyword arguments (line 405)
        kwargs_160137 = {}
        # Getting the type of 'self' (line 405)
        self_160134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 13), 'self', False)
        # Obtaining the member 'zeros' of a type (line 405)
        zeros_160135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 13), self_160134, 'zeros')
        # Calling zeros(args, kwargs) (line 405)
        zeros_call_result_160138 = invoke(stypy.reporting.localization.Localization(__file__, 405, 13), zeros_160135, *[int_160136], **kwargs_160137)
        
        # Assigning a type to the variable 'm1' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'm1', zeros_call_result_160138)
        
        # Assigning a List to a Name (line 406):
        
        # Assigning a List to a Name (line 406):
        
        # Obtaining an instance of the builtin type 'list' (line 406)
        list_160139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 406)
        # Adding element type (line 406)
        int_160140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 13), list_160139, int_160140)
        # Adding element type (line 406)
        int_160141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 13), list_160139, int_160141)
        # Adding element type (line 406)
        int_160142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 13), list_160139, int_160142)
        # Adding element type (line 406)
        int_160143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 13), list_160139, int_160143)
        # Adding element type (line 406)
        int_160144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 13), list_160139, int_160144)
        # Adding element type (line 406)
        int_160145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 13), list_160139, int_160145)
        
        # Assigning a type to the variable 'm2' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'm2', list_160139)
        
        # Assigning a List to a Name (line 407):
        
        # Assigning a List to a Name (line 407):
        
        # Obtaining an instance of the builtin type 'list' (line 407)
        list_160146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 407)
        # Adding element type (line 407)
        
        # Obtaining an instance of the builtin type 'list' (line 407)
        list_160147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 407)
        # Adding element type (line 407)
        int_160148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 14), list_160147, int_160148)
        # Adding element type (line 407)
        int_160149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 18), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 14), list_160147, int_160149)
        # Adding element type (line 407)
        int_160150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 14), list_160147, int_160150)
        # Adding element type (line 407)
        int_160151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 24), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 14), list_160147, int_160151)
        # Adding element type (line 407)
        int_160152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 27), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 14), list_160147, int_160152)
        # Adding element type (line 407)
        int_160153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 30), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 14), list_160147, int_160153)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 13), list_160146, list_160147)
        # Adding element type (line 407)
        
        # Obtaining an instance of the builtin type 'list' (line 407)
        list_160154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 407)
        # Adding element type (line 407)
        int_160155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 34), list_160154, int_160155)
        # Adding element type (line 407)
        int_160156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 34), list_160154, int_160156)
        # Adding element type (line 407)
        int_160157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 41), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 34), list_160154, int_160157)
        # Adding element type (line 407)
        int_160158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 34), list_160154, int_160158)
        # Adding element type (line 407)
        int_160159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 34), list_160154, int_160159)
        # Adding element type (line 407)
        int_160160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 34), list_160154, int_160160)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 407, 13), list_160146, list_160154)
        
        # Assigning a type to the variable 'm3' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'm3', list_160146)
        
        # Assigning a Call to a Name (line 408):
        
        # Assigning a Call to a Name (line 408):
        
        # Call to ones(...): (line 408)
        # Processing the call arguments (line 408)
        int_160163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 23), 'int')
        # Processing the call keyword arguments (line 408)
        kwargs_160164 = {}
        # Getting the type of 'self' (line 408)
        self_160161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 13), 'self', False)
        # Obtaining the member 'ones' of a type (line 408)
        ones_160162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 13), self_160161, 'ones')
        # Calling ones(args, kwargs) (line 408)
        ones_call_result_160165 = invoke(stypy.reporting.localization.Localization(__file__, 408, 13), ones_160162, *[int_160163], **kwargs_160164)
        
        # Assigning a type to the variable 'm4' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'm4', ones_call_result_160165)
        
        # Assigning a List to a Name (line 409):
        
        # Assigning a List to a Name (line 409):
        
        # Obtaining an instance of the builtin type 'list' (line 409)
        list_160166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 409)
        # Adding element type (line 409)
        int_160167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 13), list_160166, int_160167)
        # Adding element type (line 409)
        int_160168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 13), list_160166, int_160168)
        # Adding element type (line 409)
        int_160169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 13), list_160166, int_160169)
        # Adding element type (line 409)
        int_160170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 23), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 13), list_160166, int_160170)
        # Adding element type (line 409)
        int_160171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 26), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 13), list_160166, int_160171)
        # Adding element type (line 409)
        int_160172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 29), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 409, 13), list_160166, int_160172)
        
        # Assigning a type to the variable 'm5' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'm5', list_160166)
        
        # Call to assert_array_equal(...): (line 410)
        # Processing the call arguments (line 410)
        
        # Call to average(...): (line 410)
        # Processing the call arguments (line 410)
        
        # Call to masked_array(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 'x' (line 410)
        x_160179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 63), 'x', False)
        # Getting the type of 'm1' (line 410)
        m1_160180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 66), 'm1', False)
        # Processing the call keyword arguments (line 410)
        kwargs_160181 = {}
        # Getting the type of 'self' (line 410)
        self_160177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 45), 'self', False)
        # Obtaining the member 'masked_array' of a type (line 410)
        masked_array_160178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 45), self_160177, 'masked_array')
        # Calling masked_array(args, kwargs) (line 410)
        masked_array_call_result_160182 = invoke(stypy.reporting.localization.Localization(__file__, 410, 45), masked_array_160178, *[x_160179, m1_160180], **kwargs_160181)
        
        # Processing the call keyword arguments (line 410)
        int_160183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 76), 'int')
        keyword_160184 = int_160183
        kwargs_160185 = {'axis': keyword_160184}
        # Getting the type of 'self' (line 410)
        self_160175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 32), 'self', False)
        # Obtaining the member 'average' of a type (line 410)
        average_160176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 32), self_160175, 'average')
        # Calling average(args, kwargs) (line 410)
        average_call_result_160186 = invoke(stypy.reporting.localization.Localization(__file__, 410, 32), average_160176, *[masked_array_call_result_160182], **kwargs_160185)
        
        float_160187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 80), 'float')
        # Processing the call keyword arguments (line 410)
        kwargs_160188 = {}
        # Getting the type of 'self' (line 410)
        self_160173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 410)
        assert_array_equal_160174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 8), self_160173, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 410)
        assert_array_equal_call_result_160189 = invoke(stypy.reporting.localization.Localization(__file__, 410, 8), assert_array_equal_160174, *[average_call_result_160186, float_160187], **kwargs_160188)
        
        
        # Call to assert_array_equal(...): (line 411)
        # Processing the call arguments (line 411)
        
        # Call to average(...): (line 411)
        # Processing the call arguments (line 411)
        
        # Call to masked_array(...): (line 411)
        # Processing the call arguments (line 411)
        # Getting the type of 'x' (line 411)
        x_160196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 63), 'x', False)
        # Getting the type of 'm2' (line 411)
        m2_160197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 66), 'm2', False)
        # Processing the call keyword arguments (line 411)
        kwargs_160198 = {}
        # Getting the type of 'self' (line 411)
        self_160194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 45), 'self', False)
        # Obtaining the member 'masked_array' of a type (line 411)
        masked_array_160195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 45), self_160194, 'masked_array')
        # Calling masked_array(args, kwargs) (line 411)
        masked_array_call_result_160199 = invoke(stypy.reporting.localization.Localization(__file__, 411, 45), masked_array_160195, *[x_160196, m2_160197], **kwargs_160198)
        
        # Processing the call keyword arguments (line 411)
        int_160200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 76), 'int')
        keyword_160201 = int_160200
        kwargs_160202 = {'axis': keyword_160201}
        # Getting the type of 'self' (line 411)
        self_160192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 32), 'self', False)
        # Obtaining the member 'average' of a type (line 411)
        average_160193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 32), self_160192, 'average')
        # Calling average(args, kwargs) (line 411)
        average_call_result_160203 = invoke(stypy.reporting.localization.Localization(__file__, 411, 32), average_160193, *[masked_array_call_result_160199], **kwargs_160202)
        
        float_160204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 80), 'float')
        # Processing the call keyword arguments (line 411)
        kwargs_160205 = {}
        # Getting the type of 'self' (line 411)
        self_160190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 411)
        assert_array_equal_160191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), self_160190, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 411)
        assert_array_equal_call_result_160206 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), assert_array_equal_160191, *[average_call_result_160203, float_160204], **kwargs_160205)
        
        
        # Call to assert_array_equal(...): (line 412)
        # Processing the call arguments (line 412)
        
        # Call to average(...): (line 412)
        # Processing the call arguments (line 412)
        
        # Call to masked_array(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 'x' (line 412)
        x_160213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 63), 'x', False)
        # Getting the type of 'm5' (line 412)
        m5_160214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 66), 'm5', False)
        # Processing the call keyword arguments (line 412)
        kwargs_160215 = {}
        # Getting the type of 'self' (line 412)
        self_160211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 45), 'self', False)
        # Obtaining the member 'masked_array' of a type (line 412)
        masked_array_160212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 45), self_160211, 'masked_array')
        # Calling masked_array(args, kwargs) (line 412)
        masked_array_call_result_160216 = invoke(stypy.reporting.localization.Localization(__file__, 412, 45), masked_array_160212, *[x_160213, m5_160214], **kwargs_160215)
        
        # Processing the call keyword arguments (line 412)
        int_160217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 76), 'int')
        keyword_160218 = int_160217
        kwargs_160219 = {'axis': keyword_160218}
        # Getting the type of 'self' (line 412)
        self_160209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 32), 'self', False)
        # Obtaining the member 'average' of a type (line 412)
        average_160210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 32), self_160209, 'average')
        # Calling average(args, kwargs) (line 412)
        average_call_result_160220 = invoke(stypy.reporting.localization.Localization(__file__, 412, 32), average_160210, *[masked_array_call_result_160216], **kwargs_160219)
        
        float_160221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 80), 'float')
        # Processing the call keyword arguments (line 412)
        kwargs_160222 = {}
        # Getting the type of 'self' (line 412)
        self_160207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 412)
        assert_array_equal_160208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 8), self_160207, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 412)
        assert_array_equal_call_result_160223 = invoke(stypy.reporting.localization.Localization(__file__, 412, 8), assert_array_equal_160208, *[average_call_result_160220, float_160221], **kwargs_160222)
        
        
        # Call to assert_array_equal(...): (line 413)
        # Processing the call arguments (line 413)
        
        # Call to count(...): (line 413)
        # Processing the call arguments (line 413)
        
        # Call to average(...): (line 413)
        # Processing the call arguments (line 413)
        
        # Call to masked_array(...): (line 413)
        # Processing the call arguments (line 413)
        # Getting the type of 'x' (line 413)
        x_160232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 74), 'x', False)
        # Getting the type of 'm4' (line 413)
        m4_160233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 77), 'm4', False)
        # Processing the call keyword arguments (line 413)
        kwargs_160234 = {}
        # Getting the type of 'self' (line 413)
        self_160230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 56), 'self', False)
        # Obtaining the member 'masked_array' of a type (line 413)
        masked_array_160231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 56), self_160230, 'masked_array')
        # Calling masked_array(args, kwargs) (line 413)
        masked_array_call_result_160235 = invoke(stypy.reporting.localization.Localization(__file__, 413, 56), masked_array_160231, *[x_160232, m4_160233], **kwargs_160234)
        
        # Processing the call keyword arguments (line 413)
        int_160236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 87), 'int')
        keyword_160237 = int_160236
        kwargs_160238 = {'axis': keyword_160237}
        # Getting the type of 'self' (line 413)
        self_160228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 43), 'self', False)
        # Obtaining the member 'average' of a type (line 413)
        average_160229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 43), self_160228, 'average')
        # Calling average(args, kwargs) (line 413)
        average_call_result_160239 = invoke(stypy.reporting.localization.Localization(__file__, 413, 43), average_160229, *[masked_array_call_result_160235], **kwargs_160238)
        
        # Processing the call keyword arguments (line 413)
        kwargs_160240 = {}
        # Getting the type of 'self' (line 413)
        self_160226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 32), 'self', False)
        # Obtaining the member 'count' of a type (line 413)
        count_160227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 32), self_160226, 'count')
        # Calling count(args, kwargs) (line 413)
        count_call_result_160241 = invoke(stypy.reporting.localization.Localization(__file__, 413, 32), count_160227, *[average_call_result_160239], **kwargs_160240)
        
        int_160242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 92), 'int')
        # Processing the call keyword arguments (line 413)
        kwargs_160243 = {}
        # Getting the type of 'self' (line 413)
        self_160224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 413)
        assert_array_equal_160225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 8), self_160224, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 413)
        assert_array_equal_call_result_160244 = invoke(stypy.reporting.localization.Localization(__file__, 413, 8), assert_array_equal_160225, *[count_call_result_160241, int_160242], **kwargs_160243)
        
        
        # Assigning a Call to a Name (line 414):
        
        # Assigning a Call to a Name (line 414):
        
        # Call to masked_array(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'y' (line 414)
        y_160247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 30), 'y', False)
        # Getting the type of 'm3' (line 414)
        m3_160248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 33), 'm3', False)
        # Processing the call keyword arguments (line 414)
        kwargs_160249 = {}
        # Getting the type of 'self' (line 414)
        self_160245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'self', False)
        # Obtaining the member 'masked_array' of a type (line 414)
        masked_array_160246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 12), self_160245, 'masked_array')
        # Calling masked_array(args, kwargs) (line 414)
        masked_array_call_result_160250 = invoke(stypy.reporting.localization.Localization(__file__, 414, 12), masked_array_160246, *[y_160247, m3_160248], **kwargs_160249)
        
        # Assigning a type to the variable 'z' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'z', masked_array_call_result_160250)
        
        # Call to assert_array_equal(...): (line 415)
        # Processing the call arguments (line 415)
        
        # Call to average(...): (line 415)
        # Processing the call arguments (line 415)
        # Getting the type of 'z' (line 415)
        z_160255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 45), 'z', False)
        # Getting the type of 'None' (line 415)
        None_160256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 48), 'None', False)
        # Processing the call keyword arguments (line 415)
        kwargs_160257 = {}
        # Getting the type of 'self' (line 415)
        self_160253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 32), 'self', False)
        # Obtaining the member 'average' of a type (line 415)
        average_160254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 32), self_160253, 'average')
        # Calling average(args, kwargs) (line 415)
        average_call_result_160258 = invoke(stypy.reporting.localization.Localization(__file__, 415, 32), average_160254, *[z_160255, None_160256], **kwargs_160257)
        
        float_160259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 55), 'float')
        float_160260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 59), 'float')
        # Applying the binary operator 'div' (line 415)
        result_div_160261 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 55), 'div', float_160259, float_160260)
        
        # Processing the call keyword arguments (line 415)
        kwargs_160262 = {}
        # Getting the type of 'self' (line 415)
        self_160251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 415)
        assert_array_equal_160252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 8), self_160251, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 415)
        assert_array_equal_call_result_160263 = invoke(stypy.reporting.localization.Localization(__file__, 415, 8), assert_array_equal_160252, *[average_call_result_160258, result_div_160261], **kwargs_160262)
        
        
        # Call to assert_array_equal(...): (line 416)
        # Processing the call arguments (line 416)
        
        # Call to average(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'z' (line 416)
        z_160268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 45), 'z', False)
        # Processing the call keyword arguments (line 416)
        int_160269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 53), 'int')
        keyword_160270 = int_160269
        kwargs_160271 = {'axis': keyword_160270}
        # Getting the type of 'self' (line 416)
        self_160266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 32), 'self', False)
        # Obtaining the member 'average' of a type (line 416)
        average_160267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 32), self_160266, 'average')
        # Calling average(args, kwargs) (line 416)
        average_call_result_160272 = invoke(stypy.reporting.localization.Localization(__file__, 416, 32), average_160267, *[z_160268], **kwargs_160271)
        
        
        # Obtaining an instance of the builtin type 'list' (line 416)
        list_160273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 416)
        # Adding element type (line 416)
        float_160274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 57), list_160273, float_160274)
        # Adding element type (line 416)
        float_160275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 57), list_160273, float_160275)
        # Adding element type (line 416)
        float_160276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 66), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 57), list_160273, float_160276)
        # Adding element type (line 416)
        float_160277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 71), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 57), list_160273, float_160277)
        # Adding element type (line 416)
        float_160278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 76), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 57), list_160273, float_160278)
        # Adding element type (line 416)
        float_160279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 81), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 416, 57), list_160273, float_160279)
        
        # Processing the call keyword arguments (line 416)
        kwargs_160280 = {}
        # Getting the type of 'self' (line 416)
        self_160264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 416)
        assert_array_equal_160265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 8), self_160264, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 416)
        assert_array_equal_call_result_160281 = invoke(stypy.reporting.localization.Localization(__file__, 416, 8), assert_array_equal_160265, *[average_call_result_160272, list_160273], **kwargs_160280)
        
        
        # Call to assert_array_equal(...): (line 417)
        # Processing the call arguments (line 417)
        
        # Call to average(...): (line 417)
        # Processing the call arguments (line 417)
        # Getting the type of 'z' (line 417)
        z_160286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 45), 'z', False)
        # Processing the call keyword arguments (line 417)
        int_160287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 53), 'int')
        keyword_160288 = int_160287
        kwargs_160289 = {'axis': keyword_160288}
        # Getting the type of 'self' (line 417)
        self_160284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 32), 'self', False)
        # Obtaining the member 'average' of a type (line 417)
        average_160285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 32), self_160284, 'average')
        # Calling average(args, kwargs) (line 417)
        average_call_result_160290 = invoke(stypy.reporting.localization.Localization(__file__, 417, 32), average_160285, *[z_160286], **kwargs_160289)
        
        
        # Obtaining an instance of the builtin type 'list' (line 417)
        list_160291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 417)
        # Adding element type (line 417)
        float_160292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 57), list_160291, float_160292)
        # Adding element type (line 417)
        float_160293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 63), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 417, 57), list_160291, float_160293)
        
        # Processing the call keyword arguments (line 417)
        kwargs_160294 = {}
        # Getting the type of 'self' (line 417)
        self_160282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 417)
        assert_array_equal_160283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 8), self_160282, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 417)
        assert_array_equal_call_result_160295 = invoke(stypy.reporting.localization.Localization(__file__, 417, 8), assert_array_equal_160283, *[average_call_result_160290, list_160291], **kwargs_160294)
        
        
        # Call to assert_array_equal(...): (line 418)
        # Processing the call arguments (line 418)
        
        # Call to average(...): (line 418)
        # Processing the call arguments (line 418)
        # Getting the type of 'z' (line 418)
        z_160300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 45), 'z', False)
        # Processing the call keyword arguments (line 418)
        int_160301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 53), 'int')
        keyword_160302 = int_160301
        # Getting the type of 'w2' (line 418)
        w2_160303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 64), 'w2', False)
        keyword_160304 = w2_160303
        kwargs_160305 = {'weights': keyword_160304, 'axis': keyword_160302}
        # Getting the type of 'self' (line 418)
        self_160298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 32), 'self', False)
        # Obtaining the member 'average' of a type (line 418)
        average_160299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 32), self_160298, 'average')
        # Calling average(args, kwargs) (line 418)
        average_call_result_160306 = invoke(stypy.reporting.localization.Localization(__file__, 418, 32), average_160299, *[z_160300], **kwargs_160305)
        
        
        # Obtaining an instance of the builtin type 'list' (line 418)
        list_160307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 69), 'list')
        # Adding type elements to the builtin type 'list' instance (line 418)
        # Adding element type (line 418)
        float_160308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 70), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 69), list_160307, float_160308)
        # Adding element type (line 418)
        float_160309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 74), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 69), list_160307, float_160309)
        # Adding element type (line 418)
        float_160310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 78), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 69), list_160307, float_160310)
        # Adding element type (line 418)
        float_160311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 83), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 69), list_160307, float_160311)
        # Adding element type (line 418)
        float_160312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 88), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 69), list_160307, float_160312)
        # Adding element type (line 418)
        float_160313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 93), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 69), list_160307, float_160313)
        
        # Processing the call keyword arguments (line 418)
        kwargs_160314 = {}
        # Getting the type of 'self' (line 418)
        self_160296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'self', False)
        # Obtaining the member 'assert_array_equal' of a type (line 418)
        assert_array_equal_160297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), self_160296, 'assert_array_equal')
        # Calling assert_array_equal(args, kwargs) (line 418)
        assert_array_equal_call_result_160315 = invoke(stypy.reporting.localization.Localization(__file__, 418, 8), assert_array_equal_160297, *[average_call_result_160306, list_160307], **kwargs_160314)
        
        
        # ################# End of 'test_99(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_99' in the type store
        # Getting the type of 'stypy_return_type' (line 375)
        stypy_return_type_160316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_160316)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_99'
        return stypy_return_type_160316


    @norecursion
    def test_A(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_A'
        module_type_store = module_type_store.open_function_context('test_A', 420, 4, False)
        # Assigning a type to the variable 'self' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ModuleTester.test_A.__dict__.__setitem__('stypy_localization', localization)
        ModuleTester.test_A.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ModuleTester.test_A.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleTester.test_A.__dict__.__setitem__('stypy_function_name', 'ModuleTester.test_A')
        ModuleTester.test_A.__dict__.__setitem__('stypy_param_names_list', [])
        ModuleTester.test_A.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleTester.test_A.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleTester.test_A.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleTester.test_A.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleTester.test_A.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleTester.test_A.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ModuleTester.test_A', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_A', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_A(...)' code ##################

        
        # Assigning a Call to a Name (line 421):
        
        # Assigning a Call to a Name (line 421):
        
        # Call to arange(...): (line 421)
        # Processing the call arguments (line 421)
        int_160319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 24), 'int')
        # Processing the call keyword arguments (line 421)
        kwargs_160320 = {}
        # Getting the type of 'self' (line 421)
        self_160317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 12), 'self', False)
        # Obtaining the member 'arange' of a type (line 421)
        arange_160318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 12), self_160317, 'arange')
        # Calling arange(args, kwargs) (line 421)
        arange_call_result_160321 = invoke(stypy.reporting.localization.Localization(__file__, 421, 12), arange_160318, *[int_160319], **kwargs_160320)
        
        # Assigning a type to the variable 'x' (line 421)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'x', arange_call_result_160321)
        
        # Assigning a Attribute to a Subscript (line 422):
        
        # Assigning a Attribute to a Subscript (line 422):
        # Getting the type of 'self' (line 422)
        self_160322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 17), 'self')
        # Obtaining the member 'masked' of a type (line 422)
        masked_160323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 17), self_160322, 'masked')
        # Getting the type of 'x' (line 422)
        x_160324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'x')
        int_160325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 10), 'int')
        int_160326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 12), 'int')
        slice_160327 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 422, 8), int_160325, int_160326, None)
        # Storing an element on a container (line 422)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 422, 8), x_160324, (slice_160327, masked_160323))
        
        # Assigning a Call to a Name (line 423):
        
        # Assigning a Call to a Name (line 423):
        
        # Call to reshape(...): (line 423)
        # Processing the call arguments (line 423)
        int_160330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 22), 'int')
        int_160331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 25), 'int')
        int_160332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 28), 'int')
        # Processing the call keyword arguments (line 423)
        kwargs_160333 = {}
        # Getting the type of 'x' (line 423)
        x_160328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'x', False)
        # Obtaining the member 'reshape' of a type (line 423)
        reshape_160329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 12), x_160328, 'reshape')
        # Calling reshape(args, kwargs) (line 423)
        reshape_call_result_160334 = invoke(stypy.reporting.localization.Localization(__file__, 423, 12), reshape_160329, *[int_160330, int_160331, int_160332], **kwargs_160333)
        
        # Assigning a type to the variable 'x' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'x', reshape_call_result_160334)
        
        # ################# End of 'test_A(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_A' in the type store
        # Getting the type of 'stypy_return_type' (line 420)
        stypy_return_type_160335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_160335)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_A'
        return stypy_return_type_160335


# Assigning a type to the variable 'ModuleTester' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'ModuleTester', ModuleTester)

if (__name__ == '__main__'):
    
    # Assigning a Str to a Name (line 427):
    
    # Assigning a Str to a Name (line 427):
    str_160336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 18), 'str', 'from __main__ import ModuleTester \nimport numpy\ntester = ModuleTester(module)\n')
    # Assigning a type to the variable 'setup_base' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'setup_base', str_160336)
    
    # Assigning a BinOp to a Name (line 430):
    
    # Assigning a BinOp to a Name (line 430):
    str_160337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 16), 'str', 'import numpy.ma.core as module\n')
    # Getting the type of 'setup_base' (line 430)
    setup_base_160338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 53), 'setup_base')
    # Applying the binary operator '+' (line 430)
    result_add_160339 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 16), '+', str_160337, setup_base_160338)
    
    # Assigning a type to the variable 'setup_cur' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'setup_cur', result_add_160339)
    
    # Assigning a Tuple to a Tuple (line 431):
    
    # Assigning a Num to a Name (line 431):
    int_160340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 24), 'int')
    # Assigning a type to the variable 'tuple_assignment_158055' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'tuple_assignment_158055', int_160340)
    
    # Assigning a Num to a Name (line 431):
    int_160341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 28), 'int')
    # Assigning a type to the variable 'tuple_assignment_158056' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'tuple_assignment_158056', int_160341)
    
    # Assigning a Name to a Name (line 431):
    # Getting the type of 'tuple_assignment_158055' (line 431)
    tuple_assignment_158055_160342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'tuple_assignment_158055')
    # Assigning a type to the variable 'nrepeat' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 5), 'nrepeat', tuple_assignment_158055_160342)
    
    # Assigning a Name to a Name (line 431):
    # Getting the type of 'tuple_assignment_158056' (line 431)
    tuple_assignment_158056_160343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'tuple_assignment_158056')
    # Assigning a type to the variable 'nloop' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 14), 'nloop', tuple_assignment_158056_160343)
    
    int_160344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 7), 'int')
    # Testing the type of an if condition (line 433)
    if_condition_160345 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 4), int_160344)
    # Assigning a type to the variable 'if_condition_160345' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'if_condition_160345', if_condition_160345)
    # SSA begins for if statement (line 433)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to range(...): (line 434)
    # Processing the call arguments (line 434)
    int_160347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 23), 'int')
    int_160348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 26), 'int')
    # Processing the call keyword arguments (line 434)
    kwargs_160349 = {}
    # Getting the type of 'range' (line 434)
    range_160346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 17), 'range', False)
    # Calling range(args, kwargs) (line 434)
    range_call_result_160350 = invoke(stypy.reporting.localization.Localization(__file__, 434, 17), range_160346, *[int_160347, int_160348], **kwargs_160349)
    
    # Testing the type of a for loop iterable (line 434)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 434, 8), range_call_result_160350)
    # Getting the type of the for loop variable (line 434)
    for_loop_var_160351 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 434, 8), range_call_result_160350)
    # Assigning a type to the variable 'i' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'i', for_loop_var_160351)
    # SSA begins for a for statement (line 434)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 435):
    
    # Assigning a BinOp to a Name (line 435):
    str_160352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 19), 'str', 'tester.test_%i()')
    # Getting the type of 'i' (line 435)
    i_160353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 40), 'i')
    # Applying the binary operator '%' (line 435)
    result_mod_160354 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 19), '%', str_160352, i_160353)
    
    # Assigning a type to the variable 'func' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 12), 'func', result_mod_160354)
    
    # Assigning a Call to a Name (line 436):
    
    # Assigning a Call to a Name (line 436):
    
    # Call to repeat(...): (line 436)
    # Processing the call arguments (line 436)
    # Getting the type of 'nrepeat' (line 436)
    nrepeat_160362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 55), 'nrepeat', False)
    # Getting the type of 'nloop' (line 436)
    nloop_160363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 64), 'nloop', False)
    int_160364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 70), 'int')
    # Applying the binary operator '*' (line 436)
    result_mul_160365 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 64), '*', nloop_160363, int_160364)
    
    # Processing the call keyword arguments (line 436)
    kwargs_160366 = {}
    
    # Call to Timer(...): (line 436)
    # Processing the call arguments (line 436)
    # Getting the type of 'func' (line 436)
    func_160357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 31), 'func', False)
    # Getting the type of 'setup_cur' (line 436)
    setup_cur_160358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 37), 'setup_cur', False)
    # Processing the call keyword arguments (line 436)
    kwargs_160359 = {}
    # Getting the type of 'timeit' (line 436)
    timeit_160355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 18), 'timeit', False)
    # Obtaining the member 'Timer' of a type (line 436)
    Timer_160356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 18), timeit_160355, 'Timer')
    # Calling Timer(args, kwargs) (line 436)
    Timer_call_result_160360 = invoke(stypy.reporting.localization.Localization(__file__, 436, 18), Timer_160356, *[func_160357, setup_cur_160358], **kwargs_160359)
    
    # Obtaining the member 'repeat' of a type (line 436)
    repeat_160361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 18), Timer_call_result_160360, 'repeat')
    # Calling repeat(args, kwargs) (line 436)
    repeat_call_result_160367 = invoke(stypy.reporting.localization.Localization(__file__, 436, 18), repeat_160361, *[nrepeat_160362, result_mul_160365], **kwargs_160366)
    
    # Assigning a type to the variable 'cur' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'cur', repeat_call_result_160367)
    
    # Assigning a Call to a Name (line 437):
    
    # Assigning a Call to a Name (line 437):
    
    # Call to sort(...): (line 437)
    # Processing the call arguments (line 437)
    # Getting the type of 'cur' (line 437)
    cur_160370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 26), 'cur', False)
    # Processing the call keyword arguments (line 437)
    kwargs_160371 = {}
    # Getting the type of 'np' (line 437)
    np_160368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 18), 'np', False)
    # Obtaining the member 'sort' of a type (line 437)
    sort_160369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 18), np_160368, 'sort')
    # Calling sort(args, kwargs) (line 437)
    sort_call_result_160372 = invoke(stypy.reporting.localization.Localization(__file__, 437, 18), sort_160369, *[cur_160370], **kwargs_160371)
    
    # Assigning a type to the variable 'cur' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'cur', sort_call_result_160372)
    
    # Call to print(...): (line 438)
    # Processing the call arguments (line 438)
    str_160374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 18), 'str', '#%i')
    # Getting the type of 'i' (line 438)
    i_160375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 26), 'i', False)
    # Applying the binary operator '%' (line 438)
    result_mod_160376 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 18), '%', str_160374, i_160375)
    
    int_160377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 30), 'int')
    str_160378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 33), 'str', '.')
    # Applying the binary operator '*' (line 438)
    result_mul_160379 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 30), '*', int_160377, str_160378)
    
    # Applying the binary operator '+' (line 438)
    result_add_160380 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 18), '+', result_mod_160376, result_mul_160379)
    
    # Processing the call keyword arguments (line 438)
    kwargs_160381 = {}
    # Getting the type of 'print' (line 438)
    print_160373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'print', False)
    # Calling print(args, kwargs) (line 438)
    print_call_result_160382 = invoke(stypy.reporting.localization.Localization(__file__, 438, 12), print_160373, *[result_add_160380], **kwargs_160381)
    
    
    # Call to print(...): (line 439)
    # Processing the call arguments (line 439)
    
    # Call to eval(...): (line 439)
    # Processing the call arguments (line 439)
    str_160385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 23), 'str', 'ModuleTester.test_%i.__doc__')
    # Getting the type of 'i' (line 439)
    i_160386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 56), 'i', False)
    # Applying the binary operator '%' (line 439)
    result_mod_160387 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 23), '%', str_160385, i_160386)
    
    # Processing the call keyword arguments (line 439)
    kwargs_160388 = {}
    # Getting the type of 'eval' (line 439)
    eval_160384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 18), 'eval', False)
    # Calling eval(args, kwargs) (line 439)
    eval_call_result_160389 = invoke(stypy.reporting.localization.Localization(__file__, 439, 18), eval_160384, *[result_mod_160387], **kwargs_160388)
    
    # Processing the call keyword arguments (line 439)
    kwargs_160390 = {}
    # Getting the type of 'print' (line 439)
    print_160383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'print', False)
    # Calling print(args, kwargs) (line 439)
    print_call_result_160391 = invoke(stypy.reporting.localization.Localization(__file__, 439, 12), print_160383, *[eval_call_result_160389], **kwargs_160390)
    
    
    # Call to print(...): (line 440)
    # Processing the call arguments (line 440)
    str_160393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 18), 'str', 'core_current : %.3f - %.3f')
    
    # Obtaining an instance of the builtin type 'tuple' (line 440)
    tuple_160394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 440)
    # Adding element type (line 440)
    
    # Obtaining the type of the subscript
    int_160395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 54), 'int')
    # Getting the type of 'cur' (line 440)
    cur_160396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 50), 'cur', False)
    # Obtaining the member '__getitem__' of a type (line 440)
    getitem___160397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 50), cur_160396, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 440)
    subscript_call_result_160398 = invoke(stypy.reporting.localization.Localization(__file__, 440, 50), getitem___160397, int_160395)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 50), tuple_160394, subscript_call_result_160398)
    # Adding element type (line 440)
    
    # Obtaining the type of the subscript
    int_160399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 62), 'int')
    # Getting the type of 'cur' (line 440)
    cur_160400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 58), 'cur', False)
    # Obtaining the member '__getitem__' of a type (line 440)
    getitem___160401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 58), cur_160400, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 440)
    subscript_call_result_160402 = invoke(stypy.reporting.localization.Localization(__file__, 440, 58), getitem___160401, int_160399)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 440, 50), tuple_160394, subscript_call_result_160402)
    
    # Applying the binary operator '%' (line 440)
    result_mod_160403 = python_operator(stypy.reporting.localization.Localization(__file__, 440, 18), '%', str_160393, tuple_160394)
    
    # Processing the call keyword arguments (line 440)
    kwargs_160404 = {}
    # Getting the type of 'print' (line 440)
    print_160392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'print', False)
    # Calling print(args, kwargs) (line 440)
    print_call_result_160405 = invoke(stypy.reporting.localization.Localization(__file__, 440, 12), print_160392, *[result_mod_160403], **kwargs_160404)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 433)
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
