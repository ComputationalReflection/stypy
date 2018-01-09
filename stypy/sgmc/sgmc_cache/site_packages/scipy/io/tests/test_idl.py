
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from os import path
4: import warnings
5: 
6: DATA_PATH = path.join(path.dirname(__file__), 'data')
7: 
8: import numpy as np
9: from numpy.testing import (assert_equal, assert_array_equal,
10:     assert_)
11: from scipy._lib._numpy_compat import suppress_warnings
12: 
13: from scipy.io.idl import readsav
14: 
15: 
16: def object_array(*args):
17:     '''Constructs a numpy array of objects'''
18:     array = np.empty(len(args), dtype=object)
19:     for i in range(len(args)):
20:         array[i] = args[i]
21:     return array
22: 
23: 
24: def assert_identical(a, b):
25:     '''Assert whether value AND type are the same'''
26:     assert_equal(a, b)
27:     if type(b) is str:
28:         assert_equal(type(a), type(b))
29:     else:
30:         assert_equal(np.asarray(a).dtype.type, np.asarray(b).dtype.type)
31: 
32: 
33: def assert_array_identical(a, b):
34:     '''Assert whether values AND type are the same'''
35:     assert_array_equal(a, b)
36:     assert_equal(a.dtype.type, b.dtype.type)
37: 
38: 
39: # Define vectorized ID function for pointer arrays
40: vect_id = np.vectorize(id)
41: 
42: 
43: class TestIdict:
44: 
45:     def test_idict(self):
46:         custom_dict = {'a': np.int16(999)}
47:         original_id = id(custom_dict)
48:         s = readsav(path.join(DATA_PATH, 'scalar_byte.sav'), idict=custom_dict, verbose=False)
49:         assert_equal(original_id, id(s))
50:         assert_('a' in s)
51:         assert_identical(s['a'], np.int16(999))
52:         assert_identical(s['i8u'], np.uint8(234))
53: 
54: 
55: class TestScalars:
56:     # Test that scalar values are read in with the correct value and type
57: 
58:     def test_byte(self):
59:         s = readsav(path.join(DATA_PATH, 'scalar_byte.sav'), verbose=False)
60:         assert_identical(s.i8u, np.uint8(234))
61: 
62:     def test_int16(self):
63:         s = readsav(path.join(DATA_PATH, 'scalar_int16.sav'), verbose=False)
64:         assert_identical(s.i16s, np.int16(-23456))
65: 
66:     def test_int32(self):
67:         s = readsav(path.join(DATA_PATH, 'scalar_int32.sav'), verbose=False)
68:         assert_identical(s.i32s, np.int32(-1234567890))
69: 
70:     def test_float32(self):
71:         s = readsav(path.join(DATA_PATH, 'scalar_float32.sav'), verbose=False)
72:         assert_identical(s.f32, np.float32(-3.1234567e+37))
73: 
74:     def test_float64(self):
75:         s = readsav(path.join(DATA_PATH, 'scalar_float64.sav'), verbose=False)
76:         assert_identical(s.f64, np.float64(-1.1976931348623157e+307))
77: 
78:     def test_complex32(self):
79:         s = readsav(path.join(DATA_PATH, 'scalar_complex32.sav'), verbose=False)
80:         assert_identical(s.c32, np.complex64(3.124442e13-2.312442e31j))
81: 
82:     def test_bytes(self):
83:         s = readsav(path.join(DATA_PATH, 'scalar_string.sav'), verbose=False)
84:         assert_identical(s.s, np.bytes_("The quick brown fox jumps over the lazy python"))
85: 
86:     def test_structure(self):
87:         pass
88: 
89:     def test_complex64(self):
90:         s = readsav(path.join(DATA_PATH, 'scalar_complex64.sav'), verbose=False)
91:         assert_identical(s.c64, np.complex128(1.1987253647623157e+112-5.1987258887729157e+307j))
92: 
93:     def test_heap_pointer(self):
94:         pass
95: 
96:     def test_object_reference(self):
97:         pass
98: 
99:     def test_uint16(self):
100:         s = readsav(path.join(DATA_PATH, 'scalar_uint16.sav'), verbose=False)
101:         assert_identical(s.i16u, np.uint16(65511))
102: 
103:     def test_uint32(self):
104:         s = readsav(path.join(DATA_PATH, 'scalar_uint32.sav'), verbose=False)
105:         assert_identical(s.i32u, np.uint32(4294967233))
106: 
107:     def test_int64(self):
108:         s = readsav(path.join(DATA_PATH, 'scalar_int64.sav'), verbose=False)
109:         assert_identical(s.i64s, np.int64(-9223372036854774567))
110: 
111:     def test_uint64(self):
112:         s = readsav(path.join(DATA_PATH, 'scalar_uint64.sav'), verbose=False)
113:         assert_identical(s.i64u, np.uint64(18446744073709529285))
114: 
115: 
116: class TestCompressed(TestScalars):
117:     # Test that compressed .sav files can be read in
118: 
119:     def test_compressed(self):
120:         s = readsav(path.join(DATA_PATH, 'various_compressed.sav'), verbose=False)
121: 
122:         assert_identical(s.i8u, np.uint8(234))
123:         assert_identical(s.f32, np.float32(-3.1234567e+37))
124:         assert_identical(s.c64, np.complex128(1.1987253647623157e+112-5.1987258887729157e+307j))
125:         assert_equal(s.array5d.shape, (4, 3, 4, 6, 5))
126:         assert_identical(s.arrays.a[0], np.array([1, 2, 3], dtype=np.int16))
127:         assert_identical(s.arrays.b[0], np.array([4., 5., 6., 7.], dtype=np.float32))
128:         assert_identical(s.arrays.c[0], np.array([np.complex64(1+2j), np.complex64(7+8j)]))
129:         assert_identical(s.arrays.d[0], np.array([b"cheese", b"bacon", b"spam"], dtype=object))
130: 
131: 
132: class TestArrayDimensions:
133:     # Test that multi-dimensional arrays are read in with the correct dimensions
134: 
135:     def test_1d(self):
136:         s = readsav(path.join(DATA_PATH, 'array_float32_1d.sav'), verbose=False)
137:         assert_equal(s.array1d.shape, (123, ))
138: 
139:     def test_2d(self):
140:         s = readsav(path.join(DATA_PATH, 'array_float32_2d.sav'), verbose=False)
141:         assert_equal(s.array2d.shape, (22, 12))
142: 
143:     def test_3d(self):
144:         s = readsav(path.join(DATA_PATH, 'array_float32_3d.sav'), verbose=False)
145:         assert_equal(s.array3d.shape, (11, 22, 12))
146: 
147:     def test_4d(self):
148:         s = readsav(path.join(DATA_PATH, 'array_float32_4d.sav'), verbose=False)
149:         assert_equal(s.array4d.shape, (4, 5, 8, 7))
150: 
151:     def test_5d(self):
152:         s = readsav(path.join(DATA_PATH, 'array_float32_5d.sav'), verbose=False)
153:         assert_equal(s.array5d.shape, (4, 3, 4, 6, 5))
154: 
155:     def test_6d(self):
156:         s = readsav(path.join(DATA_PATH, 'array_float32_6d.sav'), verbose=False)
157:         assert_equal(s.array6d.shape, (3, 6, 4, 5, 3, 4))
158: 
159:     def test_7d(self):
160:         s = readsav(path.join(DATA_PATH, 'array_float32_7d.sav'), verbose=False)
161:         assert_equal(s.array7d.shape, (2, 1, 2, 3, 4, 3, 2))
162: 
163:     def test_8d(self):
164:         s = readsav(path.join(DATA_PATH, 'array_float32_8d.sav'), verbose=False)
165:         assert_equal(s.array8d.shape, (4, 3, 2, 1, 2, 3, 5, 4))
166: 
167: 
168: class TestStructures:
169: 
170:     def test_scalars(self):
171:         s = readsav(path.join(DATA_PATH, 'struct_scalars.sav'), verbose=False)
172:         assert_identical(s.scalars.a, np.array(np.int16(1)))
173:         assert_identical(s.scalars.b, np.array(np.int32(2)))
174:         assert_identical(s.scalars.c, np.array(np.float32(3.)))
175:         assert_identical(s.scalars.d, np.array(np.float64(4.)))
176:         assert_identical(s.scalars.e, np.array([b"spam"], dtype=object))
177:         assert_identical(s.scalars.f, np.array(np.complex64(-1.+3j)))
178: 
179:     def test_scalars_replicated(self):
180:         s = readsav(path.join(DATA_PATH, 'struct_scalars_replicated.sav'), verbose=False)
181:         assert_identical(s.scalars_rep.a, np.repeat(np.int16(1), 5))
182:         assert_identical(s.scalars_rep.b, np.repeat(np.int32(2), 5))
183:         assert_identical(s.scalars_rep.c, np.repeat(np.float32(3.), 5))
184:         assert_identical(s.scalars_rep.d, np.repeat(np.float64(4.), 5))
185:         assert_identical(s.scalars_rep.e, np.repeat(b"spam", 5).astype(object))
186:         assert_identical(s.scalars_rep.f, np.repeat(np.complex64(-1.+3j), 5))
187: 
188:     def test_scalars_replicated_3d(self):
189:         s = readsav(path.join(DATA_PATH, 'struct_scalars_replicated_3d.sav'), verbose=False)
190:         assert_identical(s.scalars_rep.a, np.repeat(np.int16(1), 24).reshape(4, 3, 2))
191:         assert_identical(s.scalars_rep.b, np.repeat(np.int32(2), 24).reshape(4, 3, 2))
192:         assert_identical(s.scalars_rep.c, np.repeat(np.float32(3.), 24).reshape(4, 3, 2))
193:         assert_identical(s.scalars_rep.d, np.repeat(np.float64(4.), 24).reshape(4, 3, 2))
194:         assert_identical(s.scalars_rep.e, np.repeat(b"spam", 24).reshape(4, 3, 2).astype(object))
195:         assert_identical(s.scalars_rep.f, np.repeat(np.complex64(-1.+3j), 24).reshape(4, 3, 2))
196: 
197:     def test_arrays(self):
198:         s = readsav(path.join(DATA_PATH, 'struct_arrays.sav'), verbose=False)
199:         assert_array_identical(s.arrays.a[0], np.array([1, 2, 3], dtype=np.int16))
200:         assert_array_identical(s.arrays.b[0], np.array([4., 5., 6., 7.], dtype=np.float32))
201:         assert_array_identical(s.arrays.c[0], np.array([np.complex64(1+2j), np.complex64(7+8j)]))
202:         assert_array_identical(s.arrays.d[0], np.array([b"cheese", b"bacon", b"spam"], dtype=object))
203: 
204:     def test_arrays_replicated(self):
205:         s = readsav(path.join(DATA_PATH, 'struct_arrays_replicated.sav'), verbose=False)
206: 
207:         # Check column types
208:         assert_(s.arrays_rep.a.dtype.type is np.object_)
209:         assert_(s.arrays_rep.b.dtype.type is np.object_)
210:         assert_(s.arrays_rep.c.dtype.type is np.object_)
211:         assert_(s.arrays_rep.d.dtype.type is np.object_)
212: 
213:         # Check column shapes
214:         assert_equal(s.arrays_rep.a.shape, (5, ))
215:         assert_equal(s.arrays_rep.b.shape, (5, ))
216:         assert_equal(s.arrays_rep.c.shape, (5, ))
217:         assert_equal(s.arrays_rep.d.shape, (5, ))
218: 
219:         # Check values
220:         for i in range(5):
221:             assert_array_identical(s.arrays_rep.a[i],
222:                                    np.array([1, 2, 3], dtype=np.int16))
223:             assert_array_identical(s.arrays_rep.b[i],
224:                                    np.array([4., 5., 6., 7.], dtype=np.float32))
225:             assert_array_identical(s.arrays_rep.c[i],
226:                                    np.array([np.complex64(1+2j),
227:                                              np.complex64(7+8j)]))
228:             assert_array_identical(s.arrays_rep.d[i],
229:                                    np.array([b"cheese", b"bacon", b"spam"],
230:                                             dtype=object))
231: 
232:     def test_arrays_replicated_3d(self):
233:         s = readsav(path.join(DATA_PATH, 'struct_arrays_replicated_3d.sav'), verbose=False)
234: 
235:         # Check column types
236:         assert_(s.arrays_rep.a.dtype.type is np.object_)
237:         assert_(s.arrays_rep.b.dtype.type is np.object_)
238:         assert_(s.arrays_rep.c.dtype.type is np.object_)
239:         assert_(s.arrays_rep.d.dtype.type is np.object_)
240: 
241:         # Check column shapes
242:         assert_equal(s.arrays_rep.a.shape, (4, 3, 2))
243:         assert_equal(s.arrays_rep.b.shape, (4, 3, 2))
244:         assert_equal(s.arrays_rep.c.shape, (4, 3, 2))
245:         assert_equal(s.arrays_rep.d.shape, (4, 3, 2))
246: 
247:         # Check values
248:         for i in range(4):
249:             for j in range(3):
250:                 for k in range(2):
251:                     assert_array_identical(s.arrays_rep.a[i, j, k],
252:                                            np.array([1, 2, 3], dtype=np.int16))
253:                     assert_array_identical(s.arrays_rep.b[i, j, k],
254:                                            np.array([4., 5., 6., 7.],
255:                                                     dtype=np.float32))
256:                     assert_array_identical(s.arrays_rep.c[i, j, k],
257:                                            np.array([np.complex64(1+2j),
258:                                                      np.complex64(7+8j)]))
259:                     assert_array_identical(s.arrays_rep.d[i, j, k],
260:                                            np.array([b"cheese", b"bacon", b"spam"],
261:                                                     dtype=object))
262: 
263:     def test_inheritance(self):
264:         s = readsav(path.join(DATA_PATH, 'struct_inherit.sav'), verbose=False)
265:         assert_identical(s.fc.x, np.array([0], dtype=np.int16))
266:         assert_identical(s.fc.y, np.array([0], dtype=np.int16))
267:         assert_identical(s.fc.r, np.array([0], dtype=np.int16))
268:         assert_identical(s.fc.c, np.array([4], dtype=np.int16))
269: 
270:     def test_arrays_corrupt_idl80(self):
271:         # test byte arrays with missing nbyte information from IDL 8.0 .sav file
272:         with suppress_warnings() as sup:
273:             sup.filter(UserWarning, "Not able to verify number of bytes from header")
274:             s = readsav(path.join(DATA_PATH,'struct_arrays_byte_idl80.sav'),
275:                         verbose=False)
276: 
277:         assert_identical(s.y.x[0], np.array([55,66], dtype=np.uint8))
278: 
279: 
280: class TestPointers:
281:     # Check that pointers in .sav files produce references to the same object in Python
282: 
283:     def test_pointers(self):
284:         s = readsav(path.join(DATA_PATH, 'scalar_heap_pointer.sav'), verbose=False)
285:         assert_identical(s.c64_pointer1, np.complex128(1.1987253647623157e+112-5.1987258887729157e+307j))
286:         assert_identical(s.c64_pointer2, np.complex128(1.1987253647623157e+112-5.1987258887729157e+307j))
287:         assert_(s.c64_pointer1 is s.c64_pointer2)
288: 
289: 
290: class TestPointerArray:
291:     # Test that pointers in arrays are correctly read in
292: 
293:     def test_1d(self):
294:         s = readsav(path.join(DATA_PATH, 'array_float32_pointer_1d.sav'), verbose=False)
295:         assert_equal(s.array1d.shape, (123, ))
296:         assert_(np.all(s.array1d == np.float32(4.)))
297:         assert_(np.all(vect_id(s.array1d) == id(s.array1d[0])))
298: 
299:     def test_2d(self):
300:         s = readsav(path.join(DATA_PATH, 'array_float32_pointer_2d.sav'), verbose=False)
301:         assert_equal(s.array2d.shape, (22, 12))
302:         assert_(np.all(s.array2d == np.float32(4.)))
303:         assert_(np.all(vect_id(s.array2d) == id(s.array2d[0,0])))
304: 
305:     def test_3d(self):
306:         s = readsav(path.join(DATA_PATH, 'array_float32_pointer_3d.sav'), verbose=False)
307:         assert_equal(s.array3d.shape, (11, 22, 12))
308:         assert_(np.all(s.array3d == np.float32(4.)))
309:         assert_(np.all(vect_id(s.array3d) == id(s.array3d[0,0,0])))
310: 
311:     def test_4d(self):
312:         s = readsav(path.join(DATA_PATH, 'array_float32_pointer_4d.sav'), verbose=False)
313:         assert_equal(s.array4d.shape, (4, 5, 8, 7))
314:         assert_(np.all(s.array4d == np.float32(4.)))
315:         assert_(np.all(vect_id(s.array4d) == id(s.array4d[0,0,0,0])))
316: 
317:     def test_5d(self):
318:         s = readsav(path.join(DATA_PATH, 'array_float32_pointer_5d.sav'), verbose=False)
319:         assert_equal(s.array5d.shape, (4, 3, 4, 6, 5))
320:         assert_(np.all(s.array5d == np.float32(4.)))
321:         assert_(np.all(vect_id(s.array5d) == id(s.array5d[0,0,0,0,0])))
322: 
323:     def test_6d(self):
324:         s = readsav(path.join(DATA_PATH, 'array_float32_pointer_6d.sav'), verbose=False)
325:         assert_equal(s.array6d.shape, (3, 6, 4, 5, 3, 4))
326:         assert_(np.all(s.array6d == np.float32(4.)))
327:         assert_(np.all(vect_id(s.array6d) == id(s.array6d[0,0,0,0,0,0])))
328: 
329:     def test_7d(self):
330:         s = readsav(path.join(DATA_PATH, 'array_float32_pointer_7d.sav'), verbose=False)
331:         assert_equal(s.array7d.shape, (2, 1, 2, 3, 4, 3, 2))
332:         assert_(np.all(s.array7d == np.float32(4.)))
333:         assert_(np.all(vect_id(s.array7d) == id(s.array7d[0,0,0,0,0,0,0])))
334: 
335:     def test_8d(self):
336:         s = readsav(path.join(DATA_PATH, 'array_float32_pointer_8d.sav'), verbose=False)
337:         assert_equal(s.array8d.shape, (4, 3, 2, 1, 2, 3, 5, 4))
338:         assert_(np.all(s.array8d == np.float32(4.)))
339:         assert_(np.all(vect_id(s.array8d) == id(s.array8d[0,0,0,0,0,0,0,0])))
340: 
341: 
342: class TestPointerStructures:
343:     # Test that structures are correctly read in
344: 
345:     def test_scalars(self):
346:         s = readsav(path.join(DATA_PATH, 'struct_pointers.sav'), verbose=False)
347:         assert_identical(s.pointers.g, np.array(np.float32(4.), dtype=np.object_))
348:         assert_identical(s.pointers.h, np.array(np.float32(4.), dtype=np.object_))
349:         assert_(id(s.pointers.g[0]) == id(s.pointers.h[0]))
350: 
351:     def test_pointers_replicated(self):
352:         s = readsav(path.join(DATA_PATH, 'struct_pointers_replicated.sav'), verbose=False)
353:         assert_identical(s.pointers_rep.g, np.repeat(np.float32(4.), 5).astype(np.object_))
354:         assert_identical(s.pointers_rep.h, np.repeat(np.float32(4.), 5).astype(np.object_))
355:         assert_(np.all(vect_id(s.pointers_rep.g) == vect_id(s.pointers_rep.h)))
356: 
357:     def test_pointers_replicated_3d(self):
358:         s = readsav(path.join(DATA_PATH, 'struct_pointers_replicated_3d.sav'), verbose=False)
359:         s_expect = np.repeat(np.float32(4.), 24).reshape(4, 3, 2).astype(np.object_)
360:         assert_identical(s.pointers_rep.g, s_expect)
361:         assert_identical(s.pointers_rep.h, s_expect)
362:         assert_(np.all(vect_id(s.pointers_rep.g) == vect_id(s.pointers_rep.h)))
363: 
364:     def test_arrays(self):
365:         s = readsav(path.join(DATA_PATH, 'struct_pointer_arrays.sav'), verbose=False)
366:         assert_array_identical(s.arrays.g[0], np.repeat(np.float32(4.), 2).astype(np.object_))
367:         assert_array_identical(s.arrays.h[0], np.repeat(np.float32(4.), 3).astype(np.object_))
368:         assert_(np.all(vect_id(s.arrays.g[0]) == id(s.arrays.g[0][0])))
369:         assert_(np.all(vect_id(s.arrays.h[0]) == id(s.arrays.h[0][0])))
370:         assert_(id(s.arrays.g[0][0]) == id(s.arrays.h[0][0]))
371: 
372:     def test_arrays_replicated(self):
373:         s = readsav(path.join(DATA_PATH, 'struct_pointer_arrays_replicated.sav'), verbose=False)
374: 
375:         # Check column types
376:         assert_(s.arrays_rep.g.dtype.type is np.object_)
377:         assert_(s.arrays_rep.h.dtype.type is np.object_)
378: 
379:         # Check column shapes
380:         assert_equal(s.arrays_rep.g.shape, (5, ))
381:         assert_equal(s.arrays_rep.h.shape, (5, ))
382: 
383:         # Check values
384:         for i in range(5):
385:             assert_array_identical(s.arrays_rep.g[i], np.repeat(np.float32(4.), 2).astype(np.object_))
386:             assert_array_identical(s.arrays_rep.h[i], np.repeat(np.float32(4.), 3).astype(np.object_))
387:             assert_(np.all(vect_id(s.arrays_rep.g[i]) == id(s.arrays_rep.g[0][0])))
388:             assert_(np.all(vect_id(s.arrays_rep.h[i]) == id(s.arrays_rep.h[0][0])))
389: 
390:     def test_arrays_replicated_3d(self):
391:         pth = path.join(DATA_PATH, 'struct_pointer_arrays_replicated_3d.sav')
392:         s = readsav(pth, verbose=False)
393: 
394:         # Check column types
395:         assert_(s.arrays_rep.g.dtype.type is np.object_)
396:         assert_(s.arrays_rep.h.dtype.type is np.object_)
397: 
398:         # Check column shapes
399:         assert_equal(s.arrays_rep.g.shape, (4, 3, 2))
400:         assert_equal(s.arrays_rep.h.shape, (4, 3, 2))
401: 
402:         # Check values
403:         for i in range(4):
404:             for j in range(3):
405:                 for k in range(2):
406:                     assert_array_identical(s.arrays_rep.g[i, j, k],
407:                             np.repeat(np.float32(4.), 2).astype(np.object_))
408:                     assert_array_identical(s.arrays_rep.h[i, j, k],
409:                             np.repeat(np.float32(4.), 3).astype(np.object_))
410:                     assert_(np.all(vect_id(s.arrays_rep.g[i, j, k]) == id(s.arrays_rep.g[0, 0, 0][0])))
411:                     assert_(np.all(vect_id(s.arrays_rep.h[i, j, k]) == id(s.arrays_rep.h[0, 0, 0][0])))
412: class TestTags:
413:     '''Test that sav files with description tag read at all'''
414: 
415:     def test_description(self):
416:         s = readsav(path.join(DATA_PATH, 'scalar_byte_descr.sav'), verbose=False)
417:         assert_identical(s.i8u, np.uint8(234))
418: 
419: 
420: def test_null_pointer():
421:     # Regression test for null pointers.
422:     s = readsav(path.join(DATA_PATH, 'null_pointer.sav'), verbose=False)
423:     assert_identical(s.point, None)
424:     assert_identical(s.check, np.int16(5))
425: 
426: 
427: def test_invalid_pointer():
428:     # Regression test for invalid pointers (gh-4613).
429: 
430:     # In some files in the wild, pointers can sometimes refer to a heap
431:     # variable that does not exist. In that case, we now gracefully fail for
432:     # that variable and replace the variable with None and emit a warning.
433:     # Since it's difficult to artificially produce such files, the file used
434:     # here has been edited to force the pointer reference to be invalid.
435:     with warnings.catch_warnings(record=True) as w:
436:         warnings.simplefilter("always")
437:         s = readsav(path.join(DATA_PATH, 'invalid_pointer.sav'), verbose=False)
438:     assert_(len(w) == 1)
439:     assert_(str(w[0].message) == ("Variable referenced by pointer not found in "
440:                                   "heap: variable will be set to None"))
441:     assert_identical(s['a'], np.array([None, None]))
442: 
443: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from os import path' statement (line 3)
try:
    from os import path

except:
    path = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', None, module_type_store, ['path'], [path])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import warnings' statement (line 4)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'warnings', warnings, module_type_store)


# Assigning a Call to a Name (line 6):

# Call to join(...): (line 6)
# Processing the call arguments (line 6)

# Call to dirname(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of '__file__' (line 6)
file___1607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 35), '__file__', False)
# Processing the call keyword arguments (line 6)
kwargs_1608 = {}
# Getting the type of 'path' (line 6)
path_1605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 22), 'path', False)
# Obtaining the member 'dirname' of a type (line 6)
dirname_1606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 22), path_1605, 'dirname')
# Calling dirname(args, kwargs) (line 6)
dirname_call_result_1609 = invoke(stypy.reporting.localization.Localization(__file__, 6, 22), dirname_1606, *[file___1607], **kwargs_1608)

str_1610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 46), 'str', 'data')
# Processing the call keyword arguments (line 6)
kwargs_1611 = {}
# Getting the type of 'path' (line 6)
path_1603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 12), 'path', False)
# Obtaining the member 'join' of a type (line 6)
join_1604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 12), path_1603, 'join')
# Calling join(args, kwargs) (line 6)
join_call_result_1612 = invoke(stypy.reporting.localization.Localization(__file__, 6, 12), join_1604, *[dirname_call_result_1609, str_1610], **kwargs_1611)

# Assigning a type to the variable 'DATA_PATH' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'DATA_PATH', join_call_result_1612)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import numpy' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_1613 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy')

if (type(import_1613) is not StypyTypeError):

    if (import_1613 != 'pyd_module'):
        __import__(import_1613)
        sys_modules_1614 = sys.modules[import_1613]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', sys_modules_1614.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy', import_1613)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.testing import assert_equal, assert_array_equal, assert_' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_1615 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing')

if (type(import_1615) is not StypyTypeError):

    if (import_1615 != 'pyd_module'):
        __import__(import_1615)
        sys_modules_1616 = sys.modules[import_1615]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', sys_modules_1616.module_type_store, module_type_store, ['assert_equal', 'assert_array_equal', 'assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_1616, sys_modules_1616.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_array_equal, assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_array_equal', 'assert_'], [assert_equal, assert_array_equal, assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', import_1615)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_1617 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat')

if (type(import_1617) is not StypyTypeError):

    if (import_1617 != 'pyd_module'):
        __import__(import_1617)
        sys_modules_1618 = sys.modules[import_1617]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', sys_modules_1618.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_1618, sys_modules_1618.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib._numpy_compat', import_1617)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy.io.idl import readsav' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/tests/')
import_1619 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io.idl')

if (type(import_1619) is not StypyTypeError):

    if (import_1619 != 'pyd_module'):
        __import__(import_1619)
        sys_modules_1620 = sys.modules[import_1619]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io.idl', sys_modules_1620.module_type_store, module_type_store, ['readsav'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_1620, sys_modules_1620.module_type_store, module_type_store)
    else:
        from scipy.io.idl import readsav

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io.idl', None, module_type_store, ['readsav'], [readsav])

else:
    # Assigning a type to the variable 'scipy.io.idl' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy.io.idl', import_1619)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/tests/')


@norecursion
def object_array(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'object_array'
    module_type_store = module_type_store.open_function_context('object_array', 16, 0, False)
    
    # Passed parameters checking function
    object_array.stypy_localization = localization
    object_array.stypy_type_of_self = None
    object_array.stypy_type_store = module_type_store
    object_array.stypy_function_name = 'object_array'
    object_array.stypy_param_names_list = []
    object_array.stypy_varargs_param_name = 'args'
    object_array.stypy_kwargs_param_name = None
    object_array.stypy_call_defaults = defaults
    object_array.stypy_call_varargs = varargs
    object_array.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'object_array', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'object_array', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'object_array(...)' code ##################

    str_1621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'str', 'Constructs a numpy array of objects')
    
    # Assigning a Call to a Name (line 18):
    
    # Call to empty(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Call to len(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'args' (line 18)
    args_1625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'args', False)
    # Processing the call keyword arguments (line 18)
    kwargs_1626 = {}
    # Getting the type of 'len' (line 18)
    len_1624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'len', False)
    # Calling len(args, kwargs) (line 18)
    len_call_result_1627 = invoke(stypy.reporting.localization.Localization(__file__, 18, 21), len_1624, *[args_1625], **kwargs_1626)
    
    # Processing the call keyword arguments (line 18)
    # Getting the type of 'object' (line 18)
    object_1628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 38), 'object', False)
    keyword_1629 = object_1628
    kwargs_1630 = {'dtype': keyword_1629}
    # Getting the type of 'np' (line 18)
    np_1622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'np', False)
    # Obtaining the member 'empty' of a type (line 18)
    empty_1623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 12), np_1622, 'empty')
    # Calling empty(args, kwargs) (line 18)
    empty_call_result_1631 = invoke(stypy.reporting.localization.Localization(__file__, 18, 12), empty_1623, *[len_call_result_1627], **kwargs_1630)
    
    # Assigning a type to the variable 'array' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'array', empty_call_result_1631)
    
    
    # Call to range(...): (line 19)
    # Processing the call arguments (line 19)
    
    # Call to len(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'args' (line 19)
    args_1634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'args', False)
    # Processing the call keyword arguments (line 19)
    kwargs_1635 = {}
    # Getting the type of 'len' (line 19)
    len_1633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'len', False)
    # Calling len(args, kwargs) (line 19)
    len_call_result_1636 = invoke(stypy.reporting.localization.Localization(__file__, 19, 19), len_1633, *[args_1634], **kwargs_1635)
    
    # Processing the call keyword arguments (line 19)
    kwargs_1637 = {}
    # Getting the type of 'range' (line 19)
    range_1632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), 'range', False)
    # Calling range(args, kwargs) (line 19)
    range_call_result_1638 = invoke(stypy.reporting.localization.Localization(__file__, 19, 13), range_1632, *[len_call_result_1636], **kwargs_1637)
    
    # Testing the type of a for loop iterable (line 19)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 19, 4), range_call_result_1638)
    # Getting the type of the for loop variable (line 19)
    for_loop_var_1639 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 19, 4), range_call_result_1638)
    # Assigning a type to the variable 'i' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'i', for_loop_var_1639)
    # SSA begins for a for statement (line 19)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Subscript (line 20):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 20)
    i_1640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 24), 'i')
    # Getting the type of 'args' (line 20)
    args_1641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 19), 'args')
    # Obtaining the member '__getitem__' of a type (line 20)
    getitem___1642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 19), args_1641, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 20)
    subscript_call_result_1643 = invoke(stypy.reporting.localization.Localization(__file__, 20, 19), getitem___1642, i_1640)
    
    # Getting the type of 'array' (line 20)
    array_1644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'array')
    # Getting the type of 'i' (line 20)
    i_1645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 14), 'i')
    # Storing an element on a container (line 20)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 8), array_1644, (i_1645, subscript_call_result_1643))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'array' (line 21)
    array_1646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'array')
    # Assigning a type to the variable 'stypy_return_type' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type', array_1646)
    
    # ################# End of 'object_array(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'object_array' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_1647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1647)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'object_array'
    return stypy_return_type_1647

# Assigning a type to the variable 'object_array' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'object_array', object_array)

@norecursion
def assert_identical(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'assert_identical'
    module_type_store = module_type_store.open_function_context('assert_identical', 24, 0, False)
    
    # Passed parameters checking function
    assert_identical.stypy_localization = localization
    assert_identical.stypy_type_of_self = None
    assert_identical.stypy_type_store = module_type_store
    assert_identical.stypy_function_name = 'assert_identical'
    assert_identical.stypy_param_names_list = ['a', 'b']
    assert_identical.stypy_varargs_param_name = None
    assert_identical.stypy_kwargs_param_name = None
    assert_identical.stypy_call_defaults = defaults
    assert_identical.stypy_call_varargs = varargs
    assert_identical.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_identical', ['a', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_identical', localization, ['a', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_identical(...)' code ##################

    str_1648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'str', 'Assert whether value AND type are the same')
    
    # Call to assert_equal(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'a' (line 26)
    a_1650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 17), 'a', False)
    # Getting the type of 'b' (line 26)
    b_1651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'b', False)
    # Processing the call keyword arguments (line 26)
    kwargs_1652 = {}
    # Getting the type of 'assert_equal' (line 26)
    assert_equal_1649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 26)
    assert_equal_call_result_1653 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), assert_equal_1649, *[a_1650, b_1651], **kwargs_1652)
    
    
    # Type idiom detected: calculating its left and rigth part (line 27)
    # Getting the type of 'b' (line 27)
    b_1654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'b')
    # Getting the type of 'str' (line 27)
    str_1655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'str')
    
    (may_be_1656, more_types_in_union_1657) = may_be_type(b_1654, str_1655)

    if may_be_1656:

        if more_types_in_union_1657:
            # Runtime conditional SSA (line 27)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'b' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'b', str_1655())
        
        # Call to assert_equal(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Call to type(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'a' (line 28)
        a_1660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'a', False)
        # Processing the call keyword arguments (line 28)
        kwargs_1661 = {}
        # Getting the type of 'type' (line 28)
        type_1659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 21), 'type', False)
        # Calling type(args, kwargs) (line 28)
        type_call_result_1662 = invoke(stypy.reporting.localization.Localization(__file__, 28, 21), type_1659, *[a_1660], **kwargs_1661)
        
        
        # Call to type(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'b' (line 28)
        b_1664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 35), 'b', False)
        # Processing the call keyword arguments (line 28)
        kwargs_1665 = {}
        # Getting the type of 'type' (line 28)
        type_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 30), 'type', False)
        # Calling type(args, kwargs) (line 28)
        type_call_result_1666 = invoke(stypy.reporting.localization.Localization(__file__, 28, 30), type_1663, *[b_1664], **kwargs_1665)
        
        # Processing the call keyword arguments (line 28)
        kwargs_1667 = {}
        # Getting the type of 'assert_equal' (line 28)
        assert_equal_1658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 28)
        assert_equal_call_result_1668 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), assert_equal_1658, *[type_call_result_1662, type_call_result_1666], **kwargs_1667)
        

        if more_types_in_union_1657:
            # Runtime conditional SSA for else branch (line 27)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_1656) or more_types_in_union_1657):
        # Getting the type of 'b' (line 27)
        b_1669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'b')
        # Assigning a type to the variable 'b' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'b', remove_type_from_union(b_1669, str_1655))
        
        # Call to assert_equal(...): (line 30)
        # Processing the call arguments (line 30)
        
        # Call to asarray(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'a' (line 30)
        a_1673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 32), 'a', False)
        # Processing the call keyword arguments (line 30)
        kwargs_1674 = {}
        # Getting the type of 'np' (line 30)
        np_1671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'np', False)
        # Obtaining the member 'asarray' of a type (line 30)
        asarray_1672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 21), np_1671, 'asarray')
        # Calling asarray(args, kwargs) (line 30)
        asarray_call_result_1675 = invoke(stypy.reporting.localization.Localization(__file__, 30, 21), asarray_1672, *[a_1673], **kwargs_1674)
        
        # Obtaining the member 'dtype' of a type (line 30)
        dtype_1676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 21), asarray_call_result_1675, 'dtype')
        # Obtaining the member 'type' of a type (line 30)
        type_1677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 21), dtype_1676, 'type')
        
        # Call to asarray(...): (line 30)
        # Processing the call arguments (line 30)
        # Getting the type of 'b' (line 30)
        b_1680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 58), 'b', False)
        # Processing the call keyword arguments (line 30)
        kwargs_1681 = {}
        # Getting the type of 'np' (line 30)
        np_1678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 47), 'np', False)
        # Obtaining the member 'asarray' of a type (line 30)
        asarray_1679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 47), np_1678, 'asarray')
        # Calling asarray(args, kwargs) (line 30)
        asarray_call_result_1682 = invoke(stypy.reporting.localization.Localization(__file__, 30, 47), asarray_1679, *[b_1680], **kwargs_1681)
        
        # Obtaining the member 'dtype' of a type (line 30)
        dtype_1683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 47), asarray_call_result_1682, 'dtype')
        # Obtaining the member 'type' of a type (line 30)
        type_1684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 47), dtype_1683, 'type')
        # Processing the call keyword arguments (line 30)
        kwargs_1685 = {}
        # Getting the type of 'assert_equal' (line 30)
        assert_equal_1670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 30)
        assert_equal_call_result_1686 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), assert_equal_1670, *[type_1677, type_1684], **kwargs_1685)
        

        if (may_be_1656 and more_types_in_union_1657):
            # SSA join for if statement (line 27)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'assert_identical(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_identical' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_1687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1687)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_identical'
    return stypy_return_type_1687

# Assigning a type to the variable 'assert_identical' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'assert_identical', assert_identical)

@norecursion
def assert_array_identical(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'assert_array_identical'
    module_type_store = module_type_store.open_function_context('assert_array_identical', 33, 0, False)
    
    # Passed parameters checking function
    assert_array_identical.stypy_localization = localization
    assert_array_identical.stypy_type_of_self = None
    assert_array_identical.stypy_type_store = module_type_store
    assert_array_identical.stypy_function_name = 'assert_array_identical'
    assert_array_identical.stypy_param_names_list = ['a', 'b']
    assert_array_identical.stypy_varargs_param_name = None
    assert_array_identical.stypy_kwargs_param_name = None
    assert_array_identical.stypy_call_defaults = defaults
    assert_array_identical.stypy_call_varargs = varargs
    assert_array_identical.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_array_identical', ['a', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_array_identical', localization, ['a', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_array_identical(...)' code ##################

    str_1688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'str', 'Assert whether values AND type are the same')
    
    # Call to assert_array_equal(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'a' (line 35)
    a_1690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'a', False)
    # Getting the type of 'b' (line 35)
    b_1691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 26), 'b', False)
    # Processing the call keyword arguments (line 35)
    kwargs_1692 = {}
    # Getting the type of 'assert_array_equal' (line 35)
    assert_array_equal_1689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'assert_array_equal', False)
    # Calling assert_array_equal(args, kwargs) (line 35)
    assert_array_equal_call_result_1693 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), assert_array_equal_1689, *[a_1690, b_1691], **kwargs_1692)
    
    
    # Call to assert_equal(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'a' (line 36)
    a_1695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'a', False)
    # Obtaining the member 'dtype' of a type (line 36)
    dtype_1696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 17), a_1695, 'dtype')
    # Obtaining the member 'type' of a type (line 36)
    type_1697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 17), dtype_1696, 'type')
    # Getting the type of 'b' (line 36)
    b_1698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'b', False)
    # Obtaining the member 'dtype' of a type (line 36)
    dtype_1699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 31), b_1698, 'dtype')
    # Obtaining the member 'type' of a type (line 36)
    type_1700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 31), dtype_1699, 'type')
    # Processing the call keyword arguments (line 36)
    kwargs_1701 = {}
    # Getting the type of 'assert_equal' (line 36)
    assert_equal_1694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 36)
    assert_equal_call_result_1702 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), assert_equal_1694, *[type_1697, type_1700], **kwargs_1701)
    
    
    # ################# End of 'assert_array_identical(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_array_identical' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_1703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1703)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_array_identical'
    return stypy_return_type_1703

# Assigning a type to the variable 'assert_array_identical' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'assert_array_identical', assert_array_identical)

# Assigning a Call to a Name (line 40):

# Call to vectorize(...): (line 40)
# Processing the call arguments (line 40)
# Getting the type of 'id' (line 40)
id_1706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 23), 'id', False)
# Processing the call keyword arguments (line 40)
kwargs_1707 = {}
# Getting the type of 'np' (line 40)
np_1704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 10), 'np', False)
# Obtaining the member 'vectorize' of a type (line 40)
vectorize_1705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 10), np_1704, 'vectorize')
# Calling vectorize(args, kwargs) (line 40)
vectorize_call_result_1708 = invoke(stypy.reporting.localization.Localization(__file__, 40, 10), vectorize_1705, *[id_1706], **kwargs_1707)

# Assigning a type to the variable 'vect_id' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'vect_id', vectorize_call_result_1708)
# Declaration of the 'TestIdict' class

class TestIdict:

    @norecursion
    def test_idict(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_idict'
        module_type_store = module_type_store.open_function_context('test_idict', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestIdict.test_idict.__dict__.__setitem__('stypy_localization', localization)
        TestIdict.test_idict.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestIdict.test_idict.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestIdict.test_idict.__dict__.__setitem__('stypy_function_name', 'TestIdict.test_idict')
        TestIdict.test_idict.__dict__.__setitem__('stypy_param_names_list', [])
        TestIdict.test_idict.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestIdict.test_idict.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestIdict.test_idict.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestIdict.test_idict.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestIdict.test_idict.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestIdict.test_idict.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIdict.test_idict', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_idict', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_idict(...)' code ##################

        
        # Assigning a Dict to a Name (line 46):
        
        # Obtaining an instance of the builtin type 'dict' (line 46)
        dict_1709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 46)
        # Adding element type (key, value) (line 46)
        str_1710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 23), 'str', 'a')
        
        # Call to int16(...): (line 46)
        # Processing the call arguments (line 46)
        int_1713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 37), 'int')
        # Processing the call keyword arguments (line 46)
        kwargs_1714 = {}
        # Getting the type of 'np' (line 46)
        np_1711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 28), 'np', False)
        # Obtaining the member 'int16' of a type (line 46)
        int16_1712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 28), np_1711, 'int16')
        # Calling int16(args, kwargs) (line 46)
        int16_call_result_1715 = invoke(stypy.reporting.localization.Localization(__file__, 46, 28), int16_1712, *[int_1713], **kwargs_1714)
        
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 22), dict_1709, (str_1710, int16_call_result_1715))
        
        # Assigning a type to the variable 'custom_dict' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'custom_dict', dict_1709)
        
        # Assigning a Call to a Name (line 47):
        
        # Call to id(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'custom_dict' (line 47)
        custom_dict_1717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'custom_dict', False)
        # Processing the call keyword arguments (line 47)
        kwargs_1718 = {}
        # Getting the type of 'id' (line 47)
        id_1716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 22), 'id', False)
        # Calling id(args, kwargs) (line 47)
        id_call_result_1719 = invoke(stypy.reporting.localization.Localization(__file__, 47, 22), id_1716, *[custom_dict_1717], **kwargs_1718)
        
        # Assigning a type to the variable 'original_id' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'original_id', id_call_result_1719)
        
        # Assigning a Call to a Name (line 48):
        
        # Call to readsav(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Call to join(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'DATA_PATH' (line 48)
        DATA_PATH_1723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 30), 'DATA_PATH', False)
        str_1724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 41), 'str', 'scalar_byte.sav')
        # Processing the call keyword arguments (line 48)
        kwargs_1725 = {}
        # Getting the type of 'path' (line 48)
        path_1721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 48)
        join_1722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 20), path_1721, 'join')
        # Calling join(args, kwargs) (line 48)
        join_call_result_1726 = invoke(stypy.reporting.localization.Localization(__file__, 48, 20), join_1722, *[DATA_PATH_1723, str_1724], **kwargs_1725)
        
        # Processing the call keyword arguments (line 48)
        # Getting the type of 'custom_dict' (line 48)
        custom_dict_1727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 67), 'custom_dict', False)
        keyword_1728 = custom_dict_1727
        # Getting the type of 'False' (line 48)
        False_1729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 88), 'False', False)
        keyword_1730 = False_1729
        kwargs_1731 = {'idict': keyword_1728, 'verbose': keyword_1730}
        # Getting the type of 'readsav' (line 48)
        readsav_1720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 48)
        readsav_call_result_1732 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), readsav_1720, *[join_call_result_1726], **kwargs_1731)
        
        # Assigning a type to the variable 's' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 's', readsav_call_result_1732)
        
        # Call to assert_equal(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'original_id' (line 49)
        original_id_1734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 21), 'original_id', False)
        
        # Call to id(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 's' (line 49)
        s_1736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 37), 's', False)
        # Processing the call keyword arguments (line 49)
        kwargs_1737 = {}
        # Getting the type of 'id' (line 49)
        id_1735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 34), 'id', False)
        # Calling id(args, kwargs) (line 49)
        id_call_result_1738 = invoke(stypy.reporting.localization.Localization(__file__, 49, 34), id_1735, *[s_1736], **kwargs_1737)
        
        # Processing the call keyword arguments (line 49)
        kwargs_1739 = {}
        # Getting the type of 'assert_equal' (line 49)
        assert_equal_1733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 49)
        assert_equal_call_result_1740 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), assert_equal_1733, *[original_id_1734, id_call_result_1738], **kwargs_1739)
        
        
        # Call to assert_(...): (line 50)
        # Processing the call arguments (line 50)
        
        str_1742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 16), 'str', 'a')
        # Getting the type of 's' (line 50)
        s_1743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 23), 's', False)
        # Applying the binary operator 'in' (line 50)
        result_contains_1744 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 16), 'in', str_1742, s_1743)
        
        # Processing the call keyword arguments (line 50)
        kwargs_1745 = {}
        # Getting the type of 'assert_' (line 50)
        assert__1741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 50)
        assert__call_result_1746 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), assert__1741, *[result_contains_1744], **kwargs_1745)
        
        
        # Call to assert_identical(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Obtaining the type of the subscript
        str_1748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 27), 'str', 'a')
        # Getting the type of 's' (line 51)
        s_1749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 's', False)
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___1750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 25), s_1749, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_1751 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), getitem___1750, str_1748)
        
        
        # Call to int16(...): (line 51)
        # Processing the call arguments (line 51)
        int_1754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 42), 'int')
        # Processing the call keyword arguments (line 51)
        kwargs_1755 = {}
        # Getting the type of 'np' (line 51)
        np_1752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 33), 'np', False)
        # Obtaining the member 'int16' of a type (line 51)
        int16_1753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 33), np_1752, 'int16')
        # Calling int16(args, kwargs) (line 51)
        int16_call_result_1756 = invoke(stypy.reporting.localization.Localization(__file__, 51, 33), int16_1753, *[int_1754], **kwargs_1755)
        
        # Processing the call keyword arguments (line 51)
        kwargs_1757 = {}
        # Getting the type of 'assert_identical' (line 51)
        assert_identical_1747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 51)
        assert_identical_call_result_1758 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), assert_identical_1747, *[subscript_call_result_1751, int16_call_result_1756], **kwargs_1757)
        
        
        # Call to assert_identical(...): (line 52)
        # Processing the call arguments (line 52)
        
        # Obtaining the type of the subscript
        str_1760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 27), 'str', 'i8u')
        # Getting the type of 's' (line 52)
        s_1761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 25), 's', False)
        # Obtaining the member '__getitem__' of a type (line 52)
        getitem___1762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 25), s_1761, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 52)
        subscript_call_result_1763 = invoke(stypy.reporting.localization.Localization(__file__, 52, 25), getitem___1762, str_1760)
        
        
        # Call to uint8(...): (line 52)
        # Processing the call arguments (line 52)
        int_1766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 44), 'int')
        # Processing the call keyword arguments (line 52)
        kwargs_1767 = {}
        # Getting the type of 'np' (line 52)
        np_1764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 35), 'np', False)
        # Obtaining the member 'uint8' of a type (line 52)
        uint8_1765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 35), np_1764, 'uint8')
        # Calling uint8(args, kwargs) (line 52)
        uint8_call_result_1768 = invoke(stypy.reporting.localization.Localization(__file__, 52, 35), uint8_1765, *[int_1766], **kwargs_1767)
        
        # Processing the call keyword arguments (line 52)
        kwargs_1769 = {}
        # Getting the type of 'assert_identical' (line 52)
        assert_identical_1759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 52)
        assert_identical_call_result_1770 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), assert_identical_1759, *[subscript_call_result_1763, uint8_call_result_1768], **kwargs_1769)
        
        
        # ################# End of 'test_idict(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_idict' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_1771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1771)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_idict'
        return stypy_return_type_1771


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 43, 0, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestIdict.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestIdict' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'TestIdict', TestIdict)
# Declaration of the 'TestScalars' class

class TestScalars:

    @norecursion
    def test_byte(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_byte'
        module_type_store = module_type_store.open_function_context('test_byte', 58, 4, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestScalars.test_byte.__dict__.__setitem__('stypy_localization', localization)
        TestScalars.test_byte.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestScalars.test_byte.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestScalars.test_byte.__dict__.__setitem__('stypy_function_name', 'TestScalars.test_byte')
        TestScalars.test_byte.__dict__.__setitem__('stypy_param_names_list', [])
        TestScalars.test_byte.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestScalars.test_byte.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestScalars.test_byte.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestScalars.test_byte.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestScalars.test_byte.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestScalars.test_byte.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScalars.test_byte', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_byte', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_byte(...)' code ##################

        
        # Assigning a Call to a Name (line 59):
        
        # Call to readsav(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Call to join(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'DATA_PATH' (line 59)
        DATA_PATH_1775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 30), 'DATA_PATH', False)
        str_1776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 41), 'str', 'scalar_byte.sav')
        # Processing the call keyword arguments (line 59)
        kwargs_1777 = {}
        # Getting the type of 'path' (line 59)
        path_1773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 59)
        join_1774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 20), path_1773, 'join')
        # Calling join(args, kwargs) (line 59)
        join_call_result_1778 = invoke(stypy.reporting.localization.Localization(__file__, 59, 20), join_1774, *[DATA_PATH_1775, str_1776], **kwargs_1777)
        
        # Processing the call keyword arguments (line 59)
        # Getting the type of 'False' (line 59)
        False_1779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 69), 'False', False)
        keyword_1780 = False_1779
        kwargs_1781 = {'verbose': keyword_1780}
        # Getting the type of 'readsav' (line 59)
        readsav_1772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 59)
        readsav_call_result_1782 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), readsav_1772, *[join_call_result_1778], **kwargs_1781)
        
        # Assigning a type to the variable 's' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 's', readsav_call_result_1782)
        
        # Call to assert_identical(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 's' (line 60)
        s_1784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 's', False)
        # Obtaining the member 'i8u' of a type (line 60)
        i8u_1785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 25), s_1784, 'i8u')
        
        # Call to uint8(...): (line 60)
        # Processing the call arguments (line 60)
        int_1788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 41), 'int')
        # Processing the call keyword arguments (line 60)
        kwargs_1789 = {}
        # Getting the type of 'np' (line 60)
        np_1786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 32), 'np', False)
        # Obtaining the member 'uint8' of a type (line 60)
        uint8_1787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 32), np_1786, 'uint8')
        # Calling uint8(args, kwargs) (line 60)
        uint8_call_result_1790 = invoke(stypy.reporting.localization.Localization(__file__, 60, 32), uint8_1787, *[int_1788], **kwargs_1789)
        
        # Processing the call keyword arguments (line 60)
        kwargs_1791 = {}
        # Getting the type of 'assert_identical' (line 60)
        assert_identical_1783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 60)
        assert_identical_call_result_1792 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), assert_identical_1783, *[i8u_1785, uint8_call_result_1790], **kwargs_1791)
        
        
        # ################# End of 'test_byte(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_byte' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_1793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1793)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_byte'
        return stypy_return_type_1793


    @norecursion
    def test_int16(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_int16'
        module_type_store = module_type_store.open_function_context('test_int16', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestScalars.test_int16.__dict__.__setitem__('stypy_localization', localization)
        TestScalars.test_int16.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestScalars.test_int16.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestScalars.test_int16.__dict__.__setitem__('stypy_function_name', 'TestScalars.test_int16')
        TestScalars.test_int16.__dict__.__setitem__('stypy_param_names_list', [])
        TestScalars.test_int16.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestScalars.test_int16.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestScalars.test_int16.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestScalars.test_int16.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestScalars.test_int16.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestScalars.test_int16.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScalars.test_int16', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_int16', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_int16(...)' code ##################

        
        # Assigning a Call to a Name (line 63):
        
        # Call to readsav(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Call to join(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'DATA_PATH' (line 63)
        DATA_PATH_1797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'DATA_PATH', False)
        str_1798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 41), 'str', 'scalar_int16.sav')
        # Processing the call keyword arguments (line 63)
        kwargs_1799 = {}
        # Getting the type of 'path' (line 63)
        path_1795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 63)
        join_1796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 20), path_1795, 'join')
        # Calling join(args, kwargs) (line 63)
        join_call_result_1800 = invoke(stypy.reporting.localization.Localization(__file__, 63, 20), join_1796, *[DATA_PATH_1797, str_1798], **kwargs_1799)
        
        # Processing the call keyword arguments (line 63)
        # Getting the type of 'False' (line 63)
        False_1801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 70), 'False', False)
        keyword_1802 = False_1801
        kwargs_1803 = {'verbose': keyword_1802}
        # Getting the type of 'readsav' (line 63)
        readsav_1794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 63)
        readsav_call_result_1804 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), readsav_1794, *[join_call_result_1800], **kwargs_1803)
        
        # Assigning a type to the variable 's' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 's', readsav_call_result_1804)
        
        # Call to assert_identical(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 's' (line 64)
        s_1806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 25), 's', False)
        # Obtaining the member 'i16s' of a type (line 64)
        i16s_1807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 25), s_1806, 'i16s')
        
        # Call to int16(...): (line 64)
        # Processing the call arguments (line 64)
        int_1810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 42), 'int')
        # Processing the call keyword arguments (line 64)
        kwargs_1811 = {}
        # Getting the type of 'np' (line 64)
        np_1808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 33), 'np', False)
        # Obtaining the member 'int16' of a type (line 64)
        int16_1809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 33), np_1808, 'int16')
        # Calling int16(args, kwargs) (line 64)
        int16_call_result_1812 = invoke(stypy.reporting.localization.Localization(__file__, 64, 33), int16_1809, *[int_1810], **kwargs_1811)
        
        # Processing the call keyword arguments (line 64)
        kwargs_1813 = {}
        # Getting the type of 'assert_identical' (line 64)
        assert_identical_1805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 64)
        assert_identical_call_result_1814 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), assert_identical_1805, *[i16s_1807, int16_call_result_1812], **kwargs_1813)
        
        
        # ################# End of 'test_int16(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_int16' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_1815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1815)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_int16'
        return stypy_return_type_1815


    @norecursion
    def test_int32(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_int32'
        module_type_store = module_type_store.open_function_context('test_int32', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestScalars.test_int32.__dict__.__setitem__('stypy_localization', localization)
        TestScalars.test_int32.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestScalars.test_int32.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestScalars.test_int32.__dict__.__setitem__('stypy_function_name', 'TestScalars.test_int32')
        TestScalars.test_int32.__dict__.__setitem__('stypy_param_names_list', [])
        TestScalars.test_int32.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestScalars.test_int32.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestScalars.test_int32.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestScalars.test_int32.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestScalars.test_int32.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestScalars.test_int32.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScalars.test_int32', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_int32', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_int32(...)' code ##################

        
        # Assigning a Call to a Name (line 67):
        
        # Call to readsav(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Call to join(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'DATA_PATH' (line 67)
        DATA_PATH_1819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 30), 'DATA_PATH', False)
        str_1820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 41), 'str', 'scalar_int32.sav')
        # Processing the call keyword arguments (line 67)
        kwargs_1821 = {}
        # Getting the type of 'path' (line 67)
        path_1817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 67)
        join_1818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 20), path_1817, 'join')
        # Calling join(args, kwargs) (line 67)
        join_call_result_1822 = invoke(stypy.reporting.localization.Localization(__file__, 67, 20), join_1818, *[DATA_PATH_1819, str_1820], **kwargs_1821)
        
        # Processing the call keyword arguments (line 67)
        # Getting the type of 'False' (line 67)
        False_1823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 70), 'False', False)
        keyword_1824 = False_1823
        kwargs_1825 = {'verbose': keyword_1824}
        # Getting the type of 'readsav' (line 67)
        readsav_1816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 67)
        readsav_call_result_1826 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), readsav_1816, *[join_call_result_1822], **kwargs_1825)
        
        # Assigning a type to the variable 's' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 's', readsav_call_result_1826)
        
        # Call to assert_identical(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 's' (line 68)
        s_1828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 25), 's', False)
        # Obtaining the member 'i32s' of a type (line 68)
        i32s_1829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 25), s_1828, 'i32s')
        
        # Call to int32(...): (line 68)
        # Processing the call arguments (line 68)
        int_1832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 42), 'int')
        # Processing the call keyword arguments (line 68)
        kwargs_1833 = {}
        # Getting the type of 'np' (line 68)
        np_1830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 33), 'np', False)
        # Obtaining the member 'int32' of a type (line 68)
        int32_1831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 33), np_1830, 'int32')
        # Calling int32(args, kwargs) (line 68)
        int32_call_result_1834 = invoke(stypy.reporting.localization.Localization(__file__, 68, 33), int32_1831, *[int_1832], **kwargs_1833)
        
        # Processing the call keyword arguments (line 68)
        kwargs_1835 = {}
        # Getting the type of 'assert_identical' (line 68)
        assert_identical_1827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 68)
        assert_identical_call_result_1836 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), assert_identical_1827, *[i32s_1829, int32_call_result_1834], **kwargs_1835)
        
        
        # ################# End of 'test_int32(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_int32' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_1837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1837)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_int32'
        return stypy_return_type_1837


    @norecursion
    def test_float32(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_float32'
        module_type_store = module_type_store.open_function_context('test_float32', 70, 4, False)
        # Assigning a type to the variable 'self' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestScalars.test_float32.__dict__.__setitem__('stypy_localization', localization)
        TestScalars.test_float32.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestScalars.test_float32.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestScalars.test_float32.__dict__.__setitem__('stypy_function_name', 'TestScalars.test_float32')
        TestScalars.test_float32.__dict__.__setitem__('stypy_param_names_list', [])
        TestScalars.test_float32.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestScalars.test_float32.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestScalars.test_float32.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestScalars.test_float32.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestScalars.test_float32.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestScalars.test_float32.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScalars.test_float32', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_float32', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_float32(...)' code ##################

        
        # Assigning a Call to a Name (line 71):
        
        # Call to readsav(...): (line 71)
        # Processing the call arguments (line 71)
        
        # Call to join(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'DATA_PATH' (line 71)
        DATA_PATH_1841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 30), 'DATA_PATH', False)
        str_1842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 41), 'str', 'scalar_float32.sav')
        # Processing the call keyword arguments (line 71)
        kwargs_1843 = {}
        # Getting the type of 'path' (line 71)
        path_1839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 71)
        join_1840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 20), path_1839, 'join')
        # Calling join(args, kwargs) (line 71)
        join_call_result_1844 = invoke(stypy.reporting.localization.Localization(__file__, 71, 20), join_1840, *[DATA_PATH_1841, str_1842], **kwargs_1843)
        
        # Processing the call keyword arguments (line 71)
        # Getting the type of 'False' (line 71)
        False_1845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 72), 'False', False)
        keyword_1846 = False_1845
        kwargs_1847 = {'verbose': keyword_1846}
        # Getting the type of 'readsav' (line 71)
        readsav_1838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 71)
        readsav_call_result_1848 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), readsav_1838, *[join_call_result_1844], **kwargs_1847)
        
        # Assigning a type to the variable 's' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 's', readsav_call_result_1848)
        
        # Call to assert_identical(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 's' (line 72)
        s_1850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 's', False)
        # Obtaining the member 'f32' of a type (line 72)
        f32_1851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 25), s_1850, 'f32')
        
        # Call to float32(...): (line 72)
        # Processing the call arguments (line 72)
        float_1854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 43), 'float')
        # Processing the call keyword arguments (line 72)
        kwargs_1855 = {}
        # Getting the type of 'np' (line 72)
        np_1852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'np', False)
        # Obtaining the member 'float32' of a type (line 72)
        float32_1853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 32), np_1852, 'float32')
        # Calling float32(args, kwargs) (line 72)
        float32_call_result_1856 = invoke(stypy.reporting.localization.Localization(__file__, 72, 32), float32_1853, *[float_1854], **kwargs_1855)
        
        # Processing the call keyword arguments (line 72)
        kwargs_1857 = {}
        # Getting the type of 'assert_identical' (line 72)
        assert_identical_1849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 72)
        assert_identical_call_result_1858 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), assert_identical_1849, *[f32_1851, float32_call_result_1856], **kwargs_1857)
        
        
        # ################# End of 'test_float32(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_float32' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_1859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1859)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_float32'
        return stypy_return_type_1859


    @norecursion
    def test_float64(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_float64'
        module_type_store = module_type_store.open_function_context('test_float64', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestScalars.test_float64.__dict__.__setitem__('stypy_localization', localization)
        TestScalars.test_float64.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestScalars.test_float64.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestScalars.test_float64.__dict__.__setitem__('stypy_function_name', 'TestScalars.test_float64')
        TestScalars.test_float64.__dict__.__setitem__('stypy_param_names_list', [])
        TestScalars.test_float64.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestScalars.test_float64.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestScalars.test_float64.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestScalars.test_float64.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestScalars.test_float64.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestScalars.test_float64.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScalars.test_float64', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_float64', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_float64(...)' code ##################

        
        # Assigning a Call to a Name (line 75):
        
        # Call to readsav(...): (line 75)
        # Processing the call arguments (line 75)
        
        # Call to join(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'DATA_PATH' (line 75)
        DATA_PATH_1863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 30), 'DATA_PATH', False)
        str_1864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 41), 'str', 'scalar_float64.sav')
        # Processing the call keyword arguments (line 75)
        kwargs_1865 = {}
        # Getting the type of 'path' (line 75)
        path_1861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 75)
        join_1862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 20), path_1861, 'join')
        # Calling join(args, kwargs) (line 75)
        join_call_result_1866 = invoke(stypy.reporting.localization.Localization(__file__, 75, 20), join_1862, *[DATA_PATH_1863, str_1864], **kwargs_1865)
        
        # Processing the call keyword arguments (line 75)
        # Getting the type of 'False' (line 75)
        False_1867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 72), 'False', False)
        keyword_1868 = False_1867
        kwargs_1869 = {'verbose': keyword_1868}
        # Getting the type of 'readsav' (line 75)
        readsav_1860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 75)
        readsav_call_result_1870 = invoke(stypy.reporting.localization.Localization(__file__, 75, 12), readsav_1860, *[join_call_result_1866], **kwargs_1869)
        
        # Assigning a type to the variable 's' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 's', readsav_call_result_1870)
        
        # Call to assert_identical(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 's' (line 76)
        s_1872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 's', False)
        # Obtaining the member 'f64' of a type (line 76)
        f64_1873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 25), s_1872, 'f64')
        
        # Call to float64(...): (line 76)
        # Processing the call arguments (line 76)
        float_1876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 43), 'float')
        # Processing the call keyword arguments (line 76)
        kwargs_1877 = {}
        # Getting the type of 'np' (line 76)
        np_1874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 32), 'np', False)
        # Obtaining the member 'float64' of a type (line 76)
        float64_1875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 32), np_1874, 'float64')
        # Calling float64(args, kwargs) (line 76)
        float64_call_result_1878 = invoke(stypy.reporting.localization.Localization(__file__, 76, 32), float64_1875, *[float_1876], **kwargs_1877)
        
        # Processing the call keyword arguments (line 76)
        kwargs_1879 = {}
        # Getting the type of 'assert_identical' (line 76)
        assert_identical_1871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 76)
        assert_identical_call_result_1880 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), assert_identical_1871, *[f64_1873, float64_call_result_1878], **kwargs_1879)
        
        
        # ################# End of 'test_float64(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_float64' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_1881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1881)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_float64'
        return stypy_return_type_1881


    @norecursion
    def test_complex32(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_complex32'
        module_type_store = module_type_store.open_function_context('test_complex32', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestScalars.test_complex32.__dict__.__setitem__('stypy_localization', localization)
        TestScalars.test_complex32.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestScalars.test_complex32.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestScalars.test_complex32.__dict__.__setitem__('stypy_function_name', 'TestScalars.test_complex32')
        TestScalars.test_complex32.__dict__.__setitem__('stypy_param_names_list', [])
        TestScalars.test_complex32.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestScalars.test_complex32.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestScalars.test_complex32.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestScalars.test_complex32.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestScalars.test_complex32.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestScalars.test_complex32.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScalars.test_complex32', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_complex32', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_complex32(...)' code ##################

        
        # Assigning a Call to a Name (line 79):
        
        # Call to readsav(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Call to join(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'DATA_PATH' (line 79)
        DATA_PATH_1885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'DATA_PATH', False)
        str_1886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 41), 'str', 'scalar_complex32.sav')
        # Processing the call keyword arguments (line 79)
        kwargs_1887 = {}
        # Getting the type of 'path' (line 79)
        path_1883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 79)
        join_1884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 20), path_1883, 'join')
        # Calling join(args, kwargs) (line 79)
        join_call_result_1888 = invoke(stypy.reporting.localization.Localization(__file__, 79, 20), join_1884, *[DATA_PATH_1885, str_1886], **kwargs_1887)
        
        # Processing the call keyword arguments (line 79)
        # Getting the type of 'False' (line 79)
        False_1889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 74), 'False', False)
        keyword_1890 = False_1889
        kwargs_1891 = {'verbose': keyword_1890}
        # Getting the type of 'readsav' (line 79)
        readsav_1882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 79)
        readsav_call_result_1892 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), readsav_1882, *[join_call_result_1888], **kwargs_1891)
        
        # Assigning a type to the variable 's' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 's', readsav_call_result_1892)
        
        # Call to assert_identical(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 's' (line 80)
        s_1894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 25), 's', False)
        # Obtaining the member 'c32' of a type (line 80)
        c32_1895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 25), s_1894, 'c32')
        
        # Call to complex64(...): (line 80)
        # Processing the call arguments (line 80)
        float_1898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 45), 'float')
        complex_1899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 57), 'complex')
        # Applying the binary operator '-' (line 80)
        result_sub_1900 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 45), '-', float_1898, complex_1899)
        
        # Processing the call keyword arguments (line 80)
        kwargs_1901 = {}
        # Getting the type of 'np' (line 80)
        np_1896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 32), 'np', False)
        # Obtaining the member 'complex64' of a type (line 80)
        complex64_1897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 32), np_1896, 'complex64')
        # Calling complex64(args, kwargs) (line 80)
        complex64_call_result_1902 = invoke(stypy.reporting.localization.Localization(__file__, 80, 32), complex64_1897, *[result_sub_1900], **kwargs_1901)
        
        # Processing the call keyword arguments (line 80)
        kwargs_1903 = {}
        # Getting the type of 'assert_identical' (line 80)
        assert_identical_1893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 80)
        assert_identical_call_result_1904 = invoke(stypy.reporting.localization.Localization(__file__, 80, 8), assert_identical_1893, *[c32_1895, complex64_call_result_1902], **kwargs_1903)
        
        
        # ################# End of 'test_complex32(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_complex32' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_1905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1905)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_complex32'
        return stypy_return_type_1905


    @norecursion
    def test_bytes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_bytes'
        module_type_store = module_type_store.open_function_context('test_bytes', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestScalars.test_bytes.__dict__.__setitem__('stypy_localization', localization)
        TestScalars.test_bytes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestScalars.test_bytes.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestScalars.test_bytes.__dict__.__setitem__('stypy_function_name', 'TestScalars.test_bytes')
        TestScalars.test_bytes.__dict__.__setitem__('stypy_param_names_list', [])
        TestScalars.test_bytes.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestScalars.test_bytes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestScalars.test_bytes.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestScalars.test_bytes.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestScalars.test_bytes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestScalars.test_bytes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScalars.test_bytes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_bytes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_bytes(...)' code ##################

        
        # Assigning a Call to a Name (line 83):
        
        # Call to readsav(...): (line 83)
        # Processing the call arguments (line 83)
        
        # Call to join(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'DATA_PATH' (line 83)
        DATA_PATH_1909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'DATA_PATH', False)
        str_1910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 41), 'str', 'scalar_string.sav')
        # Processing the call keyword arguments (line 83)
        kwargs_1911 = {}
        # Getting the type of 'path' (line 83)
        path_1907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 83)
        join_1908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 20), path_1907, 'join')
        # Calling join(args, kwargs) (line 83)
        join_call_result_1912 = invoke(stypy.reporting.localization.Localization(__file__, 83, 20), join_1908, *[DATA_PATH_1909, str_1910], **kwargs_1911)
        
        # Processing the call keyword arguments (line 83)
        # Getting the type of 'False' (line 83)
        False_1913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 71), 'False', False)
        keyword_1914 = False_1913
        kwargs_1915 = {'verbose': keyword_1914}
        # Getting the type of 'readsav' (line 83)
        readsav_1906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 83)
        readsav_call_result_1916 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), readsav_1906, *[join_call_result_1912], **kwargs_1915)
        
        # Assigning a type to the variable 's' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 's', readsav_call_result_1916)
        
        # Call to assert_identical(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 's' (line 84)
        s_1918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 25), 's', False)
        # Obtaining the member 's' of a type (line 84)
        s_1919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 25), s_1918, 's')
        
        # Call to bytes_(...): (line 84)
        # Processing the call arguments (line 84)
        str_1922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 40), 'str', 'The quick brown fox jumps over the lazy python')
        # Processing the call keyword arguments (line 84)
        kwargs_1923 = {}
        # Getting the type of 'np' (line 84)
        np_1920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 30), 'np', False)
        # Obtaining the member 'bytes_' of a type (line 84)
        bytes__1921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 30), np_1920, 'bytes_')
        # Calling bytes_(args, kwargs) (line 84)
        bytes__call_result_1924 = invoke(stypy.reporting.localization.Localization(__file__, 84, 30), bytes__1921, *[str_1922], **kwargs_1923)
        
        # Processing the call keyword arguments (line 84)
        kwargs_1925 = {}
        # Getting the type of 'assert_identical' (line 84)
        assert_identical_1917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 84)
        assert_identical_call_result_1926 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), assert_identical_1917, *[s_1919, bytes__call_result_1924], **kwargs_1925)
        
        
        # ################# End of 'test_bytes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_bytes' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_1927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1927)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_bytes'
        return stypy_return_type_1927


    @norecursion
    def test_structure(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_structure'
        module_type_store = module_type_store.open_function_context('test_structure', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestScalars.test_structure.__dict__.__setitem__('stypy_localization', localization)
        TestScalars.test_structure.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestScalars.test_structure.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestScalars.test_structure.__dict__.__setitem__('stypy_function_name', 'TestScalars.test_structure')
        TestScalars.test_structure.__dict__.__setitem__('stypy_param_names_list', [])
        TestScalars.test_structure.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestScalars.test_structure.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestScalars.test_structure.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestScalars.test_structure.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestScalars.test_structure.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestScalars.test_structure.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScalars.test_structure', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_structure', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_structure(...)' code ##################

        pass
        
        # ################# End of 'test_structure(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_structure' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_1928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1928)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_structure'
        return stypy_return_type_1928


    @norecursion
    def test_complex64(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_complex64'
        module_type_store = module_type_store.open_function_context('test_complex64', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestScalars.test_complex64.__dict__.__setitem__('stypy_localization', localization)
        TestScalars.test_complex64.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestScalars.test_complex64.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestScalars.test_complex64.__dict__.__setitem__('stypy_function_name', 'TestScalars.test_complex64')
        TestScalars.test_complex64.__dict__.__setitem__('stypy_param_names_list', [])
        TestScalars.test_complex64.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestScalars.test_complex64.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestScalars.test_complex64.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestScalars.test_complex64.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestScalars.test_complex64.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestScalars.test_complex64.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScalars.test_complex64', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_complex64', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_complex64(...)' code ##################

        
        # Assigning a Call to a Name (line 90):
        
        # Call to readsav(...): (line 90)
        # Processing the call arguments (line 90)
        
        # Call to join(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'DATA_PATH' (line 90)
        DATA_PATH_1932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), 'DATA_PATH', False)
        str_1933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 41), 'str', 'scalar_complex64.sav')
        # Processing the call keyword arguments (line 90)
        kwargs_1934 = {}
        # Getting the type of 'path' (line 90)
        path_1930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 90)
        join_1931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 20), path_1930, 'join')
        # Calling join(args, kwargs) (line 90)
        join_call_result_1935 = invoke(stypy.reporting.localization.Localization(__file__, 90, 20), join_1931, *[DATA_PATH_1932, str_1933], **kwargs_1934)
        
        # Processing the call keyword arguments (line 90)
        # Getting the type of 'False' (line 90)
        False_1936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 74), 'False', False)
        keyword_1937 = False_1936
        kwargs_1938 = {'verbose': keyword_1937}
        # Getting the type of 'readsav' (line 90)
        readsav_1929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 90)
        readsav_call_result_1939 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), readsav_1929, *[join_call_result_1935], **kwargs_1938)
        
        # Assigning a type to the variable 's' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 's', readsav_call_result_1939)
        
        # Call to assert_identical(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 's' (line 91)
        s_1941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 25), 's', False)
        # Obtaining the member 'c64' of a type (line 91)
        c64_1942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 25), s_1941, 'c64')
        
        # Call to complex128(...): (line 91)
        # Processing the call arguments (line 91)
        float_1945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 46), 'float')
        complex_1946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 70), 'complex')
        # Applying the binary operator '-' (line 91)
        result_sub_1947 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 46), '-', float_1945, complex_1946)
        
        # Processing the call keyword arguments (line 91)
        kwargs_1948 = {}
        # Getting the type of 'np' (line 91)
        np_1943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'np', False)
        # Obtaining the member 'complex128' of a type (line 91)
        complex128_1944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 32), np_1943, 'complex128')
        # Calling complex128(args, kwargs) (line 91)
        complex128_call_result_1949 = invoke(stypy.reporting.localization.Localization(__file__, 91, 32), complex128_1944, *[result_sub_1947], **kwargs_1948)
        
        # Processing the call keyword arguments (line 91)
        kwargs_1950 = {}
        # Getting the type of 'assert_identical' (line 91)
        assert_identical_1940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 91)
        assert_identical_call_result_1951 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), assert_identical_1940, *[c64_1942, complex128_call_result_1949], **kwargs_1950)
        
        
        # ################# End of 'test_complex64(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_complex64' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_1952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1952)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_complex64'
        return stypy_return_type_1952


    @norecursion
    def test_heap_pointer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_heap_pointer'
        module_type_store = module_type_store.open_function_context('test_heap_pointer', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestScalars.test_heap_pointer.__dict__.__setitem__('stypy_localization', localization)
        TestScalars.test_heap_pointer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestScalars.test_heap_pointer.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestScalars.test_heap_pointer.__dict__.__setitem__('stypy_function_name', 'TestScalars.test_heap_pointer')
        TestScalars.test_heap_pointer.__dict__.__setitem__('stypy_param_names_list', [])
        TestScalars.test_heap_pointer.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestScalars.test_heap_pointer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestScalars.test_heap_pointer.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestScalars.test_heap_pointer.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestScalars.test_heap_pointer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestScalars.test_heap_pointer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScalars.test_heap_pointer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_heap_pointer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_heap_pointer(...)' code ##################

        pass
        
        # ################# End of 'test_heap_pointer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_heap_pointer' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_1953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1953)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_heap_pointer'
        return stypy_return_type_1953


    @norecursion
    def test_object_reference(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_object_reference'
        module_type_store = module_type_store.open_function_context('test_object_reference', 96, 4, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestScalars.test_object_reference.__dict__.__setitem__('stypy_localization', localization)
        TestScalars.test_object_reference.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestScalars.test_object_reference.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestScalars.test_object_reference.__dict__.__setitem__('stypy_function_name', 'TestScalars.test_object_reference')
        TestScalars.test_object_reference.__dict__.__setitem__('stypy_param_names_list', [])
        TestScalars.test_object_reference.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestScalars.test_object_reference.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestScalars.test_object_reference.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestScalars.test_object_reference.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestScalars.test_object_reference.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestScalars.test_object_reference.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScalars.test_object_reference', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_object_reference', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_object_reference(...)' code ##################

        pass
        
        # ################# End of 'test_object_reference(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_object_reference' in the type store
        # Getting the type of 'stypy_return_type' (line 96)
        stypy_return_type_1954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1954)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_object_reference'
        return stypy_return_type_1954


    @norecursion
    def test_uint16(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_uint16'
        module_type_store = module_type_store.open_function_context('test_uint16', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestScalars.test_uint16.__dict__.__setitem__('stypy_localization', localization)
        TestScalars.test_uint16.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestScalars.test_uint16.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestScalars.test_uint16.__dict__.__setitem__('stypy_function_name', 'TestScalars.test_uint16')
        TestScalars.test_uint16.__dict__.__setitem__('stypy_param_names_list', [])
        TestScalars.test_uint16.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestScalars.test_uint16.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestScalars.test_uint16.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestScalars.test_uint16.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestScalars.test_uint16.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestScalars.test_uint16.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScalars.test_uint16', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_uint16', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_uint16(...)' code ##################

        
        # Assigning a Call to a Name (line 100):
        
        # Call to readsav(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Call to join(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'DATA_PATH' (line 100)
        DATA_PATH_1958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 30), 'DATA_PATH', False)
        str_1959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 41), 'str', 'scalar_uint16.sav')
        # Processing the call keyword arguments (line 100)
        kwargs_1960 = {}
        # Getting the type of 'path' (line 100)
        path_1956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 100)
        join_1957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 20), path_1956, 'join')
        # Calling join(args, kwargs) (line 100)
        join_call_result_1961 = invoke(stypy.reporting.localization.Localization(__file__, 100, 20), join_1957, *[DATA_PATH_1958, str_1959], **kwargs_1960)
        
        # Processing the call keyword arguments (line 100)
        # Getting the type of 'False' (line 100)
        False_1962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 71), 'False', False)
        keyword_1963 = False_1962
        kwargs_1964 = {'verbose': keyword_1963}
        # Getting the type of 'readsav' (line 100)
        readsav_1955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 100)
        readsav_call_result_1965 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), readsav_1955, *[join_call_result_1961], **kwargs_1964)
        
        # Assigning a type to the variable 's' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 's', readsav_call_result_1965)
        
        # Call to assert_identical(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 's' (line 101)
        s_1967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 25), 's', False)
        # Obtaining the member 'i16u' of a type (line 101)
        i16u_1968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 25), s_1967, 'i16u')
        
        # Call to uint16(...): (line 101)
        # Processing the call arguments (line 101)
        int_1971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 43), 'int')
        # Processing the call keyword arguments (line 101)
        kwargs_1972 = {}
        # Getting the type of 'np' (line 101)
        np_1969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 33), 'np', False)
        # Obtaining the member 'uint16' of a type (line 101)
        uint16_1970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 33), np_1969, 'uint16')
        # Calling uint16(args, kwargs) (line 101)
        uint16_call_result_1973 = invoke(stypy.reporting.localization.Localization(__file__, 101, 33), uint16_1970, *[int_1971], **kwargs_1972)
        
        # Processing the call keyword arguments (line 101)
        kwargs_1974 = {}
        # Getting the type of 'assert_identical' (line 101)
        assert_identical_1966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 101)
        assert_identical_call_result_1975 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), assert_identical_1966, *[i16u_1968, uint16_call_result_1973], **kwargs_1974)
        
        
        # ################# End of 'test_uint16(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_uint16' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_1976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1976)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_uint16'
        return stypy_return_type_1976


    @norecursion
    def test_uint32(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_uint32'
        module_type_store = module_type_store.open_function_context('test_uint32', 103, 4, False)
        # Assigning a type to the variable 'self' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestScalars.test_uint32.__dict__.__setitem__('stypy_localization', localization)
        TestScalars.test_uint32.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestScalars.test_uint32.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestScalars.test_uint32.__dict__.__setitem__('stypy_function_name', 'TestScalars.test_uint32')
        TestScalars.test_uint32.__dict__.__setitem__('stypy_param_names_list', [])
        TestScalars.test_uint32.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestScalars.test_uint32.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestScalars.test_uint32.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestScalars.test_uint32.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestScalars.test_uint32.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestScalars.test_uint32.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScalars.test_uint32', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_uint32', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_uint32(...)' code ##################

        
        # Assigning a Call to a Name (line 104):
        
        # Call to readsav(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Call to join(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'DATA_PATH' (line 104)
        DATA_PATH_1980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'DATA_PATH', False)
        str_1981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 41), 'str', 'scalar_uint32.sav')
        # Processing the call keyword arguments (line 104)
        kwargs_1982 = {}
        # Getting the type of 'path' (line 104)
        path_1978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 104)
        join_1979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 20), path_1978, 'join')
        # Calling join(args, kwargs) (line 104)
        join_call_result_1983 = invoke(stypy.reporting.localization.Localization(__file__, 104, 20), join_1979, *[DATA_PATH_1980, str_1981], **kwargs_1982)
        
        # Processing the call keyword arguments (line 104)
        # Getting the type of 'False' (line 104)
        False_1984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 71), 'False', False)
        keyword_1985 = False_1984
        kwargs_1986 = {'verbose': keyword_1985}
        # Getting the type of 'readsav' (line 104)
        readsav_1977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 104)
        readsav_call_result_1987 = invoke(stypy.reporting.localization.Localization(__file__, 104, 12), readsav_1977, *[join_call_result_1983], **kwargs_1986)
        
        # Assigning a type to the variable 's' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 's', readsav_call_result_1987)
        
        # Call to assert_identical(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 's' (line 105)
        s_1989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 25), 's', False)
        # Obtaining the member 'i32u' of a type (line 105)
        i32u_1990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 25), s_1989, 'i32u')
        
        # Call to uint32(...): (line 105)
        # Processing the call arguments (line 105)
        long_1993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 43), 'long')
        # Processing the call keyword arguments (line 105)
        kwargs_1994 = {}
        # Getting the type of 'np' (line 105)
        np_1991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 33), 'np', False)
        # Obtaining the member 'uint32' of a type (line 105)
        uint32_1992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 33), np_1991, 'uint32')
        # Calling uint32(args, kwargs) (line 105)
        uint32_call_result_1995 = invoke(stypy.reporting.localization.Localization(__file__, 105, 33), uint32_1992, *[long_1993], **kwargs_1994)
        
        # Processing the call keyword arguments (line 105)
        kwargs_1996 = {}
        # Getting the type of 'assert_identical' (line 105)
        assert_identical_1988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 105)
        assert_identical_call_result_1997 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), assert_identical_1988, *[i32u_1990, uint32_call_result_1995], **kwargs_1996)
        
        
        # ################# End of 'test_uint32(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_uint32' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_1998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1998)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_uint32'
        return stypy_return_type_1998


    @norecursion
    def test_int64(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_int64'
        module_type_store = module_type_store.open_function_context('test_int64', 107, 4, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestScalars.test_int64.__dict__.__setitem__('stypy_localization', localization)
        TestScalars.test_int64.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestScalars.test_int64.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestScalars.test_int64.__dict__.__setitem__('stypy_function_name', 'TestScalars.test_int64')
        TestScalars.test_int64.__dict__.__setitem__('stypy_param_names_list', [])
        TestScalars.test_int64.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestScalars.test_int64.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestScalars.test_int64.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestScalars.test_int64.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestScalars.test_int64.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestScalars.test_int64.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScalars.test_int64', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_int64', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_int64(...)' code ##################

        
        # Assigning a Call to a Name (line 108):
        
        # Call to readsav(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Call to join(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'DATA_PATH' (line 108)
        DATA_PATH_2002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 30), 'DATA_PATH', False)
        str_2003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 41), 'str', 'scalar_int64.sav')
        # Processing the call keyword arguments (line 108)
        kwargs_2004 = {}
        # Getting the type of 'path' (line 108)
        path_2000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 108)
        join_2001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 20), path_2000, 'join')
        # Calling join(args, kwargs) (line 108)
        join_call_result_2005 = invoke(stypy.reporting.localization.Localization(__file__, 108, 20), join_2001, *[DATA_PATH_2002, str_2003], **kwargs_2004)
        
        # Processing the call keyword arguments (line 108)
        # Getting the type of 'False' (line 108)
        False_2006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 70), 'False', False)
        keyword_2007 = False_2006
        kwargs_2008 = {'verbose': keyword_2007}
        # Getting the type of 'readsav' (line 108)
        readsav_1999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 108)
        readsav_call_result_2009 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), readsav_1999, *[join_call_result_2005], **kwargs_2008)
        
        # Assigning a type to the variable 's' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 's', readsav_call_result_2009)
        
        # Call to assert_identical(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 's' (line 109)
        s_2011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 25), 's', False)
        # Obtaining the member 'i64s' of a type (line 109)
        i64s_2012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 25), s_2011, 'i64s')
        
        # Call to int64(...): (line 109)
        # Processing the call arguments (line 109)
        long_2015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 42), 'long')
        # Processing the call keyword arguments (line 109)
        kwargs_2016 = {}
        # Getting the type of 'np' (line 109)
        np_2013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 33), 'np', False)
        # Obtaining the member 'int64' of a type (line 109)
        int64_2014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 33), np_2013, 'int64')
        # Calling int64(args, kwargs) (line 109)
        int64_call_result_2017 = invoke(stypy.reporting.localization.Localization(__file__, 109, 33), int64_2014, *[long_2015], **kwargs_2016)
        
        # Processing the call keyword arguments (line 109)
        kwargs_2018 = {}
        # Getting the type of 'assert_identical' (line 109)
        assert_identical_2010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 109)
        assert_identical_call_result_2019 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), assert_identical_2010, *[i64s_2012, int64_call_result_2017], **kwargs_2018)
        
        
        # ################# End of 'test_int64(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_int64' in the type store
        # Getting the type of 'stypy_return_type' (line 107)
        stypy_return_type_2020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2020)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_int64'
        return stypy_return_type_2020


    @norecursion
    def test_uint64(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_uint64'
        module_type_store = module_type_store.open_function_context('test_uint64', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestScalars.test_uint64.__dict__.__setitem__('stypy_localization', localization)
        TestScalars.test_uint64.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestScalars.test_uint64.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestScalars.test_uint64.__dict__.__setitem__('stypy_function_name', 'TestScalars.test_uint64')
        TestScalars.test_uint64.__dict__.__setitem__('stypy_param_names_list', [])
        TestScalars.test_uint64.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestScalars.test_uint64.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestScalars.test_uint64.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestScalars.test_uint64.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestScalars.test_uint64.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestScalars.test_uint64.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScalars.test_uint64', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_uint64', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_uint64(...)' code ##################

        
        # Assigning a Call to a Name (line 112):
        
        # Call to readsav(...): (line 112)
        # Processing the call arguments (line 112)
        
        # Call to join(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'DATA_PATH' (line 112)
        DATA_PATH_2024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 30), 'DATA_PATH', False)
        str_2025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 41), 'str', 'scalar_uint64.sav')
        # Processing the call keyword arguments (line 112)
        kwargs_2026 = {}
        # Getting the type of 'path' (line 112)
        path_2022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 112)
        join_2023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 20), path_2022, 'join')
        # Calling join(args, kwargs) (line 112)
        join_call_result_2027 = invoke(stypy.reporting.localization.Localization(__file__, 112, 20), join_2023, *[DATA_PATH_2024, str_2025], **kwargs_2026)
        
        # Processing the call keyword arguments (line 112)
        # Getting the type of 'False' (line 112)
        False_2028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 71), 'False', False)
        keyword_2029 = False_2028
        kwargs_2030 = {'verbose': keyword_2029}
        # Getting the type of 'readsav' (line 112)
        readsav_2021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 112)
        readsav_call_result_2031 = invoke(stypy.reporting.localization.Localization(__file__, 112, 12), readsav_2021, *[join_call_result_2027], **kwargs_2030)
        
        # Assigning a type to the variable 's' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 's', readsav_call_result_2031)
        
        # Call to assert_identical(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 's' (line 113)
        s_2033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 's', False)
        # Obtaining the member 'i64u' of a type (line 113)
        i64u_2034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 25), s_2033, 'i64u')
        
        # Call to uint64(...): (line 113)
        # Processing the call arguments (line 113)
        long_2037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 43), 'long')
        # Processing the call keyword arguments (line 113)
        kwargs_2038 = {}
        # Getting the type of 'np' (line 113)
        np_2035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 33), 'np', False)
        # Obtaining the member 'uint64' of a type (line 113)
        uint64_2036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 33), np_2035, 'uint64')
        # Calling uint64(args, kwargs) (line 113)
        uint64_call_result_2039 = invoke(stypy.reporting.localization.Localization(__file__, 113, 33), uint64_2036, *[long_2037], **kwargs_2038)
        
        # Processing the call keyword arguments (line 113)
        kwargs_2040 = {}
        # Getting the type of 'assert_identical' (line 113)
        assert_identical_2032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 113)
        assert_identical_call_result_2041 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), assert_identical_2032, *[i64u_2034, uint64_call_result_2039], **kwargs_2040)
        
        
        # ################# End of 'test_uint64(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_uint64' in the type store
        # Getting the type of 'stypy_return_type' (line 111)
        stypy_return_type_2042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2042)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_uint64'
        return stypy_return_type_2042


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 55, 0, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestScalars.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestScalars' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'TestScalars', TestScalars)
# Declaration of the 'TestCompressed' class
# Getting the type of 'TestScalars' (line 116)
TestScalars_2043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 21), 'TestScalars')

class TestCompressed(TestScalars_2043, ):

    @norecursion
    def test_compressed(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_compressed'
        module_type_store = module_type_store.open_function_context('test_compressed', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCompressed.test_compressed.__dict__.__setitem__('stypy_localization', localization)
        TestCompressed.test_compressed.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCompressed.test_compressed.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCompressed.test_compressed.__dict__.__setitem__('stypy_function_name', 'TestCompressed.test_compressed')
        TestCompressed.test_compressed.__dict__.__setitem__('stypy_param_names_list', [])
        TestCompressed.test_compressed.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCompressed.test_compressed.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCompressed.test_compressed.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCompressed.test_compressed.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCompressed.test_compressed.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCompressed.test_compressed.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCompressed.test_compressed', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_compressed', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_compressed(...)' code ##################

        
        # Assigning a Call to a Name (line 120):
        
        # Call to readsav(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Call to join(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'DATA_PATH' (line 120)
        DATA_PATH_2047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 30), 'DATA_PATH', False)
        str_2048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 41), 'str', 'various_compressed.sav')
        # Processing the call keyword arguments (line 120)
        kwargs_2049 = {}
        # Getting the type of 'path' (line 120)
        path_2045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 120)
        join_2046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 20), path_2045, 'join')
        # Calling join(args, kwargs) (line 120)
        join_call_result_2050 = invoke(stypy.reporting.localization.Localization(__file__, 120, 20), join_2046, *[DATA_PATH_2047, str_2048], **kwargs_2049)
        
        # Processing the call keyword arguments (line 120)
        # Getting the type of 'False' (line 120)
        False_2051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 76), 'False', False)
        keyword_2052 = False_2051
        kwargs_2053 = {'verbose': keyword_2052}
        # Getting the type of 'readsav' (line 120)
        readsav_2044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 120)
        readsav_call_result_2054 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), readsav_2044, *[join_call_result_2050], **kwargs_2053)
        
        # Assigning a type to the variable 's' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 's', readsav_call_result_2054)
        
        # Call to assert_identical(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 's' (line 122)
        s_2056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 25), 's', False)
        # Obtaining the member 'i8u' of a type (line 122)
        i8u_2057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 25), s_2056, 'i8u')
        
        # Call to uint8(...): (line 122)
        # Processing the call arguments (line 122)
        int_2060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 41), 'int')
        # Processing the call keyword arguments (line 122)
        kwargs_2061 = {}
        # Getting the type of 'np' (line 122)
        np_2058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 32), 'np', False)
        # Obtaining the member 'uint8' of a type (line 122)
        uint8_2059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 32), np_2058, 'uint8')
        # Calling uint8(args, kwargs) (line 122)
        uint8_call_result_2062 = invoke(stypy.reporting.localization.Localization(__file__, 122, 32), uint8_2059, *[int_2060], **kwargs_2061)
        
        # Processing the call keyword arguments (line 122)
        kwargs_2063 = {}
        # Getting the type of 'assert_identical' (line 122)
        assert_identical_2055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 122)
        assert_identical_call_result_2064 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), assert_identical_2055, *[i8u_2057, uint8_call_result_2062], **kwargs_2063)
        
        
        # Call to assert_identical(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 's' (line 123)
        s_2066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 25), 's', False)
        # Obtaining the member 'f32' of a type (line 123)
        f32_2067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 25), s_2066, 'f32')
        
        # Call to float32(...): (line 123)
        # Processing the call arguments (line 123)
        float_2070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 43), 'float')
        # Processing the call keyword arguments (line 123)
        kwargs_2071 = {}
        # Getting the type of 'np' (line 123)
        np_2068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 32), 'np', False)
        # Obtaining the member 'float32' of a type (line 123)
        float32_2069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 32), np_2068, 'float32')
        # Calling float32(args, kwargs) (line 123)
        float32_call_result_2072 = invoke(stypy.reporting.localization.Localization(__file__, 123, 32), float32_2069, *[float_2070], **kwargs_2071)
        
        # Processing the call keyword arguments (line 123)
        kwargs_2073 = {}
        # Getting the type of 'assert_identical' (line 123)
        assert_identical_2065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 123)
        assert_identical_call_result_2074 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), assert_identical_2065, *[f32_2067, float32_call_result_2072], **kwargs_2073)
        
        
        # Call to assert_identical(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 's' (line 124)
        s_2076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 's', False)
        # Obtaining the member 'c64' of a type (line 124)
        c64_2077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 25), s_2076, 'c64')
        
        # Call to complex128(...): (line 124)
        # Processing the call arguments (line 124)
        float_2080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 46), 'float')
        complex_2081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 70), 'complex')
        # Applying the binary operator '-' (line 124)
        result_sub_2082 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 46), '-', float_2080, complex_2081)
        
        # Processing the call keyword arguments (line 124)
        kwargs_2083 = {}
        # Getting the type of 'np' (line 124)
        np_2078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 32), 'np', False)
        # Obtaining the member 'complex128' of a type (line 124)
        complex128_2079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 32), np_2078, 'complex128')
        # Calling complex128(args, kwargs) (line 124)
        complex128_call_result_2084 = invoke(stypy.reporting.localization.Localization(__file__, 124, 32), complex128_2079, *[result_sub_2082], **kwargs_2083)
        
        # Processing the call keyword arguments (line 124)
        kwargs_2085 = {}
        # Getting the type of 'assert_identical' (line 124)
        assert_identical_2075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 124)
        assert_identical_call_result_2086 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), assert_identical_2075, *[c64_2077, complex128_call_result_2084], **kwargs_2085)
        
        
        # Call to assert_equal(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 's' (line 125)
        s_2088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 21), 's', False)
        # Obtaining the member 'array5d' of a type (line 125)
        array5d_2089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 21), s_2088, 'array5d')
        # Obtaining the member 'shape' of a type (line 125)
        shape_2090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 21), array5d_2089, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 125)
        tuple_2091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 125)
        # Adding element type (line 125)
        int_2092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 39), tuple_2091, int_2092)
        # Adding element type (line 125)
        int_2093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 39), tuple_2091, int_2093)
        # Adding element type (line 125)
        int_2094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 39), tuple_2091, int_2094)
        # Adding element type (line 125)
        int_2095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 39), tuple_2091, int_2095)
        # Adding element type (line 125)
        int_2096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 39), tuple_2091, int_2096)
        
        # Processing the call keyword arguments (line 125)
        kwargs_2097 = {}
        # Getting the type of 'assert_equal' (line 125)
        assert_equal_2087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 125)
        assert_equal_call_result_2098 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), assert_equal_2087, *[shape_2090, tuple_2091], **kwargs_2097)
        
        
        # Call to assert_identical(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Obtaining the type of the subscript
        int_2100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 36), 'int')
        # Getting the type of 's' (line 126)
        s_2101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 25), 's', False)
        # Obtaining the member 'arrays' of a type (line 126)
        arrays_2102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 25), s_2101, 'arrays')
        # Obtaining the member 'a' of a type (line 126)
        a_2103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 25), arrays_2102, 'a')
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___2104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 25), a_2103, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 126)
        subscript_call_result_2105 = invoke(stypy.reporting.localization.Localization(__file__, 126, 25), getitem___2104, int_2100)
        
        
        # Call to array(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Obtaining an instance of the builtin type 'list' (line 126)
        list_2108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 126)
        # Adding element type (line 126)
        int_2109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 49), list_2108, int_2109)
        # Adding element type (line 126)
        int_2110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 49), list_2108, int_2110)
        # Adding element type (line 126)
        int_2111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 49), list_2108, int_2111)
        
        # Processing the call keyword arguments (line 126)
        # Getting the type of 'np' (line 126)
        np_2112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 66), 'np', False)
        # Obtaining the member 'int16' of a type (line 126)
        int16_2113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 66), np_2112, 'int16')
        keyword_2114 = int16_2113
        kwargs_2115 = {'dtype': keyword_2114}
        # Getting the type of 'np' (line 126)
        np_2106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 40), 'np', False)
        # Obtaining the member 'array' of a type (line 126)
        array_2107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 40), np_2106, 'array')
        # Calling array(args, kwargs) (line 126)
        array_call_result_2116 = invoke(stypy.reporting.localization.Localization(__file__, 126, 40), array_2107, *[list_2108], **kwargs_2115)
        
        # Processing the call keyword arguments (line 126)
        kwargs_2117 = {}
        # Getting the type of 'assert_identical' (line 126)
        assert_identical_2099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 126)
        assert_identical_call_result_2118 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), assert_identical_2099, *[subscript_call_result_2105, array_call_result_2116], **kwargs_2117)
        
        
        # Call to assert_identical(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Obtaining the type of the subscript
        int_2120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 36), 'int')
        # Getting the type of 's' (line 127)
        s_2121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 25), 's', False)
        # Obtaining the member 'arrays' of a type (line 127)
        arrays_2122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 25), s_2121, 'arrays')
        # Obtaining the member 'b' of a type (line 127)
        b_2123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 25), arrays_2122, 'b')
        # Obtaining the member '__getitem__' of a type (line 127)
        getitem___2124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 25), b_2123, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 127)
        subscript_call_result_2125 = invoke(stypy.reporting.localization.Localization(__file__, 127, 25), getitem___2124, int_2120)
        
        
        # Call to array(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Obtaining an instance of the builtin type 'list' (line 127)
        list_2128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 127)
        # Adding element type (line 127)
        float_2129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 50), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 49), list_2128, float_2129)
        # Adding element type (line 127)
        float_2130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 54), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 49), list_2128, float_2130)
        # Adding element type (line 127)
        float_2131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 58), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 49), list_2128, float_2131)
        # Adding element type (line 127)
        float_2132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 62), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 49), list_2128, float_2132)
        
        # Processing the call keyword arguments (line 127)
        # Getting the type of 'np' (line 127)
        np_2133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 73), 'np', False)
        # Obtaining the member 'float32' of a type (line 127)
        float32_2134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 73), np_2133, 'float32')
        keyword_2135 = float32_2134
        kwargs_2136 = {'dtype': keyword_2135}
        # Getting the type of 'np' (line 127)
        np_2126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 40), 'np', False)
        # Obtaining the member 'array' of a type (line 127)
        array_2127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 40), np_2126, 'array')
        # Calling array(args, kwargs) (line 127)
        array_call_result_2137 = invoke(stypy.reporting.localization.Localization(__file__, 127, 40), array_2127, *[list_2128], **kwargs_2136)
        
        # Processing the call keyword arguments (line 127)
        kwargs_2138 = {}
        # Getting the type of 'assert_identical' (line 127)
        assert_identical_2119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 127)
        assert_identical_call_result_2139 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), assert_identical_2119, *[subscript_call_result_2125, array_call_result_2137], **kwargs_2138)
        
        
        # Call to assert_identical(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining the type of the subscript
        int_2141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 36), 'int')
        # Getting the type of 's' (line 128)
        s_2142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 25), 's', False)
        # Obtaining the member 'arrays' of a type (line 128)
        arrays_2143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 25), s_2142, 'arrays')
        # Obtaining the member 'c' of a type (line 128)
        c_2144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 25), arrays_2143, 'c')
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___2145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 25), c_2144, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_2146 = invoke(stypy.reporting.localization.Localization(__file__, 128, 25), getitem___2145, int_2141)
        
        
        # Call to array(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Obtaining an instance of the builtin type 'list' (line 128)
        list_2149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 128)
        # Adding element type (line 128)
        
        # Call to complex64(...): (line 128)
        # Processing the call arguments (line 128)
        int_2152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 63), 'int')
        complex_2153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 65), 'complex')
        # Applying the binary operator '+' (line 128)
        result_add_2154 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 63), '+', int_2152, complex_2153)
        
        # Processing the call keyword arguments (line 128)
        kwargs_2155 = {}
        # Getting the type of 'np' (line 128)
        np_2150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 50), 'np', False)
        # Obtaining the member 'complex64' of a type (line 128)
        complex64_2151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 50), np_2150, 'complex64')
        # Calling complex64(args, kwargs) (line 128)
        complex64_call_result_2156 = invoke(stypy.reporting.localization.Localization(__file__, 128, 50), complex64_2151, *[result_add_2154], **kwargs_2155)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 49), list_2149, complex64_call_result_2156)
        # Adding element type (line 128)
        
        # Call to complex64(...): (line 128)
        # Processing the call arguments (line 128)
        int_2159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 83), 'int')
        complex_2160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 85), 'complex')
        # Applying the binary operator '+' (line 128)
        result_add_2161 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 83), '+', int_2159, complex_2160)
        
        # Processing the call keyword arguments (line 128)
        kwargs_2162 = {}
        # Getting the type of 'np' (line 128)
        np_2157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 70), 'np', False)
        # Obtaining the member 'complex64' of a type (line 128)
        complex64_2158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 70), np_2157, 'complex64')
        # Calling complex64(args, kwargs) (line 128)
        complex64_call_result_2163 = invoke(stypy.reporting.localization.Localization(__file__, 128, 70), complex64_2158, *[result_add_2161], **kwargs_2162)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 49), list_2149, complex64_call_result_2163)
        
        # Processing the call keyword arguments (line 128)
        kwargs_2164 = {}
        # Getting the type of 'np' (line 128)
        np_2147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 40), 'np', False)
        # Obtaining the member 'array' of a type (line 128)
        array_2148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 40), np_2147, 'array')
        # Calling array(args, kwargs) (line 128)
        array_call_result_2165 = invoke(stypy.reporting.localization.Localization(__file__, 128, 40), array_2148, *[list_2149], **kwargs_2164)
        
        # Processing the call keyword arguments (line 128)
        kwargs_2166 = {}
        # Getting the type of 'assert_identical' (line 128)
        assert_identical_2140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 128)
        assert_identical_call_result_2167 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), assert_identical_2140, *[subscript_call_result_2146, array_call_result_2165], **kwargs_2166)
        
        
        # Call to assert_identical(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Obtaining the type of the subscript
        int_2169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 36), 'int')
        # Getting the type of 's' (line 129)
        s_2170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 's', False)
        # Obtaining the member 'arrays' of a type (line 129)
        arrays_2171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 25), s_2170, 'arrays')
        # Obtaining the member 'd' of a type (line 129)
        d_2172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 25), arrays_2171, 'd')
        # Obtaining the member '__getitem__' of a type (line 129)
        getitem___2173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 25), d_2172, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 129)
        subscript_call_result_2174 = invoke(stypy.reporting.localization.Localization(__file__, 129, 25), getitem___2173, int_2169)
        
        
        # Call to array(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Obtaining an instance of the builtin type 'list' (line 129)
        list_2177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 129)
        # Adding element type (line 129)
        str_2178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 50), 'str', 'cheese')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 49), list_2177, str_2178)
        # Adding element type (line 129)
        str_2179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 61), 'str', 'bacon')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 49), list_2177, str_2179)
        # Adding element type (line 129)
        str_2180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 71), 'str', 'spam')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 49), list_2177, str_2180)
        
        # Processing the call keyword arguments (line 129)
        # Getting the type of 'object' (line 129)
        object_2181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 87), 'object', False)
        keyword_2182 = object_2181
        kwargs_2183 = {'dtype': keyword_2182}
        # Getting the type of 'np' (line 129)
        np_2175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 40), 'np', False)
        # Obtaining the member 'array' of a type (line 129)
        array_2176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 40), np_2175, 'array')
        # Calling array(args, kwargs) (line 129)
        array_call_result_2184 = invoke(stypy.reporting.localization.Localization(__file__, 129, 40), array_2176, *[list_2177], **kwargs_2183)
        
        # Processing the call keyword arguments (line 129)
        kwargs_2185 = {}
        # Getting the type of 'assert_identical' (line 129)
        assert_identical_2168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 129)
        assert_identical_call_result_2186 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), assert_identical_2168, *[subscript_call_result_2174, array_call_result_2184], **kwargs_2185)
        
        
        # ################# End of 'test_compressed(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_compressed' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_2187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2187)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_compressed'
        return stypy_return_type_2187


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 116, 0, False)
        # Assigning a type to the variable 'self' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCompressed.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestCompressed' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'TestCompressed', TestCompressed)
# Declaration of the 'TestArrayDimensions' class

class TestArrayDimensions:

    @norecursion
    def test_1d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_1d'
        module_type_store = module_type_store.open_function_context('test_1d', 135, 4, False)
        # Assigning a type to the variable 'self' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArrayDimensions.test_1d.__dict__.__setitem__('stypy_localization', localization)
        TestArrayDimensions.test_1d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArrayDimensions.test_1d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArrayDimensions.test_1d.__dict__.__setitem__('stypy_function_name', 'TestArrayDimensions.test_1d')
        TestArrayDimensions.test_1d.__dict__.__setitem__('stypy_param_names_list', [])
        TestArrayDimensions.test_1d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArrayDimensions.test_1d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArrayDimensions.test_1d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArrayDimensions.test_1d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArrayDimensions.test_1d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArrayDimensions.test_1d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArrayDimensions.test_1d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_1d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_1d(...)' code ##################

        
        # Assigning a Call to a Name (line 136):
        
        # Call to readsav(...): (line 136)
        # Processing the call arguments (line 136)
        
        # Call to join(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'DATA_PATH' (line 136)
        DATA_PATH_2191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 30), 'DATA_PATH', False)
        str_2192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 41), 'str', 'array_float32_1d.sav')
        # Processing the call keyword arguments (line 136)
        kwargs_2193 = {}
        # Getting the type of 'path' (line 136)
        path_2189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 136)
        join_2190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 20), path_2189, 'join')
        # Calling join(args, kwargs) (line 136)
        join_call_result_2194 = invoke(stypy.reporting.localization.Localization(__file__, 136, 20), join_2190, *[DATA_PATH_2191, str_2192], **kwargs_2193)
        
        # Processing the call keyword arguments (line 136)
        # Getting the type of 'False' (line 136)
        False_2195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 74), 'False', False)
        keyword_2196 = False_2195
        kwargs_2197 = {'verbose': keyword_2196}
        # Getting the type of 'readsav' (line 136)
        readsav_2188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 136)
        readsav_call_result_2198 = invoke(stypy.reporting.localization.Localization(__file__, 136, 12), readsav_2188, *[join_call_result_2194], **kwargs_2197)
        
        # Assigning a type to the variable 's' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 's', readsav_call_result_2198)
        
        # Call to assert_equal(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 's' (line 137)
        s_2200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 21), 's', False)
        # Obtaining the member 'array1d' of a type (line 137)
        array1d_2201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 21), s_2200, 'array1d')
        # Obtaining the member 'shape' of a type (line 137)
        shape_2202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 21), array1d_2201, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 137)
        tuple_2203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 137)
        # Adding element type (line 137)
        int_2204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 39), tuple_2203, int_2204)
        
        # Processing the call keyword arguments (line 137)
        kwargs_2205 = {}
        # Getting the type of 'assert_equal' (line 137)
        assert_equal_2199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 137)
        assert_equal_call_result_2206 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), assert_equal_2199, *[shape_2202, tuple_2203], **kwargs_2205)
        
        
        # ################# End of 'test_1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_1d' in the type store
        # Getting the type of 'stypy_return_type' (line 135)
        stypy_return_type_2207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2207)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_1d'
        return stypy_return_type_2207


    @norecursion
    def test_2d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_2d'
        module_type_store = module_type_store.open_function_context('test_2d', 139, 4, False)
        # Assigning a type to the variable 'self' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArrayDimensions.test_2d.__dict__.__setitem__('stypy_localization', localization)
        TestArrayDimensions.test_2d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArrayDimensions.test_2d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArrayDimensions.test_2d.__dict__.__setitem__('stypy_function_name', 'TestArrayDimensions.test_2d')
        TestArrayDimensions.test_2d.__dict__.__setitem__('stypy_param_names_list', [])
        TestArrayDimensions.test_2d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArrayDimensions.test_2d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArrayDimensions.test_2d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArrayDimensions.test_2d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArrayDimensions.test_2d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArrayDimensions.test_2d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArrayDimensions.test_2d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_2d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_2d(...)' code ##################

        
        # Assigning a Call to a Name (line 140):
        
        # Call to readsav(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Call to join(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'DATA_PATH' (line 140)
        DATA_PATH_2211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 30), 'DATA_PATH', False)
        str_2212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 41), 'str', 'array_float32_2d.sav')
        # Processing the call keyword arguments (line 140)
        kwargs_2213 = {}
        # Getting the type of 'path' (line 140)
        path_2209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 140)
        join_2210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 20), path_2209, 'join')
        # Calling join(args, kwargs) (line 140)
        join_call_result_2214 = invoke(stypy.reporting.localization.Localization(__file__, 140, 20), join_2210, *[DATA_PATH_2211, str_2212], **kwargs_2213)
        
        # Processing the call keyword arguments (line 140)
        # Getting the type of 'False' (line 140)
        False_2215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 74), 'False', False)
        keyword_2216 = False_2215
        kwargs_2217 = {'verbose': keyword_2216}
        # Getting the type of 'readsav' (line 140)
        readsav_2208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 140)
        readsav_call_result_2218 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), readsav_2208, *[join_call_result_2214], **kwargs_2217)
        
        # Assigning a type to the variable 's' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 's', readsav_call_result_2218)
        
        # Call to assert_equal(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 's' (line 141)
        s_2220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 21), 's', False)
        # Obtaining the member 'array2d' of a type (line 141)
        array2d_2221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 21), s_2220, 'array2d')
        # Obtaining the member 'shape' of a type (line 141)
        shape_2222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 21), array2d_2221, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 141)
        tuple_2223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 141)
        # Adding element type (line 141)
        int_2224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 39), tuple_2223, int_2224)
        # Adding element type (line 141)
        int_2225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 39), tuple_2223, int_2225)
        
        # Processing the call keyword arguments (line 141)
        kwargs_2226 = {}
        # Getting the type of 'assert_equal' (line 141)
        assert_equal_2219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 141)
        assert_equal_call_result_2227 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), assert_equal_2219, *[shape_2222, tuple_2223], **kwargs_2226)
        
        
        # ################# End of 'test_2d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_2d' in the type store
        # Getting the type of 'stypy_return_type' (line 139)
        stypy_return_type_2228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2228)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_2d'
        return stypy_return_type_2228


    @norecursion
    def test_3d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_3d'
        module_type_store = module_type_store.open_function_context('test_3d', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArrayDimensions.test_3d.__dict__.__setitem__('stypy_localization', localization)
        TestArrayDimensions.test_3d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArrayDimensions.test_3d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArrayDimensions.test_3d.__dict__.__setitem__('stypy_function_name', 'TestArrayDimensions.test_3d')
        TestArrayDimensions.test_3d.__dict__.__setitem__('stypy_param_names_list', [])
        TestArrayDimensions.test_3d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArrayDimensions.test_3d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArrayDimensions.test_3d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArrayDimensions.test_3d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArrayDimensions.test_3d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArrayDimensions.test_3d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArrayDimensions.test_3d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_3d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_3d(...)' code ##################

        
        # Assigning a Call to a Name (line 144):
        
        # Call to readsav(...): (line 144)
        # Processing the call arguments (line 144)
        
        # Call to join(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'DATA_PATH' (line 144)
        DATA_PATH_2232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 30), 'DATA_PATH', False)
        str_2233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 41), 'str', 'array_float32_3d.sav')
        # Processing the call keyword arguments (line 144)
        kwargs_2234 = {}
        # Getting the type of 'path' (line 144)
        path_2230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 144)
        join_2231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 20), path_2230, 'join')
        # Calling join(args, kwargs) (line 144)
        join_call_result_2235 = invoke(stypy.reporting.localization.Localization(__file__, 144, 20), join_2231, *[DATA_PATH_2232, str_2233], **kwargs_2234)
        
        # Processing the call keyword arguments (line 144)
        # Getting the type of 'False' (line 144)
        False_2236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 74), 'False', False)
        keyword_2237 = False_2236
        kwargs_2238 = {'verbose': keyword_2237}
        # Getting the type of 'readsav' (line 144)
        readsav_2229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 144)
        readsav_call_result_2239 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), readsav_2229, *[join_call_result_2235], **kwargs_2238)
        
        # Assigning a type to the variable 's' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 's', readsav_call_result_2239)
        
        # Call to assert_equal(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 's' (line 145)
        s_2241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 21), 's', False)
        # Obtaining the member 'array3d' of a type (line 145)
        array3d_2242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 21), s_2241, 'array3d')
        # Obtaining the member 'shape' of a type (line 145)
        shape_2243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 21), array3d_2242, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 145)
        tuple_2244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 145)
        # Adding element type (line 145)
        int_2245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 39), tuple_2244, int_2245)
        # Adding element type (line 145)
        int_2246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 39), tuple_2244, int_2246)
        # Adding element type (line 145)
        int_2247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 39), tuple_2244, int_2247)
        
        # Processing the call keyword arguments (line 145)
        kwargs_2248 = {}
        # Getting the type of 'assert_equal' (line 145)
        assert_equal_2240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 145)
        assert_equal_call_result_2249 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), assert_equal_2240, *[shape_2243, tuple_2244], **kwargs_2248)
        
        
        # ################# End of 'test_3d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_3d' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_2250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2250)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_3d'
        return stypy_return_type_2250


    @norecursion
    def test_4d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_4d'
        module_type_store = module_type_store.open_function_context('test_4d', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArrayDimensions.test_4d.__dict__.__setitem__('stypy_localization', localization)
        TestArrayDimensions.test_4d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArrayDimensions.test_4d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArrayDimensions.test_4d.__dict__.__setitem__('stypy_function_name', 'TestArrayDimensions.test_4d')
        TestArrayDimensions.test_4d.__dict__.__setitem__('stypy_param_names_list', [])
        TestArrayDimensions.test_4d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArrayDimensions.test_4d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArrayDimensions.test_4d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArrayDimensions.test_4d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArrayDimensions.test_4d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArrayDimensions.test_4d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArrayDimensions.test_4d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_4d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_4d(...)' code ##################

        
        # Assigning a Call to a Name (line 148):
        
        # Call to readsav(...): (line 148)
        # Processing the call arguments (line 148)
        
        # Call to join(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'DATA_PATH' (line 148)
        DATA_PATH_2254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 30), 'DATA_PATH', False)
        str_2255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 41), 'str', 'array_float32_4d.sav')
        # Processing the call keyword arguments (line 148)
        kwargs_2256 = {}
        # Getting the type of 'path' (line 148)
        path_2252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 148)
        join_2253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 20), path_2252, 'join')
        # Calling join(args, kwargs) (line 148)
        join_call_result_2257 = invoke(stypy.reporting.localization.Localization(__file__, 148, 20), join_2253, *[DATA_PATH_2254, str_2255], **kwargs_2256)
        
        # Processing the call keyword arguments (line 148)
        # Getting the type of 'False' (line 148)
        False_2258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 74), 'False', False)
        keyword_2259 = False_2258
        kwargs_2260 = {'verbose': keyword_2259}
        # Getting the type of 'readsav' (line 148)
        readsav_2251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 148)
        readsav_call_result_2261 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), readsav_2251, *[join_call_result_2257], **kwargs_2260)
        
        # Assigning a type to the variable 's' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 's', readsav_call_result_2261)
        
        # Call to assert_equal(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 's' (line 149)
        s_2263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 21), 's', False)
        # Obtaining the member 'array4d' of a type (line 149)
        array4d_2264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 21), s_2263, 'array4d')
        # Obtaining the member 'shape' of a type (line 149)
        shape_2265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 21), array4d_2264, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 149)
        tuple_2266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 149)
        # Adding element type (line 149)
        int_2267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 39), tuple_2266, int_2267)
        # Adding element type (line 149)
        int_2268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 39), tuple_2266, int_2268)
        # Adding element type (line 149)
        int_2269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 39), tuple_2266, int_2269)
        # Adding element type (line 149)
        int_2270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 39), tuple_2266, int_2270)
        
        # Processing the call keyword arguments (line 149)
        kwargs_2271 = {}
        # Getting the type of 'assert_equal' (line 149)
        assert_equal_2262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 149)
        assert_equal_call_result_2272 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), assert_equal_2262, *[shape_2265, tuple_2266], **kwargs_2271)
        
        
        # ################# End of 'test_4d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_4d' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_2273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2273)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_4d'
        return stypy_return_type_2273


    @norecursion
    def test_5d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_5d'
        module_type_store = module_type_store.open_function_context('test_5d', 151, 4, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArrayDimensions.test_5d.__dict__.__setitem__('stypy_localization', localization)
        TestArrayDimensions.test_5d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArrayDimensions.test_5d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArrayDimensions.test_5d.__dict__.__setitem__('stypy_function_name', 'TestArrayDimensions.test_5d')
        TestArrayDimensions.test_5d.__dict__.__setitem__('stypy_param_names_list', [])
        TestArrayDimensions.test_5d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArrayDimensions.test_5d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArrayDimensions.test_5d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArrayDimensions.test_5d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArrayDimensions.test_5d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArrayDimensions.test_5d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArrayDimensions.test_5d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_5d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_5d(...)' code ##################

        
        # Assigning a Call to a Name (line 152):
        
        # Call to readsav(...): (line 152)
        # Processing the call arguments (line 152)
        
        # Call to join(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'DATA_PATH' (line 152)
        DATA_PATH_2277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 30), 'DATA_PATH', False)
        str_2278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 41), 'str', 'array_float32_5d.sav')
        # Processing the call keyword arguments (line 152)
        kwargs_2279 = {}
        # Getting the type of 'path' (line 152)
        path_2275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 152)
        join_2276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 20), path_2275, 'join')
        # Calling join(args, kwargs) (line 152)
        join_call_result_2280 = invoke(stypy.reporting.localization.Localization(__file__, 152, 20), join_2276, *[DATA_PATH_2277, str_2278], **kwargs_2279)
        
        # Processing the call keyword arguments (line 152)
        # Getting the type of 'False' (line 152)
        False_2281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 74), 'False', False)
        keyword_2282 = False_2281
        kwargs_2283 = {'verbose': keyword_2282}
        # Getting the type of 'readsav' (line 152)
        readsav_2274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 152)
        readsav_call_result_2284 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), readsav_2274, *[join_call_result_2280], **kwargs_2283)
        
        # Assigning a type to the variable 's' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 's', readsav_call_result_2284)
        
        # Call to assert_equal(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 's' (line 153)
        s_2286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 's', False)
        # Obtaining the member 'array5d' of a type (line 153)
        array5d_2287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 21), s_2286, 'array5d')
        # Obtaining the member 'shape' of a type (line 153)
        shape_2288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 21), array5d_2287, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 153)
        tuple_2289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 153)
        # Adding element type (line 153)
        int_2290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 39), tuple_2289, int_2290)
        # Adding element type (line 153)
        int_2291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 39), tuple_2289, int_2291)
        # Adding element type (line 153)
        int_2292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 39), tuple_2289, int_2292)
        # Adding element type (line 153)
        int_2293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 39), tuple_2289, int_2293)
        # Adding element type (line 153)
        int_2294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 39), tuple_2289, int_2294)
        
        # Processing the call keyword arguments (line 153)
        kwargs_2295 = {}
        # Getting the type of 'assert_equal' (line 153)
        assert_equal_2285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 153)
        assert_equal_call_result_2296 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), assert_equal_2285, *[shape_2288, tuple_2289], **kwargs_2295)
        
        
        # ################# End of 'test_5d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_5d' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_2297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2297)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_5d'
        return stypy_return_type_2297


    @norecursion
    def test_6d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_6d'
        module_type_store = module_type_store.open_function_context('test_6d', 155, 4, False)
        # Assigning a type to the variable 'self' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArrayDimensions.test_6d.__dict__.__setitem__('stypy_localization', localization)
        TestArrayDimensions.test_6d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArrayDimensions.test_6d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArrayDimensions.test_6d.__dict__.__setitem__('stypy_function_name', 'TestArrayDimensions.test_6d')
        TestArrayDimensions.test_6d.__dict__.__setitem__('stypy_param_names_list', [])
        TestArrayDimensions.test_6d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArrayDimensions.test_6d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArrayDimensions.test_6d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArrayDimensions.test_6d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArrayDimensions.test_6d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArrayDimensions.test_6d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArrayDimensions.test_6d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_6d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_6d(...)' code ##################

        
        # Assigning a Call to a Name (line 156):
        
        # Call to readsav(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Call to join(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'DATA_PATH' (line 156)
        DATA_PATH_2301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 30), 'DATA_PATH', False)
        str_2302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 41), 'str', 'array_float32_6d.sav')
        # Processing the call keyword arguments (line 156)
        kwargs_2303 = {}
        # Getting the type of 'path' (line 156)
        path_2299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 156)
        join_2300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 20), path_2299, 'join')
        # Calling join(args, kwargs) (line 156)
        join_call_result_2304 = invoke(stypy.reporting.localization.Localization(__file__, 156, 20), join_2300, *[DATA_PATH_2301, str_2302], **kwargs_2303)
        
        # Processing the call keyword arguments (line 156)
        # Getting the type of 'False' (line 156)
        False_2305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 74), 'False', False)
        keyword_2306 = False_2305
        kwargs_2307 = {'verbose': keyword_2306}
        # Getting the type of 'readsav' (line 156)
        readsav_2298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 156)
        readsav_call_result_2308 = invoke(stypy.reporting.localization.Localization(__file__, 156, 12), readsav_2298, *[join_call_result_2304], **kwargs_2307)
        
        # Assigning a type to the variable 's' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 's', readsav_call_result_2308)
        
        # Call to assert_equal(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 's' (line 157)
        s_2310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 's', False)
        # Obtaining the member 'array6d' of a type (line 157)
        array6d_2311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 21), s_2310, 'array6d')
        # Obtaining the member 'shape' of a type (line 157)
        shape_2312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 21), array6d_2311, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 157)
        tuple_2313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 157)
        # Adding element type (line 157)
        int_2314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 39), tuple_2313, int_2314)
        # Adding element type (line 157)
        int_2315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 39), tuple_2313, int_2315)
        # Adding element type (line 157)
        int_2316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 39), tuple_2313, int_2316)
        # Adding element type (line 157)
        int_2317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 39), tuple_2313, int_2317)
        # Adding element type (line 157)
        int_2318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 39), tuple_2313, int_2318)
        # Adding element type (line 157)
        int_2319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 39), tuple_2313, int_2319)
        
        # Processing the call keyword arguments (line 157)
        kwargs_2320 = {}
        # Getting the type of 'assert_equal' (line 157)
        assert_equal_2309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 157)
        assert_equal_call_result_2321 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), assert_equal_2309, *[shape_2312, tuple_2313], **kwargs_2320)
        
        
        # ################# End of 'test_6d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_6d' in the type store
        # Getting the type of 'stypy_return_type' (line 155)
        stypy_return_type_2322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2322)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_6d'
        return stypy_return_type_2322


    @norecursion
    def test_7d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_7d'
        module_type_store = module_type_store.open_function_context('test_7d', 159, 4, False)
        # Assigning a type to the variable 'self' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArrayDimensions.test_7d.__dict__.__setitem__('stypy_localization', localization)
        TestArrayDimensions.test_7d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArrayDimensions.test_7d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArrayDimensions.test_7d.__dict__.__setitem__('stypy_function_name', 'TestArrayDimensions.test_7d')
        TestArrayDimensions.test_7d.__dict__.__setitem__('stypy_param_names_list', [])
        TestArrayDimensions.test_7d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArrayDimensions.test_7d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArrayDimensions.test_7d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArrayDimensions.test_7d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArrayDimensions.test_7d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArrayDimensions.test_7d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArrayDimensions.test_7d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_7d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_7d(...)' code ##################

        
        # Assigning a Call to a Name (line 160):
        
        # Call to readsav(...): (line 160)
        # Processing the call arguments (line 160)
        
        # Call to join(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'DATA_PATH' (line 160)
        DATA_PATH_2326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 30), 'DATA_PATH', False)
        str_2327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 41), 'str', 'array_float32_7d.sav')
        # Processing the call keyword arguments (line 160)
        kwargs_2328 = {}
        # Getting the type of 'path' (line 160)
        path_2324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 160)
        join_2325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 20), path_2324, 'join')
        # Calling join(args, kwargs) (line 160)
        join_call_result_2329 = invoke(stypy.reporting.localization.Localization(__file__, 160, 20), join_2325, *[DATA_PATH_2326, str_2327], **kwargs_2328)
        
        # Processing the call keyword arguments (line 160)
        # Getting the type of 'False' (line 160)
        False_2330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 74), 'False', False)
        keyword_2331 = False_2330
        kwargs_2332 = {'verbose': keyword_2331}
        # Getting the type of 'readsav' (line 160)
        readsav_2323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 160)
        readsav_call_result_2333 = invoke(stypy.reporting.localization.Localization(__file__, 160, 12), readsav_2323, *[join_call_result_2329], **kwargs_2332)
        
        # Assigning a type to the variable 's' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 's', readsav_call_result_2333)
        
        # Call to assert_equal(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 's' (line 161)
        s_2335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 21), 's', False)
        # Obtaining the member 'array7d' of a type (line 161)
        array7d_2336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 21), s_2335, 'array7d')
        # Obtaining the member 'shape' of a type (line 161)
        shape_2337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 21), array7d_2336, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 161)
        tuple_2338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 161)
        # Adding element type (line 161)
        int_2339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 39), tuple_2338, int_2339)
        # Adding element type (line 161)
        int_2340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 39), tuple_2338, int_2340)
        # Adding element type (line 161)
        int_2341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 39), tuple_2338, int_2341)
        # Adding element type (line 161)
        int_2342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 39), tuple_2338, int_2342)
        # Adding element type (line 161)
        int_2343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 39), tuple_2338, int_2343)
        # Adding element type (line 161)
        int_2344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 39), tuple_2338, int_2344)
        # Adding element type (line 161)
        int_2345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 39), tuple_2338, int_2345)
        
        # Processing the call keyword arguments (line 161)
        kwargs_2346 = {}
        # Getting the type of 'assert_equal' (line 161)
        assert_equal_2334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 161)
        assert_equal_call_result_2347 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), assert_equal_2334, *[shape_2337, tuple_2338], **kwargs_2346)
        
        
        # ################# End of 'test_7d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_7d' in the type store
        # Getting the type of 'stypy_return_type' (line 159)
        stypy_return_type_2348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2348)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_7d'
        return stypy_return_type_2348


    @norecursion
    def test_8d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_8d'
        module_type_store = module_type_store.open_function_context('test_8d', 163, 4, False)
        # Assigning a type to the variable 'self' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestArrayDimensions.test_8d.__dict__.__setitem__('stypy_localization', localization)
        TestArrayDimensions.test_8d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestArrayDimensions.test_8d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestArrayDimensions.test_8d.__dict__.__setitem__('stypy_function_name', 'TestArrayDimensions.test_8d')
        TestArrayDimensions.test_8d.__dict__.__setitem__('stypy_param_names_list', [])
        TestArrayDimensions.test_8d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestArrayDimensions.test_8d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestArrayDimensions.test_8d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestArrayDimensions.test_8d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestArrayDimensions.test_8d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestArrayDimensions.test_8d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArrayDimensions.test_8d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_8d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_8d(...)' code ##################

        
        # Assigning a Call to a Name (line 164):
        
        # Call to readsav(...): (line 164)
        # Processing the call arguments (line 164)
        
        # Call to join(...): (line 164)
        # Processing the call arguments (line 164)
        # Getting the type of 'DATA_PATH' (line 164)
        DATA_PATH_2352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 30), 'DATA_PATH', False)
        str_2353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 41), 'str', 'array_float32_8d.sav')
        # Processing the call keyword arguments (line 164)
        kwargs_2354 = {}
        # Getting the type of 'path' (line 164)
        path_2350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 164)
        join_2351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 20), path_2350, 'join')
        # Calling join(args, kwargs) (line 164)
        join_call_result_2355 = invoke(stypy.reporting.localization.Localization(__file__, 164, 20), join_2351, *[DATA_PATH_2352, str_2353], **kwargs_2354)
        
        # Processing the call keyword arguments (line 164)
        # Getting the type of 'False' (line 164)
        False_2356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 74), 'False', False)
        keyword_2357 = False_2356
        kwargs_2358 = {'verbose': keyword_2357}
        # Getting the type of 'readsav' (line 164)
        readsav_2349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 164)
        readsav_call_result_2359 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), readsav_2349, *[join_call_result_2355], **kwargs_2358)
        
        # Assigning a type to the variable 's' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 's', readsav_call_result_2359)
        
        # Call to assert_equal(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 's' (line 165)
        s_2361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 21), 's', False)
        # Obtaining the member 'array8d' of a type (line 165)
        array8d_2362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 21), s_2361, 'array8d')
        # Obtaining the member 'shape' of a type (line 165)
        shape_2363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 21), array8d_2362, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 165)
        tuple_2364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 165)
        # Adding element type (line 165)
        int_2365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 39), tuple_2364, int_2365)
        # Adding element type (line 165)
        int_2366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 39), tuple_2364, int_2366)
        # Adding element type (line 165)
        int_2367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 39), tuple_2364, int_2367)
        # Adding element type (line 165)
        int_2368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 39), tuple_2364, int_2368)
        # Adding element type (line 165)
        int_2369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 39), tuple_2364, int_2369)
        # Adding element type (line 165)
        int_2370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 39), tuple_2364, int_2370)
        # Adding element type (line 165)
        int_2371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 39), tuple_2364, int_2371)
        # Adding element type (line 165)
        int_2372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 39), tuple_2364, int_2372)
        
        # Processing the call keyword arguments (line 165)
        kwargs_2373 = {}
        # Getting the type of 'assert_equal' (line 165)
        assert_equal_2360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 165)
        assert_equal_call_result_2374 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), assert_equal_2360, *[shape_2363, tuple_2364], **kwargs_2373)
        
        
        # ################# End of 'test_8d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_8d' in the type store
        # Getting the type of 'stypy_return_type' (line 163)
        stypy_return_type_2375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2375)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_8d'
        return stypy_return_type_2375


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 132, 0, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestArrayDimensions.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestArrayDimensions' (line 132)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'TestArrayDimensions', TestArrayDimensions)
# Declaration of the 'TestStructures' class

class TestStructures:

    @norecursion
    def test_scalars(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scalars'
        module_type_store = module_type_store.open_function_context('test_scalars', 170, 4, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestStructures.test_scalars.__dict__.__setitem__('stypy_localization', localization)
        TestStructures.test_scalars.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestStructures.test_scalars.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestStructures.test_scalars.__dict__.__setitem__('stypy_function_name', 'TestStructures.test_scalars')
        TestStructures.test_scalars.__dict__.__setitem__('stypy_param_names_list', [])
        TestStructures.test_scalars.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestStructures.test_scalars.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestStructures.test_scalars.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestStructures.test_scalars.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestStructures.test_scalars.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestStructures.test_scalars.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestStructures.test_scalars', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scalars', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scalars(...)' code ##################

        
        # Assigning a Call to a Name (line 171):
        
        # Call to readsav(...): (line 171)
        # Processing the call arguments (line 171)
        
        # Call to join(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'DATA_PATH' (line 171)
        DATA_PATH_2379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 30), 'DATA_PATH', False)
        str_2380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 41), 'str', 'struct_scalars.sav')
        # Processing the call keyword arguments (line 171)
        kwargs_2381 = {}
        # Getting the type of 'path' (line 171)
        path_2377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 171)
        join_2378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 20), path_2377, 'join')
        # Calling join(args, kwargs) (line 171)
        join_call_result_2382 = invoke(stypy.reporting.localization.Localization(__file__, 171, 20), join_2378, *[DATA_PATH_2379, str_2380], **kwargs_2381)
        
        # Processing the call keyword arguments (line 171)
        # Getting the type of 'False' (line 171)
        False_2383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 72), 'False', False)
        keyword_2384 = False_2383
        kwargs_2385 = {'verbose': keyword_2384}
        # Getting the type of 'readsav' (line 171)
        readsav_2376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 171)
        readsav_call_result_2386 = invoke(stypy.reporting.localization.Localization(__file__, 171, 12), readsav_2376, *[join_call_result_2382], **kwargs_2385)
        
        # Assigning a type to the variable 's' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 's', readsav_call_result_2386)
        
        # Call to assert_identical(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 's' (line 172)
        s_2388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 25), 's', False)
        # Obtaining the member 'scalars' of a type (line 172)
        scalars_2389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 25), s_2388, 'scalars')
        # Obtaining the member 'a' of a type (line 172)
        a_2390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 25), scalars_2389, 'a')
        
        # Call to array(...): (line 172)
        # Processing the call arguments (line 172)
        
        # Call to int16(...): (line 172)
        # Processing the call arguments (line 172)
        int_2395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 56), 'int')
        # Processing the call keyword arguments (line 172)
        kwargs_2396 = {}
        # Getting the type of 'np' (line 172)
        np_2393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 47), 'np', False)
        # Obtaining the member 'int16' of a type (line 172)
        int16_2394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 47), np_2393, 'int16')
        # Calling int16(args, kwargs) (line 172)
        int16_call_result_2397 = invoke(stypy.reporting.localization.Localization(__file__, 172, 47), int16_2394, *[int_2395], **kwargs_2396)
        
        # Processing the call keyword arguments (line 172)
        kwargs_2398 = {}
        # Getting the type of 'np' (line 172)
        np_2391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 38), 'np', False)
        # Obtaining the member 'array' of a type (line 172)
        array_2392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 38), np_2391, 'array')
        # Calling array(args, kwargs) (line 172)
        array_call_result_2399 = invoke(stypy.reporting.localization.Localization(__file__, 172, 38), array_2392, *[int16_call_result_2397], **kwargs_2398)
        
        # Processing the call keyword arguments (line 172)
        kwargs_2400 = {}
        # Getting the type of 'assert_identical' (line 172)
        assert_identical_2387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 172)
        assert_identical_call_result_2401 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), assert_identical_2387, *[a_2390, array_call_result_2399], **kwargs_2400)
        
        
        # Call to assert_identical(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 's' (line 173)
        s_2403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 25), 's', False)
        # Obtaining the member 'scalars' of a type (line 173)
        scalars_2404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 25), s_2403, 'scalars')
        # Obtaining the member 'b' of a type (line 173)
        b_2405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 25), scalars_2404, 'b')
        
        # Call to array(...): (line 173)
        # Processing the call arguments (line 173)
        
        # Call to int32(...): (line 173)
        # Processing the call arguments (line 173)
        int_2410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 56), 'int')
        # Processing the call keyword arguments (line 173)
        kwargs_2411 = {}
        # Getting the type of 'np' (line 173)
        np_2408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 47), 'np', False)
        # Obtaining the member 'int32' of a type (line 173)
        int32_2409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 47), np_2408, 'int32')
        # Calling int32(args, kwargs) (line 173)
        int32_call_result_2412 = invoke(stypy.reporting.localization.Localization(__file__, 173, 47), int32_2409, *[int_2410], **kwargs_2411)
        
        # Processing the call keyword arguments (line 173)
        kwargs_2413 = {}
        # Getting the type of 'np' (line 173)
        np_2406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 38), 'np', False)
        # Obtaining the member 'array' of a type (line 173)
        array_2407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 38), np_2406, 'array')
        # Calling array(args, kwargs) (line 173)
        array_call_result_2414 = invoke(stypy.reporting.localization.Localization(__file__, 173, 38), array_2407, *[int32_call_result_2412], **kwargs_2413)
        
        # Processing the call keyword arguments (line 173)
        kwargs_2415 = {}
        # Getting the type of 'assert_identical' (line 173)
        assert_identical_2402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 173)
        assert_identical_call_result_2416 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), assert_identical_2402, *[b_2405, array_call_result_2414], **kwargs_2415)
        
        
        # Call to assert_identical(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 's' (line 174)
        s_2418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 25), 's', False)
        # Obtaining the member 'scalars' of a type (line 174)
        scalars_2419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 25), s_2418, 'scalars')
        # Obtaining the member 'c' of a type (line 174)
        c_2420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 25), scalars_2419, 'c')
        
        # Call to array(...): (line 174)
        # Processing the call arguments (line 174)
        
        # Call to float32(...): (line 174)
        # Processing the call arguments (line 174)
        float_2425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 58), 'float')
        # Processing the call keyword arguments (line 174)
        kwargs_2426 = {}
        # Getting the type of 'np' (line 174)
        np_2423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 47), 'np', False)
        # Obtaining the member 'float32' of a type (line 174)
        float32_2424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 47), np_2423, 'float32')
        # Calling float32(args, kwargs) (line 174)
        float32_call_result_2427 = invoke(stypy.reporting.localization.Localization(__file__, 174, 47), float32_2424, *[float_2425], **kwargs_2426)
        
        # Processing the call keyword arguments (line 174)
        kwargs_2428 = {}
        # Getting the type of 'np' (line 174)
        np_2421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 38), 'np', False)
        # Obtaining the member 'array' of a type (line 174)
        array_2422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 38), np_2421, 'array')
        # Calling array(args, kwargs) (line 174)
        array_call_result_2429 = invoke(stypy.reporting.localization.Localization(__file__, 174, 38), array_2422, *[float32_call_result_2427], **kwargs_2428)
        
        # Processing the call keyword arguments (line 174)
        kwargs_2430 = {}
        # Getting the type of 'assert_identical' (line 174)
        assert_identical_2417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 174)
        assert_identical_call_result_2431 = invoke(stypy.reporting.localization.Localization(__file__, 174, 8), assert_identical_2417, *[c_2420, array_call_result_2429], **kwargs_2430)
        
        
        # Call to assert_identical(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 's' (line 175)
        s_2433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 's', False)
        # Obtaining the member 'scalars' of a type (line 175)
        scalars_2434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 25), s_2433, 'scalars')
        # Obtaining the member 'd' of a type (line 175)
        d_2435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 25), scalars_2434, 'd')
        
        # Call to array(...): (line 175)
        # Processing the call arguments (line 175)
        
        # Call to float64(...): (line 175)
        # Processing the call arguments (line 175)
        float_2440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 58), 'float')
        # Processing the call keyword arguments (line 175)
        kwargs_2441 = {}
        # Getting the type of 'np' (line 175)
        np_2438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 47), 'np', False)
        # Obtaining the member 'float64' of a type (line 175)
        float64_2439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 47), np_2438, 'float64')
        # Calling float64(args, kwargs) (line 175)
        float64_call_result_2442 = invoke(stypy.reporting.localization.Localization(__file__, 175, 47), float64_2439, *[float_2440], **kwargs_2441)
        
        # Processing the call keyword arguments (line 175)
        kwargs_2443 = {}
        # Getting the type of 'np' (line 175)
        np_2436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 38), 'np', False)
        # Obtaining the member 'array' of a type (line 175)
        array_2437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 38), np_2436, 'array')
        # Calling array(args, kwargs) (line 175)
        array_call_result_2444 = invoke(stypy.reporting.localization.Localization(__file__, 175, 38), array_2437, *[float64_call_result_2442], **kwargs_2443)
        
        # Processing the call keyword arguments (line 175)
        kwargs_2445 = {}
        # Getting the type of 'assert_identical' (line 175)
        assert_identical_2432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 175)
        assert_identical_call_result_2446 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), assert_identical_2432, *[d_2435, array_call_result_2444], **kwargs_2445)
        
        
        # Call to assert_identical(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 's' (line 176)
        s_2448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 25), 's', False)
        # Obtaining the member 'scalars' of a type (line 176)
        scalars_2449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 25), s_2448, 'scalars')
        # Obtaining the member 'e' of a type (line 176)
        e_2450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 25), scalars_2449, 'e')
        
        # Call to array(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_2453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        str_2454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 48), 'str', 'spam')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 47), list_2453, str_2454)
        
        # Processing the call keyword arguments (line 176)
        # Getting the type of 'object' (line 176)
        object_2455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 64), 'object', False)
        keyword_2456 = object_2455
        kwargs_2457 = {'dtype': keyword_2456}
        # Getting the type of 'np' (line 176)
        np_2451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 38), 'np', False)
        # Obtaining the member 'array' of a type (line 176)
        array_2452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 38), np_2451, 'array')
        # Calling array(args, kwargs) (line 176)
        array_call_result_2458 = invoke(stypy.reporting.localization.Localization(__file__, 176, 38), array_2452, *[list_2453], **kwargs_2457)
        
        # Processing the call keyword arguments (line 176)
        kwargs_2459 = {}
        # Getting the type of 'assert_identical' (line 176)
        assert_identical_2447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 176)
        assert_identical_call_result_2460 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), assert_identical_2447, *[e_2450, array_call_result_2458], **kwargs_2459)
        
        
        # Call to assert_identical(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 's' (line 177)
        s_2462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 25), 's', False)
        # Obtaining the member 'scalars' of a type (line 177)
        scalars_2463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 25), s_2462, 'scalars')
        # Obtaining the member 'f' of a type (line 177)
        f_2464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 25), scalars_2463, 'f')
        
        # Call to array(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Call to complex64(...): (line 177)
        # Processing the call arguments (line 177)
        float_2469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 60), 'float')
        complex_2470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 64), 'complex')
        # Applying the binary operator '+' (line 177)
        result_add_2471 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 60), '+', float_2469, complex_2470)
        
        # Processing the call keyword arguments (line 177)
        kwargs_2472 = {}
        # Getting the type of 'np' (line 177)
        np_2467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 47), 'np', False)
        # Obtaining the member 'complex64' of a type (line 177)
        complex64_2468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 47), np_2467, 'complex64')
        # Calling complex64(args, kwargs) (line 177)
        complex64_call_result_2473 = invoke(stypy.reporting.localization.Localization(__file__, 177, 47), complex64_2468, *[result_add_2471], **kwargs_2472)
        
        # Processing the call keyword arguments (line 177)
        kwargs_2474 = {}
        # Getting the type of 'np' (line 177)
        np_2465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 38), 'np', False)
        # Obtaining the member 'array' of a type (line 177)
        array_2466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 38), np_2465, 'array')
        # Calling array(args, kwargs) (line 177)
        array_call_result_2475 = invoke(stypy.reporting.localization.Localization(__file__, 177, 38), array_2466, *[complex64_call_result_2473], **kwargs_2474)
        
        # Processing the call keyword arguments (line 177)
        kwargs_2476 = {}
        # Getting the type of 'assert_identical' (line 177)
        assert_identical_2461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 177)
        assert_identical_call_result_2477 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), assert_identical_2461, *[f_2464, array_call_result_2475], **kwargs_2476)
        
        
        # ################# End of 'test_scalars(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scalars' in the type store
        # Getting the type of 'stypy_return_type' (line 170)
        stypy_return_type_2478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2478)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scalars'
        return stypy_return_type_2478


    @norecursion
    def test_scalars_replicated(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scalars_replicated'
        module_type_store = module_type_store.open_function_context('test_scalars_replicated', 179, 4, False)
        # Assigning a type to the variable 'self' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestStructures.test_scalars_replicated.__dict__.__setitem__('stypy_localization', localization)
        TestStructures.test_scalars_replicated.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestStructures.test_scalars_replicated.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestStructures.test_scalars_replicated.__dict__.__setitem__('stypy_function_name', 'TestStructures.test_scalars_replicated')
        TestStructures.test_scalars_replicated.__dict__.__setitem__('stypy_param_names_list', [])
        TestStructures.test_scalars_replicated.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestStructures.test_scalars_replicated.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestStructures.test_scalars_replicated.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestStructures.test_scalars_replicated.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestStructures.test_scalars_replicated.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestStructures.test_scalars_replicated.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestStructures.test_scalars_replicated', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scalars_replicated', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scalars_replicated(...)' code ##################

        
        # Assigning a Call to a Name (line 180):
        
        # Call to readsav(...): (line 180)
        # Processing the call arguments (line 180)
        
        # Call to join(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'DATA_PATH' (line 180)
        DATA_PATH_2482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 30), 'DATA_PATH', False)
        str_2483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 41), 'str', 'struct_scalars_replicated.sav')
        # Processing the call keyword arguments (line 180)
        kwargs_2484 = {}
        # Getting the type of 'path' (line 180)
        path_2480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 180)
        join_2481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 20), path_2480, 'join')
        # Calling join(args, kwargs) (line 180)
        join_call_result_2485 = invoke(stypy.reporting.localization.Localization(__file__, 180, 20), join_2481, *[DATA_PATH_2482, str_2483], **kwargs_2484)
        
        # Processing the call keyword arguments (line 180)
        # Getting the type of 'False' (line 180)
        False_2486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 83), 'False', False)
        keyword_2487 = False_2486
        kwargs_2488 = {'verbose': keyword_2487}
        # Getting the type of 'readsav' (line 180)
        readsav_2479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 180)
        readsav_call_result_2489 = invoke(stypy.reporting.localization.Localization(__file__, 180, 12), readsav_2479, *[join_call_result_2485], **kwargs_2488)
        
        # Assigning a type to the variable 's' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 's', readsav_call_result_2489)
        
        # Call to assert_identical(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 's' (line 181)
        s_2491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 25), 's', False)
        # Obtaining the member 'scalars_rep' of a type (line 181)
        scalars_rep_2492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 25), s_2491, 'scalars_rep')
        # Obtaining the member 'a' of a type (line 181)
        a_2493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 25), scalars_rep_2492, 'a')
        
        # Call to repeat(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Call to int16(...): (line 181)
        # Processing the call arguments (line 181)
        int_2498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 61), 'int')
        # Processing the call keyword arguments (line 181)
        kwargs_2499 = {}
        # Getting the type of 'np' (line 181)
        np_2496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 52), 'np', False)
        # Obtaining the member 'int16' of a type (line 181)
        int16_2497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 52), np_2496, 'int16')
        # Calling int16(args, kwargs) (line 181)
        int16_call_result_2500 = invoke(stypy.reporting.localization.Localization(__file__, 181, 52), int16_2497, *[int_2498], **kwargs_2499)
        
        int_2501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 65), 'int')
        # Processing the call keyword arguments (line 181)
        kwargs_2502 = {}
        # Getting the type of 'np' (line 181)
        np_2494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 42), 'np', False)
        # Obtaining the member 'repeat' of a type (line 181)
        repeat_2495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 42), np_2494, 'repeat')
        # Calling repeat(args, kwargs) (line 181)
        repeat_call_result_2503 = invoke(stypy.reporting.localization.Localization(__file__, 181, 42), repeat_2495, *[int16_call_result_2500, int_2501], **kwargs_2502)
        
        # Processing the call keyword arguments (line 181)
        kwargs_2504 = {}
        # Getting the type of 'assert_identical' (line 181)
        assert_identical_2490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 181)
        assert_identical_call_result_2505 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), assert_identical_2490, *[a_2493, repeat_call_result_2503], **kwargs_2504)
        
        
        # Call to assert_identical(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 's' (line 182)
        s_2507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 25), 's', False)
        # Obtaining the member 'scalars_rep' of a type (line 182)
        scalars_rep_2508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 25), s_2507, 'scalars_rep')
        # Obtaining the member 'b' of a type (line 182)
        b_2509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 25), scalars_rep_2508, 'b')
        
        # Call to repeat(...): (line 182)
        # Processing the call arguments (line 182)
        
        # Call to int32(...): (line 182)
        # Processing the call arguments (line 182)
        int_2514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 61), 'int')
        # Processing the call keyword arguments (line 182)
        kwargs_2515 = {}
        # Getting the type of 'np' (line 182)
        np_2512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 52), 'np', False)
        # Obtaining the member 'int32' of a type (line 182)
        int32_2513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 52), np_2512, 'int32')
        # Calling int32(args, kwargs) (line 182)
        int32_call_result_2516 = invoke(stypy.reporting.localization.Localization(__file__, 182, 52), int32_2513, *[int_2514], **kwargs_2515)
        
        int_2517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 65), 'int')
        # Processing the call keyword arguments (line 182)
        kwargs_2518 = {}
        # Getting the type of 'np' (line 182)
        np_2510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 42), 'np', False)
        # Obtaining the member 'repeat' of a type (line 182)
        repeat_2511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 42), np_2510, 'repeat')
        # Calling repeat(args, kwargs) (line 182)
        repeat_call_result_2519 = invoke(stypy.reporting.localization.Localization(__file__, 182, 42), repeat_2511, *[int32_call_result_2516, int_2517], **kwargs_2518)
        
        # Processing the call keyword arguments (line 182)
        kwargs_2520 = {}
        # Getting the type of 'assert_identical' (line 182)
        assert_identical_2506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 182)
        assert_identical_call_result_2521 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), assert_identical_2506, *[b_2509, repeat_call_result_2519], **kwargs_2520)
        
        
        # Call to assert_identical(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 's' (line 183)
        s_2523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 25), 's', False)
        # Obtaining the member 'scalars_rep' of a type (line 183)
        scalars_rep_2524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 25), s_2523, 'scalars_rep')
        # Obtaining the member 'c' of a type (line 183)
        c_2525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 25), scalars_rep_2524, 'c')
        
        # Call to repeat(...): (line 183)
        # Processing the call arguments (line 183)
        
        # Call to float32(...): (line 183)
        # Processing the call arguments (line 183)
        float_2530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 63), 'float')
        # Processing the call keyword arguments (line 183)
        kwargs_2531 = {}
        # Getting the type of 'np' (line 183)
        np_2528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 52), 'np', False)
        # Obtaining the member 'float32' of a type (line 183)
        float32_2529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 52), np_2528, 'float32')
        # Calling float32(args, kwargs) (line 183)
        float32_call_result_2532 = invoke(stypy.reporting.localization.Localization(__file__, 183, 52), float32_2529, *[float_2530], **kwargs_2531)
        
        int_2533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 68), 'int')
        # Processing the call keyword arguments (line 183)
        kwargs_2534 = {}
        # Getting the type of 'np' (line 183)
        np_2526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 42), 'np', False)
        # Obtaining the member 'repeat' of a type (line 183)
        repeat_2527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 42), np_2526, 'repeat')
        # Calling repeat(args, kwargs) (line 183)
        repeat_call_result_2535 = invoke(stypy.reporting.localization.Localization(__file__, 183, 42), repeat_2527, *[float32_call_result_2532, int_2533], **kwargs_2534)
        
        # Processing the call keyword arguments (line 183)
        kwargs_2536 = {}
        # Getting the type of 'assert_identical' (line 183)
        assert_identical_2522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 183)
        assert_identical_call_result_2537 = invoke(stypy.reporting.localization.Localization(__file__, 183, 8), assert_identical_2522, *[c_2525, repeat_call_result_2535], **kwargs_2536)
        
        
        # Call to assert_identical(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 's' (line 184)
        s_2539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 25), 's', False)
        # Obtaining the member 'scalars_rep' of a type (line 184)
        scalars_rep_2540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 25), s_2539, 'scalars_rep')
        # Obtaining the member 'd' of a type (line 184)
        d_2541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 25), scalars_rep_2540, 'd')
        
        # Call to repeat(...): (line 184)
        # Processing the call arguments (line 184)
        
        # Call to float64(...): (line 184)
        # Processing the call arguments (line 184)
        float_2546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 63), 'float')
        # Processing the call keyword arguments (line 184)
        kwargs_2547 = {}
        # Getting the type of 'np' (line 184)
        np_2544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 52), 'np', False)
        # Obtaining the member 'float64' of a type (line 184)
        float64_2545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 52), np_2544, 'float64')
        # Calling float64(args, kwargs) (line 184)
        float64_call_result_2548 = invoke(stypy.reporting.localization.Localization(__file__, 184, 52), float64_2545, *[float_2546], **kwargs_2547)
        
        int_2549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 68), 'int')
        # Processing the call keyword arguments (line 184)
        kwargs_2550 = {}
        # Getting the type of 'np' (line 184)
        np_2542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 42), 'np', False)
        # Obtaining the member 'repeat' of a type (line 184)
        repeat_2543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 42), np_2542, 'repeat')
        # Calling repeat(args, kwargs) (line 184)
        repeat_call_result_2551 = invoke(stypy.reporting.localization.Localization(__file__, 184, 42), repeat_2543, *[float64_call_result_2548, int_2549], **kwargs_2550)
        
        # Processing the call keyword arguments (line 184)
        kwargs_2552 = {}
        # Getting the type of 'assert_identical' (line 184)
        assert_identical_2538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 184)
        assert_identical_call_result_2553 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), assert_identical_2538, *[d_2541, repeat_call_result_2551], **kwargs_2552)
        
        
        # Call to assert_identical(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 's' (line 185)
        s_2555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 25), 's', False)
        # Obtaining the member 'scalars_rep' of a type (line 185)
        scalars_rep_2556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 25), s_2555, 'scalars_rep')
        # Obtaining the member 'e' of a type (line 185)
        e_2557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 25), scalars_rep_2556, 'e')
        
        # Call to astype(...): (line 185)
        # Processing the call arguments (line 185)
        # Getting the type of 'object' (line 185)
        object_2565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 71), 'object', False)
        # Processing the call keyword arguments (line 185)
        kwargs_2566 = {}
        
        # Call to repeat(...): (line 185)
        # Processing the call arguments (line 185)
        str_2560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 52), 'str', 'spam')
        int_2561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 61), 'int')
        # Processing the call keyword arguments (line 185)
        kwargs_2562 = {}
        # Getting the type of 'np' (line 185)
        np_2558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 42), 'np', False)
        # Obtaining the member 'repeat' of a type (line 185)
        repeat_2559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 42), np_2558, 'repeat')
        # Calling repeat(args, kwargs) (line 185)
        repeat_call_result_2563 = invoke(stypy.reporting.localization.Localization(__file__, 185, 42), repeat_2559, *[str_2560, int_2561], **kwargs_2562)
        
        # Obtaining the member 'astype' of a type (line 185)
        astype_2564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 42), repeat_call_result_2563, 'astype')
        # Calling astype(args, kwargs) (line 185)
        astype_call_result_2567 = invoke(stypy.reporting.localization.Localization(__file__, 185, 42), astype_2564, *[object_2565], **kwargs_2566)
        
        # Processing the call keyword arguments (line 185)
        kwargs_2568 = {}
        # Getting the type of 'assert_identical' (line 185)
        assert_identical_2554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 185)
        assert_identical_call_result_2569 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), assert_identical_2554, *[e_2557, astype_call_result_2567], **kwargs_2568)
        
        
        # Call to assert_identical(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 's' (line 186)
        s_2571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 's', False)
        # Obtaining the member 'scalars_rep' of a type (line 186)
        scalars_rep_2572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 25), s_2571, 'scalars_rep')
        # Obtaining the member 'f' of a type (line 186)
        f_2573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 25), scalars_rep_2572, 'f')
        
        # Call to repeat(...): (line 186)
        # Processing the call arguments (line 186)
        
        # Call to complex64(...): (line 186)
        # Processing the call arguments (line 186)
        float_2578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 65), 'float')
        complex_2579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 69), 'complex')
        # Applying the binary operator '+' (line 186)
        result_add_2580 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 65), '+', float_2578, complex_2579)
        
        # Processing the call keyword arguments (line 186)
        kwargs_2581 = {}
        # Getting the type of 'np' (line 186)
        np_2576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 52), 'np', False)
        # Obtaining the member 'complex64' of a type (line 186)
        complex64_2577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 52), np_2576, 'complex64')
        # Calling complex64(args, kwargs) (line 186)
        complex64_call_result_2582 = invoke(stypy.reporting.localization.Localization(__file__, 186, 52), complex64_2577, *[result_add_2580], **kwargs_2581)
        
        int_2583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 74), 'int')
        # Processing the call keyword arguments (line 186)
        kwargs_2584 = {}
        # Getting the type of 'np' (line 186)
        np_2574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 42), 'np', False)
        # Obtaining the member 'repeat' of a type (line 186)
        repeat_2575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 42), np_2574, 'repeat')
        # Calling repeat(args, kwargs) (line 186)
        repeat_call_result_2585 = invoke(stypy.reporting.localization.Localization(__file__, 186, 42), repeat_2575, *[complex64_call_result_2582, int_2583], **kwargs_2584)
        
        # Processing the call keyword arguments (line 186)
        kwargs_2586 = {}
        # Getting the type of 'assert_identical' (line 186)
        assert_identical_2570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 186)
        assert_identical_call_result_2587 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), assert_identical_2570, *[f_2573, repeat_call_result_2585], **kwargs_2586)
        
        
        # ################# End of 'test_scalars_replicated(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scalars_replicated' in the type store
        # Getting the type of 'stypy_return_type' (line 179)
        stypy_return_type_2588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2588)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scalars_replicated'
        return stypy_return_type_2588


    @norecursion
    def test_scalars_replicated_3d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scalars_replicated_3d'
        module_type_store = module_type_store.open_function_context('test_scalars_replicated_3d', 188, 4, False)
        # Assigning a type to the variable 'self' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestStructures.test_scalars_replicated_3d.__dict__.__setitem__('stypy_localization', localization)
        TestStructures.test_scalars_replicated_3d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestStructures.test_scalars_replicated_3d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestStructures.test_scalars_replicated_3d.__dict__.__setitem__('stypy_function_name', 'TestStructures.test_scalars_replicated_3d')
        TestStructures.test_scalars_replicated_3d.__dict__.__setitem__('stypy_param_names_list', [])
        TestStructures.test_scalars_replicated_3d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestStructures.test_scalars_replicated_3d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestStructures.test_scalars_replicated_3d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestStructures.test_scalars_replicated_3d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestStructures.test_scalars_replicated_3d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestStructures.test_scalars_replicated_3d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestStructures.test_scalars_replicated_3d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scalars_replicated_3d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scalars_replicated_3d(...)' code ##################

        
        # Assigning a Call to a Name (line 189):
        
        # Call to readsav(...): (line 189)
        # Processing the call arguments (line 189)
        
        # Call to join(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'DATA_PATH' (line 189)
        DATA_PATH_2592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 30), 'DATA_PATH', False)
        str_2593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 41), 'str', 'struct_scalars_replicated_3d.sav')
        # Processing the call keyword arguments (line 189)
        kwargs_2594 = {}
        # Getting the type of 'path' (line 189)
        path_2590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 189)
        join_2591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 20), path_2590, 'join')
        # Calling join(args, kwargs) (line 189)
        join_call_result_2595 = invoke(stypy.reporting.localization.Localization(__file__, 189, 20), join_2591, *[DATA_PATH_2592, str_2593], **kwargs_2594)
        
        # Processing the call keyword arguments (line 189)
        # Getting the type of 'False' (line 189)
        False_2596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 86), 'False', False)
        keyword_2597 = False_2596
        kwargs_2598 = {'verbose': keyword_2597}
        # Getting the type of 'readsav' (line 189)
        readsav_2589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 189)
        readsav_call_result_2599 = invoke(stypy.reporting.localization.Localization(__file__, 189, 12), readsav_2589, *[join_call_result_2595], **kwargs_2598)
        
        # Assigning a type to the variable 's' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 's', readsav_call_result_2599)
        
        # Call to assert_identical(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 's' (line 190)
        s_2601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 25), 's', False)
        # Obtaining the member 'scalars_rep' of a type (line 190)
        scalars_rep_2602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 25), s_2601, 'scalars_rep')
        # Obtaining the member 'a' of a type (line 190)
        a_2603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 25), scalars_rep_2602, 'a')
        
        # Call to reshape(...): (line 190)
        # Processing the call arguments (line 190)
        int_2615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 77), 'int')
        int_2616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 80), 'int')
        int_2617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 83), 'int')
        # Processing the call keyword arguments (line 190)
        kwargs_2618 = {}
        
        # Call to repeat(...): (line 190)
        # Processing the call arguments (line 190)
        
        # Call to int16(...): (line 190)
        # Processing the call arguments (line 190)
        int_2608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 61), 'int')
        # Processing the call keyword arguments (line 190)
        kwargs_2609 = {}
        # Getting the type of 'np' (line 190)
        np_2606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 52), 'np', False)
        # Obtaining the member 'int16' of a type (line 190)
        int16_2607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 52), np_2606, 'int16')
        # Calling int16(args, kwargs) (line 190)
        int16_call_result_2610 = invoke(stypy.reporting.localization.Localization(__file__, 190, 52), int16_2607, *[int_2608], **kwargs_2609)
        
        int_2611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 65), 'int')
        # Processing the call keyword arguments (line 190)
        kwargs_2612 = {}
        # Getting the type of 'np' (line 190)
        np_2604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 42), 'np', False)
        # Obtaining the member 'repeat' of a type (line 190)
        repeat_2605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 42), np_2604, 'repeat')
        # Calling repeat(args, kwargs) (line 190)
        repeat_call_result_2613 = invoke(stypy.reporting.localization.Localization(__file__, 190, 42), repeat_2605, *[int16_call_result_2610, int_2611], **kwargs_2612)
        
        # Obtaining the member 'reshape' of a type (line 190)
        reshape_2614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 42), repeat_call_result_2613, 'reshape')
        # Calling reshape(args, kwargs) (line 190)
        reshape_call_result_2619 = invoke(stypy.reporting.localization.Localization(__file__, 190, 42), reshape_2614, *[int_2615, int_2616, int_2617], **kwargs_2618)
        
        # Processing the call keyword arguments (line 190)
        kwargs_2620 = {}
        # Getting the type of 'assert_identical' (line 190)
        assert_identical_2600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 190)
        assert_identical_call_result_2621 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), assert_identical_2600, *[a_2603, reshape_call_result_2619], **kwargs_2620)
        
        
        # Call to assert_identical(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 's' (line 191)
        s_2623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 25), 's', False)
        # Obtaining the member 'scalars_rep' of a type (line 191)
        scalars_rep_2624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 25), s_2623, 'scalars_rep')
        # Obtaining the member 'b' of a type (line 191)
        b_2625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 25), scalars_rep_2624, 'b')
        
        # Call to reshape(...): (line 191)
        # Processing the call arguments (line 191)
        int_2637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 77), 'int')
        int_2638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 80), 'int')
        int_2639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 83), 'int')
        # Processing the call keyword arguments (line 191)
        kwargs_2640 = {}
        
        # Call to repeat(...): (line 191)
        # Processing the call arguments (line 191)
        
        # Call to int32(...): (line 191)
        # Processing the call arguments (line 191)
        int_2630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 61), 'int')
        # Processing the call keyword arguments (line 191)
        kwargs_2631 = {}
        # Getting the type of 'np' (line 191)
        np_2628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 52), 'np', False)
        # Obtaining the member 'int32' of a type (line 191)
        int32_2629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 52), np_2628, 'int32')
        # Calling int32(args, kwargs) (line 191)
        int32_call_result_2632 = invoke(stypy.reporting.localization.Localization(__file__, 191, 52), int32_2629, *[int_2630], **kwargs_2631)
        
        int_2633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 65), 'int')
        # Processing the call keyword arguments (line 191)
        kwargs_2634 = {}
        # Getting the type of 'np' (line 191)
        np_2626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 42), 'np', False)
        # Obtaining the member 'repeat' of a type (line 191)
        repeat_2627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 42), np_2626, 'repeat')
        # Calling repeat(args, kwargs) (line 191)
        repeat_call_result_2635 = invoke(stypy.reporting.localization.Localization(__file__, 191, 42), repeat_2627, *[int32_call_result_2632, int_2633], **kwargs_2634)
        
        # Obtaining the member 'reshape' of a type (line 191)
        reshape_2636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 42), repeat_call_result_2635, 'reshape')
        # Calling reshape(args, kwargs) (line 191)
        reshape_call_result_2641 = invoke(stypy.reporting.localization.Localization(__file__, 191, 42), reshape_2636, *[int_2637, int_2638, int_2639], **kwargs_2640)
        
        # Processing the call keyword arguments (line 191)
        kwargs_2642 = {}
        # Getting the type of 'assert_identical' (line 191)
        assert_identical_2622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 191)
        assert_identical_call_result_2643 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), assert_identical_2622, *[b_2625, reshape_call_result_2641], **kwargs_2642)
        
        
        # Call to assert_identical(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 's' (line 192)
        s_2645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 25), 's', False)
        # Obtaining the member 'scalars_rep' of a type (line 192)
        scalars_rep_2646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 25), s_2645, 'scalars_rep')
        # Obtaining the member 'c' of a type (line 192)
        c_2647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 25), scalars_rep_2646, 'c')
        
        # Call to reshape(...): (line 192)
        # Processing the call arguments (line 192)
        int_2659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 80), 'int')
        int_2660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 83), 'int')
        int_2661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 86), 'int')
        # Processing the call keyword arguments (line 192)
        kwargs_2662 = {}
        
        # Call to repeat(...): (line 192)
        # Processing the call arguments (line 192)
        
        # Call to float32(...): (line 192)
        # Processing the call arguments (line 192)
        float_2652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 63), 'float')
        # Processing the call keyword arguments (line 192)
        kwargs_2653 = {}
        # Getting the type of 'np' (line 192)
        np_2650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 52), 'np', False)
        # Obtaining the member 'float32' of a type (line 192)
        float32_2651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 52), np_2650, 'float32')
        # Calling float32(args, kwargs) (line 192)
        float32_call_result_2654 = invoke(stypy.reporting.localization.Localization(__file__, 192, 52), float32_2651, *[float_2652], **kwargs_2653)
        
        int_2655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 68), 'int')
        # Processing the call keyword arguments (line 192)
        kwargs_2656 = {}
        # Getting the type of 'np' (line 192)
        np_2648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 42), 'np', False)
        # Obtaining the member 'repeat' of a type (line 192)
        repeat_2649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 42), np_2648, 'repeat')
        # Calling repeat(args, kwargs) (line 192)
        repeat_call_result_2657 = invoke(stypy.reporting.localization.Localization(__file__, 192, 42), repeat_2649, *[float32_call_result_2654, int_2655], **kwargs_2656)
        
        # Obtaining the member 'reshape' of a type (line 192)
        reshape_2658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 42), repeat_call_result_2657, 'reshape')
        # Calling reshape(args, kwargs) (line 192)
        reshape_call_result_2663 = invoke(stypy.reporting.localization.Localization(__file__, 192, 42), reshape_2658, *[int_2659, int_2660, int_2661], **kwargs_2662)
        
        # Processing the call keyword arguments (line 192)
        kwargs_2664 = {}
        # Getting the type of 'assert_identical' (line 192)
        assert_identical_2644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 192)
        assert_identical_call_result_2665 = invoke(stypy.reporting.localization.Localization(__file__, 192, 8), assert_identical_2644, *[c_2647, reshape_call_result_2663], **kwargs_2664)
        
        
        # Call to assert_identical(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 's' (line 193)
        s_2667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 25), 's', False)
        # Obtaining the member 'scalars_rep' of a type (line 193)
        scalars_rep_2668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 25), s_2667, 'scalars_rep')
        # Obtaining the member 'd' of a type (line 193)
        d_2669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 25), scalars_rep_2668, 'd')
        
        # Call to reshape(...): (line 193)
        # Processing the call arguments (line 193)
        int_2681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 80), 'int')
        int_2682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 83), 'int')
        int_2683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 86), 'int')
        # Processing the call keyword arguments (line 193)
        kwargs_2684 = {}
        
        # Call to repeat(...): (line 193)
        # Processing the call arguments (line 193)
        
        # Call to float64(...): (line 193)
        # Processing the call arguments (line 193)
        float_2674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 63), 'float')
        # Processing the call keyword arguments (line 193)
        kwargs_2675 = {}
        # Getting the type of 'np' (line 193)
        np_2672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 52), 'np', False)
        # Obtaining the member 'float64' of a type (line 193)
        float64_2673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 52), np_2672, 'float64')
        # Calling float64(args, kwargs) (line 193)
        float64_call_result_2676 = invoke(stypy.reporting.localization.Localization(__file__, 193, 52), float64_2673, *[float_2674], **kwargs_2675)
        
        int_2677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 68), 'int')
        # Processing the call keyword arguments (line 193)
        kwargs_2678 = {}
        # Getting the type of 'np' (line 193)
        np_2670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 42), 'np', False)
        # Obtaining the member 'repeat' of a type (line 193)
        repeat_2671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 42), np_2670, 'repeat')
        # Calling repeat(args, kwargs) (line 193)
        repeat_call_result_2679 = invoke(stypy.reporting.localization.Localization(__file__, 193, 42), repeat_2671, *[float64_call_result_2676, int_2677], **kwargs_2678)
        
        # Obtaining the member 'reshape' of a type (line 193)
        reshape_2680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 42), repeat_call_result_2679, 'reshape')
        # Calling reshape(args, kwargs) (line 193)
        reshape_call_result_2685 = invoke(stypy.reporting.localization.Localization(__file__, 193, 42), reshape_2680, *[int_2681, int_2682, int_2683], **kwargs_2684)
        
        # Processing the call keyword arguments (line 193)
        kwargs_2686 = {}
        # Getting the type of 'assert_identical' (line 193)
        assert_identical_2666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 193)
        assert_identical_call_result_2687 = invoke(stypy.reporting.localization.Localization(__file__, 193, 8), assert_identical_2666, *[d_2669, reshape_call_result_2685], **kwargs_2686)
        
        
        # Call to assert_identical(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 's' (line 194)
        s_2689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 25), 's', False)
        # Obtaining the member 'scalars_rep' of a type (line 194)
        scalars_rep_2690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 25), s_2689, 'scalars_rep')
        # Obtaining the member 'e' of a type (line 194)
        e_2691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 25), scalars_rep_2690, 'e')
        
        # Call to astype(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'object' (line 194)
        object_2705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 89), 'object', False)
        # Processing the call keyword arguments (line 194)
        kwargs_2706 = {}
        
        # Call to reshape(...): (line 194)
        # Processing the call arguments (line 194)
        int_2699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 73), 'int')
        int_2700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 76), 'int')
        int_2701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 79), 'int')
        # Processing the call keyword arguments (line 194)
        kwargs_2702 = {}
        
        # Call to repeat(...): (line 194)
        # Processing the call arguments (line 194)
        str_2694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 52), 'str', 'spam')
        int_2695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 61), 'int')
        # Processing the call keyword arguments (line 194)
        kwargs_2696 = {}
        # Getting the type of 'np' (line 194)
        np_2692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 42), 'np', False)
        # Obtaining the member 'repeat' of a type (line 194)
        repeat_2693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 42), np_2692, 'repeat')
        # Calling repeat(args, kwargs) (line 194)
        repeat_call_result_2697 = invoke(stypy.reporting.localization.Localization(__file__, 194, 42), repeat_2693, *[str_2694, int_2695], **kwargs_2696)
        
        # Obtaining the member 'reshape' of a type (line 194)
        reshape_2698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 42), repeat_call_result_2697, 'reshape')
        # Calling reshape(args, kwargs) (line 194)
        reshape_call_result_2703 = invoke(stypy.reporting.localization.Localization(__file__, 194, 42), reshape_2698, *[int_2699, int_2700, int_2701], **kwargs_2702)
        
        # Obtaining the member 'astype' of a type (line 194)
        astype_2704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 42), reshape_call_result_2703, 'astype')
        # Calling astype(args, kwargs) (line 194)
        astype_call_result_2707 = invoke(stypy.reporting.localization.Localization(__file__, 194, 42), astype_2704, *[object_2705], **kwargs_2706)
        
        # Processing the call keyword arguments (line 194)
        kwargs_2708 = {}
        # Getting the type of 'assert_identical' (line 194)
        assert_identical_2688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 194)
        assert_identical_call_result_2709 = invoke(stypy.reporting.localization.Localization(__file__, 194, 8), assert_identical_2688, *[e_2691, astype_call_result_2707], **kwargs_2708)
        
        
        # Call to assert_identical(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 's' (line 195)
        s_2711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 25), 's', False)
        # Obtaining the member 'scalars_rep' of a type (line 195)
        scalars_rep_2712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 25), s_2711, 'scalars_rep')
        # Obtaining the member 'f' of a type (line 195)
        f_2713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 25), scalars_rep_2712, 'f')
        
        # Call to reshape(...): (line 195)
        # Processing the call arguments (line 195)
        int_2727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 86), 'int')
        int_2728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 89), 'int')
        int_2729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 92), 'int')
        # Processing the call keyword arguments (line 195)
        kwargs_2730 = {}
        
        # Call to repeat(...): (line 195)
        # Processing the call arguments (line 195)
        
        # Call to complex64(...): (line 195)
        # Processing the call arguments (line 195)
        float_2718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 65), 'float')
        complex_2719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 69), 'complex')
        # Applying the binary operator '+' (line 195)
        result_add_2720 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 65), '+', float_2718, complex_2719)
        
        # Processing the call keyword arguments (line 195)
        kwargs_2721 = {}
        # Getting the type of 'np' (line 195)
        np_2716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 52), 'np', False)
        # Obtaining the member 'complex64' of a type (line 195)
        complex64_2717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 52), np_2716, 'complex64')
        # Calling complex64(args, kwargs) (line 195)
        complex64_call_result_2722 = invoke(stypy.reporting.localization.Localization(__file__, 195, 52), complex64_2717, *[result_add_2720], **kwargs_2721)
        
        int_2723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 74), 'int')
        # Processing the call keyword arguments (line 195)
        kwargs_2724 = {}
        # Getting the type of 'np' (line 195)
        np_2714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 42), 'np', False)
        # Obtaining the member 'repeat' of a type (line 195)
        repeat_2715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 42), np_2714, 'repeat')
        # Calling repeat(args, kwargs) (line 195)
        repeat_call_result_2725 = invoke(stypy.reporting.localization.Localization(__file__, 195, 42), repeat_2715, *[complex64_call_result_2722, int_2723], **kwargs_2724)
        
        # Obtaining the member 'reshape' of a type (line 195)
        reshape_2726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 42), repeat_call_result_2725, 'reshape')
        # Calling reshape(args, kwargs) (line 195)
        reshape_call_result_2731 = invoke(stypy.reporting.localization.Localization(__file__, 195, 42), reshape_2726, *[int_2727, int_2728, int_2729], **kwargs_2730)
        
        # Processing the call keyword arguments (line 195)
        kwargs_2732 = {}
        # Getting the type of 'assert_identical' (line 195)
        assert_identical_2710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 195)
        assert_identical_call_result_2733 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), assert_identical_2710, *[f_2713, reshape_call_result_2731], **kwargs_2732)
        
        
        # ################# End of 'test_scalars_replicated_3d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scalars_replicated_3d' in the type store
        # Getting the type of 'stypy_return_type' (line 188)
        stypy_return_type_2734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2734)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scalars_replicated_3d'
        return stypy_return_type_2734


    @norecursion
    def test_arrays(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_arrays'
        module_type_store = module_type_store.open_function_context('test_arrays', 197, 4, False)
        # Assigning a type to the variable 'self' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestStructures.test_arrays.__dict__.__setitem__('stypy_localization', localization)
        TestStructures.test_arrays.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestStructures.test_arrays.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestStructures.test_arrays.__dict__.__setitem__('stypy_function_name', 'TestStructures.test_arrays')
        TestStructures.test_arrays.__dict__.__setitem__('stypy_param_names_list', [])
        TestStructures.test_arrays.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestStructures.test_arrays.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestStructures.test_arrays.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestStructures.test_arrays.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestStructures.test_arrays.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestStructures.test_arrays.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestStructures.test_arrays', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_arrays', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_arrays(...)' code ##################

        
        # Assigning a Call to a Name (line 198):
        
        # Call to readsav(...): (line 198)
        # Processing the call arguments (line 198)
        
        # Call to join(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'DATA_PATH' (line 198)
        DATA_PATH_2738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 30), 'DATA_PATH', False)
        str_2739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 41), 'str', 'struct_arrays.sav')
        # Processing the call keyword arguments (line 198)
        kwargs_2740 = {}
        # Getting the type of 'path' (line 198)
        path_2736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 198)
        join_2737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 20), path_2736, 'join')
        # Calling join(args, kwargs) (line 198)
        join_call_result_2741 = invoke(stypy.reporting.localization.Localization(__file__, 198, 20), join_2737, *[DATA_PATH_2738, str_2739], **kwargs_2740)
        
        # Processing the call keyword arguments (line 198)
        # Getting the type of 'False' (line 198)
        False_2742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 71), 'False', False)
        keyword_2743 = False_2742
        kwargs_2744 = {'verbose': keyword_2743}
        # Getting the type of 'readsav' (line 198)
        readsav_2735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 198)
        readsav_call_result_2745 = invoke(stypy.reporting.localization.Localization(__file__, 198, 12), readsav_2735, *[join_call_result_2741], **kwargs_2744)
        
        # Assigning a type to the variable 's' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 's', readsav_call_result_2745)
        
        # Call to assert_array_identical(...): (line 199)
        # Processing the call arguments (line 199)
        
        # Obtaining the type of the subscript
        int_2747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 42), 'int')
        # Getting the type of 's' (line 199)
        s_2748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 31), 's', False)
        # Obtaining the member 'arrays' of a type (line 199)
        arrays_2749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 31), s_2748, 'arrays')
        # Obtaining the member 'a' of a type (line 199)
        a_2750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 31), arrays_2749, 'a')
        # Obtaining the member '__getitem__' of a type (line 199)
        getitem___2751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 31), a_2750, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 199)
        subscript_call_result_2752 = invoke(stypy.reporting.localization.Localization(__file__, 199, 31), getitem___2751, int_2747)
        
        
        # Call to array(...): (line 199)
        # Processing the call arguments (line 199)
        
        # Obtaining an instance of the builtin type 'list' (line 199)
        list_2755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 199)
        # Adding element type (line 199)
        int_2756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 55), list_2755, int_2756)
        # Adding element type (line 199)
        int_2757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 55), list_2755, int_2757)
        # Adding element type (line 199)
        int_2758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 55), list_2755, int_2758)
        
        # Processing the call keyword arguments (line 199)
        # Getting the type of 'np' (line 199)
        np_2759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 72), 'np', False)
        # Obtaining the member 'int16' of a type (line 199)
        int16_2760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 72), np_2759, 'int16')
        keyword_2761 = int16_2760
        kwargs_2762 = {'dtype': keyword_2761}
        # Getting the type of 'np' (line 199)
        np_2753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 46), 'np', False)
        # Obtaining the member 'array' of a type (line 199)
        array_2754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 46), np_2753, 'array')
        # Calling array(args, kwargs) (line 199)
        array_call_result_2763 = invoke(stypy.reporting.localization.Localization(__file__, 199, 46), array_2754, *[list_2755], **kwargs_2762)
        
        # Processing the call keyword arguments (line 199)
        kwargs_2764 = {}
        # Getting the type of 'assert_array_identical' (line 199)
        assert_array_identical_2746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 199)
        assert_array_identical_call_result_2765 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), assert_array_identical_2746, *[subscript_call_result_2752, array_call_result_2763], **kwargs_2764)
        
        
        # Call to assert_array_identical(...): (line 200)
        # Processing the call arguments (line 200)
        
        # Obtaining the type of the subscript
        int_2767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 42), 'int')
        # Getting the type of 's' (line 200)
        s_2768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 31), 's', False)
        # Obtaining the member 'arrays' of a type (line 200)
        arrays_2769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 31), s_2768, 'arrays')
        # Obtaining the member 'b' of a type (line 200)
        b_2770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 31), arrays_2769, 'b')
        # Obtaining the member '__getitem__' of a type (line 200)
        getitem___2771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 31), b_2770, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 200)
        subscript_call_result_2772 = invoke(stypy.reporting.localization.Localization(__file__, 200, 31), getitem___2771, int_2767)
        
        
        # Call to array(...): (line 200)
        # Processing the call arguments (line 200)
        
        # Obtaining an instance of the builtin type 'list' (line 200)
        list_2775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 200)
        # Adding element type (line 200)
        float_2776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 56), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 55), list_2775, float_2776)
        # Adding element type (line 200)
        float_2777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 60), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 55), list_2775, float_2777)
        # Adding element type (line 200)
        float_2778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 64), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 55), list_2775, float_2778)
        # Adding element type (line 200)
        float_2779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 68), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 200, 55), list_2775, float_2779)
        
        # Processing the call keyword arguments (line 200)
        # Getting the type of 'np' (line 200)
        np_2780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 79), 'np', False)
        # Obtaining the member 'float32' of a type (line 200)
        float32_2781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 79), np_2780, 'float32')
        keyword_2782 = float32_2781
        kwargs_2783 = {'dtype': keyword_2782}
        # Getting the type of 'np' (line 200)
        np_2773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 46), 'np', False)
        # Obtaining the member 'array' of a type (line 200)
        array_2774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 46), np_2773, 'array')
        # Calling array(args, kwargs) (line 200)
        array_call_result_2784 = invoke(stypy.reporting.localization.Localization(__file__, 200, 46), array_2774, *[list_2775], **kwargs_2783)
        
        # Processing the call keyword arguments (line 200)
        kwargs_2785 = {}
        # Getting the type of 'assert_array_identical' (line 200)
        assert_array_identical_2766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 200)
        assert_array_identical_call_result_2786 = invoke(stypy.reporting.localization.Localization(__file__, 200, 8), assert_array_identical_2766, *[subscript_call_result_2772, array_call_result_2784], **kwargs_2785)
        
        
        # Call to assert_array_identical(...): (line 201)
        # Processing the call arguments (line 201)
        
        # Obtaining the type of the subscript
        int_2788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 42), 'int')
        # Getting the type of 's' (line 201)
        s_2789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 31), 's', False)
        # Obtaining the member 'arrays' of a type (line 201)
        arrays_2790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 31), s_2789, 'arrays')
        # Obtaining the member 'c' of a type (line 201)
        c_2791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 31), arrays_2790, 'c')
        # Obtaining the member '__getitem__' of a type (line 201)
        getitem___2792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 31), c_2791, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 201)
        subscript_call_result_2793 = invoke(stypy.reporting.localization.Localization(__file__, 201, 31), getitem___2792, int_2788)
        
        
        # Call to array(...): (line 201)
        # Processing the call arguments (line 201)
        
        # Obtaining an instance of the builtin type 'list' (line 201)
        list_2796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 201)
        # Adding element type (line 201)
        
        # Call to complex64(...): (line 201)
        # Processing the call arguments (line 201)
        int_2799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 69), 'int')
        complex_2800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 71), 'complex')
        # Applying the binary operator '+' (line 201)
        result_add_2801 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 69), '+', int_2799, complex_2800)
        
        # Processing the call keyword arguments (line 201)
        kwargs_2802 = {}
        # Getting the type of 'np' (line 201)
        np_2797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 56), 'np', False)
        # Obtaining the member 'complex64' of a type (line 201)
        complex64_2798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 56), np_2797, 'complex64')
        # Calling complex64(args, kwargs) (line 201)
        complex64_call_result_2803 = invoke(stypy.reporting.localization.Localization(__file__, 201, 56), complex64_2798, *[result_add_2801], **kwargs_2802)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 55), list_2796, complex64_call_result_2803)
        # Adding element type (line 201)
        
        # Call to complex64(...): (line 201)
        # Processing the call arguments (line 201)
        int_2806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 89), 'int')
        complex_2807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 91), 'complex')
        # Applying the binary operator '+' (line 201)
        result_add_2808 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 89), '+', int_2806, complex_2807)
        
        # Processing the call keyword arguments (line 201)
        kwargs_2809 = {}
        # Getting the type of 'np' (line 201)
        np_2804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 76), 'np', False)
        # Obtaining the member 'complex64' of a type (line 201)
        complex64_2805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 76), np_2804, 'complex64')
        # Calling complex64(args, kwargs) (line 201)
        complex64_call_result_2810 = invoke(stypy.reporting.localization.Localization(__file__, 201, 76), complex64_2805, *[result_add_2808], **kwargs_2809)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 55), list_2796, complex64_call_result_2810)
        
        # Processing the call keyword arguments (line 201)
        kwargs_2811 = {}
        # Getting the type of 'np' (line 201)
        np_2794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 46), 'np', False)
        # Obtaining the member 'array' of a type (line 201)
        array_2795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 46), np_2794, 'array')
        # Calling array(args, kwargs) (line 201)
        array_call_result_2812 = invoke(stypy.reporting.localization.Localization(__file__, 201, 46), array_2795, *[list_2796], **kwargs_2811)
        
        # Processing the call keyword arguments (line 201)
        kwargs_2813 = {}
        # Getting the type of 'assert_array_identical' (line 201)
        assert_array_identical_2787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 201)
        assert_array_identical_call_result_2814 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), assert_array_identical_2787, *[subscript_call_result_2793, array_call_result_2812], **kwargs_2813)
        
        
        # Call to assert_array_identical(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Obtaining the type of the subscript
        int_2816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 42), 'int')
        # Getting the type of 's' (line 202)
        s_2817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 31), 's', False)
        # Obtaining the member 'arrays' of a type (line 202)
        arrays_2818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 31), s_2817, 'arrays')
        # Obtaining the member 'd' of a type (line 202)
        d_2819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 31), arrays_2818, 'd')
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___2820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 31), d_2819, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_2821 = invoke(stypy.reporting.localization.Localization(__file__, 202, 31), getitem___2820, int_2816)
        
        
        # Call to array(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Obtaining an instance of the builtin type 'list' (line 202)
        list_2824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 202)
        # Adding element type (line 202)
        str_2825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 56), 'str', 'cheese')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 55), list_2824, str_2825)
        # Adding element type (line 202)
        str_2826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 67), 'str', 'bacon')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 55), list_2824, str_2826)
        # Adding element type (line 202)
        str_2827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 77), 'str', 'spam')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 55), list_2824, str_2827)
        
        # Processing the call keyword arguments (line 202)
        # Getting the type of 'object' (line 202)
        object_2828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 93), 'object', False)
        keyword_2829 = object_2828
        kwargs_2830 = {'dtype': keyword_2829}
        # Getting the type of 'np' (line 202)
        np_2822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 46), 'np', False)
        # Obtaining the member 'array' of a type (line 202)
        array_2823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 46), np_2822, 'array')
        # Calling array(args, kwargs) (line 202)
        array_call_result_2831 = invoke(stypy.reporting.localization.Localization(__file__, 202, 46), array_2823, *[list_2824], **kwargs_2830)
        
        # Processing the call keyword arguments (line 202)
        kwargs_2832 = {}
        # Getting the type of 'assert_array_identical' (line 202)
        assert_array_identical_2815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 202)
        assert_array_identical_call_result_2833 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), assert_array_identical_2815, *[subscript_call_result_2821, array_call_result_2831], **kwargs_2832)
        
        
        # ################# End of 'test_arrays(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_arrays' in the type store
        # Getting the type of 'stypy_return_type' (line 197)
        stypy_return_type_2834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2834)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_arrays'
        return stypy_return_type_2834


    @norecursion
    def test_arrays_replicated(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_arrays_replicated'
        module_type_store = module_type_store.open_function_context('test_arrays_replicated', 204, 4, False)
        # Assigning a type to the variable 'self' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestStructures.test_arrays_replicated.__dict__.__setitem__('stypy_localization', localization)
        TestStructures.test_arrays_replicated.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestStructures.test_arrays_replicated.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestStructures.test_arrays_replicated.__dict__.__setitem__('stypy_function_name', 'TestStructures.test_arrays_replicated')
        TestStructures.test_arrays_replicated.__dict__.__setitem__('stypy_param_names_list', [])
        TestStructures.test_arrays_replicated.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestStructures.test_arrays_replicated.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestStructures.test_arrays_replicated.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestStructures.test_arrays_replicated.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestStructures.test_arrays_replicated.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestStructures.test_arrays_replicated.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestStructures.test_arrays_replicated', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_arrays_replicated', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_arrays_replicated(...)' code ##################

        
        # Assigning a Call to a Name (line 205):
        
        # Call to readsav(...): (line 205)
        # Processing the call arguments (line 205)
        
        # Call to join(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'DATA_PATH' (line 205)
        DATA_PATH_2838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 30), 'DATA_PATH', False)
        str_2839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 41), 'str', 'struct_arrays_replicated.sav')
        # Processing the call keyword arguments (line 205)
        kwargs_2840 = {}
        # Getting the type of 'path' (line 205)
        path_2836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 205)
        join_2837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 20), path_2836, 'join')
        # Calling join(args, kwargs) (line 205)
        join_call_result_2841 = invoke(stypy.reporting.localization.Localization(__file__, 205, 20), join_2837, *[DATA_PATH_2838, str_2839], **kwargs_2840)
        
        # Processing the call keyword arguments (line 205)
        # Getting the type of 'False' (line 205)
        False_2842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 82), 'False', False)
        keyword_2843 = False_2842
        kwargs_2844 = {'verbose': keyword_2843}
        # Getting the type of 'readsav' (line 205)
        readsav_2835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 205)
        readsav_call_result_2845 = invoke(stypy.reporting.localization.Localization(__file__, 205, 12), readsav_2835, *[join_call_result_2841], **kwargs_2844)
        
        # Assigning a type to the variable 's' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 's', readsav_call_result_2845)
        
        # Call to assert_(...): (line 208)
        # Processing the call arguments (line 208)
        
        # Getting the type of 's' (line 208)
        s_2847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 208)
        arrays_rep_2848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 16), s_2847, 'arrays_rep')
        # Obtaining the member 'a' of a type (line 208)
        a_2849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 16), arrays_rep_2848, 'a')
        # Obtaining the member 'dtype' of a type (line 208)
        dtype_2850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 16), a_2849, 'dtype')
        # Obtaining the member 'type' of a type (line 208)
        type_2851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 16), dtype_2850, 'type')
        # Getting the type of 'np' (line 208)
        np_2852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 45), 'np', False)
        # Obtaining the member 'object_' of a type (line 208)
        object__2853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 45), np_2852, 'object_')
        # Applying the binary operator 'is' (line 208)
        result_is__2854 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 16), 'is', type_2851, object__2853)
        
        # Processing the call keyword arguments (line 208)
        kwargs_2855 = {}
        # Getting the type of 'assert_' (line 208)
        assert__2846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 208)
        assert__call_result_2856 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), assert__2846, *[result_is__2854], **kwargs_2855)
        
        
        # Call to assert_(...): (line 209)
        # Processing the call arguments (line 209)
        
        # Getting the type of 's' (line 209)
        s_2858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 16), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 209)
        arrays_rep_2859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 16), s_2858, 'arrays_rep')
        # Obtaining the member 'b' of a type (line 209)
        b_2860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 16), arrays_rep_2859, 'b')
        # Obtaining the member 'dtype' of a type (line 209)
        dtype_2861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 16), b_2860, 'dtype')
        # Obtaining the member 'type' of a type (line 209)
        type_2862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 16), dtype_2861, 'type')
        # Getting the type of 'np' (line 209)
        np_2863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 45), 'np', False)
        # Obtaining the member 'object_' of a type (line 209)
        object__2864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 45), np_2863, 'object_')
        # Applying the binary operator 'is' (line 209)
        result_is__2865 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 16), 'is', type_2862, object__2864)
        
        # Processing the call keyword arguments (line 209)
        kwargs_2866 = {}
        # Getting the type of 'assert_' (line 209)
        assert__2857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 209)
        assert__call_result_2867 = invoke(stypy.reporting.localization.Localization(__file__, 209, 8), assert__2857, *[result_is__2865], **kwargs_2866)
        
        
        # Call to assert_(...): (line 210)
        # Processing the call arguments (line 210)
        
        # Getting the type of 's' (line 210)
        s_2869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 210)
        arrays_rep_2870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 16), s_2869, 'arrays_rep')
        # Obtaining the member 'c' of a type (line 210)
        c_2871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 16), arrays_rep_2870, 'c')
        # Obtaining the member 'dtype' of a type (line 210)
        dtype_2872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 16), c_2871, 'dtype')
        # Obtaining the member 'type' of a type (line 210)
        type_2873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 16), dtype_2872, 'type')
        # Getting the type of 'np' (line 210)
        np_2874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 45), 'np', False)
        # Obtaining the member 'object_' of a type (line 210)
        object__2875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 45), np_2874, 'object_')
        # Applying the binary operator 'is' (line 210)
        result_is__2876 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 16), 'is', type_2873, object__2875)
        
        # Processing the call keyword arguments (line 210)
        kwargs_2877 = {}
        # Getting the type of 'assert_' (line 210)
        assert__2868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 210)
        assert__call_result_2878 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), assert__2868, *[result_is__2876], **kwargs_2877)
        
        
        # Call to assert_(...): (line 211)
        # Processing the call arguments (line 211)
        
        # Getting the type of 's' (line 211)
        s_2880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 211)
        arrays_rep_2881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 16), s_2880, 'arrays_rep')
        # Obtaining the member 'd' of a type (line 211)
        d_2882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 16), arrays_rep_2881, 'd')
        # Obtaining the member 'dtype' of a type (line 211)
        dtype_2883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 16), d_2882, 'dtype')
        # Obtaining the member 'type' of a type (line 211)
        type_2884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 16), dtype_2883, 'type')
        # Getting the type of 'np' (line 211)
        np_2885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 45), 'np', False)
        # Obtaining the member 'object_' of a type (line 211)
        object__2886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 45), np_2885, 'object_')
        # Applying the binary operator 'is' (line 211)
        result_is__2887 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 16), 'is', type_2884, object__2886)
        
        # Processing the call keyword arguments (line 211)
        kwargs_2888 = {}
        # Getting the type of 'assert_' (line 211)
        assert__2879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 211)
        assert__call_result_2889 = invoke(stypy.reporting.localization.Localization(__file__, 211, 8), assert__2879, *[result_is__2887], **kwargs_2888)
        
        
        # Call to assert_equal(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 's' (line 214)
        s_2891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 21), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 214)
        arrays_rep_2892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 21), s_2891, 'arrays_rep')
        # Obtaining the member 'a' of a type (line 214)
        a_2893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 21), arrays_rep_2892, 'a')
        # Obtaining the member 'shape' of a type (line 214)
        shape_2894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 21), a_2893, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 214)
        tuple_2895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 214)
        # Adding element type (line 214)
        int_2896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 44), tuple_2895, int_2896)
        
        # Processing the call keyword arguments (line 214)
        kwargs_2897 = {}
        # Getting the type of 'assert_equal' (line 214)
        assert_equal_2890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 214)
        assert_equal_call_result_2898 = invoke(stypy.reporting.localization.Localization(__file__, 214, 8), assert_equal_2890, *[shape_2894, tuple_2895], **kwargs_2897)
        
        
        # Call to assert_equal(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 's' (line 215)
        s_2900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 21), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 215)
        arrays_rep_2901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 21), s_2900, 'arrays_rep')
        # Obtaining the member 'b' of a type (line 215)
        b_2902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 21), arrays_rep_2901, 'b')
        # Obtaining the member 'shape' of a type (line 215)
        shape_2903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 21), b_2902, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 215)
        tuple_2904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 215)
        # Adding element type (line 215)
        int_2905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 44), tuple_2904, int_2905)
        
        # Processing the call keyword arguments (line 215)
        kwargs_2906 = {}
        # Getting the type of 'assert_equal' (line 215)
        assert_equal_2899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 215)
        assert_equal_call_result_2907 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), assert_equal_2899, *[shape_2903, tuple_2904], **kwargs_2906)
        
        
        # Call to assert_equal(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 's' (line 216)
        s_2909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 21), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 216)
        arrays_rep_2910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 21), s_2909, 'arrays_rep')
        # Obtaining the member 'c' of a type (line 216)
        c_2911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 21), arrays_rep_2910, 'c')
        # Obtaining the member 'shape' of a type (line 216)
        shape_2912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 21), c_2911, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 216)
        tuple_2913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 216)
        # Adding element type (line 216)
        int_2914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 44), tuple_2913, int_2914)
        
        # Processing the call keyword arguments (line 216)
        kwargs_2915 = {}
        # Getting the type of 'assert_equal' (line 216)
        assert_equal_2908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 216)
        assert_equal_call_result_2916 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), assert_equal_2908, *[shape_2912, tuple_2913], **kwargs_2915)
        
        
        # Call to assert_equal(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 's' (line 217)
        s_2918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 217)
        arrays_rep_2919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 21), s_2918, 'arrays_rep')
        # Obtaining the member 'd' of a type (line 217)
        d_2920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 21), arrays_rep_2919, 'd')
        # Obtaining the member 'shape' of a type (line 217)
        shape_2921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 21), d_2920, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 217)
        tuple_2922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 217)
        # Adding element type (line 217)
        int_2923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 217, 44), tuple_2922, int_2923)
        
        # Processing the call keyword arguments (line 217)
        kwargs_2924 = {}
        # Getting the type of 'assert_equal' (line 217)
        assert_equal_2917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 217)
        assert_equal_call_result_2925 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), assert_equal_2917, *[shape_2921, tuple_2922], **kwargs_2924)
        
        
        
        # Call to range(...): (line 220)
        # Processing the call arguments (line 220)
        int_2927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 23), 'int')
        # Processing the call keyword arguments (line 220)
        kwargs_2928 = {}
        # Getting the type of 'range' (line 220)
        range_2926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 17), 'range', False)
        # Calling range(args, kwargs) (line 220)
        range_call_result_2929 = invoke(stypy.reporting.localization.Localization(__file__, 220, 17), range_2926, *[int_2927], **kwargs_2928)
        
        # Testing the type of a for loop iterable (line 220)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 220, 8), range_call_result_2929)
        # Getting the type of the for loop variable (line 220)
        for_loop_var_2930 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 220, 8), range_call_result_2929)
        # Assigning a type to the variable 'i' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'i', for_loop_var_2930)
        # SSA begins for a for statement (line 220)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_array_identical(...): (line 221)
        # Processing the call arguments (line 221)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 221)
        i_2932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 50), 'i', False)
        # Getting the type of 's' (line 221)
        s_2933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 35), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 221)
        arrays_rep_2934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 35), s_2933, 'arrays_rep')
        # Obtaining the member 'a' of a type (line 221)
        a_2935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 35), arrays_rep_2934, 'a')
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___2936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 35), a_2935, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_2937 = invoke(stypy.reporting.localization.Localization(__file__, 221, 35), getitem___2936, i_2932)
        
        
        # Call to array(...): (line 222)
        # Processing the call arguments (line 222)
        
        # Obtaining an instance of the builtin type 'list' (line 222)
        list_2940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 222)
        # Adding element type (line 222)
        int_2941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 44), list_2940, int_2941)
        # Adding element type (line 222)
        int_2942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 44), list_2940, int_2942)
        # Adding element type (line 222)
        int_2943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 44), list_2940, int_2943)
        
        # Processing the call keyword arguments (line 222)
        # Getting the type of 'np' (line 222)
        np_2944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 61), 'np', False)
        # Obtaining the member 'int16' of a type (line 222)
        int16_2945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 61), np_2944, 'int16')
        keyword_2946 = int16_2945
        kwargs_2947 = {'dtype': keyword_2946}
        # Getting the type of 'np' (line 222)
        np_2938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 222)
        array_2939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 35), np_2938, 'array')
        # Calling array(args, kwargs) (line 222)
        array_call_result_2948 = invoke(stypy.reporting.localization.Localization(__file__, 222, 35), array_2939, *[list_2940], **kwargs_2947)
        
        # Processing the call keyword arguments (line 221)
        kwargs_2949 = {}
        # Getting the type of 'assert_array_identical' (line 221)
        assert_array_identical_2931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 221)
        assert_array_identical_call_result_2950 = invoke(stypy.reporting.localization.Localization(__file__, 221, 12), assert_array_identical_2931, *[subscript_call_result_2937, array_call_result_2948], **kwargs_2949)
        
        
        # Call to assert_array_identical(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 223)
        i_2952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 50), 'i', False)
        # Getting the type of 's' (line 223)
        s_2953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 35), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 223)
        arrays_rep_2954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 35), s_2953, 'arrays_rep')
        # Obtaining the member 'b' of a type (line 223)
        b_2955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 35), arrays_rep_2954, 'b')
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___2956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 35), b_2955, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_2957 = invoke(stypy.reporting.localization.Localization(__file__, 223, 35), getitem___2956, i_2952)
        
        
        # Call to array(...): (line 224)
        # Processing the call arguments (line 224)
        
        # Obtaining an instance of the builtin type 'list' (line 224)
        list_2960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 224)
        # Adding element type (line 224)
        float_2961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 45), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 44), list_2960, float_2961)
        # Adding element type (line 224)
        float_2962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 49), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 44), list_2960, float_2962)
        # Adding element type (line 224)
        float_2963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 44), list_2960, float_2963)
        # Adding element type (line 224)
        float_2964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 44), list_2960, float_2964)
        
        # Processing the call keyword arguments (line 224)
        # Getting the type of 'np' (line 224)
        np_2965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 68), 'np', False)
        # Obtaining the member 'float32' of a type (line 224)
        float32_2966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 68), np_2965, 'float32')
        keyword_2967 = float32_2966
        kwargs_2968 = {'dtype': keyword_2967}
        # Getting the type of 'np' (line 224)
        np_2958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 224)
        array_2959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 35), np_2958, 'array')
        # Calling array(args, kwargs) (line 224)
        array_call_result_2969 = invoke(stypy.reporting.localization.Localization(__file__, 224, 35), array_2959, *[list_2960], **kwargs_2968)
        
        # Processing the call keyword arguments (line 223)
        kwargs_2970 = {}
        # Getting the type of 'assert_array_identical' (line 223)
        assert_array_identical_2951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 223)
        assert_array_identical_call_result_2971 = invoke(stypy.reporting.localization.Localization(__file__, 223, 12), assert_array_identical_2951, *[subscript_call_result_2957, array_call_result_2969], **kwargs_2970)
        
        
        # Call to assert_array_identical(...): (line 225)
        # Processing the call arguments (line 225)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 225)
        i_2973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 50), 'i', False)
        # Getting the type of 's' (line 225)
        s_2974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 35), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 225)
        arrays_rep_2975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 35), s_2974, 'arrays_rep')
        # Obtaining the member 'c' of a type (line 225)
        c_2976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 35), arrays_rep_2975, 'c')
        # Obtaining the member '__getitem__' of a type (line 225)
        getitem___2977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 35), c_2976, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 225)
        subscript_call_result_2978 = invoke(stypy.reporting.localization.Localization(__file__, 225, 35), getitem___2977, i_2973)
        
        
        # Call to array(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Obtaining an instance of the builtin type 'list' (line 226)
        list_2981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 226)
        # Adding element type (line 226)
        
        # Call to complex64(...): (line 226)
        # Processing the call arguments (line 226)
        int_2984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 58), 'int')
        complex_2985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 60), 'complex')
        # Applying the binary operator '+' (line 226)
        result_add_2986 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 58), '+', int_2984, complex_2985)
        
        # Processing the call keyword arguments (line 226)
        kwargs_2987 = {}
        # Getting the type of 'np' (line 226)
        np_2982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 45), 'np', False)
        # Obtaining the member 'complex64' of a type (line 226)
        complex64_2983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 45), np_2982, 'complex64')
        # Calling complex64(args, kwargs) (line 226)
        complex64_call_result_2988 = invoke(stypy.reporting.localization.Localization(__file__, 226, 45), complex64_2983, *[result_add_2986], **kwargs_2987)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 44), list_2981, complex64_call_result_2988)
        # Adding element type (line 226)
        
        # Call to complex64(...): (line 227)
        # Processing the call arguments (line 227)
        int_2991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 58), 'int')
        complex_2992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 60), 'complex')
        # Applying the binary operator '+' (line 227)
        result_add_2993 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 58), '+', int_2991, complex_2992)
        
        # Processing the call keyword arguments (line 227)
        kwargs_2994 = {}
        # Getting the type of 'np' (line 227)
        np_2989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 45), 'np', False)
        # Obtaining the member 'complex64' of a type (line 227)
        complex64_2990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 45), np_2989, 'complex64')
        # Calling complex64(args, kwargs) (line 227)
        complex64_call_result_2995 = invoke(stypy.reporting.localization.Localization(__file__, 227, 45), complex64_2990, *[result_add_2993], **kwargs_2994)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 44), list_2981, complex64_call_result_2995)
        
        # Processing the call keyword arguments (line 226)
        kwargs_2996 = {}
        # Getting the type of 'np' (line 226)
        np_2979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 226)
        array_2980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 35), np_2979, 'array')
        # Calling array(args, kwargs) (line 226)
        array_call_result_2997 = invoke(stypy.reporting.localization.Localization(__file__, 226, 35), array_2980, *[list_2981], **kwargs_2996)
        
        # Processing the call keyword arguments (line 225)
        kwargs_2998 = {}
        # Getting the type of 'assert_array_identical' (line 225)
        assert_array_identical_2972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 225)
        assert_array_identical_call_result_2999 = invoke(stypy.reporting.localization.Localization(__file__, 225, 12), assert_array_identical_2972, *[subscript_call_result_2978, array_call_result_2997], **kwargs_2998)
        
        
        # Call to assert_array_identical(...): (line 228)
        # Processing the call arguments (line 228)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 228)
        i_3001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 50), 'i', False)
        # Getting the type of 's' (line 228)
        s_3002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 35), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 228)
        arrays_rep_3003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 35), s_3002, 'arrays_rep')
        # Obtaining the member 'd' of a type (line 228)
        d_3004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 35), arrays_rep_3003, 'd')
        # Obtaining the member '__getitem__' of a type (line 228)
        getitem___3005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 35), d_3004, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 228)
        subscript_call_result_3006 = invoke(stypy.reporting.localization.Localization(__file__, 228, 35), getitem___3005, i_3001)
        
        
        # Call to array(...): (line 229)
        # Processing the call arguments (line 229)
        
        # Obtaining an instance of the builtin type 'list' (line 229)
        list_3009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 229)
        # Adding element type (line 229)
        str_3010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 45), 'str', 'cheese')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 44), list_3009, str_3010)
        # Adding element type (line 229)
        str_3011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 56), 'str', 'bacon')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 44), list_3009, str_3011)
        # Adding element type (line 229)
        str_3012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 66), 'str', 'spam')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 44), list_3009, str_3012)
        
        # Processing the call keyword arguments (line 229)
        # Getting the type of 'object' (line 230)
        object_3013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 50), 'object', False)
        keyword_3014 = object_3013
        kwargs_3015 = {'dtype': keyword_3014}
        # Getting the type of 'np' (line 229)
        np_3007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 229)
        array_3008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 35), np_3007, 'array')
        # Calling array(args, kwargs) (line 229)
        array_call_result_3016 = invoke(stypy.reporting.localization.Localization(__file__, 229, 35), array_3008, *[list_3009], **kwargs_3015)
        
        # Processing the call keyword arguments (line 228)
        kwargs_3017 = {}
        # Getting the type of 'assert_array_identical' (line 228)
        assert_array_identical_3000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 228)
        assert_array_identical_call_result_3018 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), assert_array_identical_3000, *[subscript_call_result_3006, array_call_result_3016], **kwargs_3017)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_arrays_replicated(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_arrays_replicated' in the type store
        # Getting the type of 'stypy_return_type' (line 204)
        stypy_return_type_3019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3019)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_arrays_replicated'
        return stypy_return_type_3019


    @norecursion
    def test_arrays_replicated_3d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_arrays_replicated_3d'
        module_type_store = module_type_store.open_function_context('test_arrays_replicated_3d', 232, 4, False)
        # Assigning a type to the variable 'self' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_localization', localization)
        TestStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_function_name', 'TestStructures.test_arrays_replicated_3d')
        TestStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_param_names_list', [])
        TestStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestStructures.test_arrays_replicated_3d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_arrays_replicated_3d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_arrays_replicated_3d(...)' code ##################

        
        # Assigning a Call to a Name (line 233):
        
        # Call to readsav(...): (line 233)
        # Processing the call arguments (line 233)
        
        # Call to join(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'DATA_PATH' (line 233)
        DATA_PATH_3023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 30), 'DATA_PATH', False)
        str_3024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 41), 'str', 'struct_arrays_replicated_3d.sav')
        # Processing the call keyword arguments (line 233)
        kwargs_3025 = {}
        # Getting the type of 'path' (line 233)
        path_3021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 233)
        join_3022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 20), path_3021, 'join')
        # Calling join(args, kwargs) (line 233)
        join_call_result_3026 = invoke(stypy.reporting.localization.Localization(__file__, 233, 20), join_3022, *[DATA_PATH_3023, str_3024], **kwargs_3025)
        
        # Processing the call keyword arguments (line 233)
        # Getting the type of 'False' (line 233)
        False_3027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 85), 'False', False)
        keyword_3028 = False_3027
        kwargs_3029 = {'verbose': keyword_3028}
        # Getting the type of 'readsav' (line 233)
        readsav_3020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 233)
        readsav_call_result_3030 = invoke(stypy.reporting.localization.Localization(__file__, 233, 12), readsav_3020, *[join_call_result_3026], **kwargs_3029)
        
        # Assigning a type to the variable 's' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 's', readsav_call_result_3030)
        
        # Call to assert_(...): (line 236)
        # Processing the call arguments (line 236)
        
        # Getting the type of 's' (line 236)
        s_3032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 236)
        arrays_rep_3033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 16), s_3032, 'arrays_rep')
        # Obtaining the member 'a' of a type (line 236)
        a_3034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 16), arrays_rep_3033, 'a')
        # Obtaining the member 'dtype' of a type (line 236)
        dtype_3035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 16), a_3034, 'dtype')
        # Obtaining the member 'type' of a type (line 236)
        type_3036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 16), dtype_3035, 'type')
        # Getting the type of 'np' (line 236)
        np_3037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 45), 'np', False)
        # Obtaining the member 'object_' of a type (line 236)
        object__3038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 45), np_3037, 'object_')
        # Applying the binary operator 'is' (line 236)
        result_is__3039 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 16), 'is', type_3036, object__3038)
        
        # Processing the call keyword arguments (line 236)
        kwargs_3040 = {}
        # Getting the type of 'assert_' (line 236)
        assert__3031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 236)
        assert__call_result_3041 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), assert__3031, *[result_is__3039], **kwargs_3040)
        
        
        # Call to assert_(...): (line 237)
        # Processing the call arguments (line 237)
        
        # Getting the type of 's' (line 237)
        s_3043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 237)
        arrays_rep_3044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 16), s_3043, 'arrays_rep')
        # Obtaining the member 'b' of a type (line 237)
        b_3045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 16), arrays_rep_3044, 'b')
        # Obtaining the member 'dtype' of a type (line 237)
        dtype_3046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 16), b_3045, 'dtype')
        # Obtaining the member 'type' of a type (line 237)
        type_3047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 16), dtype_3046, 'type')
        # Getting the type of 'np' (line 237)
        np_3048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 45), 'np', False)
        # Obtaining the member 'object_' of a type (line 237)
        object__3049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 45), np_3048, 'object_')
        # Applying the binary operator 'is' (line 237)
        result_is__3050 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 16), 'is', type_3047, object__3049)
        
        # Processing the call keyword arguments (line 237)
        kwargs_3051 = {}
        # Getting the type of 'assert_' (line 237)
        assert__3042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 237)
        assert__call_result_3052 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), assert__3042, *[result_is__3050], **kwargs_3051)
        
        
        # Call to assert_(...): (line 238)
        # Processing the call arguments (line 238)
        
        # Getting the type of 's' (line 238)
        s_3054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 238)
        arrays_rep_3055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 16), s_3054, 'arrays_rep')
        # Obtaining the member 'c' of a type (line 238)
        c_3056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 16), arrays_rep_3055, 'c')
        # Obtaining the member 'dtype' of a type (line 238)
        dtype_3057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 16), c_3056, 'dtype')
        # Obtaining the member 'type' of a type (line 238)
        type_3058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 16), dtype_3057, 'type')
        # Getting the type of 'np' (line 238)
        np_3059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 45), 'np', False)
        # Obtaining the member 'object_' of a type (line 238)
        object__3060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 45), np_3059, 'object_')
        # Applying the binary operator 'is' (line 238)
        result_is__3061 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 16), 'is', type_3058, object__3060)
        
        # Processing the call keyword arguments (line 238)
        kwargs_3062 = {}
        # Getting the type of 'assert_' (line 238)
        assert__3053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 238)
        assert__call_result_3063 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), assert__3053, *[result_is__3061], **kwargs_3062)
        
        
        # Call to assert_(...): (line 239)
        # Processing the call arguments (line 239)
        
        # Getting the type of 's' (line 239)
        s_3065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 239)
        arrays_rep_3066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 16), s_3065, 'arrays_rep')
        # Obtaining the member 'd' of a type (line 239)
        d_3067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 16), arrays_rep_3066, 'd')
        # Obtaining the member 'dtype' of a type (line 239)
        dtype_3068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 16), d_3067, 'dtype')
        # Obtaining the member 'type' of a type (line 239)
        type_3069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 16), dtype_3068, 'type')
        # Getting the type of 'np' (line 239)
        np_3070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 45), 'np', False)
        # Obtaining the member 'object_' of a type (line 239)
        object__3071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 45), np_3070, 'object_')
        # Applying the binary operator 'is' (line 239)
        result_is__3072 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 16), 'is', type_3069, object__3071)
        
        # Processing the call keyword arguments (line 239)
        kwargs_3073 = {}
        # Getting the type of 'assert_' (line 239)
        assert__3064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 239)
        assert__call_result_3074 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), assert__3064, *[result_is__3072], **kwargs_3073)
        
        
        # Call to assert_equal(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 's' (line 242)
        s_3076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 21), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 242)
        arrays_rep_3077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 21), s_3076, 'arrays_rep')
        # Obtaining the member 'a' of a type (line 242)
        a_3078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 21), arrays_rep_3077, 'a')
        # Obtaining the member 'shape' of a type (line 242)
        shape_3079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 21), a_3078, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 242)
        tuple_3080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 242)
        # Adding element type (line 242)
        int_3081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 44), tuple_3080, int_3081)
        # Adding element type (line 242)
        int_3082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 44), tuple_3080, int_3082)
        # Adding element type (line 242)
        int_3083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 242, 44), tuple_3080, int_3083)
        
        # Processing the call keyword arguments (line 242)
        kwargs_3084 = {}
        # Getting the type of 'assert_equal' (line 242)
        assert_equal_3075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 242)
        assert_equal_call_result_3085 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), assert_equal_3075, *[shape_3079, tuple_3080], **kwargs_3084)
        
        
        # Call to assert_equal(...): (line 243)
        # Processing the call arguments (line 243)
        # Getting the type of 's' (line 243)
        s_3087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 21), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 243)
        arrays_rep_3088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 21), s_3087, 'arrays_rep')
        # Obtaining the member 'b' of a type (line 243)
        b_3089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 21), arrays_rep_3088, 'b')
        # Obtaining the member 'shape' of a type (line 243)
        shape_3090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 21), b_3089, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 243)
        tuple_3091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 243)
        # Adding element type (line 243)
        int_3092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 44), tuple_3091, int_3092)
        # Adding element type (line 243)
        int_3093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 44), tuple_3091, int_3093)
        # Adding element type (line 243)
        int_3094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 44), tuple_3091, int_3094)
        
        # Processing the call keyword arguments (line 243)
        kwargs_3095 = {}
        # Getting the type of 'assert_equal' (line 243)
        assert_equal_3086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 243)
        assert_equal_call_result_3096 = invoke(stypy.reporting.localization.Localization(__file__, 243, 8), assert_equal_3086, *[shape_3090, tuple_3091], **kwargs_3095)
        
        
        # Call to assert_equal(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 's' (line 244)
        s_3098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 21), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 244)
        arrays_rep_3099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 21), s_3098, 'arrays_rep')
        # Obtaining the member 'c' of a type (line 244)
        c_3100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 21), arrays_rep_3099, 'c')
        # Obtaining the member 'shape' of a type (line 244)
        shape_3101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 21), c_3100, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 244)
        tuple_3102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 244)
        # Adding element type (line 244)
        int_3103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 44), tuple_3102, int_3103)
        # Adding element type (line 244)
        int_3104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 44), tuple_3102, int_3104)
        # Adding element type (line 244)
        int_3105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 44), tuple_3102, int_3105)
        
        # Processing the call keyword arguments (line 244)
        kwargs_3106 = {}
        # Getting the type of 'assert_equal' (line 244)
        assert_equal_3097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 244)
        assert_equal_call_result_3107 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), assert_equal_3097, *[shape_3101, tuple_3102], **kwargs_3106)
        
        
        # Call to assert_equal(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 's' (line 245)
        s_3109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 21), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 245)
        arrays_rep_3110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 21), s_3109, 'arrays_rep')
        # Obtaining the member 'd' of a type (line 245)
        d_3111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 21), arrays_rep_3110, 'd')
        # Obtaining the member 'shape' of a type (line 245)
        shape_3112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 21), d_3111, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 245)
        tuple_3113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 245)
        # Adding element type (line 245)
        int_3114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 44), tuple_3113, int_3114)
        # Adding element type (line 245)
        int_3115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 44), tuple_3113, int_3115)
        # Adding element type (line 245)
        int_3116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 44), tuple_3113, int_3116)
        
        # Processing the call keyword arguments (line 245)
        kwargs_3117 = {}
        # Getting the type of 'assert_equal' (line 245)
        assert_equal_3108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 245)
        assert_equal_call_result_3118 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), assert_equal_3108, *[shape_3112, tuple_3113], **kwargs_3117)
        
        
        
        # Call to range(...): (line 248)
        # Processing the call arguments (line 248)
        int_3120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 23), 'int')
        # Processing the call keyword arguments (line 248)
        kwargs_3121 = {}
        # Getting the type of 'range' (line 248)
        range_3119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 17), 'range', False)
        # Calling range(args, kwargs) (line 248)
        range_call_result_3122 = invoke(stypy.reporting.localization.Localization(__file__, 248, 17), range_3119, *[int_3120], **kwargs_3121)
        
        # Testing the type of a for loop iterable (line 248)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 248, 8), range_call_result_3122)
        # Getting the type of the for loop variable (line 248)
        for_loop_var_3123 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 248, 8), range_call_result_3122)
        # Assigning a type to the variable 'i' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'i', for_loop_var_3123)
        # SSA begins for a for statement (line 248)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 249)
        # Processing the call arguments (line 249)
        int_3125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 27), 'int')
        # Processing the call keyword arguments (line 249)
        kwargs_3126 = {}
        # Getting the type of 'range' (line 249)
        range_3124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 21), 'range', False)
        # Calling range(args, kwargs) (line 249)
        range_call_result_3127 = invoke(stypy.reporting.localization.Localization(__file__, 249, 21), range_3124, *[int_3125], **kwargs_3126)
        
        # Testing the type of a for loop iterable (line 249)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 249, 12), range_call_result_3127)
        # Getting the type of the for loop variable (line 249)
        for_loop_var_3128 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 249, 12), range_call_result_3127)
        # Assigning a type to the variable 'j' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'j', for_loop_var_3128)
        # SSA begins for a for statement (line 249)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 250)
        # Processing the call arguments (line 250)
        int_3130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 31), 'int')
        # Processing the call keyword arguments (line 250)
        kwargs_3131 = {}
        # Getting the type of 'range' (line 250)
        range_3129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 25), 'range', False)
        # Calling range(args, kwargs) (line 250)
        range_call_result_3132 = invoke(stypy.reporting.localization.Localization(__file__, 250, 25), range_3129, *[int_3130], **kwargs_3131)
        
        # Testing the type of a for loop iterable (line 250)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 250, 16), range_call_result_3132)
        # Getting the type of the for loop variable (line 250)
        for_loop_var_3133 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 250, 16), range_call_result_3132)
        # Assigning a type to the variable 'k' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'k', for_loop_var_3133)
        # SSA begins for a for statement (line 250)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_array_identical(...): (line 251)
        # Processing the call arguments (line 251)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 251)
        tuple_3135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 251)
        # Adding element type (line 251)
        # Getting the type of 'i' (line 251)
        i_3136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 58), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 58), tuple_3135, i_3136)
        # Adding element type (line 251)
        # Getting the type of 'j' (line 251)
        j_3137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 61), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 58), tuple_3135, j_3137)
        # Adding element type (line 251)
        # Getting the type of 'k' (line 251)
        k_3138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 64), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 58), tuple_3135, k_3138)
        
        # Getting the type of 's' (line 251)
        s_3139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 43), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 251)
        arrays_rep_3140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 43), s_3139, 'arrays_rep')
        # Obtaining the member 'a' of a type (line 251)
        a_3141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 43), arrays_rep_3140, 'a')
        # Obtaining the member '__getitem__' of a type (line 251)
        getitem___3142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 43), a_3141, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 251)
        subscript_call_result_3143 = invoke(stypy.reporting.localization.Localization(__file__, 251, 43), getitem___3142, tuple_3135)
        
        
        # Call to array(...): (line 252)
        # Processing the call arguments (line 252)
        
        # Obtaining an instance of the builtin type 'list' (line 252)
        list_3146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 252)
        # Adding element type (line 252)
        int_3147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 53), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 52), list_3146, int_3147)
        # Adding element type (line 252)
        int_3148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 56), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 52), list_3146, int_3148)
        # Adding element type (line 252)
        int_3149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 59), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 52), list_3146, int_3149)
        
        # Processing the call keyword arguments (line 252)
        # Getting the type of 'np' (line 252)
        np_3150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 69), 'np', False)
        # Obtaining the member 'int16' of a type (line 252)
        int16_3151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 69), np_3150, 'int16')
        keyword_3152 = int16_3151
        kwargs_3153 = {'dtype': keyword_3152}
        # Getting the type of 'np' (line 252)
        np_3144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 43), 'np', False)
        # Obtaining the member 'array' of a type (line 252)
        array_3145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 43), np_3144, 'array')
        # Calling array(args, kwargs) (line 252)
        array_call_result_3154 = invoke(stypy.reporting.localization.Localization(__file__, 252, 43), array_3145, *[list_3146], **kwargs_3153)
        
        # Processing the call keyword arguments (line 251)
        kwargs_3155 = {}
        # Getting the type of 'assert_array_identical' (line 251)
        assert_array_identical_3134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 20), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 251)
        assert_array_identical_call_result_3156 = invoke(stypy.reporting.localization.Localization(__file__, 251, 20), assert_array_identical_3134, *[subscript_call_result_3143, array_call_result_3154], **kwargs_3155)
        
        
        # Call to assert_array_identical(...): (line 253)
        # Processing the call arguments (line 253)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 253)
        tuple_3158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 253)
        # Adding element type (line 253)
        # Getting the type of 'i' (line 253)
        i_3159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 58), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 58), tuple_3158, i_3159)
        # Adding element type (line 253)
        # Getting the type of 'j' (line 253)
        j_3160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 61), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 58), tuple_3158, j_3160)
        # Adding element type (line 253)
        # Getting the type of 'k' (line 253)
        k_3161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 64), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 58), tuple_3158, k_3161)
        
        # Getting the type of 's' (line 253)
        s_3162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 43), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 253)
        arrays_rep_3163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 43), s_3162, 'arrays_rep')
        # Obtaining the member 'b' of a type (line 253)
        b_3164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 43), arrays_rep_3163, 'b')
        # Obtaining the member '__getitem__' of a type (line 253)
        getitem___3165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 43), b_3164, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 253)
        subscript_call_result_3166 = invoke(stypy.reporting.localization.Localization(__file__, 253, 43), getitem___3165, tuple_3158)
        
        
        # Call to array(...): (line 254)
        # Processing the call arguments (line 254)
        
        # Obtaining an instance of the builtin type 'list' (line 254)
        list_3169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 254)
        # Adding element type (line 254)
        float_3170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 53), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 52), list_3169, float_3170)
        # Adding element type (line 254)
        float_3171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 57), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 52), list_3169, float_3171)
        # Adding element type (line 254)
        float_3172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 61), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 52), list_3169, float_3172)
        # Adding element type (line 254)
        float_3173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 65), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 254, 52), list_3169, float_3173)
        
        # Processing the call keyword arguments (line 254)
        # Getting the type of 'np' (line 255)
        np_3174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 58), 'np', False)
        # Obtaining the member 'float32' of a type (line 255)
        float32_3175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 58), np_3174, 'float32')
        keyword_3176 = float32_3175
        kwargs_3177 = {'dtype': keyword_3176}
        # Getting the type of 'np' (line 254)
        np_3167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 43), 'np', False)
        # Obtaining the member 'array' of a type (line 254)
        array_3168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 43), np_3167, 'array')
        # Calling array(args, kwargs) (line 254)
        array_call_result_3178 = invoke(stypy.reporting.localization.Localization(__file__, 254, 43), array_3168, *[list_3169], **kwargs_3177)
        
        # Processing the call keyword arguments (line 253)
        kwargs_3179 = {}
        # Getting the type of 'assert_array_identical' (line 253)
        assert_array_identical_3157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 20), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 253)
        assert_array_identical_call_result_3180 = invoke(stypy.reporting.localization.Localization(__file__, 253, 20), assert_array_identical_3157, *[subscript_call_result_3166, array_call_result_3178], **kwargs_3179)
        
        
        # Call to assert_array_identical(...): (line 256)
        # Processing the call arguments (line 256)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 256)
        tuple_3182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 256)
        # Adding element type (line 256)
        # Getting the type of 'i' (line 256)
        i_3183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 58), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 58), tuple_3182, i_3183)
        # Adding element type (line 256)
        # Getting the type of 'j' (line 256)
        j_3184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 61), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 58), tuple_3182, j_3184)
        # Adding element type (line 256)
        # Getting the type of 'k' (line 256)
        k_3185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 64), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 58), tuple_3182, k_3185)
        
        # Getting the type of 's' (line 256)
        s_3186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 43), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 256)
        arrays_rep_3187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 43), s_3186, 'arrays_rep')
        # Obtaining the member 'c' of a type (line 256)
        c_3188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 43), arrays_rep_3187, 'c')
        # Obtaining the member '__getitem__' of a type (line 256)
        getitem___3189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 43), c_3188, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 256)
        subscript_call_result_3190 = invoke(stypy.reporting.localization.Localization(__file__, 256, 43), getitem___3189, tuple_3182)
        
        
        # Call to array(...): (line 257)
        # Processing the call arguments (line 257)
        
        # Obtaining an instance of the builtin type 'list' (line 257)
        list_3193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 257)
        # Adding element type (line 257)
        
        # Call to complex64(...): (line 257)
        # Processing the call arguments (line 257)
        int_3196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 66), 'int')
        complex_3197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 68), 'complex')
        # Applying the binary operator '+' (line 257)
        result_add_3198 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 66), '+', int_3196, complex_3197)
        
        # Processing the call keyword arguments (line 257)
        kwargs_3199 = {}
        # Getting the type of 'np' (line 257)
        np_3194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 53), 'np', False)
        # Obtaining the member 'complex64' of a type (line 257)
        complex64_3195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 53), np_3194, 'complex64')
        # Calling complex64(args, kwargs) (line 257)
        complex64_call_result_3200 = invoke(stypy.reporting.localization.Localization(__file__, 257, 53), complex64_3195, *[result_add_3198], **kwargs_3199)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 52), list_3193, complex64_call_result_3200)
        # Adding element type (line 257)
        
        # Call to complex64(...): (line 258)
        # Processing the call arguments (line 258)
        int_3203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 66), 'int')
        complex_3204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 68), 'complex')
        # Applying the binary operator '+' (line 258)
        result_add_3205 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 66), '+', int_3203, complex_3204)
        
        # Processing the call keyword arguments (line 258)
        kwargs_3206 = {}
        # Getting the type of 'np' (line 258)
        np_3201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 53), 'np', False)
        # Obtaining the member 'complex64' of a type (line 258)
        complex64_3202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 53), np_3201, 'complex64')
        # Calling complex64(args, kwargs) (line 258)
        complex64_call_result_3207 = invoke(stypy.reporting.localization.Localization(__file__, 258, 53), complex64_3202, *[result_add_3205], **kwargs_3206)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 52), list_3193, complex64_call_result_3207)
        
        # Processing the call keyword arguments (line 257)
        kwargs_3208 = {}
        # Getting the type of 'np' (line 257)
        np_3191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 43), 'np', False)
        # Obtaining the member 'array' of a type (line 257)
        array_3192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 43), np_3191, 'array')
        # Calling array(args, kwargs) (line 257)
        array_call_result_3209 = invoke(stypy.reporting.localization.Localization(__file__, 257, 43), array_3192, *[list_3193], **kwargs_3208)
        
        # Processing the call keyword arguments (line 256)
        kwargs_3210 = {}
        # Getting the type of 'assert_array_identical' (line 256)
        assert_array_identical_3181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 20), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 256)
        assert_array_identical_call_result_3211 = invoke(stypy.reporting.localization.Localization(__file__, 256, 20), assert_array_identical_3181, *[subscript_call_result_3190, array_call_result_3209], **kwargs_3210)
        
        
        # Call to assert_array_identical(...): (line 259)
        # Processing the call arguments (line 259)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 259)
        tuple_3213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 259)
        # Adding element type (line 259)
        # Getting the type of 'i' (line 259)
        i_3214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 58), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 58), tuple_3213, i_3214)
        # Adding element type (line 259)
        # Getting the type of 'j' (line 259)
        j_3215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 61), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 58), tuple_3213, j_3215)
        # Adding element type (line 259)
        # Getting the type of 'k' (line 259)
        k_3216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 64), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 58), tuple_3213, k_3216)
        
        # Getting the type of 's' (line 259)
        s_3217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 43), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 259)
        arrays_rep_3218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 43), s_3217, 'arrays_rep')
        # Obtaining the member 'd' of a type (line 259)
        d_3219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 43), arrays_rep_3218, 'd')
        # Obtaining the member '__getitem__' of a type (line 259)
        getitem___3220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 43), d_3219, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 259)
        subscript_call_result_3221 = invoke(stypy.reporting.localization.Localization(__file__, 259, 43), getitem___3220, tuple_3213)
        
        
        # Call to array(...): (line 260)
        # Processing the call arguments (line 260)
        
        # Obtaining an instance of the builtin type 'list' (line 260)
        list_3224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 52), 'list')
        # Adding type elements to the builtin type 'list' instance (line 260)
        # Adding element type (line 260)
        str_3225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 53), 'str', 'cheese')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 52), list_3224, str_3225)
        # Adding element type (line 260)
        str_3226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 64), 'str', 'bacon')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 52), list_3224, str_3226)
        # Adding element type (line 260)
        str_3227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 74), 'str', 'spam')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 52), list_3224, str_3227)
        
        # Processing the call keyword arguments (line 260)
        # Getting the type of 'object' (line 261)
        object_3228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 58), 'object', False)
        keyword_3229 = object_3228
        kwargs_3230 = {'dtype': keyword_3229}
        # Getting the type of 'np' (line 260)
        np_3222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 43), 'np', False)
        # Obtaining the member 'array' of a type (line 260)
        array_3223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 43), np_3222, 'array')
        # Calling array(args, kwargs) (line 260)
        array_call_result_3231 = invoke(stypy.reporting.localization.Localization(__file__, 260, 43), array_3223, *[list_3224], **kwargs_3230)
        
        # Processing the call keyword arguments (line 259)
        kwargs_3232 = {}
        # Getting the type of 'assert_array_identical' (line 259)
        assert_array_identical_3212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 20), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 259)
        assert_array_identical_call_result_3233 = invoke(stypy.reporting.localization.Localization(__file__, 259, 20), assert_array_identical_3212, *[subscript_call_result_3221, array_call_result_3231], **kwargs_3232)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_arrays_replicated_3d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_arrays_replicated_3d' in the type store
        # Getting the type of 'stypy_return_type' (line 232)
        stypy_return_type_3234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3234)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_arrays_replicated_3d'
        return stypy_return_type_3234


    @norecursion
    def test_inheritance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_inheritance'
        module_type_store = module_type_store.open_function_context('test_inheritance', 263, 4, False)
        # Assigning a type to the variable 'self' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestStructures.test_inheritance.__dict__.__setitem__('stypy_localization', localization)
        TestStructures.test_inheritance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestStructures.test_inheritance.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestStructures.test_inheritance.__dict__.__setitem__('stypy_function_name', 'TestStructures.test_inheritance')
        TestStructures.test_inheritance.__dict__.__setitem__('stypy_param_names_list', [])
        TestStructures.test_inheritance.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestStructures.test_inheritance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestStructures.test_inheritance.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestStructures.test_inheritance.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestStructures.test_inheritance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestStructures.test_inheritance.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestStructures.test_inheritance', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_inheritance', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_inheritance(...)' code ##################

        
        # Assigning a Call to a Name (line 264):
        
        # Call to readsav(...): (line 264)
        # Processing the call arguments (line 264)
        
        # Call to join(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'DATA_PATH' (line 264)
        DATA_PATH_3238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 30), 'DATA_PATH', False)
        str_3239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 41), 'str', 'struct_inherit.sav')
        # Processing the call keyword arguments (line 264)
        kwargs_3240 = {}
        # Getting the type of 'path' (line 264)
        path_3236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 264)
        join_3237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 20), path_3236, 'join')
        # Calling join(args, kwargs) (line 264)
        join_call_result_3241 = invoke(stypy.reporting.localization.Localization(__file__, 264, 20), join_3237, *[DATA_PATH_3238, str_3239], **kwargs_3240)
        
        # Processing the call keyword arguments (line 264)
        # Getting the type of 'False' (line 264)
        False_3242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 72), 'False', False)
        keyword_3243 = False_3242
        kwargs_3244 = {'verbose': keyword_3243}
        # Getting the type of 'readsav' (line 264)
        readsav_3235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 264)
        readsav_call_result_3245 = invoke(stypy.reporting.localization.Localization(__file__, 264, 12), readsav_3235, *[join_call_result_3241], **kwargs_3244)
        
        # Assigning a type to the variable 's' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 's', readsav_call_result_3245)
        
        # Call to assert_identical(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 's' (line 265)
        s_3247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 25), 's', False)
        # Obtaining the member 'fc' of a type (line 265)
        fc_3248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 25), s_3247, 'fc')
        # Obtaining the member 'x' of a type (line 265)
        x_3249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 25), fc_3248, 'x')
        
        # Call to array(...): (line 265)
        # Processing the call arguments (line 265)
        
        # Obtaining an instance of the builtin type 'list' (line 265)
        list_3252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 265)
        # Adding element type (line 265)
        int_3253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 42), list_3252, int_3253)
        
        # Processing the call keyword arguments (line 265)
        # Getting the type of 'np' (line 265)
        np_3254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 53), 'np', False)
        # Obtaining the member 'int16' of a type (line 265)
        int16_3255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 53), np_3254, 'int16')
        keyword_3256 = int16_3255
        kwargs_3257 = {'dtype': keyword_3256}
        # Getting the type of 'np' (line 265)
        np_3250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 33), 'np', False)
        # Obtaining the member 'array' of a type (line 265)
        array_3251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 33), np_3250, 'array')
        # Calling array(args, kwargs) (line 265)
        array_call_result_3258 = invoke(stypy.reporting.localization.Localization(__file__, 265, 33), array_3251, *[list_3252], **kwargs_3257)
        
        # Processing the call keyword arguments (line 265)
        kwargs_3259 = {}
        # Getting the type of 'assert_identical' (line 265)
        assert_identical_3246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 265)
        assert_identical_call_result_3260 = invoke(stypy.reporting.localization.Localization(__file__, 265, 8), assert_identical_3246, *[x_3249, array_call_result_3258], **kwargs_3259)
        
        
        # Call to assert_identical(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 's' (line 266)
        s_3262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 25), 's', False)
        # Obtaining the member 'fc' of a type (line 266)
        fc_3263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 25), s_3262, 'fc')
        # Obtaining the member 'y' of a type (line 266)
        y_3264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 25), fc_3263, 'y')
        
        # Call to array(...): (line 266)
        # Processing the call arguments (line 266)
        
        # Obtaining an instance of the builtin type 'list' (line 266)
        list_3267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 266)
        # Adding element type (line 266)
        int_3268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 42), list_3267, int_3268)
        
        # Processing the call keyword arguments (line 266)
        # Getting the type of 'np' (line 266)
        np_3269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 53), 'np', False)
        # Obtaining the member 'int16' of a type (line 266)
        int16_3270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 53), np_3269, 'int16')
        keyword_3271 = int16_3270
        kwargs_3272 = {'dtype': keyword_3271}
        # Getting the type of 'np' (line 266)
        np_3265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 33), 'np', False)
        # Obtaining the member 'array' of a type (line 266)
        array_3266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 33), np_3265, 'array')
        # Calling array(args, kwargs) (line 266)
        array_call_result_3273 = invoke(stypy.reporting.localization.Localization(__file__, 266, 33), array_3266, *[list_3267], **kwargs_3272)
        
        # Processing the call keyword arguments (line 266)
        kwargs_3274 = {}
        # Getting the type of 'assert_identical' (line 266)
        assert_identical_3261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 266)
        assert_identical_call_result_3275 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), assert_identical_3261, *[y_3264, array_call_result_3273], **kwargs_3274)
        
        
        # Call to assert_identical(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 's' (line 267)
        s_3277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 25), 's', False)
        # Obtaining the member 'fc' of a type (line 267)
        fc_3278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 25), s_3277, 'fc')
        # Obtaining the member 'r' of a type (line 267)
        r_3279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 25), fc_3278, 'r')
        
        # Call to array(...): (line 267)
        # Processing the call arguments (line 267)
        
        # Obtaining an instance of the builtin type 'list' (line 267)
        list_3282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 267)
        # Adding element type (line 267)
        int_3283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 42), list_3282, int_3283)
        
        # Processing the call keyword arguments (line 267)
        # Getting the type of 'np' (line 267)
        np_3284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 53), 'np', False)
        # Obtaining the member 'int16' of a type (line 267)
        int16_3285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 53), np_3284, 'int16')
        keyword_3286 = int16_3285
        kwargs_3287 = {'dtype': keyword_3286}
        # Getting the type of 'np' (line 267)
        np_3280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 33), 'np', False)
        # Obtaining the member 'array' of a type (line 267)
        array_3281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 33), np_3280, 'array')
        # Calling array(args, kwargs) (line 267)
        array_call_result_3288 = invoke(stypy.reporting.localization.Localization(__file__, 267, 33), array_3281, *[list_3282], **kwargs_3287)
        
        # Processing the call keyword arguments (line 267)
        kwargs_3289 = {}
        # Getting the type of 'assert_identical' (line 267)
        assert_identical_3276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 267)
        assert_identical_call_result_3290 = invoke(stypy.reporting.localization.Localization(__file__, 267, 8), assert_identical_3276, *[r_3279, array_call_result_3288], **kwargs_3289)
        
        
        # Call to assert_identical(...): (line 268)
        # Processing the call arguments (line 268)
        # Getting the type of 's' (line 268)
        s_3292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 25), 's', False)
        # Obtaining the member 'fc' of a type (line 268)
        fc_3293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 25), s_3292, 'fc')
        # Obtaining the member 'c' of a type (line 268)
        c_3294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 25), fc_3293, 'c')
        
        # Call to array(...): (line 268)
        # Processing the call arguments (line 268)
        
        # Obtaining an instance of the builtin type 'list' (line 268)
        list_3297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 268)
        # Adding element type (line 268)
        int_3298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 42), list_3297, int_3298)
        
        # Processing the call keyword arguments (line 268)
        # Getting the type of 'np' (line 268)
        np_3299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 53), 'np', False)
        # Obtaining the member 'int16' of a type (line 268)
        int16_3300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 53), np_3299, 'int16')
        keyword_3301 = int16_3300
        kwargs_3302 = {'dtype': keyword_3301}
        # Getting the type of 'np' (line 268)
        np_3295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 33), 'np', False)
        # Obtaining the member 'array' of a type (line 268)
        array_3296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 33), np_3295, 'array')
        # Calling array(args, kwargs) (line 268)
        array_call_result_3303 = invoke(stypy.reporting.localization.Localization(__file__, 268, 33), array_3296, *[list_3297], **kwargs_3302)
        
        # Processing the call keyword arguments (line 268)
        kwargs_3304 = {}
        # Getting the type of 'assert_identical' (line 268)
        assert_identical_3291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 268)
        assert_identical_call_result_3305 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), assert_identical_3291, *[c_3294, array_call_result_3303], **kwargs_3304)
        
        
        # ################# End of 'test_inheritance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_inheritance' in the type store
        # Getting the type of 'stypy_return_type' (line 263)
        stypy_return_type_3306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3306)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_inheritance'
        return stypy_return_type_3306


    @norecursion
    def test_arrays_corrupt_idl80(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_arrays_corrupt_idl80'
        module_type_store = module_type_store.open_function_context('test_arrays_corrupt_idl80', 270, 4, False)
        # Assigning a type to the variable 'self' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestStructures.test_arrays_corrupt_idl80.__dict__.__setitem__('stypy_localization', localization)
        TestStructures.test_arrays_corrupt_idl80.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestStructures.test_arrays_corrupt_idl80.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestStructures.test_arrays_corrupt_idl80.__dict__.__setitem__('stypy_function_name', 'TestStructures.test_arrays_corrupt_idl80')
        TestStructures.test_arrays_corrupt_idl80.__dict__.__setitem__('stypy_param_names_list', [])
        TestStructures.test_arrays_corrupt_idl80.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestStructures.test_arrays_corrupt_idl80.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestStructures.test_arrays_corrupt_idl80.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestStructures.test_arrays_corrupt_idl80.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestStructures.test_arrays_corrupt_idl80.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestStructures.test_arrays_corrupt_idl80.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestStructures.test_arrays_corrupt_idl80', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_arrays_corrupt_idl80', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_arrays_corrupt_idl80(...)' code ##################

        
        # Call to suppress_warnings(...): (line 272)
        # Processing the call keyword arguments (line 272)
        kwargs_3308 = {}
        # Getting the type of 'suppress_warnings' (line 272)
        suppress_warnings_3307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 13), 'suppress_warnings', False)
        # Calling suppress_warnings(args, kwargs) (line 272)
        suppress_warnings_call_result_3309 = invoke(stypy.reporting.localization.Localization(__file__, 272, 13), suppress_warnings_3307, *[], **kwargs_3308)
        
        with_3310 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 272, 13), suppress_warnings_call_result_3309, 'with parameter', '__enter__', '__exit__')

        if with_3310:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 272)
            enter___3311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 13), suppress_warnings_call_result_3309, '__enter__')
            with_enter_3312 = invoke(stypy.reporting.localization.Localization(__file__, 272, 13), enter___3311)
            # Assigning a type to the variable 'sup' (line 272)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 13), 'sup', with_enter_3312)
            
            # Call to filter(...): (line 273)
            # Processing the call arguments (line 273)
            # Getting the type of 'UserWarning' (line 273)
            UserWarning_3315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 23), 'UserWarning', False)
            str_3316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 36), 'str', 'Not able to verify number of bytes from header')
            # Processing the call keyword arguments (line 273)
            kwargs_3317 = {}
            # Getting the type of 'sup' (line 273)
            sup_3313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'sup', False)
            # Obtaining the member 'filter' of a type (line 273)
            filter_3314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 12), sup_3313, 'filter')
            # Calling filter(args, kwargs) (line 273)
            filter_call_result_3318 = invoke(stypy.reporting.localization.Localization(__file__, 273, 12), filter_3314, *[UserWarning_3315, str_3316], **kwargs_3317)
            
            
            # Assigning a Call to a Name (line 274):
            
            # Call to readsav(...): (line 274)
            # Processing the call arguments (line 274)
            
            # Call to join(...): (line 274)
            # Processing the call arguments (line 274)
            # Getting the type of 'DATA_PATH' (line 274)
            DATA_PATH_3322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 34), 'DATA_PATH', False)
            str_3323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 44), 'str', 'struct_arrays_byte_idl80.sav')
            # Processing the call keyword arguments (line 274)
            kwargs_3324 = {}
            # Getting the type of 'path' (line 274)
            path_3320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 24), 'path', False)
            # Obtaining the member 'join' of a type (line 274)
            join_3321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 24), path_3320, 'join')
            # Calling join(args, kwargs) (line 274)
            join_call_result_3325 = invoke(stypy.reporting.localization.Localization(__file__, 274, 24), join_3321, *[DATA_PATH_3322, str_3323], **kwargs_3324)
            
            # Processing the call keyword arguments (line 274)
            # Getting the type of 'False' (line 275)
            False_3326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 32), 'False', False)
            keyword_3327 = False_3326
            kwargs_3328 = {'verbose': keyword_3327}
            # Getting the type of 'readsav' (line 274)
            readsav_3319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 16), 'readsav', False)
            # Calling readsav(args, kwargs) (line 274)
            readsav_call_result_3329 = invoke(stypy.reporting.localization.Localization(__file__, 274, 16), readsav_3319, *[join_call_result_3325], **kwargs_3328)
            
            # Assigning a type to the variable 's' (line 274)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 's', readsav_call_result_3329)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 272)
            exit___3330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 13), suppress_warnings_call_result_3309, '__exit__')
            with_exit_3331 = invoke(stypy.reporting.localization.Localization(__file__, 272, 13), exit___3330, None, None, None)

        
        # Call to assert_identical(...): (line 277)
        # Processing the call arguments (line 277)
        
        # Obtaining the type of the subscript
        int_3333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 31), 'int')
        # Getting the type of 's' (line 277)
        s_3334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 25), 's', False)
        # Obtaining the member 'y' of a type (line 277)
        y_3335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 25), s_3334, 'y')
        # Obtaining the member 'x' of a type (line 277)
        x_3336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 25), y_3335, 'x')
        # Obtaining the member '__getitem__' of a type (line 277)
        getitem___3337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 25), x_3336, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 277)
        subscript_call_result_3338 = invoke(stypy.reporting.localization.Localization(__file__, 277, 25), getitem___3337, int_3333)
        
        
        # Call to array(...): (line 277)
        # Processing the call arguments (line 277)
        
        # Obtaining an instance of the builtin type 'list' (line 277)
        list_3341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 277)
        # Adding element type (line 277)
        int_3342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 44), list_3341, int_3342)
        # Adding element type (line 277)
        int_3343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 44), list_3341, int_3343)
        
        # Processing the call keyword arguments (line 277)
        # Getting the type of 'np' (line 277)
        np_3344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 59), 'np', False)
        # Obtaining the member 'uint8' of a type (line 277)
        uint8_3345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 59), np_3344, 'uint8')
        keyword_3346 = uint8_3345
        kwargs_3347 = {'dtype': keyword_3346}
        # Getting the type of 'np' (line 277)
        np_3339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 35), 'np', False)
        # Obtaining the member 'array' of a type (line 277)
        array_3340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 35), np_3339, 'array')
        # Calling array(args, kwargs) (line 277)
        array_call_result_3348 = invoke(stypy.reporting.localization.Localization(__file__, 277, 35), array_3340, *[list_3341], **kwargs_3347)
        
        # Processing the call keyword arguments (line 277)
        kwargs_3349 = {}
        # Getting the type of 'assert_identical' (line 277)
        assert_identical_3332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 277)
        assert_identical_call_result_3350 = invoke(stypy.reporting.localization.Localization(__file__, 277, 8), assert_identical_3332, *[subscript_call_result_3338, array_call_result_3348], **kwargs_3349)
        
        
        # ################# End of 'test_arrays_corrupt_idl80(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_arrays_corrupt_idl80' in the type store
        # Getting the type of 'stypy_return_type' (line 270)
        stypy_return_type_3351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3351)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_arrays_corrupt_idl80'
        return stypy_return_type_3351


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 168, 0, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestStructures.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestStructures' (line 168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'TestStructures', TestStructures)
# Declaration of the 'TestPointers' class

class TestPointers:

    @norecursion
    def test_pointers(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_pointers'
        module_type_store = module_type_store.open_function_context('test_pointers', 283, 4, False)
        # Assigning a type to the variable 'self' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPointers.test_pointers.__dict__.__setitem__('stypy_localization', localization)
        TestPointers.test_pointers.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPointers.test_pointers.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPointers.test_pointers.__dict__.__setitem__('stypy_function_name', 'TestPointers.test_pointers')
        TestPointers.test_pointers.__dict__.__setitem__('stypy_param_names_list', [])
        TestPointers.test_pointers.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPointers.test_pointers.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPointers.test_pointers.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPointers.test_pointers.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPointers.test_pointers.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPointers.test_pointers.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointers.test_pointers', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_pointers', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_pointers(...)' code ##################

        
        # Assigning a Call to a Name (line 284):
        
        # Call to readsav(...): (line 284)
        # Processing the call arguments (line 284)
        
        # Call to join(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'DATA_PATH' (line 284)
        DATA_PATH_3355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 30), 'DATA_PATH', False)
        str_3356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 41), 'str', 'scalar_heap_pointer.sav')
        # Processing the call keyword arguments (line 284)
        kwargs_3357 = {}
        # Getting the type of 'path' (line 284)
        path_3353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 284)
        join_3354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 20), path_3353, 'join')
        # Calling join(args, kwargs) (line 284)
        join_call_result_3358 = invoke(stypy.reporting.localization.Localization(__file__, 284, 20), join_3354, *[DATA_PATH_3355, str_3356], **kwargs_3357)
        
        # Processing the call keyword arguments (line 284)
        # Getting the type of 'False' (line 284)
        False_3359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 77), 'False', False)
        keyword_3360 = False_3359
        kwargs_3361 = {'verbose': keyword_3360}
        # Getting the type of 'readsav' (line 284)
        readsav_3352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 284)
        readsav_call_result_3362 = invoke(stypy.reporting.localization.Localization(__file__, 284, 12), readsav_3352, *[join_call_result_3358], **kwargs_3361)
        
        # Assigning a type to the variable 's' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 's', readsav_call_result_3362)
        
        # Call to assert_identical(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 's' (line 285)
        s_3364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 25), 's', False)
        # Obtaining the member 'c64_pointer1' of a type (line 285)
        c64_pointer1_3365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 25), s_3364, 'c64_pointer1')
        
        # Call to complex128(...): (line 285)
        # Processing the call arguments (line 285)
        float_3368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 55), 'float')
        complex_3369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 79), 'complex')
        # Applying the binary operator '-' (line 285)
        result_sub_3370 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 55), '-', float_3368, complex_3369)
        
        # Processing the call keyword arguments (line 285)
        kwargs_3371 = {}
        # Getting the type of 'np' (line 285)
        np_3366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 41), 'np', False)
        # Obtaining the member 'complex128' of a type (line 285)
        complex128_3367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 41), np_3366, 'complex128')
        # Calling complex128(args, kwargs) (line 285)
        complex128_call_result_3372 = invoke(stypy.reporting.localization.Localization(__file__, 285, 41), complex128_3367, *[result_sub_3370], **kwargs_3371)
        
        # Processing the call keyword arguments (line 285)
        kwargs_3373 = {}
        # Getting the type of 'assert_identical' (line 285)
        assert_identical_3363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 285)
        assert_identical_call_result_3374 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), assert_identical_3363, *[c64_pointer1_3365, complex128_call_result_3372], **kwargs_3373)
        
        
        # Call to assert_identical(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 's' (line 286)
        s_3376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 25), 's', False)
        # Obtaining the member 'c64_pointer2' of a type (line 286)
        c64_pointer2_3377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 25), s_3376, 'c64_pointer2')
        
        # Call to complex128(...): (line 286)
        # Processing the call arguments (line 286)
        float_3380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 55), 'float')
        complex_3381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 79), 'complex')
        # Applying the binary operator '-' (line 286)
        result_sub_3382 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 55), '-', float_3380, complex_3381)
        
        # Processing the call keyword arguments (line 286)
        kwargs_3383 = {}
        # Getting the type of 'np' (line 286)
        np_3378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 41), 'np', False)
        # Obtaining the member 'complex128' of a type (line 286)
        complex128_3379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 41), np_3378, 'complex128')
        # Calling complex128(args, kwargs) (line 286)
        complex128_call_result_3384 = invoke(stypy.reporting.localization.Localization(__file__, 286, 41), complex128_3379, *[result_sub_3382], **kwargs_3383)
        
        # Processing the call keyword arguments (line 286)
        kwargs_3385 = {}
        # Getting the type of 'assert_identical' (line 286)
        assert_identical_3375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 286)
        assert_identical_call_result_3386 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), assert_identical_3375, *[c64_pointer2_3377, complex128_call_result_3384], **kwargs_3385)
        
        
        # Call to assert_(...): (line 287)
        # Processing the call arguments (line 287)
        
        # Getting the type of 's' (line 287)
        s_3388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 16), 's', False)
        # Obtaining the member 'c64_pointer1' of a type (line 287)
        c64_pointer1_3389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 16), s_3388, 'c64_pointer1')
        # Getting the type of 's' (line 287)
        s_3390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 34), 's', False)
        # Obtaining the member 'c64_pointer2' of a type (line 287)
        c64_pointer2_3391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 34), s_3390, 'c64_pointer2')
        # Applying the binary operator 'is' (line 287)
        result_is__3392 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 16), 'is', c64_pointer1_3389, c64_pointer2_3391)
        
        # Processing the call keyword arguments (line 287)
        kwargs_3393 = {}
        # Getting the type of 'assert_' (line 287)
        assert__3387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 287)
        assert__call_result_3394 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), assert__3387, *[result_is__3392], **kwargs_3393)
        
        
        # ################# End of 'test_pointers(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_pointers' in the type store
        # Getting the type of 'stypy_return_type' (line 283)
        stypy_return_type_3395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3395)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_pointers'
        return stypy_return_type_3395


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 280, 0, False)
        # Assigning a type to the variable 'self' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointers.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestPointers' (line 280)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 0), 'TestPointers', TestPointers)
# Declaration of the 'TestPointerArray' class

class TestPointerArray:

    @norecursion
    def test_1d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_1d'
        module_type_store = module_type_store.open_function_context('test_1d', 293, 4, False)
        # Assigning a type to the variable 'self' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPointerArray.test_1d.__dict__.__setitem__('stypy_localization', localization)
        TestPointerArray.test_1d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPointerArray.test_1d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPointerArray.test_1d.__dict__.__setitem__('stypy_function_name', 'TestPointerArray.test_1d')
        TestPointerArray.test_1d.__dict__.__setitem__('stypy_param_names_list', [])
        TestPointerArray.test_1d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPointerArray.test_1d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPointerArray.test_1d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPointerArray.test_1d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPointerArray.test_1d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPointerArray.test_1d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointerArray.test_1d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_1d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_1d(...)' code ##################

        
        # Assigning a Call to a Name (line 294):
        
        # Call to readsav(...): (line 294)
        # Processing the call arguments (line 294)
        
        # Call to join(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'DATA_PATH' (line 294)
        DATA_PATH_3399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 30), 'DATA_PATH', False)
        str_3400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 41), 'str', 'array_float32_pointer_1d.sav')
        # Processing the call keyword arguments (line 294)
        kwargs_3401 = {}
        # Getting the type of 'path' (line 294)
        path_3397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 294)
        join_3398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 20), path_3397, 'join')
        # Calling join(args, kwargs) (line 294)
        join_call_result_3402 = invoke(stypy.reporting.localization.Localization(__file__, 294, 20), join_3398, *[DATA_PATH_3399, str_3400], **kwargs_3401)
        
        # Processing the call keyword arguments (line 294)
        # Getting the type of 'False' (line 294)
        False_3403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 82), 'False', False)
        keyword_3404 = False_3403
        kwargs_3405 = {'verbose': keyword_3404}
        # Getting the type of 'readsav' (line 294)
        readsav_3396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 294)
        readsav_call_result_3406 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), readsav_3396, *[join_call_result_3402], **kwargs_3405)
        
        # Assigning a type to the variable 's' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 's', readsav_call_result_3406)
        
        # Call to assert_equal(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 's' (line 295)
        s_3408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 21), 's', False)
        # Obtaining the member 'array1d' of a type (line 295)
        array1d_3409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 21), s_3408, 'array1d')
        # Obtaining the member 'shape' of a type (line 295)
        shape_3410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 21), array1d_3409, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 295)
        tuple_3411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 295)
        # Adding element type (line 295)
        int_3412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 39), tuple_3411, int_3412)
        
        # Processing the call keyword arguments (line 295)
        kwargs_3413 = {}
        # Getting the type of 'assert_equal' (line 295)
        assert_equal_3407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 295)
        assert_equal_call_result_3414 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), assert_equal_3407, *[shape_3410, tuple_3411], **kwargs_3413)
        
        
        # Call to assert_(...): (line 296)
        # Processing the call arguments (line 296)
        
        # Call to all(...): (line 296)
        # Processing the call arguments (line 296)
        
        # Getting the type of 's' (line 296)
        s_3418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 23), 's', False)
        # Obtaining the member 'array1d' of a type (line 296)
        array1d_3419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 23), s_3418, 'array1d')
        
        # Call to float32(...): (line 296)
        # Processing the call arguments (line 296)
        float_3422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 47), 'float')
        # Processing the call keyword arguments (line 296)
        kwargs_3423 = {}
        # Getting the type of 'np' (line 296)
        np_3420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 36), 'np', False)
        # Obtaining the member 'float32' of a type (line 296)
        float32_3421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 36), np_3420, 'float32')
        # Calling float32(args, kwargs) (line 296)
        float32_call_result_3424 = invoke(stypy.reporting.localization.Localization(__file__, 296, 36), float32_3421, *[float_3422], **kwargs_3423)
        
        # Applying the binary operator '==' (line 296)
        result_eq_3425 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 23), '==', array1d_3419, float32_call_result_3424)
        
        # Processing the call keyword arguments (line 296)
        kwargs_3426 = {}
        # Getting the type of 'np' (line 296)
        np_3416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 296)
        all_3417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 16), np_3416, 'all')
        # Calling all(args, kwargs) (line 296)
        all_call_result_3427 = invoke(stypy.reporting.localization.Localization(__file__, 296, 16), all_3417, *[result_eq_3425], **kwargs_3426)
        
        # Processing the call keyword arguments (line 296)
        kwargs_3428 = {}
        # Getting the type of 'assert_' (line 296)
        assert__3415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 296)
        assert__call_result_3429 = invoke(stypy.reporting.localization.Localization(__file__, 296, 8), assert__3415, *[all_call_result_3427], **kwargs_3428)
        
        
        # Call to assert_(...): (line 297)
        # Processing the call arguments (line 297)
        
        # Call to all(...): (line 297)
        # Processing the call arguments (line 297)
        
        
        # Call to vect_id(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 's' (line 297)
        s_3434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 31), 's', False)
        # Obtaining the member 'array1d' of a type (line 297)
        array1d_3435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 31), s_3434, 'array1d')
        # Processing the call keyword arguments (line 297)
        kwargs_3436 = {}
        # Getting the type of 'vect_id' (line 297)
        vect_id_3433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 23), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 297)
        vect_id_call_result_3437 = invoke(stypy.reporting.localization.Localization(__file__, 297, 23), vect_id_3433, *[array1d_3435], **kwargs_3436)
        
        
        # Call to id(...): (line 297)
        # Processing the call arguments (line 297)
        
        # Obtaining the type of the subscript
        int_3439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 58), 'int')
        # Getting the type of 's' (line 297)
        s_3440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 48), 's', False)
        # Obtaining the member 'array1d' of a type (line 297)
        array1d_3441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 48), s_3440, 'array1d')
        # Obtaining the member '__getitem__' of a type (line 297)
        getitem___3442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 48), array1d_3441, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 297)
        subscript_call_result_3443 = invoke(stypy.reporting.localization.Localization(__file__, 297, 48), getitem___3442, int_3439)
        
        # Processing the call keyword arguments (line 297)
        kwargs_3444 = {}
        # Getting the type of 'id' (line 297)
        id_3438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 45), 'id', False)
        # Calling id(args, kwargs) (line 297)
        id_call_result_3445 = invoke(stypy.reporting.localization.Localization(__file__, 297, 45), id_3438, *[subscript_call_result_3443], **kwargs_3444)
        
        # Applying the binary operator '==' (line 297)
        result_eq_3446 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 23), '==', vect_id_call_result_3437, id_call_result_3445)
        
        # Processing the call keyword arguments (line 297)
        kwargs_3447 = {}
        # Getting the type of 'np' (line 297)
        np_3431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 297)
        all_3432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 16), np_3431, 'all')
        # Calling all(args, kwargs) (line 297)
        all_call_result_3448 = invoke(stypy.reporting.localization.Localization(__file__, 297, 16), all_3432, *[result_eq_3446], **kwargs_3447)
        
        # Processing the call keyword arguments (line 297)
        kwargs_3449 = {}
        # Getting the type of 'assert_' (line 297)
        assert__3430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 297)
        assert__call_result_3450 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), assert__3430, *[all_call_result_3448], **kwargs_3449)
        
        
        # ################# End of 'test_1d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_1d' in the type store
        # Getting the type of 'stypy_return_type' (line 293)
        stypy_return_type_3451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3451)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_1d'
        return stypy_return_type_3451


    @norecursion
    def test_2d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_2d'
        module_type_store = module_type_store.open_function_context('test_2d', 299, 4, False)
        # Assigning a type to the variable 'self' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPointerArray.test_2d.__dict__.__setitem__('stypy_localization', localization)
        TestPointerArray.test_2d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPointerArray.test_2d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPointerArray.test_2d.__dict__.__setitem__('stypy_function_name', 'TestPointerArray.test_2d')
        TestPointerArray.test_2d.__dict__.__setitem__('stypy_param_names_list', [])
        TestPointerArray.test_2d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPointerArray.test_2d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPointerArray.test_2d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPointerArray.test_2d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPointerArray.test_2d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPointerArray.test_2d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointerArray.test_2d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_2d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_2d(...)' code ##################

        
        # Assigning a Call to a Name (line 300):
        
        # Call to readsav(...): (line 300)
        # Processing the call arguments (line 300)
        
        # Call to join(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'DATA_PATH' (line 300)
        DATA_PATH_3455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 30), 'DATA_PATH', False)
        str_3456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 41), 'str', 'array_float32_pointer_2d.sav')
        # Processing the call keyword arguments (line 300)
        kwargs_3457 = {}
        # Getting the type of 'path' (line 300)
        path_3453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 300)
        join_3454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 20), path_3453, 'join')
        # Calling join(args, kwargs) (line 300)
        join_call_result_3458 = invoke(stypy.reporting.localization.Localization(__file__, 300, 20), join_3454, *[DATA_PATH_3455, str_3456], **kwargs_3457)
        
        # Processing the call keyword arguments (line 300)
        # Getting the type of 'False' (line 300)
        False_3459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 82), 'False', False)
        keyword_3460 = False_3459
        kwargs_3461 = {'verbose': keyword_3460}
        # Getting the type of 'readsav' (line 300)
        readsav_3452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 300)
        readsav_call_result_3462 = invoke(stypy.reporting.localization.Localization(__file__, 300, 12), readsav_3452, *[join_call_result_3458], **kwargs_3461)
        
        # Assigning a type to the variable 's' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 's', readsav_call_result_3462)
        
        # Call to assert_equal(...): (line 301)
        # Processing the call arguments (line 301)
        # Getting the type of 's' (line 301)
        s_3464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 21), 's', False)
        # Obtaining the member 'array2d' of a type (line 301)
        array2d_3465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 21), s_3464, 'array2d')
        # Obtaining the member 'shape' of a type (line 301)
        shape_3466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 21), array2d_3465, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 301)
        tuple_3467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 301)
        # Adding element type (line 301)
        int_3468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 39), tuple_3467, int_3468)
        # Adding element type (line 301)
        int_3469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 301, 39), tuple_3467, int_3469)
        
        # Processing the call keyword arguments (line 301)
        kwargs_3470 = {}
        # Getting the type of 'assert_equal' (line 301)
        assert_equal_3463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 301)
        assert_equal_call_result_3471 = invoke(stypy.reporting.localization.Localization(__file__, 301, 8), assert_equal_3463, *[shape_3466, tuple_3467], **kwargs_3470)
        
        
        # Call to assert_(...): (line 302)
        # Processing the call arguments (line 302)
        
        # Call to all(...): (line 302)
        # Processing the call arguments (line 302)
        
        # Getting the type of 's' (line 302)
        s_3475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 23), 's', False)
        # Obtaining the member 'array2d' of a type (line 302)
        array2d_3476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 23), s_3475, 'array2d')
        
        # Call to float32(...): (line 302)
        # Processing the call arguments (line 302)
        float_3479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 47), 'float')
        # Processing the call keyword arguments (line 302)
        kwargs_3480 = {}
        # Getting the type of 'np' (line 302)
        np_3477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 36), 'np', False)
        # Obtaining the member 'float32' of a type (line 302)
        float32_3478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 36), np_3477, 'float32')
        # Calling float32(args, kwargs) (line 302)
        float32_call_result_3481 = invoke(stypy.reporting.localization.Localization(__file__, 302, 36), float32_3478, *[float_3479], **kwargs_3480)
        
        # Applying the binary operator '==' (line 302)
        result_eq_3482 = python_operator(stypy.reporting.localization.Localization(__file__, 302, 23), '==', array2d_3476, float32_call_result_3481)
        
        # Processing the call keyword arguments (line 302)
        kwargs_3483 = {}
        # Getting the type of 'np' (line 302)
        np_3473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 302)
        all_3474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 16), np_3473, 'all')
        # Calling all(args, kwargs) (line 302)
        all_call_result_3484 = invoke(stypy.reporting.localization.Localization(__file__, 302, 16), all_3474, *[result_eq_3482], **kwargs_3483)
        
        # Processing the call keyword arguments (line 302)
        kwargs_3485 = {}
        # Getting the type of 'assert_' (line 302)
        assert__3472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 302)
        assert__call_result_3486 = invoke(stypy.reporting.localization.Localization(__file__, 302, 8), assert__3472, *[all_call_result_3484], **kwargs_3485)
        
        
        # Call to assert_(...): (line 303)
        # Processing the call arguments (line 303)
        
        # Call to all(...): (line 303)
        # Processing the call arguments (line 303)
        
        
        # Call to vect_id(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 's' (line 303)
        s_3491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 31), 's', False)
        # Obtaining the member 'array2d' of a type (line 303)
        array2d_3492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 31), s_3491, 'array2d')
        # Processing the call keyword arguments (line 303)
        kwargs_3493 = {}
        # Getting the type of 'vect_id' (line 303)
        vect_id_3490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 23), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 303)
        vect_id_call_result_3494 = invoke(stypy.reporting.localization.Localization(__file__, 303, 23), vect_id_3490, *[array2d_3492], **kwargs_3493)
        
        
        # Call to id(...): (line 303)
        # Processing the call arguments (line 303)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 303)
        tuple_3496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 303)
        # Adding element type (line 303)
        int_3497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 58), tuple_3496, int_3497)
        # Adding element type (line 303)
        int_3498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 303, 58), tuple_3496, int_3498)
        
        # Getting the type of 's' (line 303)
        s_3499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 48), 's', False)
        # Obtaining the member 'array2d' of a type (line 303)
        array2d_3500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 48), s_3499, 'array2d')
        # Obtaining the member '__getitem__' of a type (line 303)
        getitem___3501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 48), array2d_3500, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 303)
        subscript_call_result_3502 = invoke(stypy.reporting.localization.Localization(__file__, 303, 48), getitem___3501, tuple_3496)
        
        # Processing the call keyword arguments (line 303)
        kwargs_3503 = {}
        # Getting the type of 'id' (line 303)
        id_3495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 45), 'id', False)
        # Calling id(args, kwargs) (line 303)
        id_call_result_3504 = invoke(stypy.reporting.localization.Localization(__file__, 303, 45), id_3495, *[subscript_call_result_3502], **kwargs_3503)
        
        # Applying the binary operator '==' (line 303)
        result_eq_3505 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 23), '==', vect_id_call_result_3494, id_call_result_3504)
        
        # Processing the call keyword arguments (line 303)
        kwargs_3506 = {}
        # Getting the type of 'np' (line 303)
        np_3488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 303)
        all_3489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 16), np_3488, 'all')
        # Calling all(args, kwargs) (line 303)
        all_call_result_3507 = invoke(stypy.reporting.localization.Localization(__file__, 303, 16), all_3489, *[result_eq_3505], **kwargs_3506)
        
        # Processing the call keyword arguments (line 303)
        kwargs_3508 = {}
        # Getting the type of 'assert_' (line 303)
        assert__3487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 303)
        assert__call_result_3509 = invoke(stypy.reporting.localization.Localization(__file__, 303, 8), assert__3487, *[all_call_result_3507], **kwargs_3508)
        
        
        # ################# End of 'test_2d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_2d' in the type store
        # Getting the type of 'stypy_return_type' (line 299)
        stypy_return_type_3510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3510)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_2d'
        return stypy_return_type_3510


    @norecursion
    def test_3d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_3d'
        module_type_store = module_type_store.open_function_context('test_3d', 305, 4, False)
        # Assigning a type to the variable 'self' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPointerArray.test_3d.__dict__.__setitem__('stypy_localization', localization)
        TestPointerArray.test_3d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPointerArray.test_3d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPointerArray.test_3d.__dict__.__setitem__('stypy_function_name', 'TestPointerArray.test_3d')
        TestPointerArray.test_3d.__dict__.__setitem__('stypy_param_names_list', [])
        TestPointerArray.test_3d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPointerArray.test_3d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPointerArray.test_3d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPointerArray.test_3d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPointerArray.test_3d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPointerArray.test_3d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointerArray.test_3d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_3d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_3d(...)' code ##################

        
        # Assigning a Call to a Name (line 306):
        
        # Call to readsav(...): (line 306)
        # Processing the call arguments (line 306)
        
        # Call to join(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'DATA_PATH' (line 306)
        DATA_PATH_3514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 30), 'DATA_PATH', False)
        str_3515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 41), 'str', 'array_float32_pointer_3d.sav')
        # Processing the call keyword arguments (line 306)
        kwargs_3516 = {}
        # Getting the type of 'path' (line 306)
        path_3512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 306)
        join_3513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 20), path_3512, 'join')
        # Calling join(args, kwargs) (line 306)
        join_call_result_3517 = invoke(stypy.reporting.localization.Localization(__file__, 306, 20), join_3513, *[DATA_PATH_3514, str_3515], **kwargs_3516)
        
        # Processing the call keyword arguments (line 306)
        # Getting the type of 'False' (line 306)
        False_3518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 82), 'False', False)
        keyword_3519 = False_3518
        kwargs_3520 = {'verbose': keyword_3519}
        # Getting the type of 'readsav' (line 306)
        readsav_3511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 306)
        readsav_call_result_3521 = invoke(stypy.reporting.localization.Localization(__file__, 306, 12), readsav_3511, *[join_call_result_3517], **kwargs_3520)
        
        # Assigning a type to the variable 's' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 's', readsav_call_result_3521)
        
        # Call to assert_equal(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 's' (line 307)
        s_3523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 21), 's', False)
        # Obtaining the member 'array3d' of a type (line 307)
        array3d_3524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 21), s_3523, 'array3d')
        # Obtaining the member 'shape' of a type (line 307)
        shape_3525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 21), array3d_3524, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 307)
        tuple_3526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 307)
        # Adding element type (line 307)
        int_3527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 39), tuple_3526, int_3527)
        # Adding element type (line 307)
        int_3528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 43), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 39), tuple_3526, int_3528)
        # Adding element type (line 307)
        int_3529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 307, 39), tuple_3526, int_3529)
        
        # Processing the call keyword arguments (line 307)
        kwargs_3530 = {}
        # Getting the type of 'assert_equal' (line 307)
        assert_equal_3522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 307)
        assert_equal_call_result_3531 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), assert_equal_3522, *[shape_3525, tuple_3526], **kwargs_3530)
        
        
        # Call to assert_(...): (line 308)
        # Processing the call arguments (line 308)
        
        # Call to all(...): (line 308)
        # Processing the call arguments (line 308)
        
        # Getting the type of 's' (line 308)
        s_3535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 23), 's', False)
        # Obtaining the member 'array3d' of a type (line 308)
        array3d_3536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 23), s_3535, 'array3d')
        
        # Call to float32(...): (line 308)
        # Processing the call arguments (line 308)
        float_3539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 47), 'float')
        # Processing the call keyword arguments (line 308)
        kwargs_3540 = {}
        # Getting the type of 'np' (line 308)
        np_3537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 36), 'np', False)
        # Obtaining the member 'float32' of a type (line 308)
        float32_3538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 36), np_3537, 'float32')
        # Calling float32(args, kwargs) (line 308)
        float32_call_result_3541 = invoke(stypy.reporting.localization.Localization(__file__, 308, 36), float32_3538, *[float_3539], **kwargs_3540)
        
        # Applying the binary operator '==' (line 308)
        result_eq_3542 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 23), '==', array3d_3536, float32_call_result_3541)
        
        # Processing the call keyword arguments (line 308)
        kwargs_3543 = {}
        # Getting the type of 'np' (line 308)
        np_3533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 308)
        all_3534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 16), np_3533, 'all')
        # Calling all(args, kwargs) (line 308)
        all_call_result_3544 = invoke(stypy.reporting.localization.Localization(__file__, 308, 16), all_3534, *[result_eq_3542], **kwargs_3543)
        
        # Processing the call keyword arguments (line 308)
        kwargs_3545 = {}
        # Getting the type of 'assert_' (line 308)
        assert__3532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 308)
        assert__call_result_3546 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), assert__3532, *[all_call_result_3544], **kwargs_3545)
        
        
        # Call to assert_(...): (line 309)
        # Processing the call arguments (line 309)
        
        # Call to all(...): (line 309)
        # Processing the call arguments (line 309)
        
        
        # Call to vect_id(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 's' (line 309)
        s_3551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 31), 's', False)
        # Obtaining the member 'array3d' of a type (line 309)
        array3d_3552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 31), s_3551, 'array3d')
        # Processing the call keyword arguments (line 309)
        kwargs_3553 = {}
        # Getting the type of 'vect_id' (line 309)
        vect_id_3550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 23), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 309)
        vect_id_call_result_3554 = invoke(stypy.reporting.localization.Localization(__file__, 309, 23), vect_id_3550, *[array3d_3552], **kwargs_3553)
        
        
        # Call to id(...): (line 309)
        # Processing the call arguments (line 309)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 309)
        tuple_3556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 309)
        # Adding element type (line 309)
        int_3557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 58), tuple_3556, int_3557)
        # Adding element type (line 309)
        int_3558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 58), tuple_3556, int_3558)
        # Adding element type (line 309)
        int_3559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 58), tuple_3556, int_3559)
        
        # Getting the type of 's' (line 309)
        s_3560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 48), 's', False)
        # Obtaining the member 'array3d' of a type (line 309)
        array3d_3561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 48), s_3560, 'array3d')
        # Obtaining the member '__getitem__' of a type (line 309)
        getitem___3562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 48), array3d_3561, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 309)
        subscript_call_result_3563 = invoke(stypy.reporting.localization.Localization(__file__, 309, 48), getitem___3562, tuple_3556)
        
        # Processing the call keyword arguments (line 309)
        kwargs_3564 = {}
        # Getting the type of 'id' (line 309)
        id_3555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 45), 'id', False)
        # Calling id(args, kwargs) (line 309)
        id_call_result_3565 = invoke(stypy.reporting.localization.Localization(__file__, 309, 45), id_3555, *[subscript_call_result_3563], **kwargs_3564)
        
        # Applying the binary operator '==' (line 309)
        result_eq_3566 = python_operator(stypy.reporting.localization.Localization(__file__, 309, 23), '==', vect_id_call_result_3554, id_call_result_3565)
        
        # Processing the call keyword arguments (line 309)
        kwargs_3567 = {}
        # Getting the type of 'np' (line 309)
        np_3548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 309)
        all_3549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 16), np_3548, 'all')
        # Calling all(args, kwargs) (line 309)
        all_call_result_3568 = invoke(stypy.reporting.localization.Localization(__file__, 309, 16), all_3549, *[result_eq_3566], **kwargs_3567)
        
        # Processing the call keyword arguments (line 309)
        kwargs_3569 = {}
        # Getting the type of 'assert_' (line 309)
        assert__3547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 309)
        assert__call_result_3570 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), assert__3547, *[all_call_result_3568], **kwargs_3569)
        
        
        # ################# End of 'test_3d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_3d' in the type store
        # Getting the type of 'stypy_return_type' (line 305)
        stypy_return_type_3571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3571)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_3d'
        return stypy_return_type_3571


    @norecursion
    def test_4d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_4d'
        module_type_store = module_type_store.open_function_context('test_4d', 311, 4, False)
        # Assigning a type to the variable 'self' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPointerArray.test_4d.__dict__.__setitem__('stypy_localization', localization)
        TestPointerArray.test_4d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPointerArray.test_4d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPointerArray.test_4d.__dict__.__setitem__('stypy_function_name', 'TestPointerArray.test_4d')
        TestPointerArray.test_4d.__dict__.__setitem__('stypy_param_names_list', [])
        TestPointerArray.test_4d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPointerArray.test_4d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPointerArray.test_4d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPointerArray.test_4d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPointerArray.test_4d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPointerArray.test_4d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointerArray.test_4d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_4d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_4d(...)' code ##################

        
        # Assigning a Call to a Name (line 312):
        
        # Call to readsav(...): (line 312)
        # Processing the call arguments (line 312)
        
        # Call to join(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 'DATA_PATH' (line 312)
        DATA_PATH_3575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 30), 'DATA_PATH', False)
        str_3576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 41), 'str', 'array_float32_pointer_4d.sav')
        # Processing the call keyword arguments (line 312)
        kwargs_3577 = {}
        # Getting the type of 'path' (line 312)
        path_3573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 312)
        join_3574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 20), path_3573, 'join')
        # Calling join(args, kwargs) (line 312)
        join_call_result_3578 = invoke(stypy.reporting.localization.Localization(__file__, 312, 20), join_3574, *[DATA_PATH_3575, str_3576], **kwargs_3577)
        
        # Processing the call keyword arguments (line 312)
        # Getting the type of 'False' (line 312)
        False_3579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 82), 'False', False)
        keyword_3580 = False_3579
        kwargs_3581 = {'verbose': keyword_3580}
        # Getting the type of 'readsav' (line 312)
        readsav_3572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 312)
        readsav_call_result_3582 = invoke(stypy.reporting.localization.Localization(__file__, 312, 12), readsav_3572, *[join_call_result_3578], **kwargs_3581)
        
        # Assigning a type to the variable 's' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 's', readsav_call_result_3582)
        
        # Call to assert_equal(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 's' (line 313)
        s_3584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 21), 's', False)
        # Obtaining the member 'array4d' of a type (line 313)
        array4d_3585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 21), s_3584, 'array4d')
        # Obtaining the member 'shape' of a type (line 313)
        shape_3586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 21), array4d_3585, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 313)
        tuple_3587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 313)
        # Adding element type (line 313)
        int_3588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 39), tuple_3587, int_3588)
        # Adding element type (line 313)
        int_3589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 39), tuple_3587, int_3589)
        # Adding element type (line 313)
        int_3590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 39), tuple_3587, int_3590)
        # Adding element type (line 313)
        int_3591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 39), tuple_3587, int_3591)
        
        # Processing the call keyword arguments (line 313)
        kwargs_3592 = {}
        # Getting the type of 'assert_equal' (line 313)
        assert_equal_3583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 313)
        assert_equal_call_result_3593 = invoke(stypy.reporting.localization.Localization(__file__, 313, 8), assert_equal_3583, *[shape_3586, tuple_3587], **kwargs_3592)
        
        
        # Call to assert_(...): (line 314)
        # Processing the call arguments (line 314)
        
        # Call to all(...): (line 314)
        # Processing the call arguments (line 314)
        
        # Getting the type of 's' (line 314)
        s_3597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 23), 's', False)
        # Obtaining the member 'array4d' of a type (line 314)
        array4d_3598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 23), s_3597, 'array4d')
        
        # Call to float32(...): (line 314)
        # Processing the call arguments (line 314)
        float_3601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 47), 'float')
        # Processing the call keyword arguments (line 314)
        kwargs_3602 = {}
        # Getting the type of 'np' (line 314)
        np_3599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 36), 'np', False)
        # Obtaining the member 'float32' of a type (line 314)
        float32_3600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 36), np_3599, 'float32')
        # Calling float32(args, kwargs) (line 314)
        float32_call_result_3603 = invoke(stypy.reporting.localization.Localization(__file__, 314, 36), float32_3600, *[float_3601], **kwargs_3602)
        
        # Applying the binary operator '==' (line 314)
        result_eq_3604 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 23), '==', array4d_3598, float32_call_result_3603)
        
        # Processing the call keyword arguments (line 314)
        kwargs_3605 = {}
        # Getting the type of 'np' (line 314)
        np_3595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 314)
        all_3596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 16), np_3595, 'all')
        # Calling all(args, kwargs) (line 314)
        all_call_result_3606 = invoke(stypy.reporting.localization.Localization(__file__, 314, 16), all_3596, *[result_eq_3604], **kwargs_3605)
        
        # Processing the call keyword arguments (line 314)
        kwargs_3607 = {}
        # Getting the type of 'assert_' (line 314)
        assert__3594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 314)
        assert__call_result_3608 = invoke(stypy.reporting.localization.Localization(__file__, 314, 8), assert__3594, *[all_call_result_3606], **kwargs_3607)
        
        
        # Call to assert_(...): (line 315)
        # Processing the call arguments (line 315)
        
        # Call to all(...): (line 315)
        # Processing the call arguments (line 315)
        
        
        # Call to vect_id(...): (line 315)
        # Processing the call arguments (line 315)
        # Getting the type of 's' (line 315)
        s_3613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 31), 's', False)
        # Obtaining the member 'array4d' of a type (line 315)
        array4d_3614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 31), s_3613, 'array4d')
        # Processing the call keyword arguments (line 315)
        kwargs_3615 = {}
        # Getting the type of 'vect_id' (line 315)
        vect_id_3612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 23), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 315)
        vect_id_call_result_3616 = invoke(stypy.reporting.localization.Localization(__file__, 315, 23), vect_id_3612, *[array4d_3614], **kwargs_3615)
        
        
        # Call to id(...): (line 315)
        # Processing the call arguments (line 315)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 315)
        tuple_3618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 315)
        # Adding element type (line 315)
        int_3619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 58), tuple_3618, int_3619)
        # Adding element type (line 315)
        int_3620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 58), tuple_3618, int_3620)
        # Adding element type (line 315)
        int_3621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 58), tuple_3618, int_3621)
        # Adding element type (line 315)
        int_3622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 315, 58), tuple_3618, int_3622)
        
        # Getting the type of 's' (line 315)
        s_3623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 48), 's', False)
        # Obtaining the member 'array4d' of a type (line 315)
        array4d_3624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 48), s_3623, 'array4d')
        # Obtaining the member '__getitem__' of a type (line 315)
        getitem___3625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 48), array4d_3624, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 315)
        subscript_call_result_3626 = invoke(stypy.reporting.localization.Localization(__file__, 315, 48), getitem___3625, tuple_3618)
        
        # Processing the call keyword arguments (line 315)
        kwargs_3627 = {}
        # Getting the type of 'id' (line 315)
        id_3617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 45), 'id', False)
        # Calling id(args, kwargs) (line 315)
        id_call_result_3628 = invoke(stypy.reporting.localization.Localization(__file__, 315, 45), id_3617, *[subscript_call_result_3626], **kwargs_3627)
        
        # Applying the binary operator '==' (line 315)
        result_eq_3629 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 23), '==', vect_id_call_result_3616, id_call_result_3628)
        
        # Processing the call keyword arguments (line 315)
        kwargs_3630 = {}
        # Getting the type of 'np' (line 315)
        np_3610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 315)
        all_3611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 16), np_3610, 'all')
        # Calling all(args, kwargs) (line 315)
        all_call_result_3631 = invoke(stypy.reporting.localization.Localization(__file__, 315, 16), all_3611, *[result_eq_3629], **kwargs_3630)
        
        # Processing the call keyword arguments (line 315)
        kwargs_3632 = {}
        # Getting the type of 'assert_' (line 315)
        assert__3609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 315)
        assert__call_result_3633 = invoke(stypy.reporting.localization.Localization(__file__, 315, 8), assert__3609, *[all_call_result_3631], **kwargs_3632)
        
        
        # ################# End of 'test_4d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_4d' in the type store
        # Getting the type of 'stypy_return_type' (line 311)
        stypy_return_type_3634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3634)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_4d'
        return stypy_return_type_3634


    @norecursion
    def test_5d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_5d'
        module_type_store = module_type_store.open_function_context('test_5d', 317, 4, False)
        # Assigning a type to the variable 'self' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPointerArray.test_5d.__dict__.__setitem__('stypy_localization', localization)
        TestPointerArray.test_5d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPointerArray.test_5d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPointerArray.test_5d.__dict__.__setitem__('stypy_function_name', 'TestPointerArray.test_5d')
        TestPointerArray.test_5d.__dict__.__setitem__('stypy_param_names_list', [])
        TestPointerArray.test_5d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPointerArray.test_5d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPointerArray.test_5d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPointerArray.test_5d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPointerArray.test_5d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPointerArray.test_5d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointerArray.test_5d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_5d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_5d(...)' code ##################

        
        # Assigning a Call to a Name (line 318):
        
        # Call to readsav(...): (line 318)
        # Processing the call arguments (line 318)
        
        # Call to join(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'DATA_PATH' (line 318)
        DATA_PATH_3638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 30), 'DATA_PATH', False)
        str_3639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 41), 'str', 'array_float32_pointer_5d.sav')
        # Processing the call keyword arguments (line 318)
        kwargs_3640 = {}
        # Getting the type of 'path' (line 318)
        path_3636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 318)
        join_3637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 20), path_3636, 'join')
        # Calling join(args, kwargs) (line 318)
        join_call_result_3641 = invoke(stypy.reporting.localization.Localization(__file__, 318, 20), join_3637, *[DATA_PATH_3638, str_3639], **kwargs_3640)
        
        # Processing the call keyword arguments (line 318)
        # Getting the type of 'False' (line 318)
        False_3642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 82), 'False', False)
        keyword_3643 = False_3642
        kwargs_3644 = {'verbose': keyword_3643}
        # Getting the type of 'readsav' (line 318)
        readsav_3635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 318)
        readsav_call_result_3645 = invoke(stypy.reporting.localization.Localization(__file__, 318, 12), readsav_3635, *[join_call_result_3641], **kwargs_3644)
        
        # Assigning a type to the variable 's' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 's', readsav_call_result_3645)
        
        # Call to assert_equal(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 's' (line 319)
        s_3647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 21), 's', False)
        # Obtaining the member 'array5d' of a type (line 319)
        array5d_3648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 21), s_3647, 'array5d')
        # Obtaining the member 'shape' of a type (line 319)
        shape_3649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 21), array5d_3648, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 319)
        tuple_3650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 319)
        # Adding element type (line 319)
        int_3651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 39), tuple_3650, int_3651)
        # Adding element type (line 319)
        int_3652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 39), tuple_3650, int_3652)
        # Adding element type (line 319)
        int_3653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 39), tuple_3650, int_3653)
        # Adding element type (line 319)
        int_3654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 39), tuple_3650, int_3654)
        # Adding element type (line 319)
        int_3655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 39), tuple_3650, int_3655)
        
        # Processing the call keyword arguments (line 319)
        kwargs_3656 = {}
        # Getting the type of 'assert_equal' (line 319)
        assert_equal_3646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 319)
        assert_equal_call_result_3657 = invoke(stypy.reporting.localization.Localization(__file__, 319, 8), assert_equal_3646, *[shape_3649, tuple_3650], **kwargs_3656)
        
        
        # Call to assert_(...): (line 320)
        # Processing the call arguments (line 320)
        
        # Call to all(...): (line 320)
        # Processing the call arguments (line 320)
        
        # Getting the type of 's' (line 320)
        s_3661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 23), 's', False)
        # Obtaining the member 'array5d' of a type (line 320)
        array5d_3662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 23), s_3661, 'array5d')
        
        # Call to float32(...): (line 320)
        # Processing the call arguments (line 320)
        float_3665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 47), 'float')
        # Processing the call keyword arguments (line 320)
        kwargs_3666 = {}
        # Getting the type of 'np' (line 320)
        np_3663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 36), 'np', False)
        # Obtaining the member 'float32' of a type (line 320)
        float32_3664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 36), np_3663, 'float32')
        # Calling float32(args, kwargs) (line 320)
        float32_call_result_3667 = invoke(stypy.reporting.localization.Localization(__file__, 320, 36), float32_3664, *[float_3665], **kwargs_3666)
        
        # Applying the binary operator '==' (line 320)
        result_eq_3668 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 23), '==', array5d_3662, float32_call_result_3667)
        
        # Processing the call keyword arguments (line 320)
        kwargs_3669 = {}
        # Getting the type of 'np' (line 320)
        np_3659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 320)
        all_3660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 16), np_3659, 'all')
        # Calling all(args, kwargs) (line 320)
        all_call_result_3670 = invoke(stypy.reporting.localization.Localization(__file__, 320, 16), all_3660, *[result_eq_3668], **kwargs_3669)
        
        # Processing the call keyword arguments (line 320)
        kwargs_3671 = {}
        # Getting the type of 'assert_' (line 320)
        assert__3658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 320)
        assert__call_result_3672 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), assert__3658, *[all_call_result_3670], **kwargs_3671)
        
        
        # Call to assert_(...): (line 321)
        # Processing the call arguments (line 321)
        
        # Call to all(...): (line 321)
        # Processing the call arguments (line 321)
        
        
        # Call to vect_id(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 's' (line 321)
        s_3677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 31), 's', False)
        # Obtaining the member 'array5d' of a type (line 321)
        array5d_3678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 31), s_3677, 'array5d')
        # Processing the call keyword arguments (line 321)
        kwargs_3679 = {}
        # Getting the type of 'vect_id' (line 321)
        vect_id_3676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 23), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 321)
        vect_id_call_result_3680 = invoke(stypy.reporting.localization.Localization(__file__, 321, 23), vect_id_3676, *[array5d_3678], **kwargs_3679)
        
        
        # Call to id(...): (line 321)
        # Processing the call arguments (line 321)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 321)
        tuple_3682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 321)
        # Adding element type (line 321)
        int_3683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 58), tuple_3682, int_3683)
        # Adding element type (line 321)
        int_3684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 58), tuple_3682, int_3684)
        # Adding element type (line 321)
        int_3685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 58), tuple_3682, int_3685)
        # Adding element type (line 321)
        int_3686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 58), tuple_3682, int_3686)
        # Adding element type (line 321)
        int_3687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 321, 58), tuple_3682, int_3687)
        
        # Getting the type of 's' (line 321)
        s_3688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 48), 's', False)
        # Obtaining the member 'array5d' of a type (line 321)
        array5d_3689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 48), s_3688, 'array5d')
        # Obtaining the member '__getitem__' of a type (line 321)
        getitem___3690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 48), array5d_3689, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 321)
        subscript_call_result_3691 = invoke(stypy.reporting.localization.Localization(__file__, 321, 48), getitem___3690, tuple_3682)
        
        # Processing the call keyword arguments (line 321)
        kwargs_3692 = {}
        # Getting the type of 'id' (line 321)
        id_3681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 45), 'id', False)
        # Calling id(args, kwargs) (line 321)
        id_call_result_3693 = invoke(stypy.reporting.localization.Localization(__file__, 321, 45), id_3681, *[subscript_call_result_3691], **kwargs_3692)
        
        # Applying the binary operator '==' (line 321)
        result_eq_3694 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 23), '==', vect_id_call_result_3680, id_call_result_3693)
        
        # Processing the call keyword arguments (line 321)
        kwargs_3695 = {}
        # Getting the type of 'np' (line 321)
        np_3674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 321)
        all_3675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 16), np_3674, 'all')
        # Calling all(args, kwargs) (line 321)
        all_call_result_3696 = invoke(stypy.reporting.localization.Localization(__file__, 321, 16), all_3675, *[result_eq_3694], **kwargs_3695)
        
        # Processing the call keyword arguments (line 321)
        kwargs_3697 = {}
        # Getting the type of 'assert_' (line 321)
        assert__3673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 321)
        assert__call_result_3698 = invoke(stypy.reporting.localization.Localization(__file__, 321, 8), assert__3673, *[all_call_result_3696], **kwargs_3697)
        
        
        # ################# End of 'test_5d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_5d' in the type store
        # Getting the type of 'stypy_return_type' (line 317)
        stypy_return_type_3699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3699)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_5d'
        return stypy_return_type_3699


    @norecursion
    def test_6d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_6d'
        module_type_store = module_type_store.open_function_context('test_6d', 323, 4, False)
        # Assigning a type to the variable 'self' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPointerArray.test_6d.__dict__.__setitem__('stypy_localization', localization)
        TestPointerArray.test_6d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPointerArray.test_6d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPointerArray.test_6d.__dict__.__setitem__('stypy_function_name', 'TestPointerArray.test_6d')
        TestPointerArray.test_6d.__dict__.__setitem__('stypy_param_names_list', [])
        TestPointerArray.test_6d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPointerArray.test_6d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPointerArray.test_6d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPointerArray.test_6d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPointerArray.test_6d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPointerArray.test_6d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointerArray.test_6d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_6d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_6d(...)' code ##################

        
        # Assigning a Call to a Name (line 324):
        
        # Call to readsav(...): (line 324)
        # Processing the call arguments (line 324)
        
        # Call to join(...): (line 324)
        # Processing the call arguments (line 324)
        # Getting the type of 'DATA_PATH' (line 324)
        DATA_PATH_3703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 30), 'DATA_PATH', False)
        str_3704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 41), 'str', 'array_float32_pointer_6d.sav')
        # Processing the call keyword arguments (line 324)
        kwargs_3705 = {}
        # Getting the type of 'path' (line 324)
        path_3701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 324)
        join_3702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 20), path_3701, 'join')
        # Calling join(args, kwargs) (line 324)
        join_call_result_3706 = invoke(stypy.reporting.localization.Localization(__file__, 324, 20), join_3702, *[DATA_PATH_3703, str_3704], **kwargs_3705)
        
        # Processing the call keyword arguments (line 324)
        # Getting the type of 'False' (line 324)
        False_3707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 82), 'False', False)
        keyword_3708 = False_3707
        kwargs_3709 = {'verbose': keyword_3708}
        # Getting the type of 'readsav' (line 324)
        readsav_3700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 324)
        readsav_call_result_3710 = invoke(stypy.reporting.localization.Localization(__file__, 324, 12), readsav_3700, *[join_call_result_3706], **kwargs_3709)
        
        # Assigning a type to the variable 's' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 's', readsav_call_result_3710)
        
        # Call to assert_equal(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 's' (line 325)
        s_3712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 21), 's', False)
        # Obtaining the member 'array6d' of a type (line 325)
        array6d_3713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 21), s_3712, 'array6d')
        # Obtaining the member 'shape' of a type (line 325)
        shape_3714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 21), array6d_3713, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 325)
        tuple_3715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 325)
        # Adding element type (line 325)
        int_3716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 39), tuple_3715, int_3716)
        # Adding element type (line 325)
        int_3717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 39), tuple_3715, int_3717)
        # Adding element type (line 325)
        int_3718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 39), tuple_3715, int_3718)
        # Adding element type (line 325)
        int_3719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 39), tuple_3715, int_3719)
        # Adding element type (line 325)
        int_3720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 39), tuple_3715, int_3720)
        # Adding element type (line 325)
        int_3721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 39), tuple_3715, int_3721)
        
        # Processing the call keyword arguments (line 325)
        kwargs_3722 = {}
        # Getting the type of 'assert_equal' (line 325)
        assert_equal_3711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 325)
        assert_equal_call_result_3723 = invoke(stypy.reporting.localization.Localization(__file__, 325, 8), assert_equal_3711, *[shape_3714, tuple_3715], **kwargs_3722)
        
        
        # Call to assert_(...): (line 326)
        # Processing the call arguments (line 326)
        
        # Call to all(...): (line 326)
        # Processing the call arguments (line 326)
        
        # Getting the type of 's' (line 326)
        s_3727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 23), 's', False)
        # Obtaining the member 'array6d' of a type (line 326)
        array6d_3728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 23), s_3727, 'array6d')
        
        # Call to float32(...): (line 326)
        # Processing the call arguments (line 326)
        float_3731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 47), 'float')
        # Processing the call keyword arguments (line 326)
        kwargs_3732 = {}
        # Getting the type of 'np' (line 326)
        np_3729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'np', False)
        # Obtaining the member 'float32' of a type (line 326)
        float32_3730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 36), np_3729, 'float32')
        # Calling float32(args, kwargs) (line 326)
        float32_call_result_3733 = invoke(stypy.reporting.localization.Localization(__file__, 326, 36), float32_3730, *[float_3731], **kwargs_3732)
        
        # Applying the binary operator '==' (line 326)
        result_eq_3734 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 23), '==', array6d_3728, float32_call_result_3733)
        
        # Processing the call keyword arguments (line 326)
        kwargs_3735 = {}
        # Getting the type of 'np' (line 326)
        np_3725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 326)
        all_3726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 16), np_3725, 'all')
        # Calling all(args, kwargs) (line 326)
        all_call_result_3736 = invoke(stypy.reporting.localization.Localization(__file__, 326, 16), all_3726, *[result_eq_3734], **kwargs_3735)
        
        # Processing the call keyword arguments (line 326)
        kwargs_3737 = {}
        # Getting the type of 'assert_' (line 326)
        assert__3724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 326)
        assert__call_result_3738 = invoke(stypy.reporting.localization.Localization(__file__, 326, 8), assert__3724, *[all_call_result_3736], **kwargs_3737)
        
        
        # Call to assert_(...): (line 327)
        # Processing the call arguments (line 327)
        
        # Call to all(...): (line 327)
        # Processing the call arguments (line 327)
        
        
        # Call to vect_id(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 's' (line 327)
        s_3743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 31), 's', False)
        # Obtaining the member 'array6d' of a type (line 327)
        array6d_3744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 31), s_3743, 'array6d')
        # Processing the call keyword arguments (line 327)
        kwargs_3745 = {}
        # Getting the type of 'vect_id' (line 327)
        vect_id_3742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 23), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 327)
        vect_id_call_result_3746 = invoke(stypy.reporting.localization.Localization(__file__, 327, 23), vect_id_3742, *[array6d_3744], **kwargs_3745)
        
        
        # Call to id(...): (line 327)
        # Processing the call arguments (line 327)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 327)
        tuple_3748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 327)
        # Adding element type (line 327)
        int_3749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 58), tuple_3748, int_3749)
        # Adding element type (line 327)
        int_3750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 58), tuple_3748, int_3750)
        # Adding element type (line 327)
        int_3751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 58), tuple_3748, int_3751)
        # Adding element type (line 327)
        int_3752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 58), tuple_3748, int_3752)
        # Adding element type (line 327)
        int_3753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 58), tuple_3748, int_3753)
        # Adding element type (line 327)
        int_3754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 58), tuple_3748, int_3754)
        
        # Getting the type of 's' (line 327)
        s_3755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 48), 's', False)
        # Obtaining the member 'array6d' of a type (line 327)
        array6d_3756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 48), s_3755, 'array6d')
        # Obtaining the member '__getitem__' of a type (line 327)
        getitem___3757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 48), array6d_3756, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 327)
        subscript_call_result_3758 = invoke(stypy.reporting.localization.Localization(__file__, 327, 48), getitem___3757, tuple_3748)
        
        # Processing the call keyword arguments (line 327)
        kwargs_3759 = {}
        # Getting the type of 'id' (line 327)
        id_3747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 45), 'id', False)
        # Calling id(args, kwargs) (line 327)
        id_call_result_3760 = invoke(stypy.reporting.localization.Localization(__file__, 327, 45), id_3747, *[subscript_call_result_3758], **kwargs_3759)
        
        # Applying the binary operator '==' (line 327)
        result_eq_3761 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 23), '==', vect_id_call_result_3746, id_call_result_3760)
        
        # Processing the call keyword arguments (line 327)
        kwargs_3762 = {}
        # Getting the type of 'np' (line 327)
        np_3740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 327)
        all_3741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 16), np_3740, 'all')
        # Calling all(args, kwargs) (line 327)
        all_call_result_3763 = invoke(stypy.reporting.localization.Localization(__file__, 327, 16), all_3741, *[result_eq_3761], **kwargs_3762)
        
        # Processing the call keyword arguments (line 327)
        kwargs_3764 = {}
        # Getting the type of 'assert_' (line 327)
        assert__3739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 327)
        assert__call_result_3765 = invoke(stypy.reporting.localization.Localization(__file__, 327, 8), assert__3739, *[all_call_result_3763], **kwargs_3764)
        
        
        # ################# End of 'test_6d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_6d' in the type store
        # Getting the type of 'stypy_return_type' (line 323)
        stypy_return_type_3766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3766)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_6d'
        return stypy_return_type_3766


    @norecursion
    def test_7d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_7d'
        module_type_store = module_type_store.open_function_context('test_7d', 329, 4, False)
        # Assigning a type to the variable 'self' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPointerArray.test_7d.__dict__.__setitem__('stypy_localization', localization)
        TestPointerArray.test_7d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPointerArray.test_7d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPointerArray.test_7d.__dict__.__setitem__('stypy_function_name', 'TestPointerArray.test_7d')
        TestPointerArray.test_7d.__dict__.__setitem__('stypy_param_names_list', [])
        TestPointerArray.test_7d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPointerArray.test_7d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPointerArray.test_7d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPointerArray.test_7d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPointerArray.test_7d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPointerArray.test_7d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointerArray.test_7d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_7d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_7d(...)' code ##################

        
        # Assigning a Call to a Name (line 330):
        
        # Call to readsav(...): (line 330)
        # Processing the call arguments (line 330)
        
        # Call to join(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 'DATA_PATH' (line 330)
        DATA_PATH_3770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 30), 'DATA_PATH', False)
        str_3771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 41), 'str', 'array_float32_pointer_7d.sav')
        # Processing the call keyword arguments (line 330)
        kwargs_3772 = {}
        # Getting the type of 'path' (line 330)
        path_3768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 330)
        join_3769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 20), path_3768, 'join')
        # Calling join(args, kwargs) (line 330)
        join_call_result_3773 = invoke(stypy.reporting.localization.Localization(__file__, 330, 20), join_3769, *[DATA_PATH_3770, str_3771], **kwargs_3772)
        
        # Processing the call keyword arguments (line 330)
        # Getting the type of 'False' (line 330)
        False_3774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 82), 'False', False)
        keyword_3775 = False_3774
        kwargs_3776 = {'verbose': keyword_3775}
        # Getting the type of 'readsav' (line 330)
        readsav_3767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 330)
        readsav_call_result_3777 = invoke(stypy.reporting.localization.Localization(__file__, 330, 12), readsav_3767, *[join_call_result_3773], **kwargs_3776)
        
        # Assigning a type to the variable 's' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 's', readsav_call_result_3777)
        
        # Call to assert_equal(...): (line 331)
        # Processing the call arguments (line 331)
        # Getting the type of 's' (line 331)
        s_3779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 21), 's', False)
        # Obtaining the member 'array7d' of a type (line 331)
        array7d_3780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 21), s_3779, 'array7d')
        # Obtaining the member 'shape' of a type (line 331)
        shape_3781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 21), array7d_3780, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 331)
        tuple_3782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 331)
        # Adding element type (line 331)
        int_3783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 39), tuple_3782, int_3783)
        # Adding element type (line 331)
        int_3784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 39), tuple_3782, int_3784)
        # Adding element type (line 331)
        int_3785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 39), tuple_3782, int_3785)
        # Adding element type (line 331)
        int_3786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 39), tuple_3782, int_3786)
        # Adding element type (line 331)
        int_3787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 39), tuple_3782, int_3787)
        # Adding element type (line 331)
        int_3788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 39), tuple_3782, int_3788)
        # Adding element type (line 331)
        int_3789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 39), tuple_3782, int_3789)
        
        # Processing the call keyword arguments (line 331)
        kwargs_3790 = {}
        # Getting the type of 'assert_equal' (line 331)
        assert_equal_3778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 331)
        assert_equal_call_result_3791 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), assert_equal_3778, *[shape_3781, tuple_3782], **kwargs_3790)
        
        
        # Call to assert_(...): (line 332)
        # Processing the call arguments (line 332)
        
        # Call to all(...): (line 332)
        # Processing the call arguments (line 332)
        
        # Getting the type of 's' (line 332)
        s_3795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 23), 's', False)
        # Obtaining the member 'array7d' of a type (line 332)
        array7d_3796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 23), s_3795, 'array7d')
        
        # Call to float32(...): (line 332)
        # Processing the call arguments (line 332)
        float_3799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 47), 'float')
        # Processing the call keyword arguments (line 332)
        kwargs_3800 = {}
        # Getting the type of 'np' (line 332)
        np_3797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 36), 'np', False)
        # Obtaining the member 'float32' of a type (line 332)
        float32_3798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 36), np_3797, 'float32')
        # Calling float32(args, kwargs) (line 332)
        float32_call_result_3801 = invoke(stypy.reporting.localization.Localization(__file__, 332, 36), float32_3798, *[float_3799], **kwargs_3800)
        
        # Applying the binary operator '==' (line 332)
        result_eq_3802 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 23), '==', array7d_3796, float32_call_result_3801)
        
        # Processing the call keyword arguments (line 332)
        kwargs_3803 = {}
        # Getting the type of 'np' (line 332)
        np_3793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 332)
        all_3794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 16), np_3793, 'all')
        # Calling all(args, kwargs) (line 332)
        all_call_result_3804 = invoke(stypy.reporting.localization.Localization(__file__, 332, 16), all_3794, *[result_eq_3802], **kwargs_3803)
        
        # Processing the call keyword arguments (line 332)
        kwargs_3805 = {}
        # Getting the type of 'assert_' (line 332)
        assert__3792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 332)
        assert__call_result_3806 = invoke(stypy.reporting.localization.Localization(__file__, 332, 8), assert__3792, *[all_call_result_3804], **kwargs_3805)
        
        
        # Call to assert_(...): (line 333)
        # Processing the call arguments (line 333)
        
        # Call to all(...): (line 333)
        # Processing the call arguments (line 333)
        
        
        # Call to vect_id(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 's' (line 333)
        s_3811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 31), 's', False)
        # Obtaining the member 'array7d' of a type (line 333)
        array7d_3812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 31), s_3811, 'array7d')
        # Processing the call keyword arguments (line 333)
        kwargs_3813 = {}
        # Getting the type of 'vect_id' (line 333)
        vect_id_3810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 23), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 333)
        vect_id_call_result_3814 = invoke(stypy.reporting.localization.Localization(__file__, 333, 23), vect_id_3810, *[array7d_3812], **kwargs_3813)
        
        
        # Call to id(...): (line 333)
        # Processing the call arguments (line 333)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 333)
        tuple_3816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 333)
        # Adding element type (line 333)
        int_3817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 58), tuple_3816, int_3817)
        # Adding element type (line 333)
        int_3818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 58), tuple_3816, int_3818)
        # Adding element type (line 333)
        int_3819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 58), tuple_3816, int_3819)
        # Adding element type (line 333)
        int_3820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 58), tuple_3816, int_3820)
        # Adding element type (line 333)
        int_3821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 58), tuple_3816, int_3821)
        # Adding element type (line 333)
        int_3822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 58), tuple_3816, int_3822)
        # Adding element type (line 333)
        int_3823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 70), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 333, 58), tuple_3816, int_3823)
        
        # Getting the type of 's' (line 333)
        s_3824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 48), 's', False)
        # Obtaining the member 'array7d' of a type (line 333)
        array7d_3825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 48), s_3824, 'array7d')
        # Obtaining the member '__getitem__' of a type (line 333)
        getitem___3826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 48), array7d_3825, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 333)
        subscript_call_result_3827 = invoke(stypy.reporting.localization.Localization(__file__, 333, 48), getitem___3826, tuple_3816)
        
        # Processing the call keyword arguments (line 333)
        kwargs_3828 = {}
        # Getting the type of 'id' (line 333)
        id_3815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 45), 'id', False)
        # Calling id(args, kwargs) (line 333)
        id_call_result_3829 = invoke(stypy.reporting.localization.Localization(__file__, 333, 45), id_3815, *[subscript_call_result_3827], **kwargs_3828)
        
        # Applying the binary operator '==' (line 333)
        result_eq_3830 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 23), '==', vect_id_call_result_3814, id_call_result_3829)
        
        # Processing the call keyword arguments (line 333)
        kwargs_3831 = {}
        # Getting the type of 'np' (line 333)
        np_3808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 333)
        all_3809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 16), np_3808, 'all')
        # Calling all(args, kwargs) (line 333)
        all_call_result_3832 = invoke(stypy.reporting.localization.Localization(__file__, 333, 16), all_3809, *[result_eq_3830], **kwargs_3831)
        
        # Processing the call keyword arguments (line 333)
        kwargs_3833 = {}
        # Getting the type of 'assert_' (line 333)
        assert__3807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 333)
        assert__call_result_3834 = invoke(stypy.reporting.localization.Localization(__file__, 333, 8), assert__3807, *[all_call_result_3832], **kwargs_3833)
        
        
        # ################# End of 'test_7d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_7d' in the type store
        # Getting the type of 'stypy_return_type' (line 329)
        stypy_return_type_3835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3835)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_7d'
        return stypy_return_type_3835


    @norecursion
    def test_8d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_8d'
        module_type_store = module_type_store.open_function_context('test_8d', 335, 4, False)
        # Assigning a type to the variable 'self' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPointerArray.test_8d.__dict__.__setitem__('stypy_localization', localization)
        TestPointerArray.test_8d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPointerArray.test_8d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPointerArray.test_8d.__dict__.__setitem__('stypy_function_name', 'TestPointerArray.test_8d')
        TestPointerArray.test_8d.__dict__.__setitem__('stypy_param_names_list', [])
        TestPointerArray.test_8d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPointerArray.test_8d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPointerArray.test_8d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPointerArray.test_8d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPointerArray.test_8d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPointerArray.test_8d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointerArray.test_8d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_8d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_8d(...)' code ##################

        
        # Assigning a Call to a Name (line 336):
        
        # Call to readsav(...): (line 336)
        # Processing the call arguments (line 336)
        
        # Call to join(...): (line 336)
        # Processing the call arguments (line 336)
        # Getting the type of 'DATA_PATH' (line 336)
        DATA_PATH_3839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 30), 'DATA_PATH', False)
        str_3840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 41), 'str', 'array_float32_pointer_8d.sav')
        # Processing the call keyword arguments (line 336)
        kwargs_3841 = {}
        # Getting the type of 'path' (line 336)
        path_3837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 336)
        join_3838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 20), path_3837, 'join')
        # Calling join(args, kwargs) (line 336)
        join_call_result_3842 = invoke(stypy.reporting.localization.Localization(__file__, 336, 20), join_3838, *[DATA_PATH_3839, str_3840], **kwargs_3841)
        
        # Processing the call keyword arguments (line 336)
        # Getting the type of 'False' (line 336)
        False_3843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 82), 'False', False)
        keyword_3844 = False_3843
        kwargs_3845 = {'verbose': keyword_3844}
        # Getting the type of 'readsav' (line 336)
        readsav_3836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 336)
        readsav_call_result_3846 = invoke(stypy.reporting.localization.Localization(__file__, 336, 12), readsav_3836, *[join_call_result_3842], **kwargs_3845)
        
        # Assigning a type to the variable 's' (line 336)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 's', readsav_call_result_3846)
        
        # Call to assert_equal(...): (line 337)
        # Processing the call arguments (line 337)
        # Getting the type of 's' (line 337)
        s_3848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 21), 's', False)
        # Obtaining the member 'array8d' of a type (line 337)
        array8d_3849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 21), s_3848, 'array8d')
        # Obtaining the member 'shape' of a type (line 337)
        shape_3850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 21), array8d_3849, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 337)
        tuple_3851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 337)
        # Adding element type (line 337)
        int_3852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 39), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 39), tuple_3851, int_3852)
        # Adding element type (line 337)
        int_3853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 42), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 39), tuple_3851, int_3853)
        # Adding element type (line 337)
        int_3854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 45), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 39), tuple_3851, int_3854)
        # Adding element type (line 337)
        int_3855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 48), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 39), tuple_3851, int_3855)
        # Adding element type (line 337)
        int_3856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 39), tuple_3851, int_3856)
        # Adding element type (line 337)
        int_3857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 39), tuple_3851, int_3857)
        # Adding element type (line 337)
        int_3858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 39), tuple_3851, int_3858)
        # Adding element type (line 337)
        int_3859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 337, 39), tuple_3851, int_3859)
        
        # Processing the call keyword arguments (line 337)
        kwargs_3860 = {}
        # Getting the type of 'assert_equal' (line 337)
        assert_equal_3847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 337)
        assert_equal_call_result_3861 = invoke(stypy.reporting.localization.Localization(__file__, 337, 8), assert_equal_3847, *[shape_3850, tuple_3851], **kwargs_3860)
        
        
        # Call to assert_(...): (line 338)
        # Processing the call arguments (line 338)
        
        # Call to all(...): (line 338)
        # Processing the call arguments (line 338)
        
        # Getting the type of 's' (line 338)
        s_3865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 23), 's', False)
        # Obtaining the member 'array8d' of a type (line 338)
        array8d_3866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 23), s_3865, 'array8d')
        
        # Call to float32(...): (line 338)
        # Processing the call arguments (line 338)
        float_3869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 47), 'float')
        # Processing the call keyword arguments (line 338)
        kwargs_3870 = {}
        # Getting the type of 'np' (line 338)
        np_3867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 36), 'np', False)
        # Obtaining the member 'float32' of a type (line 338)
        float32_3868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 36), np_3867, 'float32')
        # Calling float32(args, kwargs) (line 338)
        float32_call_result_3871 = invoke(stypy.reporting.localization.Localization(__file__, 338, 36), float32_3868, *[float_3869], **kwargs_3870)
        
        # Applying the binary operator '==' (line 338)
        result_eq_3872 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 23), '==', array8d_3866, float32_call_result_3871)
        
        # Processing the call keyword arguments (line 338)
        kwargs_3873 = {}
        # Getting the type of 'np' (line 338)
        np_3863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 338)
        all_3864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 16), np_3863, 'all')
        # Calling all(args, kwargs) (line 338)
        all_call_result_3874 = invoke(stypy.reporting.localization.Localization(__file__, 338, 16), all_3864, *[result_eq_3872], **kwargs_3873)
        
        # Processing the call keyword arguments (line 338)
        kwargs_3875 = {}
        # Getting the type of 'assert_' (line 338)
        assert__3862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 338)
        assert__call_result_3876 = invoke(stypy.reporting.localization.Localization(__file__, 338, 8), assert__3862, *[all_call_result_3874], **kwargs_3875)
        
        
        # Call to assert_(...): (line 339)
        # Processing the call arguments (line 339)
        
        # Call to all(...): (line 339)
        # Processing the call arguments (line 339)
        
        
        # Call to vect_id(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 's' (line 339)
        s_3881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 31), 's', False)
        # Obtaining the member 'array8d' of a type (line 339)
        array8d_3882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 31), s_3881, 'array8d')
        # Processing the call keyword arguments (line 339)
        kwargs_3883 = {}
        # Getting the type of 'vect_id' (line 339)
        vect_id_3880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 23), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 339)
        vect_id_call_result_3884 = invoke(stypy.reporting.localization.Localization(__file__, 339, 23), vect_id_3880, *[array8d_3882], **kwargs_3883)
        
        
        # Call to id(...): (line 339)
        # Processing the call arguments (line 339)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 339)
        tuple_3886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 339)
        # Adding element type (line 339)
        int_3887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 58), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 58), tuple_3886, int_3887)
        # Adding element type (line 339)
        int_3888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 58), tuple_3886, int_3888)
        # Adding element type (line 339)
        int_3889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 62), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 58), tuple_3886, int_3889)
        # Adding element type (line 339)
        int_3890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 64), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 58), tuple_3886, int_3890)
        # Adding element type (line 339)
        int_3891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 66), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 58), tuple_3886, int_3891)
        # Adding element type (line 339)
        int_3892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 68), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 58), tuple_3886, int_3892)
        # Adding element type (line 339)
        int_3893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 70), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 58), tuple_3886, int_3893)
        # Adding element type (line 339)
        int_3894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 72), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 339, 58), tuple_3886, int_3894)
        
        # Getting the type of 's' (line 339)
        s_3895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 48), 's', False)
        # Obtaining the member 'array8d' of a type (line 339)
        array8d_3896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 48), s_3895, 'array8d')
        # Obtaining the member '__getitem__' of a type (line 339)
        getitem___3897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 48), array8d_3896, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 339)
        subscript_call_result_3898 = invoke(stypy.reporting.localization.Localization(__file__, 339, 48), getitem___3897, tuple_3886)
        
        # Processing the call keyword arguments (line 339)
        kwargs_3899 = {}
        # Getting the type of 'id' (line 339)
        id_3885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 45), 'id', False)
        # Calling id(args, kwargs) (line 339)
        id_call_result_3900 = invoke(stypy.reporting.localization.Localization(__file__, 339, 45), id_3885, *[subscript_call_result_3898], **kwargs_3899)
        
        # Applying the binary operator '==' (line 339)
        result_eq_3901 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 23), '==', vect_id_call_result_3884, id_call_result_3900)
        
        # Processing the call keyword arguments (line 339)
        kwargs_3902 = {}
        # Getting the type of 'np' (line 339)
        np_3878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 339)
        all_3879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 16), np_3878, 'all')
        # Calling all(args, kwargs) (line 339)
        all_call_result_3903 = invoke(stypy.reporting.localization.Localization(__file__, 339, 16), all_3879, *[result_eq_3901], **kwargs_3902)
        
        # Processing the call keyword arguments (line 339)
        kwargs_3904 = {}
        # Getting the type of 'assert_' (line 339)
        assert__3877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 339)
        assert__call_result_3905 = invoke(stypy.reporting.localization.Localization(__file__, 339, 8), assert__3877, *[all_call_result_3903], **kwargs_3904)
        
        
        # ################# End of 'test_8d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_8d' in the type store
        # Getting the type of 'stypy_return_type' (line 335)
        stypy_return_type_3906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3906)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_8d'
        return stypy_return_type_3906


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 290, 0, False)
        # Assigning a type to the variable 'self' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointerArray.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestPointerArray' (line 290)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 0), 'TestPointerArray', TestPointerArray)
# Declaration of the 'TestPointerStructures' class

class TestPointerStructures:

    @norecursion
    def test_scalars(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_scalars'
        module_type_store = module_type_store.open_function_context('test_scalars', 345, 4, False)
        # Assigning a type to the variable 'self' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPointerStructures.test_scalars.__dict__.__setitem__('stypy_localization', localization)
        TestPointerStructures.test_scalars.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPointerStructures.test_scalars.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPointerStructures.test_scalars.__dict__.__setitem__('stypy_function_name', 'TestPointerStructures.test_scalars')
        TestPointerStructures.test_scalars.__dict__.__setitem__('stypy_param_names_list', [])
        TestPointerStructures.test_scalars.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPointerStructures.test_scalars.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPointerStructures.test_scalars.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPointerStructures.test_scalars.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPointerStructures.test_scalars.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPointerStructures.test_scalars.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointerStructures.test_scalars', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_scalars', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_scalars(...)' code ##################

        
        # Assigning a Call to a Name (line 346):
        
        # Call to readsav(...): (line 346)
        # Processing the call arguments (line 346)
        
        # Call to join(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 'DATA_PATH' (line 346)
        DATA_PATH_3910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 30), 'DATA_PATH', False)
        str_3911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 41), 'str', 'struct_pointers.sav')
        # Processing the call keyword arguments (line 346)
        kwargs_3912 = {}
        # Getting the type of 'path' (line 346)
        path_3908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 346)
        join_3909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 20), path_3908, 'join')
        # Calling join(args, kwargs) (line 346)
        join_call_result_3913 = invoke(stypy.reporting.localization.Localization(__file__, 346, 20), join_3909, *[DATA_PATH_3910, str_3911], **kwargs_3912)
        
        # Processing the call keyword arguments (line 346)
        # Getting the type of 'False' (line 346)
        False_3914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 73), 'False', False)
        keyword_3915 = False_3914
        kwargs_3916 = {'verbose': keyword_3915}
        # Getting the type of 'readsav' (line 346)
        readsav_3907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 346)
        readsav_call_result_3917 = invoke(stypy.reporting.localization.Localization(__file__, 346, 12), readsav_3907, *[join_call_result_3913], **kwargs_3916)
        
        # Assigning a type to the variable 's' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 's', readsav_call_result_3917)
        
        # Call to assert_identical(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 's' (line 347)
        s_3919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 25), 's', False)
        # Obtaining the member 'pointers' of a type (line 347)
        pointers_3920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 25), s_3919, 'pointers')
        # Obtaining the member 'g' of a type (line 347)
        g_3921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 25), pointers_3920, 'g')
        
        # Call to array(...): (line 347)
        # Processing the call arguments (line 347)
        
        # Call to float32(...): (line 347)
        # Processing the call arguments (line 347)
        float_3926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 59), 'float')
        # Processing the call keyword arguments (line 347)
        kwargs_3927 = {}
        # Getting the type of 'np' (line 347)
        np_3924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 48), 'np', False)
        # Obtaining the member 'float32' of a type (line 347)
        float32_3925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 48), np_3924, 'float32')
        # Calling float32(args, kwargs) (line 347)
        float32_call_result_3928 = invoke(stypy.reporting.localization.Localization(__file__, 347, 48), float32_3925, *[float_3926], **kwargs_3927)
        
        # Processing the call keyword arguments (line 347)
        # Getting the type of 'np' (line 347)
        np_3929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 70), 'np', False)
        # Obtaining the member 'object_' of a type (line 347)
        object__3930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 70), np_3929, 'object_')
        keyword_3931 = object__3930
        kwargs_3932 = {'dtype': keyword_3931}
        # Getting the type of 'np' (line 347)
        np_3922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 39), 'np', False)
        # Obtaining the member 'array' of a type (line 347)
        array_3923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 39), np_3922, 'array')
        # Calling array(args, kwargs) (line 347)
        array_call_result_3933 = invoke(stypy.reporting.localization.Localization(__file__, 347, 39), array_3923, *[float32_call_result_3928], **kwargs_3932)
        
        # Processing the call keyword arguments (line 347)
        kwargs_3934 = {}
        # Getting the type of 'assert_identical' (line 347)
        assert_identical_3918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 347)
        assert_identical_call_result_3935 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), assert_identical_3918, *[g_3921, array_call_result_3933], **kwargs_3934)
        
        
        # Call to assert_identical(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 's' (line 348)
        s_3937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 25), 's', False)
        # Obtaining the member 'pointers' of a type (line 348)
        pointers_3938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 25), s_3937, 'pointers')
        # Obtaining the member 'h' of a type (line 348)
        h_3939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 25), pointers_3938, 'h')
        
        # Call to array(...): (line 348)
        # Processing the call arguments (line 348)
        
        # Call to float32(...): (line 348)
        # Processing the call arguments (line 348)
        float_3944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 59), 'float')
        # Processing the call keyword arguments (line 348)
        kwargs_3945 = {}
        # Getting the type of 'np' (line 348)
        np_3942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 48), 'np', False)
        # Obtaining the member 'float32' of a type (line 348)
        float32_3943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 48), np_3942, 'float32')
        # Calling float32(args, kwargs) (line 348)
        float32_call_result_3946 = invoke(stypy.reporting.localization.Localization(__file__, 348, 48), float32_3943, *[float_3944], **kwargs_3945)
        
        # Processing the call keyword arguments (line 348)
        # Getting the type of 'np' (line 348)
        np_3947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 70), 'np', False)
        # Obtaining the member 'object_' of a type (line 348)
        object__3948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 70), np_3947, 'object_')
        keyword_3949 = object__3948
        kwargs_3950 = {'dtype': keyword_3949}
        # Getting the type of 'np' (line 348)
        np_3940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 39), 'np', False)
        # Obtaining the member 'array' of a type (line 348)
        array_3941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 39), np_3940, 'array')
        # Calling array(args, kwargs) (line 348)
        array_call_result_3951 = invoke(stypy.reporting.localization.Localization(__file__, 348, 39), array_3941, *[float32_call_result_3946], **kwargs_3950)
        
        # Processing the call keyword arguments (line 348)
        kwargs_3952 = {}
        # Getting the type of 'assert_identical' (line 348)
        assert_identical_3936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 348)
        assert_identical_call_result_3953 = invoke(stypy.reporting.localization.Localization(__file__, 348, 8), assert_identical_3936, *[h_3939, array_call_result_3951], **kwargs_3952)
        
        
        # Call to assert_(...): (line 349)
        # Processing the call arguments (line 349)
        
        
        # Call to id(...): (line 349)
        # Processing the call arguments (line 349)
        
        # Obtaining the type of the subscript
        int_3956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 32), 'int')
        # Getting the type of 's' (line 349)
        s_3957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 19), 's', False)
        # Obtaining the member 'pointers' of a type (line 349)
        pointers_3958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 19), s_3957, 'pointers')
        # Obtaining the member 'g' of a type (line 349)
        g_3959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 19), pointers_3958, 'g')
        # Obtaining the member '__getitem__' of a type (line 349)
        getitem___3960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 19), g_3959, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 349)
        subscript_call_result_3961 = invoke(stypy.reporting.localization.Localization(__file__, 349, 19), getitem___3960, int_3956)
        
        # Processing the call keyword arguments (line 349)
        kwargs_3962 = {}
        # Getting the type of 'id' (line 349)
        id_3955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 16), 'id', False)
        # Calling id(args, kwargs) (line 349)
        id_call_result_3963 = invoke(stypy.reporting.localization.Localization(__file__, 349, 16), id_3955, *[subscript_call_result_3961], **kwargs_3962)
        
        
        # Call to id(...): (line 349)
        # Processing the call arguments (line 349)
        
        # Obtaining the type of the subscript
        int_3965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 55), 'int')
        # Getting the type of 's' (line 349)
        s_3966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 42), 's', False)
        # Obtaining the member 'pointers' of a type (line 349)
        pointers_3967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 42), s_3966, 'pointers')
        # Obtaining the member 'h' of a type (line 349)
        h_3968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 42), pointers_3967, 'h')
        # Obtaining the member '__getitem__' of a type (line 349)
        getitem___3969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 42), h_3968, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 349)
        subscript_call_result_3970 = invoke(stypy.reporting.localization.Localization(__file__, 349, 42), getitem___3969, int_3965)
        
        # Processing the call keyword arguments (line 349)
        kwargs_3971 = {}
        # Getting the type of 'id' (line 349)
        id_3964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 39), 'id', False)
        # Calling id(args, kwargs) (line 349)
        id_call_result_3972 = invoke(stypy.reporting.localization.Localization(__file__, 349, 39), id_3964, *[subscript_call_result_3970], **kwargs_3971)
        
        # Applying the binary operator '==' (line 349)
        result_eq_3973 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 16), '==', id_call_result_3963, id_call_result_3972)
        
        # Processing the call keyword arguments (line 349)
        kwargs_3974 = {}
        # Getting the type of 'assert_' (line 349)
        assert__3954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 349)
        assert__call_result_3975 = invoke(stypy.reporting.localization.Localization(__file__, 349, 8), assert__3954, *[result_eq_3973], **kwargs_3974)
        
        
        # ################# End of 'test_scalars(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_scalars' in the type store
        # Getting the type of 'stypy_return_type' (line 345)
        stypy_return_type_3976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3976)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_scalars'
        return stypy_return_type_3976


    @norecursion
    def test_pointers_replicated(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_pointers_replicated'
        module_type_store = module_type_store.open_function_context('test_pointers_replicated', 351, 4, False)
        # Assigning a type to the variable 'self' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPointerStructures.test_pointers_replicated.__dict__.__setitem__('stypy_localization', localization)
        TestPointerStructures.test_pointers_replicated.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPointerStructures.test_pointers_replicated.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPointerStructures.test_pointers_replicated.__dict__.__setitem__('stypy_function_name', 'TestPointerStructures.test_pointers_replicated')
        TestPointerStructures.test_pointers_replicated.__dict__.__setitem__('stypy_param_names_list', [])
        TestPointerStructures.test_pointers_replicated.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPointerStructures.test_pointers_replicated.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPointerStructures.test_pointers_replicated.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPointerStructures.test_pointers_replicated.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPointerStructures.test_pointers_replicated.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPointerStructures.test_pointers_replicated.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointerStructures.test_pointers_replicated', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_pointers_replicated', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_pointers_replicated(...)' code ##################

        
        # Assigning a Call to a Name (line 352):
        
        # Call to readsav(...): (line 352)
        # Processing the call arguments (line 352)
        
        # Call to join(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'DATA_PATH' (line 352)
        DATA_PATH_3980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 30), 'DATA_PATH', False)
        str_3981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 41), 'str', 'struct_pointers_replicated.sav')
        # Processing the call keyword arguments (line 352)
        kwargs_3982 = {}
        # Getting the type of 'path' (line 352)
        path_3978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 352)
        join_3979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 20), path_3978, 'join')
        # Calling join(args, kwargs) (line 352)
        join_call_result_3983 = invoke(stypy.reporting.localization.Localization(__file__, 352, 20), join_3979, *[DATA_PATH_3980, str_3981], **kwargs_3982)
        
        # Processing the call keyword arguments (line 352)
        # Getting the type of 'False' (line 352)
        False_3984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 84), 'False', False)
        keyword_3985 = False_3984
        kwargs_3986 = {'verbose': keyword_3985}
        # Getting the type of 'readsav' (line 352)
        readsav_3977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 352)
        readsav_call_result_3987 = invoke(stypy.reporting.localization.Localization(__file__, 352, 12), readsav_3977, *[join_call_result_3983], **kwargs_3986)
        
        # Assigning a type to the variable 's' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 8), 's', readsav_call_result_3987)
        
        # Call to assert_identical(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 's' (line 353)
        s_3989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 25), 's', False)
        # Obtaining the member 'pointers_rep' of a type (line 353)
        pointers_rep_3990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 25), s_3989, 'pointers_rep')
        # Obtaining the member 'g' of a type (line 353)
        g_3991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 25), pointers_rep_3990, 'g')
        
        # Call to astype(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'np' (line 353)
        np_4003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 79), 'np', False)
        # Obtaining the member 'object_' of a type (line 353)
        object__4004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 79), np_4003, 'object_')
        # Processing the call keyword arguments (line 353)
        kwargs_4005 = {}
        
        # Call to repeat(...): (line 353)
        # Processing the call arguments (line 353)
        
        # Call to float32(...): (line 353)
        # Processing the call arguments (line 353)
        float_3996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 64), 'float')
        # Processing the call keyword arguments (line 353)
        kwargs_3997 = {}
        # Getting the type of 'np' (line 353)
        np_3994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 53), 'np', False)
        # Obtaining the member 'float32' of a type (line 353)
        float32_3995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 53), np_3994, 'float32')
        # Calling float32(args, kwargs) (line 353)
        float32_call_result_3998 = invoke(stypy.reporting.localization.Localization(__file__, 353, 53), float32_3995, *[float_3996], **kwargs_3997)
        
        int_3999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 69), 'int')
        # Processing the call keyword arguments (line 353)
        kwargs_4000 = {}
        # Getting the type of 'np' (line 353)
        np_3992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 43), 'np', False)
        # Obtaining the member 'repeat' of a type (line 353)
        repeat_3993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 43), np_3992, 'repeat')
        # Calling repeat(args, kwargs) (line 353)
        repeat_call_result_4001 = invoke(stypy.reporting.localization.Localization(__file__, 353, 43), repeat_3993, *[float32_call_result_3998, int_3999], **kwargs_4000)
        
        # Obtaining the member 'astype' of a type (line 353)
        astype_4002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 43), repeat_call_result_4001, 'astype')
        # Calling astype(args, kwargs) (line 353)
        astype_call_result_4006 = invoke(stypy.reporting.localization.Localization(__file__, 353, 43), astype_4002, *[object__4004], **kwargs_4005)
        
        # Processing the call keyword arguments (line 353)
        kwargs_4007 = {}
        # Getting the type of 'assert_identical' (line 353)
        assert_identical_3988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 353)
        assert_identical_call_result_4008 = invoke(stypy.reporting.localization.Localization(__file__, 353, 8), assert_identical_3988, *[g_3991, astype_call_result_4006], **kwargs_4007)
        
        
        # Call to assert_identical(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 's' (line 354)
        s_4010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 25), 's', False)
        # Obtaining the member 'pointers_rep' of a type (line 354)
        pointers_rep_4011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 25), s_4010, 'pointers_rep')
        # Obtaining the member 'h' of a type (line 354)
        h_4012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 25), pointers_rep_4011, 'h')
        
        # Call to astype(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'np' (line 354)
        np_4024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 79), 'np', False)
        # Obtaining the member 'object_' of a type (line 354)
        object__4025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 79), np_4024, 'object_')
        # Processing the call keyword arguments (line 354)
        kwargs_4026 = {}
        
        # Call to repeat(...): (line 354)
        # Processing the call arguments (line 354)
        
        # Call to float32(...): (line 354)
        # Processing the call arguments (line 354)
        float_4017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 64), 'float')
        # Processing the call keyword arguments (line 354)
        kwargs_4018 = {}
        # Getting the type of 'np' (line 354)
        np_4015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 53), 'np', False)
        # Obtaining the member 'float32' of a type (line 354)
        float32_4016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 53), np_4015, 'float32')
        # Calling float32(args, kwargs) (line 354)
        float32_call_result_4019 = invoke(stypy.reporting.localization.Localization(__file__, 354, 53), float32_4016, *[float_4017], **kwargs_4018)
        
        int_4020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 69), 'int')
        # Processing the call keyword arguments (line 354)
        kwargs_4021 = {}
        # Getting the type of 'np' (line 354)
        np_4013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 43), 'np', False)
        # Obtaining the member 'repeat' of a type (line 354)
        repeat_4014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 43), np_4013, 'repeat')
        # Calling repeat(args, kwargs) (line 354)
        repeat_call_result_4022 = invoke(stypy.reporting.localization.Localization(__file__, 354, 43), repeat_4014, *[float32_call_result_4019, int_4020], **kwargs_4021)
        
        # Obtaining the member 'astype' of a type (line 354)
        astype_4023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 43), repeat_call_result_4022, 'astype')
        # Calling astype(args, kwargs) (line 354)
        astype_call_result_4027 = invoke(stypy.reporting.localization.Localization(__file__, 354, 43), astype_4023, *[object__4025], **kwargs_4026)
        
        # Processing the call keyword arguments (line 354)
        kwargs_4028 = {}
        # Getting the type of 'assert_identical' (line 354)
        assert_identical_4009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 354)
        assert_identical_call_result_4029 = invoke(stypy.reporting.localization.Localization(__file__, 354, 8), assert_identical_4009, *[h_4012, astype_call_result_4027], **kwargs_4028)
        
        
        # Call to assert_(...): (line 355)
        # Processing the call arguments (line 355)
        
        # Call to all(...): (line 355)
        # Processing the call arguments (line 355)
        
        
        # Call to vect_id(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 's' (line 355)
        s_4034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 31), 's', False)
        # Obtaining the member 'pointers_rep' of a type (line 355)
        pointers_rep_4035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 31), s_4034, 'pointers_rep')
        # Obtaining the member 'g' of a type (line 355)
        g_4036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 31), pointers_rep_4035, 'g')
        # Processing the call keyword arguments (line 355)
        kwargs_4037 = {}
        # Getting the type of 'vect_id' (line 355)
        vect_id_4033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 23), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 355)
        vect_id_call_result_4038 = invoke(stypy.reporting.localization.Localization(__file__, 355, 23), vect_id_4033, *[g_4036], **kwargs_4037)
        
        
        # Call to vect_id(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 's' (line 355)
        s_4040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 60), 's', False)
        # Obtaining the member 'pointers_rep' of a type (line 355)
        pointers_rep_4041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 60), s_4040, 'pointers_rep')
        # Obtaining the member 'h' of a type (line 355)
        h_4042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 60), pointers_rep_4041, 'h')
        # Processing the call keyword arguments (line 355)
        kwargs_4043 = {}
        # Getting the type of 'vect_id' (line 355)
        vect_id_4039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 52), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 355)
        vect_id_call_result_4044 = invoke(stypy.reporting.localization.Localization(__file__, 355, 52), vect_id_4039, *[h_4042], **kwargs_4043)
        
        # Applying the binary operator '==' (line 355)
        result_eq_4045 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 23), '==', vect_id_call_result_4038, vect_id_call_result_4044)
        
        # Processing the call keyword arguments (line 355)
        kwargs_4046 = {}
        # Getting the type of 'np' (line 355)
        np_4031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 355)
        all_4032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 16), np_4031, 'all')
        # Calling all(args, kwargs) (line 355)
        all_call_result_4047 = invoke(stypy.reporting.localization.Localization(__file__, 355, 16), all_4032, *[result_eq_4045], **kwargs_4046)
        
        # Processing the call keyword arguments (line 355)
        kwargs_4048 = {}
        # Getting the type of 'assert_' (line 355)
        assert__4030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 355)
        assert__call_result_4049 = invoke(stypy.reporting.localization.Localization(__file__, 355, 8), assert__4030, *[all_call_result_4047], **kwargs_4048)
        
        
        # ################# End of 'test_pointers_replicated(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_pointers_replicated' in the type store
        # Getting the type of 'stypy_return_type' (line 351)
        stypy_return_type_4050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4050)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_pointers_replicated'
        return stypy_return_type_4050


    @norecursion
    def test_pointers_replicated_3d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_pointers_replicated_3d'
        module_type_store = module_type_store.open_function_context('test_pointers_replicated_3d', 357, 4, False)
        # Assigning a type to the variable 'self' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPointerStructures.test_pointers_replicated_3d.__dict__.__setitem__('stypy_localization', localization)
        TestPointerStructures.test_pointers_replicated_3d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPointerStructures.test_pointers_replicated_3d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPointerStructures.test_pointers_replicated_3d.__dict__.__setitem__('stypy_function_name', 'TestPointerStructures.test_pointers_replicated_3d')
        TestPointerStructures.test_pointers_replicated_3d.__dict__.__setitem__('stypy_param_names_list', [])
        TestPointerStructures.test_pointers_replicated_3d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPointerStructures.test_pointers_replicated_3d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPointerStructures.test_pointers_replicated_3d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPointerStructures.test_pointers_replicated_3d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPointerStructures.test_pointers_replicated_3d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPointerStructures.test_pointers_replicated_3d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointerStructures.test_pointers_replicated_3d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_pointers_replicated_3d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_pointers_replicated_3d(...)' code ##################

        
        # Assigning a Call to a Name (line 358):
        
        # Call to readsav(...): (line 358)
        # Processing the call arguments (line 358)
        
        # Call to join(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'DATA_PATH' (line 358)
        DATA_PATH_4054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 30), 'DATA_PATH', False)
        str_4055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 41), 'str', 'struct_pointers_replicated_3d.sav')
        # Processing the call keyword arguments (line 358)
        kwargs_4056 = {}
        # Getting the type of 'path' (line 358)
        path_4052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 358)
        join_4053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 20), path_4052, 'join')
        # Calling join(args, kwargs) (line 358)
        join_call_result_4057 = invoke(stypy.reporting.localization.Localization(__file__, 358, 20), join_4053, *[DATA_PATH_4054, str_4055], **kwargs_4056)
        
        # Processing the call keyword arguments (line 358)
        # Getting the type of 'False' (line 358)
        False_4058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 87), 'False', False)
        keyword_4059 = False_4058
        kwargs_4060 = {'verbose': keyword_4059}
        # Getting the type of 'readsav' (line 358)
        readsav_4051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 358)
        readsav_call_result_4061 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), readsav_4051, *[join_call_result_4057], **kwargs_4060)
        
        # Assigning a type to the variable 's' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 's', readsav_call_result_4061)
        
        # Assigning a Call to a Name (line 359):
        
        # Call to astype(...): (line 359)
        # Processing the call arguments (line 359)
        # Getting the type of 'np' (line 359)
        np_4079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 73), 'np', False)
        # Obtaining the member 'object_' of a type (line 359)
        object__4080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 73), np_4079, 'object_')
        # Processing the call keyword arguments (line 359)
        kwargs_4081 = {}
        
        # Call to reshape(...): (line 359)
        # Processing the call arguments (line 359)
        int_4073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 57), 'int')
        int_4074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 60), 'int')
        int_4075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 63), 'int')
        # Processing the call keyword arguments (line 359)
        kwargs_4076 = {}
        
        # Call to repeat(...): (line 359)
        # Processing the call arguments (line 359)
        
        # Call to float32(...): (line 359)
        # Processing the call arguments (line 359)
        float_4066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 40), 'float')
        # Processing the call keyword arguments (line 359)
        kwargs_4067 = {}
        # Getting the type of 'np' (line 359)
        np_4064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 29), 'np', False)
        # Obtaining the member 'float32' of a type (line 359)
        float32_4065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 29), np_4064, 'float32')
        # Calling float32(args, kwargs) (line 359)
        float32_call_result_4068 = invoke(stypy.reporting.localization.Localization(__file__, 359, 29), float32_4065, *[float_4066], **kwargs_4067)
        
        int_4069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 45), 'int')
        # Processing the call keyword arguments (line 359)
        kwargs_4070 = {}
        # Getting the type of 'np' (line 359)
        np_4062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 19), 'np', False)
        # Obtaining the member 'repeat' of a type (line 359)
        repeat_4063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 19), np_4062, 'repeat')
        # Calling repeat(args, kwargs) (line 359)
        repeat_call_result_4071 = invoke(stypy.reporting.localization.Localization(__file__, 359, 19), repeat_4063, *[float32_call_result_4068, int_4069], **kwargs_4070)
        
        # Obtaining the member 'reshape' of a type (line 359)
        reshape_4072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 19), repeat_call_result_4071, 'reshape')
        # Calling reshape(args, kwargs) (line 359)
        reshape_call_result_4077 = invoke(stypy.reporting.localization.Localization(__file__, 359, 19), reshape_4072, *[int_4073, int_4074, int_4075], **kwargs_4076)
        
        # Obtaining the member 'astype' of a type (line 359)
        astype_4078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 19), reshape_call_result_4077, 'astype')
        # Calling astype(args, kwargs) (line 359)
        astype_call_result_4082 = invoke(stypy.reporting.localization.Localization(__file__, 359, 19), astype_4078, *[object__4080], **kwargs_4081)
        
        # Assigning a type to the variable 's_expect' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 's_expect', astype_call_result_4082)
        
        # Call to assert_identical(...): (line 360)
        # Processing the call arguments (line 360)
        # Getting the type of 's' (line 360)
        s_4084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 25), 's', False)
        # Obtaining the member 'pointers_rep' of a type (line 360)
        pointers_rep_4085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 25), s_4084, 'pointers_rep')
        # Obtaining the member 'g' of a type (line 360)
        g_4086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 25), pointers_rep_4085, 'g')
        # Getting the type of 's_expect' (line 360)
        s_expect_4087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 43), 's_expect', False)
        # Processing the call keyword arguments (line 360)
        kwargs_4088 = {}
        # Getting the type of 'assert_identical' (line 360)
        assert_identical_4083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 360)
        assert_identical_call_result_4089 = invoke(stypy.reporting.localization.Localization(__file__, 360, 8), assert_identical_4083, *[g_4086, s_expect_4087], **kwargs_4088)
        
        
        # Call to assert_identical(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 's' (line 361)
        s_4091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 25), 's', False)
        # Obtaining the member 'pointers_rep' of a type (line 361)
        pointers_rep_4092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 25), s_4091, 'pointers_rep')
        # Obtaining the member 'h' of a type (line 361)
        h_4093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 25), pointers_rep_4092, 'h')
        # Getting the type of 's_expect' (line 361)
        s_expect_4094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 43), 's_expect', False)
        # Processing the call keyword arguments (line 361)
        kwargs_4095 = {}
        # Getting the type of 'assert_identical' (line 361)
        assert_identical_4090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 361)
        assert_identical_call_result_4096 = invoke(stypy.reporting.localization.Localization(__file__, 361, 8), assert_identical_4090, *[h_4093, s_expect_4094], **kwargs_4095)
        
        
        # Call to assert_(...): (line 362)
        # Processing the call arguments (line 362)
        
        # Call to all(...): (line 362)
        # Processing the call arguments (line 362)
        
        
        # Call to vect_id(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 's' (line 362)
        s_4101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 31), 's', False)
        # Obtaining the member 'pointers_rep' of a type (line 362)
        pointers_rep_4102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 31), s_4101, 'pointers_rep')
        # Obtaining the member 'g' of a type (line 362)
        g_4103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 31), pointers_rep_4102, 'g')
        # Processing the call keyword arguments (line 362)
        kwargs_4104 = {}
        # Getting the type of 'vect_id' (line 362)
        vect_id_4100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 23), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 362)
        vect_id_call_result_4105 = invoke(stypy.reporting.localization.Localization(__file__, 362, 23), vect_id_4100, *[g_4103], **kwargs_4104)
        
        
        # Call to vect_id(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 's' (line 362)
        s_4107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 60), 's', False)
        # Obtaining the member 'pointers_rep' of a type (line 362)
        pointers_rep_4108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 60), s_4107, 'pointers_rep')
        # Obtaining the member 'h' of a type (line 362)
        h_4109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 60), pointers_rep_4108, 'h')
        # Processing the call keyword arguments (line 362)
        kwargs_4110 = {}
        # Getting the type of 'vect_id' (line 362)
        vect_id_4106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 52), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 362)
        vect_id_call_result_4111 = invoke(stypy.reporting.localization.Localization(__file__, 362, 52), vect_id_4106, *[h_4109], **kwargs_4110)
        
        # Applying the binary operator '==' (line 362)
        result_eq_4112 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 23), '==', vect_id_call_result_4105, vect_id_call_result_4111)
        
        # Processing the call keyword arguments (line 362)
        kwargs_4113 = {}
        # Getting the type of 'np' (line 362)
        np_4098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 362)
        all_4099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 16), np_4098, 'all')
        # Calling all(args, kwargs) (line 362)
        all_call_result_4114 = invoke(stypy.reporting.localization.Localization(__file__, 362, 16), all_4099, *[result_eq_4112], **kwargs_4113)
        
        # Processing the call keyword arguments (line 362)
        kwargs_4115 = {}
        # Getting the type of 'assert_' (line 362)
        assert__4097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 362)
        assert__call_result_4116 = invoke(stypy.reporting.localization.Localization(__file__, 362, 8), assert__4097, *[all_call_result_4114], **kwargs_4115)
        
        
        # ################# End of 'test_pointers_replicated_3d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_pointers_replicated_3d' in the type store
        # Getting the type of 'stypy_return_type' (line 357)
        stypy_return_type_4117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4117)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_pointers_replicated_3d'
        return stypy_return_type_4117


    @norecursion
    def test_arrays(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_arrays'
        module_type_store = module_type_store.open_function_context('test_arrays', 364, 4, False)
        # Assigning a type to the variable 'self' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPointerStructures.test_arrays.__dict__.__setitem__('stypy_localization', localization)
        TestPointerStructures.test_arrays.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPointerStructures.test_arrays.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPointerStructures.test_arrays.__dict__.__setitem__('stypy_function_name', 'TestPointerStructures.test_arrays')
        TestPointerStructures.test_arrays.__dict__.__setitem__('stypy_param_names_list', [])
        TestPointerStructures.test_arrays.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPointerStructures.test_arrays.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPointerStructures.test_arrays.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPointerStructures.test_arrays.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPointerStructures.test_arrays.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPointerStructures.test_arrays.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointerStructures.test_arrays', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_arrays', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_arrays(...)' code ##################

        
        # Assigning a Call to a Name (line 365):
        
        # Call to readsav(...): (line 365)
        # Processing the call arguments (line 365)
        
        # Call to join(...): (line 365)
        # Processing the call arguments (line 365)
        # Getting the type of 'DATA_PATH' (line 365)
        DATA_PATH_4121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 30), 'DATA_PATH', False)
        str_4122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 41), 'str', 'struct_pointer_arrays.sav')
        # Processing the call keyword arguments (line 365)
        kwargs_4123 = {}
        # Getting the type of 'path' (line 365)
        path_4119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 365)
        join_4120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 20), path_4119, 'join')
        # Calling join(args, kwargs) (line 365)
        join_call_result_4124 = invoke(stypy.reporting.localization.Localization(__file__, 365, 20), join_4120, *[DATA_PATH_4121, str_4122], **kwargs_4123)
        
        # Processing the call keyword arguments (line 365)
        # Getting the type of 'False' (line 365)
        False_4125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 79), 'False', False)
        keyword_4126 = False_4125
        kwargs_4127 = {'verbose': keyword_4126}
        # Getting the type of 'readsav' (line 365)
        readsav_4118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 365)
        readsav_call_result_4128 = invoke(stypy.reporting.localization.Localization(__file__, 365, 12), readsav_4118, *[join_call_result_4124], **kwargs_4127)
        
        # Assigning a type to the variable 's' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 's', readsav_call_result_4128)
        
        # Call to assert_array_identical(...): (line 366)
        # Processing the call arguments (line 366)
        
        # Obtaining the type of the subscript
        int_4130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 42), 'int')
        # Getting the type of 's' (line 366)
        s_4131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 31), 's', False)
        # Obtaining the member 'arrays' of a type (line 366)
        arrays_4132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 31), s_4131, 'arrays')
        # Obtaining the member 'g' of a type (line 366)
        g_4133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 31), arrays_4132, 'g')
        # Obtaining the member '__getitem__' of a type (line 366)
        getitem___4134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 31), g_4133, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 366)
        subscript_call_result_4135 = invoke(stypy.reporting.localization.Localization(__file__, 366, 31), getitem___4134, int_4130)
        
        
        # Call to astype(...): (line 366)
        # Processing the call arguments (line 366)
        # Getting the type of 'np' (line 366)
        np_4147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 82), 'np', False)
        # Obtaining the member 'object_' of a type (line 366)
        object__4148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 82), np_4147, 'object_')
        # Processing the call keyword arguments (line 366)
        kwargs_4149 = {}
        
        # Call to repeat(...): (line 366)
        # Processing the call arguments (line 366)
        
        # Call to float32(...): (line 366)
        # Processing the call arguments (line 366)
        float_4140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 67), 'float')
        # Processing the call keyword arguments (line 366)
        kwargs_4141 = {}
        # Getting the type of 'np' (line 366)
        np_4138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 56), 'np', False)
        # Obtaining the member 'float32' of a type (line 366)
        float32_4139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 56), np_4138, 'float32')
        # Calling float32(args, kwargs) (line 366)
        float32_call_result_4142 = invoke(stypy.reporting.localization.Localization(__file__, 366, 56), float32_4139, *[float_4140], **kwargs_4141)
        
        int_4143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 366, 72), 'int')
        # Processing the call keyword arguments (line 366)
        kwargs_4144 = {}
        # Getting the type of 'np' (line 366)
        np_4136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 46), 'np', False)
        # Obtaining the member 'repeat' of a type (line 366)
        repeat_4137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 46), np_4136, 'repeat')
        # Calling repeat(args, kwargs) (line 366)
        repeat_call_result_4145 = invoke(stypy.reporting.localization.Localization(__file__, 366, 46), repeat_4137, *[float32_call_result_4142, int_4143], **kwargs_4144)
        
        # Obtaining the member 'astype' of a type (line 366)
        astype_4146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 46), repeat_call_result_4145, 'astype')
        # Calling astype(args, kwargs) (line 366)
        astype_call_result_4150 = invoke(stypy.reporting.localization.Localization(__file__, 366, 46), astype_4146, *[object__4148], **kwargs_4149)
        
        # Processing the call keyword arguments (line 366)
        kwargs_4151 = {}
        # Getting the type of 'assert_array_identical' (line 366)
        assert_array_identical_4129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 8), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 366)
        assert_array_identical_call_result_4152 = invoke(stypy.reporting.localization.Localization(__file__, 366, 8), assert_array_identical_4129, *[subscript_call_result_4135, astype_call_result_4150], **kwargs_4151)
        
        
        # Call to assert_array_identical(...): (line 367)
        # Processing the call arguments (line 367)
        
        # Obtaining the type of the subscript
        int_4154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 42), 'int')
        # Getting the type of 's' (line 367)
        s_4155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 31), 's', False)
        # Obtaining the member 'arrays' of a type (line 367)
        arrays_4156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 31), s_4155, 'arrays')
        # Obtaining the member 'h' of a type (line 367)
        h_4157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 31), arrays_4156, 'h')
        # Obtaining the member '__getitem__' of a type (line 367)
        getitem___4158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 31), h_4157, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 367)
        subscript_call_result_4159 = invoke(stypy.reporting.localization.Localization(__file__, 367, 31), getitem___4158, int_4154)
        
        
        # Call to astype(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'np' (line 367)
        np_4171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 82), 'np', False)
        # Obtaining the member 'object_' of a type (line 367)
        object__4172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 82), np_4171, 'object_')
        # Processing the call keyword arguments (line 367)
        kwargs_4173 = {}
        
        # Call to repeat(...): (line 367)
        # Processing the call arguments (line 367)
        
        # Call to float32(...): (line 367)
        # Processing the call arguments (line 367)
        float_4164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 67), 'float')
        # Processing the call keyword arguments (line 367)
        kwargs_4165 = {}
        # Getting the type of 'np' (line 367)
        np_4162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 56), 'np', False)
        # Obtaining the member 'float32' of a type (line 367)
        float32_4163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 56), np_4162, 'float32')
        # Calling float32(args, kwargs) (line 367)
        float32_call_result_4166 = invoke(stypy.reporting.localization.Localization(__file__, 367, 56), float32_4163, *[float_4164], **kwargs_4165)
        
        int_4167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 72), 'int')
        # Processing the call keyword arguments (line 367)
        kwargs_4168 = {}
        # Getting the type of 'np' (line 367)
        np_4160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 46), 'np', False)
        # Obtaining the member 'repeat' of a type (line 367)
        repeat_4161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 46), np_4160, 'repeat')
        # Calling repeat(args, kwargs) (line 367)
        repeat_call_result_4169 = invoke(stypy.reporting.localization.Localization(__file__, 367, 46), repeat_4161, *[float32_call_result_4166, int_4167], **kwargs_4168)
        
        # Obtaining the member 'astype' of a type (line 367)
        astype_4170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 46), repeat_call_result_4169, 'astype')
        # Calling astype(args, kwargs) (line 367)
        astype_call_result_4174 = invoke(stypy.reporting.localization.Localization(__file__, 367, 46), astype_4170, *[object__4172], **kwargs_4173)
        
        # Processing the call keyword arguments (line 367)
        kwargs_4175 = {}
        # Getting the type of 'assert_array_identical' (line 367)
        assert_array_identical_4153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 367)
        assert_array_identical_call_result_4176 = invoke(stypy.reporting.localization.Localization(__file__, 367, 8), assert_array_identical_4153, *[subscript_call_result_4159, astype_call_result_4174], **kwargs_4175)
        
        
        # Call to assert_(...): (line 368)
        # Processing the call arguments (line 368)
        
        # Call to all(...): (line 368)
        # Processing the call arguments (line 368)
        
        
        # Call to vect_id(...): (line 368)
        # Processing the call arguments (line 368)
        
        # Obtaining the type of the subscript
        int_4181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 42), 'int')
        # Getting the type of 's' (line 368)
        s_4182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 31), 's', False)
        # Obtaining the member 'arrays' of a type (line 368)
        arrays_4183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 31), s_4182, 'arrays')
        # Obtaining the member 'g' of a type (line 368)
        g_4184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 31), arrays_4183, 'g')
        # Obtaining the member '__getitem__' of a type (line 368)
        getitem___4185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 31), g_4184, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 368)
        subscript_call_result_4186 = invoke(stypy.reporting.localization.Localization(__file__, 368, 31), getitem___4185, int_4181)
        
        # Processing the call keyword arguments (line 368)
        kwargs_4187 = {}
        # Getting the type of 'vect_id' (line 368)
        vect_id_4180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 23), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 368)
        vect_id_call_result_4188 = invoke(stypy.reporting.localization.Localization(__file__, 368, 23), vect_id_4180, *[subscript_call_result_4186], **kwargs_4187)
        
        
        # Call to id(...): (line 368)
        # Processing the call arguments (line 368)
        
        # Obtaining the type of the subscript
        int_4190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 66), 'int')
        
        # Obtaining the type of the subscript
        int_4191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 63), 'int')
        # Getting the type of 's' (line 368)
        s_4192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 52), 's', False)
        # Obtaining the member 'arrays' of a type (line 368)
        arrays_4193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 52), s_4192, 'arrays')
        # Obtaining the member 'g' of a type (line 368)
        g_4194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 52), arrays_4193, 'g')
        # Obtaining the member '__getitem__' of a type (line 368)
        getitem___4195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 52), g_4194, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 368)
        subscript_call_result_4196 = invoke(stypy.reporting.localization.Localization(__file__, 368, 52), getitem___4195, int_4191)
        
        # Obtaining the member '__getitem__' of a type (line 368)
        getitem___4197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 52), subscript_call_result_4196, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 368)
        subscript_call_result_4198 = invoke(stypy.reporting.localization.Localization(__file__, 368, 52), getitem___4197, int_4190)
        
        # Processing the call keyword arguments (line 368)
        kwargs_4199 = {}
        # Getting the type of 'id' (line 368)
        id_4189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 49), 'id', False)
        # Calling id(args, kwargs) (line 368)
        id_call_result_4200 = invoke(stypy.reporting.localization.Localization(__file__, 368, 49), id_4189, *[subscript_call_result_4198], **kwargs_4199)
        
        # Applying the binary operator '==' (line 368)
        result_eq_4201 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 23), '==', vect_id_call_result_4188, id_call_result_4200)
        
        # Processing the call keyword arguments (line 368)
        kwargs_4202 = {}
        # Getting the type of 'np' (line 368)
        np_4178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 368)
        all_4179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 16), np_4178, 'all')
        # Calling all(args, kwargs) (line 368)
        all_call_result_4203 = invoke(stypy.reporting.localization.Localization(__file__, 368, 16), all_4179, *[result_eq_4201], **kwargs_4202)
        
        # Processing the call keyword arguments (line 368)
        kwargs_4204 = {}
        # Getting the type of 'assert_' (line 368)
        assert__4177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 368)
        assert__call_result_4205 = invoke(stypy.reporting.localization.Localization(__file__, 368, 8), assert__4177, *[all_call_result_4203], **kwargs_4204)
        
        
        # Call to assert_(...): (line 369)
        # Processing the call arguments (line 369)
        
        # Call to all(...): (line 369)
        # Processing the call arguments (line 369)
        
        
        # Call to vect_id(...): (line 369)
        # Processing the call arguments (line 369)
        
        # Obtaining the type of the subscript
        int_4210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 42), 'int')
        # Getting the type of 's' (line 369)
        s_4211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 31), 's', False)
        # Obtaining the member 'arrays' of a type (line 369)
        arrays_4212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 31), s_4211, 'arrays')
        # Obtaining the member 'h' of a type (line 369)
        h_4213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 31), arrays_4212, 'h')
        # Obtaining the member '__getitem__' of a type (line 369)
        getitem___4214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 31), h_4213, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 369)
        subscript_call_result_4215 = invoke(stypy.reporting.localization.Localization(__file__, 369, 31), getitem___4214, int_4210)
        
        # Processing the call keyword arguments (line 369)
        kwargs_4216 = {}
        # Getting the type of 'vect_id' (line 369)
        vect_id_4209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 23), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 369)
        vect_id_call_result_4217 = invoke(stypy.reporting.localization.Localization(__file__, 369, 23), vect_id_4209, *[subscript_call_result_4215], **kwargs_4216)
        
        
        # Call to id(...): (line 369)
        # Processing the call arguments (line 369)
        
        # Obtaining the type of the subscript
        int_4219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 66), 'int')
        
        # Obtaining the type of the subscript
        int_4220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 63), 'int')
        # Getting the type of 's' (line 369)
        s_4221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 52), 's', False)
        # Obtaining the member 'arrays' of a type (line 369)
        arrays_4222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 52), s_4221, 'arrays')
        # Obtaining the member 'h' of a type (line 369)
        h_4223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 52), arrays_4222, 'h')
        # Obtaining the member '__getitem__' of a type (line 369)
        getitem___4224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 52), h_4223, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 369)
        subscript_call_result_4225 = invoke(stypy.reporting.localization.Localization(__file__, 369, 52), getitem___4224, int_4220)
        
        # Obtaining the member '__getitem__' of a type (line 369)
        getitem___4226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 52), subscript_call_result_4225, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 369)
        subscript_call_result_4227 = invoke(stypy.reporting.localization.Localization(__file__, 369, 52), getitem___4226, int_4219)
        
        # Processing the call keyword arguments (line 369)
        kwargs_4228 = {}
        # Getting the type of 'id' (line 369)
        id_4218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 49), 'id', False)
        # Calling id(args, kwargs) (line 369)
        id_call_result_4229 = invoke(stypy.reporting.localization.Localization(__file__, 369, 49), id_4218, *[subscript_call_result_4227], **kwargs_4228)
        
        # Applying the binary operator '==' (line 369)
        result_eq_4230 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 23), '==', vect_id_call_result_4217, id_call_result_4229)
        
        # Processing the call keyword arguments (line 369)
        kwargs_4231 = {}
        # Getting the type of 'np' (line 369)
        np_4207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 16), 'np', False)
        # Obtaining the member 'all' of a type (line 369)
        all_4208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 16), np_4207, 'all')
        # Calling all(args, kwargs) (line 369)
        all_call_result_4232 = invoke(stypy.reporting.localization.Localization(__file__, 369, 16), all_4208, *[result_eq_4230], **kwargs_4231)
        
        # Processing the call keyword arguments (line 369)
        kwargs_4233 = {}
        # Getting the type of 'assert_' (line 369)
        assert__4206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 369)
        assert__call_result_4234 = invoke(stypy.reporting.localization.Localization(__file__, 369, 8), assert__4206, *[all_call_result_4232], **kwargs_4233)
        
        
        # Call to assert_(...): (line 370)
        # Processing the call arguments (line 370)
        
        
        # Call to id(...): (line 370)
        # Processing the call arguments (line 370)
        
        # Obtaining the type of the subscript
        int_4237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 33), 'int')
        
        # Obtaining the type of the subscript
        int_4238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 30), 'int')
        # Getting the type of 's' (line 370)
        s_4239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 19), 's', False)
        # Obtaining the member 'arrays' of a type (line 370)
        arrays_4240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 19), s_4239, 'arrays')
        # Obtaining the member 'g' of a type (line 370)
        g_4241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 19), arrays_4240, 'g')
        # Obtaining the member '__getitem__' of a type (line 370)
        getitem___4242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 19), g_4241, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 370)
        subscript_call_result_4243 = invoke(stypy.reporting.localization.Localization(__file__, 370, 19), getitem___4242, int_4238)
        
        # Obtaining the member '__getitem__' of a type (line 370)
        getitem___4244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 19), subscript_call_result_4243, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 370)
        subscript_call_result_4245 = invoke(stypy.reporting.localization.Localization(__file__, 370, 19), getitem___4244, int_4237)
        
        # Processing the call keyword arguments (line 370)
        kwargs_4246 = {}
        # Getting the type of 'id' (line 370)
        id_4236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 16), 'id', False)
        # Calling id(args, kwargs) (line 370)
        id_call_result_4247 = invoke(stypy.reporting.localization.Localization(__file__, 370, 16), id_4236, *[subscript_call_result_4245], **kwargs_4246)
        
        
        # Call to id(...): (line 370)
        # Processing the call arguments (line 370)
        
        # Obtaining the type of the subscript
        int_4249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 57), 'int')
        
        # Obtaining the type of the subscript
        int_4250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 54), 'int')
        # Getting the type of 's' (line 370)
        s_4251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 43), 's', False)
        # Obtaining the member 'arrays' of a type (line 370)
        arrays_4252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 43), s_4251, 'arrays')
        # Obtaining the member 'h' of a type (line 370)
        h_4253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 43), arrays_4252, 'h')
        # Obtaining the member '__getitem__' of a type (line 370)
        getitem___4254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 43), h_4253, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 370)
        subscript_call_result_4255 = invoke(stypy.reporting.localization.Localization(__file__, 370, 43), getitem___4254, int_4250)
        
        # Obtaining the member '__getitem__' of a type (line 370)
        getitem___4256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 43), subscript_call_result_4255, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 370)
        subscript_call_result_4257 = invoke(stypy.reporting.localization.Localization(__file__, 370, 43), getitem___4256, int_4249)
        
        # Processing the call keyword arguments (line 370)
        kwargs_4258 = {}
        # Getting the type of 'id' (line 370)
        id_4248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 40), 'id', False)
        # Calling id(args, kwargs) (line 370)
        id_call_result_4259 = invoke(stypy.reporting.localization.Localization(__file__, 370, 40), id_4248, *[subscript_call_result_4257], **kwargs_4258)
        
        # Applying the binary operator '==' (line 370)
        result_eq_4260 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 16), '==', id_call_result_4247, id_call_result_4259)
        
        # Processing the call keyword arguments (line 370)
        kwargs_4261 = {}
        # Getting the type of 'assert_' (line 370)
        assert__4235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 370)
        assert__call_result_4262 = invoke(stypy.reporting.localization.Localization(__file__, 370, 8), assert__4235, *[result_eq_4260], **kwargs_4261)
        
        
        # ################# End of 'test_arrays(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_arrays' in the type store
        # Getting the type of 'stypy_return_type' (line 364)
        stypy_return_type_4263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4263)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_arrays'
        return stypy_return_type_4263


    @norecursion
    def test_arrays_replicated(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_arrays_replicated'
        module_type_store = module_type_store.open_function_context('test_arrays_replicated', 372, 4, False)
        # Assigning a type to the variable 'self' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPointerStructures.test_arrays_replicated.__dict__.__setitem__('stypy_localization', localization)
        TestPointerStructures.test_arrays_replicated.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPointerStructures.test_arrays_replicated.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPointerStructures.test_arrays_replicated.__dict__.__setitem__('stypy_function_name', 'TestPointerStructures.test_arrays_replicated')
        TestPointerStructures.test_arrays_replicated.__dict__.__setitem__('stypy_param_names_list', [])
        TestPointerStructures.test_arrays_replicated.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPointerStructures.test_arrays_replicated.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPointerStructures.test_arrays_replicated.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPointerStructures.test_arrays_replicated.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPointerStructures.test_arrays_replicated.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPointerStructures.test_arrays_replicated.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointerStructures.test_arrays_replicated', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_arrays_replicated', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_arrays_replicated(...)' code ##################

        
        # Assigning a Call to a Name (line 373):
        
        # Call to readsav(...): (line 373)
        # Processing the call arguments (line 373)
        
        # Call to join(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'DATA_PATH' (line 373)
        DATA_PATH_4267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 30), 'DATA_PATH', False)
        str_4268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 41), 'str', 'struct_pointer_arrays_replicated.sav')
        # Processing the call keyword arguments (line 373)
        kwargs_4269 = {}
        # Getting the type of 'path' (line 373)
        path_4265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 373)
        join_4266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 20), path_4265, 'join')
        # Calling join(args, kwargs) (line 373)
        join_call_result_4270 = invoke(stypy.reporting.localization.Localization(__file__, 373, 20), join_4266, *[DATA_PATH_4267, str_4268], **kwargs_4269)
        
        # Processing the call keyword arguments (line 373)
        # Getting the type of 'False' (line 373)
        False_4271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 90), 'False', False)
        keyword_4272 = False_4271
        kwargs_4273 = {'verbose': keyword_4272}
        # Getting the type of 'readsav' (line 373)
        readsav_4264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 373)
        readsav_call_result_4274 = invoke(stypy.reporting.localization.Localization(__file__, 373, 12), readsav_4264, *[join_call_result_4270], **kwargs_4273)
        
        # Assigning a type to the variable 's' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 's', readsav_call_result_4274)
        
        # Call to assert_(...): (line 376)
        # Processing the call arguments (line 376)
        
        # Getting the type of 's' (line 376)
        s_4276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 16), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 376)
        arrays_rep_4277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 16), s_4276, 'arrays_rep')
        # Obtaining the member 'g' of a type (line 376)
        g_4278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 16), arrays_rep_4277, 'g')
        # Obtaining the member 'dtype' of a type (line 376)
        dtype_4279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 16), g_4278, 'dtype')
        # Obtaining the member 'type' of a type (line 376)
        type_4280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 16), dtype_4279, 'type')
        # Getting the type of 'np' (line 376)
        np_4281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 45), 'np', False)
        # Obtaining the member 'object_' of a type (line 376)
        object__4282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 45), np_4281, 'object_')
        # Applying the binary operator 'is' (line 376)
        result_is__4283 = python_operator(stypy.reporting.localization.Localization(__file__, 376, 16), 'is', type_4280, object__4282)
        
        # Processing the call keyword arguments (line 376)
        kwargs_4284 = {}
        # Getting the type of 'assert_' (line 376)
        assert__4275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 376)
        assert__call_result_4285 = invoke(stypy.reporting.localization.Localization(__file__, 376, 8), assert__4275, *[result_is__4283], **kwargs_4284)
        
        
        # Call to assert_(...): (line 377)
        # Processing the call arguments (line 377)
        
        # Getting the type of 's' (line 377)
        s_4287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 16), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 377)
        arrays_rep_4288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 16), s_4287, 'arrays_rep')
        # Obtaining the member 'h' of a type (line 377)
        h_4289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 16), arrays_rep_4288, 'h')
        # Obtaining the member 'dtype' of a type (line 377)
        dtype_4290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 16), h_4289, 'dtype')
        # Obtaining the member 'type' of a type (line 377)
        type_4291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 16), dtype_4290, 'type')
        # Getting the type of 'np' (line 377)
        np_4292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 45), 'np', False)
        # Obtaining the member 'object_' of a type (line 377)
        object__4293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 45), np_4292, 'object_')
        # Applying the binary operator 'is' (line 377)
        result_is__4294 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 16), 'is', type_4291, object__4293)
        
        # Processing the call keyword arguments (line 377)
        kwargs_4295 = {}
        # Getting the type of 'assert_' (line 377)
        assert__4286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 377)
        assert__call_result_4296 = invoke(stypy.reporting.localization.Localization(__file__, 377, 8), assert__4286, *[result_is__4294], **kwargs_4295)
        
        
        # Call to assert_equal(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 's' (line 380)
        s_4298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 21), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 380)
        arrays_rep_4299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 21), s_4298, 'arrays_rep')
        # Obtaining the member 'g' of a type (line 380)
        g_4300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 21), arrays_rep_4299, 'g')
        # Obtaining the member 'shape' of a type (line 380)
        shape_4301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 21), g_4300, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 380)
        tuple_4302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 380)
        # Adding element type (line 380)
        int_4303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 44), tuple_4302, int_4303)
        
        # Processing the call keyword arguments (line 380)
        kwargs_4304 = {}
        # Getting the type of 'assert_equal' (line 380)
        assert_equal_4297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 380)
        assert_equal_call_result_4305 = invoke(stypy.reporting.localization.Localization(__file__, 380, 8), assert_equal_4297, *[shape_4301, tuple_4302], **kwargs_4304)
        
        
        # Call to assert_equal(...): (line 381)
        # Processing the call arguments (line 381)
        # Getting the type of 's' (line 381)
        s_4307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 21), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 381)
        arrays_rep_4308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 21), s_4307, 'arrays_rep')
        # Obtaining the member 'h' of a type (line 381)
        h_4309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 21), arrays_rep_4308, 'h')
        # Obtaining the member 'shape' of a type (line 381)
        shape_4310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 21), h_4309, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 381)
        tuple_4311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 381)
        # Adding element type (line 381)
        int_4312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 381, 44), tuple_4311, int_4312)
        
        # Processing the call keyword arguments (line 381)
        kwargs_4313 = {}
        # Getting the type of 'assert_equal' (line 381)
        assert_equal_4306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 381)
        assert_equal_call_result_4314 = invoke(stypy.reporting.localization.Localization(__file__, 381, 8), assert_equal_4306, *[shape_4310, tuple_4311], **kwargs_4313)
        
        
        
        # Call to range(...): (line 384)
        # Processing the call arguments (line 384)
        int_4316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 23), 'int')
        # Processing the call keyword arguments (line 384)
        kwargs_4317 = {}
        # Getting the type of 'range' (line 384)
        range_4315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 17), 'range', False)
        # Calling range(args, kwargs) (line 384)
        range_call_result_4318 = invoke(stypy.reporting.localization.Localization(__file__, 384, 17), range_4315, *[int_4316], **kwargs_4317)
        
        # Testing the type of a for loop iterable (line 384)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 384, 8), range_call_result_4318)
        # Getting the type of the for loop variable (line 384)
        for_loop_var_4319 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 384, 8), range_call_result_4318)
        # Assigning a type to the variable 'i' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'i', for_loop_var_4319)
        # SSA begins for a for statement (line 384)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_array_identical(...): (line 385)
        # Processing the call arguments (line 385)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 385)
        i_4321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 50), 'i', False)
        # Getting the type of 's' (line 385)
        s_4322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 35), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 385)
        arrays_rep_4323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 35), s_4322, 'arrays_rep')
        # Obtaining the member 'g' of a type (line 385)
        g_4324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 35), arrays_rep_4323, 'g')
        # Obtaining the member '__getitem__' of a type (line 385)
        getitem___4325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 35), g_4324, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 385)
        subscript_call_result_4326 = invoke(stypy.reporting.localization.Localization(__file__, 385, 35), getitem___4325, i_4321)
        
        
        # Call to astype(...): (line 385)
        # Processing the call arguments (line 385)
        # Getting the type of 'np' (line 385)
        np_4338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 90), 'np', False)
        # Obtaining the member 'object_' of a type (line 385)
        object__4339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 90), np_4338, 'object_')
        # Processing the call keyword arguments (line 385)
        kwargs_4340 = {}
        
        # Call to repeat(...): (line 385)
        # Processing the call arguments (line 385)
        
        # Call to float32(...): (line 385)
        # Processing the call arguments (line 385)
        float_4331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 75), 'float')
        # Processing the call keyword arguments (line 385)
        kwargs_4332 = {}
        # Getting the type of 'np' (line 385)
        np_4329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 64), 'np', False)
        # Obtaining the member 'float32' of a type (line 385)
        float32_4330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 64), np_4329, 'float32')
        # Calling float32(args, kwargs) (line 385)
        float32_call_result_4333 = invoke(stypy.reporting.localization.Localization(__file__, 385, 64), float32_4330, *[float_4331], **kwargs_4332)
        
        int_4334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 80), 'int')
        # Processing the call keyword arguments (line 385)
        kwargs_4335 = {}
        # Getting the type of 'np' (line 385)
        np_4327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 54), 'np', False)
        # Obtaining the member 'repeat' of a type (line 385)
        repeat_4328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 54), np_4327, 'repeat')
        # Calling repeat(args, kwargs) (line 385)
        repeat_call_result_4336 = invoke(stypy.reporting.localization.Localization(__file__, 385, 54), repeat_4328, *[float32_call_result_4333, int_4334], **kwargs_4335)
        
        # Obtaining the member 'astype' of a type (line 385)
        astype_4337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 54), repeat_call_result_4336, 'astype')
        # Calling astype(args, kwargs) (line 385)
        astype_call_result_4341 = invoke(stypy.reporting.localization.Localization(__file__, 385, 54), astype_4337, *[object__4339], **kwargs_4340)
        
        # Processing the call keyword arguments (line 385)
        kwargs_4342 = {}
        # Getting the type of 'assert_array_identical' (line 385)
        assert_array_identical_4320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 385)
        assert_array_identical_call_result_4343 = invoke(stypy.reporting.localization.Localization(__file__, 385, 12), assert_array_identical_4320, *[subscript_call_result_4326, astype_call_result_4341], **kwargs_4342)
        
        
        # Call to assert_array_identical(...): (line 386)
        # Processing the call arguments (line 386)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 386)
        i_4345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 50), 'i', False)
        # Getting the type of 's' (line 386)
        s_4346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 35), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 386)
        arrays_rep_4347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 35), s_4346, 'arrays_rep')
        # Obtaining the member 'h' of a type (line 386)
        h_4348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 35), arrays_rep_4347, 'h')
        # Obtaining the member '__getitem__' of a type (line 386)
        getitem___4349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 35), h_4348, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 386)
        subscript_call_result_4350 = invoke(stypy.reporting.localization.Localization(__file__, 386, 35), getitem___4349, i_4345)
        
        
        # Call to astype(...): (line 386)
        # Processing the call arguments (line 386)
        # Getting the type of 'np' (line 386)
        np_4362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 90), 'np', False)
        # Obtaining the member 'object_' of a type (line 386)
        object__4363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 90), np_4362, 'object_')
        # Processing the call keyword arguments (line 386)
        kwargs_4364 = {}
        
        # Call to repeat(...): (line 386)
        # Processing the call arguments (line 386)
        
        # Call to float32(...): (line 386)
        # Processing the call arguments (line 386)
        float_4355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 75), 'float')
        # Processing the call keyword arguments (line 386)
        kwargs_4356 = {}
        # Getting the type of 'np' (line 386)
        np_4353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 64), 'np', False)
        # Obtaining the member 'float32' of a type (line 386)
        float32_4354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 64), np_4353, 'float32')
        # Calling float32(args, kwargs) (line 386)
        float32_call_result_4357 = invoke(stypy.reporting.localization.Localization(__file__, 386, 64), float32_4354, *[float_4355], **kwargs_4356)
        
        int_4358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 80), 'int')
        # Processing the call keyword arguments (line 386)
        kwargs_4359 = {}
        # Getting the type of 'np' (line 386)
        np_4351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 54), 'np', False)
        # Obtaining the member 'repeat' of a type (line 386)
        repeat_4352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 54), np_4351, 'repeat')
        # Calling repeat(args, kwargs) (line 386)
        repeat_call_result_4360 = invoke(stypy.reporting.localization.Localization(__file__, 386, 54), repeat_4352, *[float32_call_result_4357, int_4358], **kwargs_4359)
        
        # Obtaining the member 'astype' of a type (line 386)
        astype_4361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 54), repeat_call_result_4360, 'astype')
        # Calling astype(args, kwargs) (line 386)
        astype_call_result_4365 = invoke(stypy.reporting.localization.Localization(__file__, 386, 54), astype_4361, *[object__4363], **kwargs_4364)
        
        # Processing the call keyword arguments (line 386)
        kwargs_4366 = {}
        # Getting the type of 'assert_array_identical' (line 386)
        assert_array_identical_4344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 386)
        assert_array_identical_call_result_4367 = invoke(stypy.reporting.localization.Localization(__file__, 386, 12), assert_array_identical_4344, *[subscript_call_result_4350, astype_call_result_4365], **kwargs_4366)
        
        
        # Call to assert_(...): (line 387)
        # Processing the call arguments (line 387)
        
        # Call to all(...): (line 387)
        # Processing the call arguments (line 387)
        
        
        # Call to vect_id(...): (line 387)
        # Processing the call arguments (line 387)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 387)
        i_4372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 50), 'i', False)
        # Getting the type of 's' (line 387)
        s_4373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 35), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 387)
        arrays_rep_4374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 35), s_4373, 'arrays_rep')
        # Obtaining the member 'g' of a type (line 387)
        g_4375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 35), arrays_rep_4374, 'g')
        # Obtaining the member '__getitem__' of a type (line 387)
        getitem___4376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 35), g_4375, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 387)
        subscript_call_result_4377 = invoke(stypy.reporting.localization.Localization(__file__, 387, 35), getitem___4376, i_4372)
        
        # Processing the call keyword arguments (line 387)
        kwargs_4378 = {}
        # Getting the type of 'vect_id' (line 387)
        vect_id_4371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 27), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 387)
        vect_id_call_result_4379 = invoke(stypy.reporting.localization.Localization(__file__, 387, 27), vect_id_4371, *[subscript_call_result_4377], **kwargs_4378)
        
        
        # Call to id(...): (line 387)
        # Processing the call arguments (line 387)
        
        # Obtaining the type of the subscript
        int_4381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 78), 'int')
        
        # Obtaining the type of the subscript
        int_4382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 75), 'int')
        # Getting the type of 's' (line 387)
        s_4383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 60), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 387)
        arrays_rep_4384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 60), s_4383, 'arrays_rep')
        # Obtaining the member 'g' of a type (line 387)
        g_4385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 60), arrays_rep_4384, 'g')
        # Obtaining the member '__getitem__' of a type (line 387)
        getitem___4386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 60), g_4385, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 387)
        subscript_call_result_4387 = invoke(stypy.reporting.localization.Localization(__file__, 387, 60), getitem___4386, int_4382)
        
        # Obtaining the member '__getitem__' of a type (line 387)
        getitem___4388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 60), subscript_call_result_4387, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 387)
        subscript_call_result_4389 = invoke(stypy.reporting.localization.Localization(__file__, 387, 60), getitem___4388, int_4381)
        
        # Processing the call keyword arguments (line 387)
        kwargs_4390 = {}
        # Getting the type of 'id' (line 387)
        id_4380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 57), 'id', False)
        # Calling id(args, kwargs) (line 387)
        id_call_result_4391 = invoke(stypy.reporting.localization.Localization(__file__, 387, 57), id_4380, *[subscript_call_result_4389], **kwargs_4390)
        
        # Applying the binary operator '==' (line 387)
        result_eq_4392 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 27), '==', vect_id_call_result_4379, id_call_result_4391)
        
        # Processing the call keyword arguments (line 387)
        kwargs_4393 = {}
        # Getting the type of 'np' (line 387)
        np_4369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 20), 'np', False)
        # Obtaining the member 'all' of a type (line 387)
        all_4370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 20), np_4369, 'all')
        # Calling all(args, kwargs) (line 387)
        all_call_result_4394 = invoke(stypy.reporting.localization.Localization(__file__, 387, 20), all_4370, *[result_eq_4392], **kwargs_4393)
        
        # Processing the call keyword arguments (line 387)
        kwargs_4395 = {}
        # Getting the type of 'assert_' (line 387)
        assert__4368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 387)
        assert__call_result_4396 = invoke(stypy.reporting.localization.Localization(__file__, 387, 12), assert__4368, *[all_call_result_4394], **kwargs_4395)
        
        
        # Call to assert_(...): (line 388)
        # Processing the call arguments (line 388)
        
        # Call to all(...): (line 388)
        # Processing the call arguments (line 388)
        
        
        # Call to vect_id(...): (line 388)
        # Processing the call arguments (line 388)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 388)
        i_4401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 50), 'i', False)
        # Getting the type of 's' (line 388)
        s_4402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 35), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 388)
        arrays_rep_4403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 35), s_4402, 'arrays_rep')
        # Obtaining the member 'h' of a type (line 388)
        h_4404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 35), arrays_rep_4403, 'h')
        # Obtaining the member '__getitem__' of a type (line 388)
        getitem___4405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 35), h_4404, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 388)
        subscript_call_result_4406 = invoke(stypy.reporting.localization.Localization(__file__, 388, 35), getitem___4405, i_4401)
        
        # Processing the call keyword arguments (line 388)
        kwargs_4407 = {}
        # Getting the type of 'vect_id' (line 388)
        vect_id_4400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 27), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 388)
        vect_id_call_result_4408 = invoke(stypy.reporting.localization.Localization(__file__, 388, 27), vect_id_4400, *[subscript_call_result_4406], **kwargs_4407)
        
        
        # Call to id(...): (line 388)
        # Processing the call arguments (line 388)
        
        # Obtaining the type of the subscript
        int_4410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 78), 'int')
        
        # Obtaining the type of the subscript
        int_4411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 75), 'int')
        # Getting the type of 's' (line 388)
        s_4412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 60), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 388)
        arrays_rep_4413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 60), s_4412, 'arrays_rep')
        # Obtaining the member 'h' of a type (line 388)
        h_4414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 60), arrays_rep_4413, 'h')
        # Obtaining the member '__getitem__' of a type (line 388)
        getitem___4415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 60), h_4414, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 388)
        subscript_call_result_4416 = invoke(stypy.reporting.localization.Localization(__file__, 388, 60), getitem___4415, int_4411)
        
        # Obtaining the member '__getitem__' of a type (line 388)
        getitem___4417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 60), subscript_call_result_4416, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 388)
        subscript_call_result_4418 = invoke(stypy.reporting.localization.Localization(__file__, 388, 60), getitem___4417, int_4410)
        
        # Processing the call keyword arguments (line 388)
        kwargs_4419 = {}
        # Getting the type of 'id' (line 388)
        id_4409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 57), 'id', False)
        # Calling id(args, kwargs) (line 388)
        id_call_result_4420 = invoke(stypy.reporting.localization.Localization(__file__, 388, 57), id_4409, *[subscript_call_result_4418], **kwargs_4419)
        
        # Applying the binary operator '==' (line 388)
        result_eq_4421 = python_operator(stypy.reporting.localization.Localization(__file__, 388, 27), '==', vect_id_call_result_4408, id_call_result_4420)
        
        # Processing the call keyword arguments (line 388)
        kwargs_4422 = {}
        # Getting the type of 'np' (line 388)
        np_4398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 20), 'np', False)
        # Obtaining the member 'all' of a type (line 388)
        all_4399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 20), np_4398, 'all')
        # Calling all(args, kwargs) (line 388)
        all_call_result_4423 = invoke(stypy.reporting.localization.Localization(__file__, 388, 20), all_4399, *[result_eq_4421], **kwargs_4422)
        
        # Processing the call keyword arguments (line 388)
        kwargs_4424 = {}
        # Getting the type of 'assert_' (line 388)
        assert__4397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 388)
        assert__call_result_4425 = invoke(stypy.reporting.localization.Localization(__file__, 388, 12), assert__4397, *[all_call_result_4423], **kwargs_4424)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_arrays_replicated(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_arrays_replicated' in the type store
        # Getting the type of 'stypy_return_type' (line 372)
        stypy_return_type_4426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4426)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_arrays_replicated'
        return stypy_return_type_4426


    @norecursion
    def test_arrays_replicated_3d(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_arrays_replicated_3d'
        module_type_store = module_type_store.open_function_context('test_arrays_replicated_3d', 390, 4, False)
        # Assigning a type to the variable 'self' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestPointerStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_localization', localization)
        TestPointerStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestPointerStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestPointerStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_function_name', 'TestPointerStructures.test_arrays_replicated_3d')
        TestPointerStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_param_names_list', [])
        TestPointerStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestPointerStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestPointerStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestPointerStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestPointerStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestPointerStructures.test_arrays_replicated_3d.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointerStructures.test_arrays_replicated_3d', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_arrays_replicated_3d', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_arrays_replicated_3d(...)' code ##################

        
        # Assigning a Call to a Name (line 391):
        
        # Call to join(...): (line 391)
        # Processing the call arguments (line 391)
        # Getting the type of 'DATA_PATH' (line 391)
        DATA_PATH_4429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 24), 'DATA_PATH', False)
        str_4430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 35), 'str', 'struct_pointer_arrays_replicated_3d.sav')
        # Processing the call keyword arguments (line 391)
        kwargs_4431 = {}
        # Getting the type of 'path' (line 391)
        path_4427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 14), 'path', False)
        # Obtaining the member 'join' of a type (line 391)
        join_4428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 14), path_4427, 'join')
        # Calling join(args, kwargs) (line 391)
        join_call_result_4432 = invoke(stypy.reporting.localization.Localization(__file__, 391, 14), join_4428, *[DATA_PATH_4429, str_4430], **kwargs_4431)
        
        # Assigning a type to the variable 'pth' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 8), 'pth', join_call_result_4432)
        
        # Assigning a Call to a Name (line 392):
        
        # Call to readsav(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'pth' (line 392)
        pth_4434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 20), 'pth', False)
        # Processing the call keyword arguments (line 392)
        # Getting the type of 'False' (line 392)
        False_4435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 33), 'False', False)
        keyword_4436 = False_4435
        kwargs_4437 = {'verbose': keyword_4436}
        # Getting the type of 'readsav' (line 392)
        readsav_4433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 392)
        readsav_call_result_4438 = invoke(stypy.reporting.localization.Localization(__file__, 392, 12), readsav_4433, *[pth_4434], **kwargs_4437)
        
        # Assigning a type to the variable 's' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 's', readsav_call_result_4438)
        
        # Call to assert_(...): (line 395)
        # Processing the call arguments (line 395)
        
        # Getting the type of 's' (line 395)
        s_4440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 16), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 395)
        arrays_rep_4441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 16), s_4440, 'arrays_rep')
        # Obtaining the member 'g' of a type (line 395)
        g_4442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 16), arrays_rep_4441, 'g')
        # Obtaining the member 'dtype' of a type (line 395)
        dtype_4443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 16), g_4442, 'dtype')
        # Obtaining the member 'type' of a type (line 395)
        type_4444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 16), dtype_4443, 'type')
        # Getting the type of 'np' (line 395)
        np_4445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 45), 'np', False)
        # Obtaining the member 'object_' of a type (line 395)
        object__4446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 45), np_4445, 'object_')
        # Applying the binary operator 'is' (line 395)
        result_is__4447 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 16), 'is', type_4444, object__4446)
        
        # Processing the call keyword arguments (line 395)
        kwargs_4448 = {}
        # Getting the type of 'assert_' (line 395)
        assert__4439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 395)
        assert__call_result_4449 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), assert__4439, *[result_is__4447], **kwargs_4448)
        
        
        # Call to assert_(...): (line 396)
        # Processing the call arguments (line 396)
        
        # Getting the type of 's' (line 396)
        s_4451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 16), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 396)
        arrays_rep_4452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 16), s_4451, 'arrays_rep')
        # Obtaining the member 'h' of a type (line 396)
        h_4453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 16), arrays_rep_4452, 'h')
        # Obtaining the member 'dtype' of a type (line 396)
        dtype_4454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 16), h_4453, 'dtype')
        # Obtaining the member 'type' of a type (line 396)
        type_4455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 16), dtype_4454, 'type')
        # Getting the type of 'np' (line 396)
        np_4456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 45), 'np', False)
        # Obtaining the member 'object_' of a type (line 396)
        object__4457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 45), np_4456, 'object_')
        # Applying the binary operator 'is' (line 396)
        result_is__4458 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 16), 'is', type_4455, object__4457)
        
        # Processing the call keyword arguments (line 396)
        kwargs_4459 = {}
        # Getting the type of 'assert_' (line 396)
        assert__4450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 396)
        assert__call_result_4460 = invoke(stypy.reporting.localization.Localization(__file__, 396, 8), assert__4450, *[result_is__4458], **kwargs_4459)
        
        
        # Call to assert_equal(...): (line 399)
        # Processing the call arguments (line 399)
        # Getting the type of 's' (line 399)
        s_4462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 21), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 399)
        arrays_rep_4463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 21), s_4462, 'arrays_rep')
        # Obtaining the member 'g' of a type (line 399)
        g_4464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 21), arrays_rep_4463, 'g')
        # Obtaining the member 'shape' of a type (line 399)
        shape_4465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 21), g_4464, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 399)
        tuple_4466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 399)
        # Adding element type (line 399)
        int_4467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 44), tuple_4466, int_4467)
        # Adding element type (line 399)
        int_4468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 44), tuple_4466, int_4468)
        # Adding element type (line 399)
        int_4469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 44), tuple_4466, int_4469)
        
        # Processing the call keyword arguments (line 399)
        kwargs_4470 = {}
        # Getting the type of 'assert_equal' (line 399)
        assert_equal_4461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 399)
        assert_equal_call_result_4471 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), assert_equal_4461, *[shape_4465, tuple_4466], **kwargs_4470)
        
        
        # Call to assert_equal(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 's' (line 400)
        s_4473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 21), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 400)
        arrays_rep_4474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 21), s_4473, 'arrays_rep')
        # Obtaining the member 'h' of a type (line 400)
        h_4475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 21), arrays_rep_4474, 'h')
        # Obtaining the member 'shape' of a type (line 400)
        shape_4476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 21), h_4475, 'shape')
        
        # Obtaining an instance of the builtin type 'tuple' (line 400)
        tuple_4477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 400)
        # Adding element type (line 400)
        int_4478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 44), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 44), tuple_4477, int_4478)
        # Adding element type (line 400)
        int_4479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 47), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 44), tuple_4477, int_4479)
        # Adding element type (line 400)
        int_4480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 44), tuple_4477, int_4480)
        
        # Processing the call keyword arguments (line 400)
        kwargs_4481 = {}
        # Getting the type of 'assert_equal' (line 400)
        assert_equal_4472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 400)
        assert_equal_call_result_4482 = invoke(stypy.reporting.localization.Localization(__file__, 400, 8), assert_equal_4472, *[shape_4476, tuple_4477], **kwargs_4481)
        
        
        
        # Call to range(...): (line 403)
        # Processing the call arguments (line 403)
        int_4484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 23), 'int')
        # Processing the call keyword arguments (line 403)
        kwargs_4485 = {}
        # Getting the type of 'range' (line 403)
        range_4483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 17), 'range', False)
        # Calling range(args, kwargs) (line 403)
        range_call_result_4486 = invoke(stypy.reporting.localization.Localization(__file__, 403, 17), range_4483, *[int_4484], **kwargs_4485)
        
        # Testing the type of a for loop iterable (line 403)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 403, 8), range_call_result_4486)
        # Getting the type of the for loop variable (line 403)
        for_loop_var_4487 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 403, 8), range_call_result_4486)
        # Assigning a type to the variable 'i' (line 403)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'i', for_loop_var_4487)
        # SSA begins for a for statement (line 403)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 404)
        # Processing the call arguments (line 404)
        int_4489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 27), 'int')
        # Processing the call keyword arguments (line 404)
        kwargs_4490 = {}
        # Getting the type of 'range' (line 404)
        range_4488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 21), 'range', False)
        # Calling range(args, kwargs) (line 404)
        range_call_result_4491 = invoke(stypy.reporting.localization.Localization(__file__, 404, 21), range_4488, *[int_4489], **kwargs_4490)
        
        # Testing the type of a for loop iterable (line 404)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 404, 12), range_call_result_4491)
        # Getting the type of the for loop variable (line 404)
        for_loop_var_4492 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 404, 12), range_call_result_4491)
        # Assigning a type to the variable 'j' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'j', for_loop_var_4492)
        # SSA begins for a for statement (line 404)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 405)
        # Processing the call arguments (line 405)
        int_4494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 31), 'int')
        # Processing the call keyword arguments (line 405)
        kwargs_4495 = {}
        # Getting the type of 'range' (line 405)
        range_4493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 25), 'range', False)
        # Calling range(args, kwargs) (line 405)
        range_call_result_4496 = invoke(stypy.reporting.localization.Localization(__file__, 405, 25), range_4493, *[int_4494], **kwargs_4495)
        
        # Testing the type of a for loop iterable (line 405)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 405, 16), range_call_result_4496)
        # Getting the type of the for loop variable (line 405)
        for_loop_var_4497 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 405, 16), range_call_result_4496)
        # Assigning a type to the variable 'k' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 16), 'k', for_loop_var_4497)
        # SSA begins for a for statement (line 405)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_array_identical(...): (line 406)
        # Processing the call arguments (line 406)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 406)
        tuple_4499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 406)
        # Adding element type (line 406)
        # Getting the type of 'i' (line 406)
        i_4500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 58), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 58), tuple_4499, i_4500)
        # Adding element type (line 406)
        # Getting the type of 'j' (line 406)
        j_4501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 61), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 58), tuple_4499, j_4501)
        # Adding element type (line 406)
        # Getting the type of 'k' (line 406)
        k_4502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 64), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 58), tuple_4499, k_4502)
        
        # Getting the type of 's' (line 406)
        s_4503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 43), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 406)
        arrays_rep_4504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 43), s_4503, 'arrays_rep')
        # Obtaining the member 'g' of a type (line 406)
        g_4505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 43), arrays_rep_4504, 'g')
        # Obtaining the member '__getitem__' of a type (line 406)
        getitem___4506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 43), g_4505, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 406)
        subscript_call_result_4507 = invoke(stypy.reporting.localization.Localization(__file__, 406, 43), getitem___4506, tuple_4499)
        
        
        # Call to astype(...): (line 407)
        # Processing the call arguments (line 407)
        # Getting the type of 'np' (line 407)
        np_4519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 64), 'np', False)
        # Obtaining the member 'object_' of a type (line 407)
        object__4520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 64), np_4519, 'object_')
        # Processing the call keyword arguments (line 407)
        kwargs_4521 = {}
        
        # Call to repeat(...): (line 407)
        # Processing the call arguments (line 407)
        
        # Call to float32(...): (line 407)
        # Processing the call arguments (line 407)
        float_4512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 49), 'float')
        # Processing the call keyword arguments (line 407)
        kwargs_4513 = {}
        # Getting the type of 'np' (line 407)
        np_4510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 38), 'np', False)
        # Obtaining the member 'float32' of a type (line 407)
        float32_4511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 38), np_4510, 'float32')
        # Calling float32(args, kwargs) (line 407)
        float32_call_result_4514 = invoke(stypy.reporting.localization.Localization(__file__, 407, 38), float32_4511, *[float_4512], **kwargs_4513)
        
        int_4515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 54), 'int')
        # Processing the call keyword arguments (line 407)
        kwargs_4516 = {}
        # Getting the type of 'np' (line 407)
        np_4508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 28), 'np', False)
        # Obtaining the member 'repeat' of a type (line 407)
        repeat_4509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 28), np_4508, 'repeat')
        # Calling repeat(args, kwargs) (line 407)
        repeat_call_result_4517 = invoke(stypy.reporting.localization.Localization(__file__, 407, 28), repeat_4509, *[float32_call_result_4514, int_4515], **kwargs_4516)
        
        # Obtaining the member 'astype' of a type (line 407)
        astype_4518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 407, 28), repeat_call_result_4517, 'astype')
        # Calling astype(args, kwargs) (line 407)
        astype_call_result_4522 = invoke(stypy.reporting.localization.Localization(__file__, 407, 28), astype_4518, *[object__4520], **kwargs_4521)
        
        # Processing the call keyword arguments (line 406)
        kwargs_4523 = {}
        # Getting the type of 'assert_array_identical' (line 406)
        assert_array_identical_4498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 20), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 406)
        assert_array_identical_call_result_4524 = invoke(stypy.reporting.localization.Localization(__file__, 406, 20), assert_array_identical_4498, *[subscript_call_result_4507, astype_call_result_4522], **kwargs_4523)
        
        
        # Call to assert_array_identical(...): (line 408)
        # Processing the call arguments (line 408)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 408)
        tuple_4526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 408)
        # Adding element type (line 408)
        # Getting the type of 'i' (line 408)
        i_4527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 58), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 58), tuple_4526, i_4527)
        # Adding element type (line 408)
        # Getting the type of 'j' (line 408)
        j_4528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 61), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 58), tuple_4526, j_4528)
        # Adding element type (line 408)
        # Getting the type of 'k' (line 408)
        k_4529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 64), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 408, 58), tuple_4526, k_4529)
        
        # Getting the type of 's' (line 408)
        s_4530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 43), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 408)
        arrays_rep_4531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 43), s_4530, 'arrays_rep')
        # Obtaining the member 'h' of a type (line 408)
        h_4532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 43), arrays_rep_4531, 'h')
        # Obtaining the member '__getitem__' of a type (line 408)
        getitem___4533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 43), h_4532, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 408)
        subscript_call_result_4534 = invoke(stypy.reporting.localization.Localization(__file__, 408, 43), getitem___4533, tuple_4526)
        
        
        # Call to astype(...): (line 409)
        # Processing the call arguments (line 409)
        # Getting the type of 'np' (line 409)
        np_4546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 64), 'np', False)
        # Obtaining the member 'object_' of a type (line 409)
        object__4547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 64), np_4546, 'object_')
        # Processing the call keyword arguments (line 409)
        kwargs_4548 = {}
        
        # Call to repeat(...): (line 409)
        # Processing the call arguments (line 409)
        
        # Call to float32(...): (line 409)
        # Processing the call arguments (line 409)
        float_4539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 49), 'float')
        # Processing the call keyword arguments (line 409)
        kwargs_4540 = {}
        # Getting the type of 'np' (line 409)
        np_4537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 38), 'np', False)
        # Obtaining the member 'float32' of a type (line 409)
        float32_4538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 38), np_4537, 'float32')
        # Calling float32(args, kwargs) (line 409)
        float32_call_result_4541 = invoke(stypy.reporting.localization.Localization(__file__, 409, 38), float32_4538, *[float_4539], **kwargs_4540)
        
        int_4542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 54), 'int')
        # Processing the call keyword arguments (line 409)
        kwargs_4543 = {}
        # Getting the type of 'np' (line 409)
        np_4535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 28), 'np', False)
        # Obtaining the member 'repeat' of a type (line 409)
        repeat_4536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 28), np_4535, 'repeat')
        # Calling repeat(args, kwargs) (line 409)
        repeat_call_result_4544 = invoke(stypy.reporting.localization.Localization(__file__, 409, 28), repeat_4536, *[float32_call_result_4541, int_4542], **kwargs_4543)
        
        # Obtaining the member 'astype' of a type (line 409)
        astype_4545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 28), repeat_call_result_4544, 'astype')
        # Calling astype(args, kwargs) (line 409)
        astype_call_result_4549 = invoke(stypy.reporting.localization.Localization(__file__, 409, 28), astype_4545, *[object__4547], **kwargs_4548)
        
        # Processing the call keyword arguments (line 408)
        kwargs_4550 = {}
        # Getting the type of 'assert_array_identical' (line 408)
        assert_array_identical_4525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 20), 'assert_array_identical', False)
        # Calling assert_array_identical(args, kwargs) (line 408)
        assert_array_identical_call_result_4551 = invoke(stypy.reporting.localization.Localization(__file__, 408, 20), assert_array_identical_4525, *[subscript_call_result_4534, astype_call_result_4549], **kwargs_4550)
        
        
        # Call to assert_(...): (line 410)
        # Processing the call arguments (line 410)
        
        # Call to all(...): (line 410)
        # Processing the call arguments (line 410)
        
        
        # Call to vect_id(...): (line 410)
        # Processing the call arguments (line 410)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 410)
        tuple_4556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 410)
        # Adding element type (line 410)
        # Getting the type of 'i' (line 410)
        i_4557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 58), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 58), tuple_4556, i_4557)
        # Adding element type (line 410)
        # Getting the type of 'j' (line 410)
        j_4558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 61), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 58), tuple_4556, j_4558)
        # Adding element type (line 410)
        # Getting the type of 'k' (line 410)
        k_4559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 64), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 58), tuple_4556, k_4559)
        
        # Getting the type of 's' (line 410)
        s_4560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 43), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 410)
        arrays_rep_4561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 43), s_4560, 'arrays_rep')
        # Obtaining the member 'g' of a type (line 410)
        g_4562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 43), arrays_rep_4561, 'g')
        # Obtaining the member '__getitem__' of a type (line 410)
        getitem___4563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 43), g_4562, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 410)
        subscript_call_result_4564 = invoke(stypy.reporting.localization.Localization(__file__, 410, 43), getitem___4563, tuple_4556)
        
        # Processing the call keyword arguments (line 410)
        kwargs_4565 = {}
        # Getting the type of 'vect_id' (line 410)
        vect_id_4555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 35), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 410)
        vect_id_call_result_4566 = invoke(stypy.reporting.localization.Localization(__file__, 410, 35), vect_id_4555, *[subscript_call_result_4564], **kwargs_4565)
        
        
        # Call to id(...): (line 410)
        # Processing the call arguments (line 410)
        
        # Obtaining the type of the subscript
        int_4568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 98), 'int')
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 410)
        tuple_4569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 89), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 410)
        # Adding element type (line 410)
        int_4570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 89), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 89), tuple_4569, int_4570)
        # Adding element type (line 410)
        int_4571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 92), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 89), tuple_4569, int_4571)
        # Adding element type (line 410)
        int_4572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 95), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 410, 89), tuple_4569, int_4572)
        
        # Getting the type of 's' (line 410)
        s_4573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 74), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 410)
        arrays_rep_4574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 74), s_4573, 'arrays_rep')
        # Obtaining the member 'g' of a type (line 410)
        g_4575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 74), arrays_rep_4574, 'g')
        # Obtaining the member '__getitem__' of a type (line 410)
        getitem___4576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 74), g_4575, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 410)
        subscript_call_result_4577 = invoke(stypy.reporting.localization.Localization(__file__, 410, 74), getitem___4576, tuple_4569)
        
        # Obtaining the member '__getitem__' of a type (line 410)
        getitem___4578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 74), subscript_call_result_4577, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 410)
        subscript_call_result_4579 = invoke(stypy.reporting.localization.Localization(__file__, 410, 74), getitem___4578, int_4568)
        
        # Processing the call keyword arguments (line 410)
        kwargs_4580 = {}
        # Getting the type of 'id' (line 410)
        id_4567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 71), 'id', False)
        # Calling id(args, kwargs) (line 410)
        id_call_result_4581 = invoke(stypy.reporting.localization.Localization(__file__, 410, 71), id_4567, *[subscript_call_result_4579], **kwargs_4580)
        
        # Applying the binary operator '==' (line 410)
        result_eq_4582 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 35), '==', vect_id_call_result_4566, id_call_result_4581)
        
        # Processing the call keyword arguments (line 410)
        kwargs_4583 = {}
        # Getting the type of 'np' (line 410)
        np_4553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 28), 'np', False)
        # Obtaining the member 'all' of a type (line 410)
        all_4554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 28), np_4553, 'all')
        # Calling all(args, kwargs) (line 410)
        all_call_result_4584 = invoke(stypy.reporting.localization.Localization(__file__, 410, 28), all_4554, *[result_eq_4582], **kwargs_4583)
        
        # Processing the call keyword arguments (line 410)
        kwargs_4585 = {}
        # Getting the type of 'assert_' (line 410)
        assert__4552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 20), 'assert_', False)
        # Calling assert_(args, kwargs) (line 410)
        assert__call_result_4586 = invoke(stypy.reporting.localization.Localization(__file__, 410, 20), assert__4552, *[all_call_result_4584], **kwargs_4585)
        
        
        # Call to assert_(...): (line 411)
        # Processing the call arguments (line 411)
        
        # Call to all(...): (line 411)
        # Processing the call arguments (line 411)
        
        
        # Call to vect_id(...): (line 411)
        # Processing the call arguments (line 411)
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 411)
        tuple_4591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 58), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 411)
        # Adding element type (line 411)
        # Getting the type of 'i' (line 411)
        i_4592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 58), 'i', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 58), tuple_4591, i_4592)
        # Adding element type (line 411)
        # Getting the type of 'j' (line 411)
        j_4593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 61), 'j', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 58), tuple_4591, j_4593)
        # Adding element type (line 411)
        # Getting the type of 'k' (line 411)
        k_4594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 64), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 58), tuple_4591, k_4594)
        
        # Getting the type of 's' (line 411)
        s_4595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 43), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 411)
        arrays_rep_4596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 43), s_4595, 'arrays_rep')
        # Obtaining the member 'h' of a type (line 411)
        h_4597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 43), arrays_rep_4596, 'h')
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___4598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 43), h_4597, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_4599 = invoke(stypy.reporting.localization.Localization(__file__, 411, 43), getitem___4598, tuple_4591)
        
        # Processing the call keyword arguments (line 411)
        kwargs_4600 = {}
        # Getting the type of 'vect_id' (line 411)
        vect_id_4590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 35), 'vect_id', False)
        # Calling vect_id(args, kwargs) (line 411)
        vect_id_call_result_4601 = invoke(stypy.reporting.localization.Localization(__file__, 411, 35), vect_id_4590, *[subscript_call_result_4599], **kwargs_4600)
        
        
        # Call to id(...): (line 411)
        # Processing the call arguments (line 411)
        
        # Obtaining the type of the subscript
        int_4603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 98), 'int')
        
        # Obtaining the type of the subscript
        
        # Obtaining an instance of the builtin type 'tuple' (line 411)
        tuple_4604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 89), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 411)
        # Adding element type (line 411)
        int_4605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 89), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 89), tuple_4604, int_4605)
        # Adding element type (line 411)
        int_4606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 92), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 89), tuple_4604, int_4606)
        # Adding element type (line 411)
        int_4607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 95), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 411, 89), tuple_4604, int_4607)
        
        # Getting the type of 's' (line 411)
        s_4608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 74), 's', False)
        # Obtaining the member 'arrays_rep' of a type (line 411)
        arrays_rep_4609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 74), s_4608, 'arrays_rep')
        # Obtaining the member 'h' of a type (line 411)
        h_4610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 74), arrays_rep_4609, 'h')
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___4611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 74), h_4610, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_4612 = invoke(stypy.reporting.localization.Localization(__file__, 411, 74), getitem___4611, tuple_4604)
        
        # Obtaining the member '__getitem__' of a type (line 411)
        getitem___4613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 74), subscript_call_result_4612, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 411)
        subscript_call_result_4614 = invoke(stypy.reporting.localization.Localization(__file__, 411, 74), getitem___4613, int_4603)
        
        # Processing the call keyword arguments (line 411)
        kwargs_4615 = {}
        # Getting the type of 'id' (line 411)
        id_4602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 71), 'id', False)
        # Calling id(args, kwargs) (line 411)
        id_call_result_4616 = invoke(stypy.reporting.localization.Localization(__file__, 411, 71), id_4602, *[subscript_call_result_4614], **kwargs_4615)
        
        # Applying the binary operator '==' (line 411)
        result_eq_4617 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 35), '==', vect_id_call_result_4601, id_call_result_4616)
        
        # Processing the call keyword arguments (line 411)
        kwargs_4618 = {}
        # Getting the type of 'np' (line 411)
        np_4588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 28), 'np', False)
        # Obtaining the member 'all' of a type (line 411)
        all_4589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 28), np_4588, 'all')
        # Calling all(args, kwargs) (line 411)
        all_call_result_4619 = invoke(stypy.reporting.localization.Localization(__file__, 411, 28), all_4589, *[result_eq_4617], **kwargs_4618)
        
        # Processing the call keyword arguments (line 411)
        kwargs_4620 = {}
        # Getting the type of 'assert_' (line 411)
        assert__4587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 20), 'assert_', False)
        # Calling assert_(args, kwargs) (line 411)
        assert__call_result_4621 = invoke(stypy.reporting.localization.Localization(__file__, 411, 20), assert__4587, *[all_call_result_4619], **kwargs_4620)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_arrays_replicated_3d(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_arrays_replicated_3d' in the type store
        # Getting the type of 'stypy_return_type' (line 390)
        stypy_return_type_4622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4622)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_arrays_replicated_3d'
        return stypy_return_type_4622


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 342, 0, False)
        # Assigning a type to the variable 'self' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestPointerStructures.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestPointerStructures' (line 342)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 0), 'TestPointerStructures', TestPointerStructures)
# Declaration of the 'TestTags' class

class TestTags:
    str_4623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 4), 'str', 'Test that sav files with description tag read at all')

    @norecursion
    def test_description(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_description'
        module_type_store = module_type_store.open_function_context('test_description', 415, 4, False)
        # Assigning a type to the variable 'self' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestTags.test_description.__dict__.__setitem__('stypy_localization', localization)
        TestTags.test_description.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestTags.test_description.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestTags.test_description.__dict__.__setitem__('stypy_function_name', 'TestTags.test_description')
        TestTags.test_description.__dict__.__setitem__('stypy_param_names_list', [])
        TestTags.test_description.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestTags.test_description.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestTags.test_description.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestTags.test_description.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestTags.test_description.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestTags.test_description.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTags.test_description', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_description', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_description(...)' code ##################

        
        # Assigning a Call to a Name (line 416):
        
        # Call to readsav(...): (line 416)
        # Processing the call arguments (line 416)
        
        # Call to join(...): (line 416)
        # Processing the call arguments (line 416)
        # Getting the type of 'DATA_PATH' (line 416)
        DATA_PATH_4627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 30), 'DATA_PATH', False)
        str_4628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 41), 'str', 'scalar_byte_descr.sav')
        # Processing the call keyword arguments (line 416)
        kwargs_4629 = {}
        # Getting the type of 'path' (line 416)
        path_4625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 416)
        join_4626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 20), path_4625, 'join')
        # Calling join(args, kwargs) (line 416)
        join_call_result_4630 = invoke(stypy.reporting.localization.Localization(__file__, 416, 20), join_4626, *[DATA_PATH_4627, str_4628], **kwargs_4629)
        
        # Processing the call keyword arguments (line 416)
        # Getting the type of 'False' (line 416)
        False_4631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 75), 'False', False)
        keyword_4632 = False_4631
        kwargs_4633 = {'verbose': keyword_4632}
        # Getting the type of 'readsav' (line 416)
        readsav_4624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 416)
        readsav_call_result_4634 = invoke(stypy.reporting.localization.Localization(__file__, 416, 12), readsav_4624, *[join_call_result_4630], **kwargs_4633)
        
        # Assigning a type to the variable 's' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 's', readsav_call_result_4634)
        
        # Call to assert_identical(...): (line 417)
        # Processing the call arguments (line 417)
        # Getting the type of 's' (line 417)
        s_4636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 25), 's', False)
        # Obtaining the member 'i8u' of a type (line 417)
        i8u_4637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 25), s_4636, 'i8u')
        
        # Call to uint8(...): (line 417)
        # Processing the call arguments (line 417)
        int_4640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 41), 'int')
        # Processing the call keyword arguments (line 417)
        kwargs_4641 = {}
        # Getting the type of 'np' (line 417)
        np_4638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 32), 'np', False)
        # Obtaining the member 'uint8' of a type (line 417)
        uint8_4639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 32), np_4638, 'uint8')
        # Calling uint8(args, kwargs) (line 417)
        uint8_call_result_4642 = invoke(stypy.reporting.localization.Localization(__file__, 417, 32), uint8_4639, *[int_4640], **kwargs_4641)
        
        # Processing the call keyword arguments (line 417)
        kwargs_4643 = {}
        # Getting the type of 'assert_identical' (line 417)
        assert_identical_4635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'assert_identical', False)
        # Calling assert_identical(args, kwargs) (line 417)
        assert_identical_call_result_4644 = invoke(stypy.reporting.localization.Localization(__file__, 417, 8), assert_identical_4635, *[i8u_4637, uint8_call_result_4642], **kwargs_4643)
        
        
        # ################# End of 'test_description(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_description' in the type store
        # Getting the type of 'stypy_return_type' (line 415)
        stypy_return_type_4645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4645)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_description'
        return stypy_return_type_4645


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 412, 0, False)
        # Assigning a type to the variable 'self' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestTags.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestTags' (line 412)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 0), 'TestTags', TestTags)

@norecursion
def test_null_pointer(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_null_pointer'
    module_type_store = module_type_store.open_function_context('test_null_pointer', 420, 0, False)
    
    # Passed parameters checking function
    test_null_pointer.stypy_localization = localization
    test_null_pointer.stypy_type_of_self = None
    test_null_pointer.stypy_type_store = module_type_store
    test_null_pointer.stypy_function_name = 'test_null_pointer'
    test_null_pointer.stypy_param_names_list = []
    test_null_pointer.stypy_varargs_param_name = None
    test_null_pointer.stypy_kwargs_param_name = None
    test_null_pointer.stypy_call_defaults = defaults
    test_null_pointer.stypy_call_varargs = varargs
    test_null_pointer.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_null_pointer', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_null_pointer', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_null_pointer(...)' code ##################

    
    # Assigning a Call to a Name (line 422):
    
    # Call to readsav(...): (line 422)
    # Processing the call arguments (line 422)
    
    # Call to join(...): (line 422)
    # Processing the call arguments (line 422)
    # Getting the type of 'DATA_PATH' (line 422)
    DATA_PATH_4649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 26), 'DATA_PATH', False)
    str_4650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 37), 'str', 'null_pointer.sav')
    # Processing the call keyword arguments (line 422)
    kwargs_4651 = {}
    # Getting the type of 'path' (line 422)
    path_4647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 16), 'path', False)
    # Obtaining the member 'join' of a type (line 422)
    join_4648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 16), path_4647, 'join')
    # Calling join(args, kwargs) (line 422)
    join_call_result_4652 = invoke(stypy.reporting.localization.Localization(__file__, 422, 16), join_4648, *[DATA_PATH_4649, str_4650], **kwargs_4651)
    
    # Processing the call keyword arguments (line 422)
    # Getting the type of 'False' (line 422)
    False_4653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 66), 'False', False)
    keyword_4654 = False_4653
    kwargs_4655 = {'verbose': keyword_4654}
    # Getting the type of 'readsav' (line 422)
    readsav_4646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'readsav', False)
    # Calling readsav(args, kwargs) (line 422)
    readsav_call_result_4656 = invoke(stypy.reporting.localization.Localization(__file__, 422, 8), readsav_4646, *[join_call_result_4652], **kwargs_4655)
    
    # Assigning a type to the variable 's' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 's', readsav_call_result_4656)
    
    # Call to assert_identical(...): (line 423)
    # Processing the call arguments (line 423)
    # Getting the type of 's' (line 423)
    s_4658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 21), 's', False)
    # Obtaining the member 'point' of a type (line 423)
    point_4659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 21), s_4658, 'point')
    # Getting the type of 'None' (line 423)
    None_4660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 30), 'None', False)
    # Processing the call keyword arguments (line 423)
    kwargs_4661 = {}
    # Getting the type of 'assert_identical' (line 423)
    assert_identical_4657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'assert_identical', False)
    # Calling assert_identical(args, kwargs) (line 423)
    assert_identical_call_result_4662 = invoke(stypy.reporting.localization.Localization(__file__, 423, 4), assert_identical_4657, *[point_4659, None_4660], **kwargs_4661)
    
    
    # Call to assert_identical(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 's' (line 424)
    s_4664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 21), 's', False)
    # Obtaining the member 'check' of a type (line 424)
    check_4665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 21), s_4664, 'check')
    
    # Call to int16(...): (line 424)
    # Processing the call arguments (line 424)
    int_4668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 39), 'int')
    # Processing the call keyword arguments (line 424)
    kwargs_4669 = {}
    # Getting the type of 'np' (line 424)
    np_4666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 30), 'np', False)
    # Obtaining the member 'int16' of a type (line 424)
    int16_4667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 30), np_4666, 'int16')
    # Calling int16(args, kwargs) (line 424)
    int16_call_result_4670 = invoke(stypy.reporting.localization.Localization(__file__, 424, 30), int16_4667, *[int_4668], **kwargs_4669)
    
    # Processing the call keyword arguments (line 424)
    kwargs_4671 = {}
    # Getting the type of 'assert_identical' (line 424)
    assert_identical_4663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'assert_identical', False)
    # Calling assert_identical(args, kwargs) (line 424)
    assert_identical_call_result_4672 = invoke(stypy.reporting.localization.Localization(__file__, 424, 4), assert_identical_4663, *[check_4665, int16_call_result_4670], **kwargs_4671)
    
    
    # ################# End of 'test_null_pointer(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_null_pointer' in the type store
    # Getting the type of 'stypy_return_type' (line 420)
    stypy_return_type_4673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4673)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_null_pointer'
    return stypy_return_type_4673

# Assigning a type to the variable 'test_null_pointer' (line 420)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 0), 'test_null_pointer', test_null_pointer)

@norecursion
def test_invalid_pointer(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_invalid_pointer'
    module_type_store = module_type_store.open_function_context('test_invalid_pointer', 427, 0, False)
    
    # Passed parameters checking function
    test_invalid_pointer.stypy_localization = localization
    test_invalid_pointer.stypy_type_of_self = None
    test_invalid_pointer.stypy_type_store = module_type_store
    test_invalid_pointer.stypy_function_name = 'test_invalid_pointer'
    test_invalid_pointer.stypy_param_names_list = []
    test_invalid_pointer.stypy_varargs_param_name = None
    test_invalid_pointer.stypy_kwargs_param_name = None
    test_invalid_pointer.stypy_call_defaults = defaults
    test_invalid_pointer.stypy_call_varargs = varargs
    test_invalid_pointer.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_invalid_pointer', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_invalid_pointer', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_invalid_pointer(...)' code ##################

    
    # Call to catch_warnings(...): (line 435)
    # Processing the call keyword arguments (line 435)
    # Getting the type of 'True' (line 435)
    True_4676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 40), 'True', False)
    keyword_4677 = True_4676
    kwargs_4678 = {'record': keyword_4677}
    # Getting the type of 'warnings' (line 435)
    warnings_4674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 9), 'warnings', False)
    # Obtaining the member 'catch_warnings' of a type (line 435)
    catch_warnings_4675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 9), warnings_4674, 'catch_warnings')
    # Calling catch_warnings(args, kwargs) (line 435)
    catch_warnings_call_result_4679 = invoke(stypy.reporting.localization.Localization(__file__, 435, 9), catch_warnings_4675, *[], **kwargs_4678)
    
    with_4680 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 435, 9), catch_warnings_call_result_4679, 'with parameter', '__enter__', '__exit__')

    if with_4680:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 435)
        enter___4681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 9), catch_warnings_call_result_4679, '__enter__')
        with_enter_4682 = invoke(stypy.reporting.localization.Localization(__file__, 435, 9), enter___4681)
        # Assigning a type to the variable 'w' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 9), 'w', with_enter_4682)
        
        # Call to simplefilter(...): (line 436)
        # Processing the call arguments (line 436)
        str_4685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 30), 'str', 'always')
        # Processing the call keyword arguments (line 436)
        kwargs_4686 = {}
        # Getting the type of 'warnings' (line 436)
        warnings_4683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'warnings', False)
        # Obtaining the member 'simplefilter' of a type (line 436)
        simplefilter_4684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 8), warnings_4683, 'simplefilter')
        # Calling simplefilter(args, kwargs) (line 436)
        simplefilter_call_result_4687 = invoke(stypy.reporting.localization.Localization(__file__, 436, 8), simplefilter_4684, *[str_4685], **kwargs_4686)
        
        
        # Assigning a Call to a Name (line 437):
        
        # Call to readsav(...): (line 437)
        # Processing the call arguments (line 437)
        
        # Call to join(...): (line 437)
        # Processing the call arguments (line 437)
        # Getting the type of 'DATA_PATH' (line 437)
        DATA_PATH_4691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 30), 'DATA_PATH', False)
        str_4692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 41), 'str', 'invalid_pointer.sav')
        # Processing the call keyword arguments (line 437)
        kwargs_4693 = {}
        # Getting the type of 'path' (line 437)
        path_4689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 20), 'path', False)
        # Obtaining the member 'join' of a type (line 437)
        join_4690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 20), path_4689, 'join')
        # Calling join(args, kwargs) (line 437)
        join_call_result_4694 = invoke(stypy.reporting.localization.Localization(__file__, 437, 20), join_4690, *[DATA_PATH_4691, str_4692], **kwargs_4693)
        
        # Processing the call keyword arguments (line 437)
        # Getting the type of 'False' (line 437)
        False_4695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 73), 'False', False)
        keyword_4696 = False_4695
        kwargs_4697 = {'verbose': keyword_4696}
        # Getting the type of 'readsav' (line 437)
        readsav_4688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'readsav', False)
        # Calling readsav(args, kwargs) (line 437)
        readsav_call_result_4698 = invoke(stypy.reporting.localization.Localization(__file__, 437, 12), readsav_4688, *[join_call_result_4694], **kwargs_4697)
        
        # Assigning a type to the variable 's' (line 437)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 's', readsav_call_result_4698)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 435)
        exit___4699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 9), catch_warnings_call_result_4679, '__exit__')
        with_exit_4700 = invoke(stypy.reporting.localization.Localization(__file__, 435, 9), exit___4699, None, None, None)

    
    # Call to assert_(...): (line 438)
    # Processing the call arguments (line 438)
    
    
    # Call to len(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'w' (line 438)
    w_4703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 16), 'w', False)
    # Processing the call keyword arguments (line 438)
    kwargs_4704 = {}
    # Getting the type of 'len' (line 438)
    len_4702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'len', False)
    # Calling len(args, kwargs) (line 438)
    len_call_result_4705 = invoke(stypy.reporting.localization.Localization(__file__, 438, 12), len_4702, *[w_4703], **kwargs_4704)
    
    int_4706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 22), 'int')
    # Applying the binary operator '==' (line 438)
    result_eq_4707 = python_operator(stypy.reporting.localization.Localization(__file__, 438, 12), '==', len_call_result_4705, int_4706)
    
    # Processing the call keyword arguments (line 438)
    kwargs_4708 = {}
    # Getting the type of 'assert_' (line 438)
    assert__4701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 438)
    assert__call_result_4709 = invoke(stypy.reporting.localization.Localization(__file__, 438, 4), assert__4701, *[result_eq_4707], **kwargs_4708)
    
    
    # Call to assert_(...): (line 439)
    # Processing the call arguments (line 439)
    
    
    # Call to str(...): (line 439)
    # Processing the call arguments (line 439)
    
    # Obtaining the type of the subscript
    int_4712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 18), 'int')
    # Getting the type of 'w' (line 439)
    w_4713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 16), 'w', False)
    # Obtaining the member '__getitem__' of a type (line 439)
    getitem___4714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 16), w_4713, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 439)
    subscript_call_result_4715 = invoke(stypy.reporting.localization.Localization(__file__, 439, 16), getitem___4714, int_4712)
    
    # Obtaining the member 'message' of a type (line 439)
    message_4716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 16), subscript_call_result_4715, 'message')
    # Processing the call keyword arguments (line 439)
    kwargs_4717 = {}
    # Getting the type of 'str' (line 439)
    str_4711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'str', False)
    # Calling str(args, kwargs) (line 439)
    str_call_result_4718 = invoke(stypy.reporting.localization.Localization(__file__, 439, 12), str_4711, *[message_4716], **kwargs_4717)
    
    str_4719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 34), 'str', 'Variable referenced by pointer not found in heap: variable will be set to None')
    # Applying the binary operator '==' (line 439)
    result_eq_4720 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 12), '==', str_call_result_4718, str_4719)
    
    # Processing the call keyword arguments (line 439)
    kwargs_4721 = {}
    # Getting the type of 'assert_' (line 439)
    assert__4710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 439)
    assert__call_result_4722 = invoke(stypy.reporting.localization.Localization(__file__, 439, 4), assert__4710, *[result_eq_4720], **kwargs_4721)
    
    
    # Call to assert_identical(...): (line 441)
    # Processing the call arguments (line 441)
    
    # Obtaining the type of the subscript
    str_4724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 23), 'str', 'a')
    # Getting the type of 's' (line 441)
    s_4725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 21), 's', False)
    # Obtaining the member '__getitem__' of a type (line 441)
    getitem___4726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 21), s_4725, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 441)
    subscript_call_result_4727 = invoke(stypy.reporting.localization.Localization(__file__, 441, 21), getitem___4726, str_4724)
    
    
    # Call to array(...): (line 441)
    # Processing the call arguments (line 441)
    
    # Obtaining an instance of the builtin type 'list' (line 441)
    list_4730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 441)
    # Adding element type (line 441)
    # Getting the type of 'None' (line 441)
    None_4731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 39), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 38), list_4730, None_4731)
    # Adding element type (line 441)
    # Getting the type of 'None' (line 441)
    None_4732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 45), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 38), list_4730, None_4732)
    
    # Processing the call keyword arguments (line 441)
    kwargs_4733 = {}
    # Getting the type of 'np' (line 441)
    np_4728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 29), 'np', False)
    # Obtaining the member 'array' of a type (line 441)
    array_4729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 29), np_4728, 'array')
    # Calling array(args, kwargs) (line 441)
    array_call_result_4734 = invoke(stypy.reporting.localization.Localization(__file__, 441, 29), array_4729, *[list_4730], **kwargs_4733)
    
    # Processing the call keyword arguments (line 441)
    kwargs_4735 = {}
    # Getting the type of 'assert_identical' (line 441)
    assert_identical_4723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'assert_identical', False)
    # Calling assert_identical(args, kwargs) (line 441)
    assert_identical_call_result_4736 = invoke(stypy.reporting.localization.Localization(__file__, 441, 4), assert_identical_4723, *[subscript_call_result_4727, array_call_result_4734], **kwargs_4735)
    
    
    # ################# End of 'test_invalid_pointer(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_invalid_pointer' in the type store
    # Getting the type of 'stypy_return_type' (line 427)
    stypy_return_type_4737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4737)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_invalid_pointer'
    return stypy_return_type_4737

# Assigning a type to the variable 'test_invalid_pointer' (line 427)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 0), 'test_invalid_pointer', test_invalid_pointer)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
