
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Code generator script to make the Cython BLAS and LAPACK wrappers
3: from the files "cython_blas_signatures.txt" and
4: "cython_lapack_signatures.txt" which contain the signatures for
5: all the BLAS/LAPACK routines that should be included in the wrappers.
6: '''
7: 
8: import os
9: from operator import itemgetter
10: 
11: BASE_DIR = os.path.abspath(os.path.dirname(__file__))
12: 
13: fortran_types = {'int': 'integer',
14:                  'c': 'complex',
15:                  'd': 'double precision',
16:                  's': 'real',
17:                  'z': 'complex*16',
18:                  'char': 'character',
19:                  'bint': 'logical'}
20: 
21: c_types = {'int': 'int',
22:            'c': 'npy_complex64',
23:            'd': 'double',
24:            's': 'float',
25:            'z': 'npy_complex128',
26:            'char': 'char',
27:            'bint': 'int',
28:            'cselect1': '_cselect1',
29:            'cselect2': '_cselect2',
30:            'dselect2': '_dselect2',
31:            'dselect3': '_dselect3',
32:            'sselect2': '_sselect2',
33:            'sselect3': '_sselect3',
34:            'zselect1': '_zselect1',
35:            'zselect2': '_zselect2'}
36: 
37: 
38: def arg_names_and_types(args):
39:     return zip(*[arg.split(' *') for arg in args.split(', ')])
40: 
41: pyx_func_template = '''
42: cdef extern from "{header_name}":
43:     void _fortran_{name} "F_FUNC({name}wrp, {upname}WRP)"({ret_type} *out, {fort_args}) nogil
44: cdef {ret_type} {name}({args}) nogil:
45:     cdef {ret_type} out
46:     _fortran_{name}(&out, {argnames})
47:     return out
48: '''
49: 
50: npy_types = {'c': 'npy_complex64', 'z': 'npy_complex128',
51:              'cselect1': '_cselect1', 'cselect2': '_cselect2',
52:              'dselect2': '_dselect2', 'dselect3': '_dselect3',
53:              'sselect2': '_sselect2', 'sselect3': '_sselect3',
54:              'zselect1': '_zselect1', 'zselect2': '_zselect2'}
55: 
56: 
57: def arg_casts(arg):
58:     if arg in ['npy_complex64', 'npy_complex128', '_cselect1', '_cselect2',
59:                '_dselect2', '_dselect3', '_sselect2', '_sselect3',
60:                '_zselect1', '_zselect2']:
61:         return '<{0}*>'.format(arg)
62:     return ''
63: 
64: 
65: def pyx_decl_func(name, ret_type, args, header_name):
66:     argtypes, argnames = arg_names_and_types(args)
67:     # Fix the case where one of the arguments has the same name as the
68:     # abbreviation for the argument type.
69:     # Otherwise the variable passed as an argument is considered overwrites
70:     # the previous typedef and Cython compilation fails.
71:     if ret_type in argnames:
72:         argnames = [n if n != ret_type else ret_type + '_' for n in argnames]
73:         argnames = [n if n not in ['lambda', 'in'] else n + '_'
74:                     for n in argnames]
75:         args = ', '.join([' *'.join([n, t])
76:                           for n, t in zip(argtypes, argnames)])
77:     argtypes = [npy_types.get(t, t) for t in argtypes]
78:     fort_args = ', '.join([' *'.join([n, t])
79:                            for n, t in zip(argtypes, argnames)])
80:     argnames = [arg_casts(t) + n for n, t in zip(argnames, argtypes)]
81:     argnames = ', '.join(argnames)
82:     c_ret_type = c_types[ret_type]
83:     args = args.replace('lambda', 'lambda_')
84:     return pyx_func_template.format(name=name, upname=name.upper(), args=args,
85:                                     fort_args=fort_args, ret_type=ret_type,
86:                                     c_ret_type=c_ret_type, argnames=argnames,
87:                                     header_name=header_name)
88: 
89: pyx_sub_template = '''cdef extern from "{header_name}":
90:     void _fortran_{name} "F_FUNC({name},{upname})"({fort_args}) nogil
91: cdef void {name}({args}) nogil:
92:     _fortran_{name}({argnames})
93: '''
94: 
95: 
96: def pyx_decl_sub(name, args, header_name):
97:     argtypes, argnames = arg_names_and_types(args)
98:     argtypes = [npy_types.get(t, t) for t in argtypes]
99:     argnames = [n if n not in ['lambda', 'in'] else n + '_' for n in argnames]
100:     fort_args = ', '.join([' *'.join([n, t])
101:                            for n, t in zip(argtypes, argnames)])
102:     argnames = [arg_casts(t) + n for n, t in zip(argnames, argtypes)]
103:     argnames = ', '.join(argnames)
104:     args = args.replace('*lambda,', '*lambda_,').replace('*in,', '*in_,')
105:     return pyx_sub_template.format(name=name, upname=name.upper(),
106:                                    args=args, fort_args=fort_args,
107:                                    argnames=argnames, header_name=header_name)
108: 
109: blas_pyx_preamble = '''# cython: boundscheck = False
110: # cython: wraparound = False
111: # cython: cdivision = True
112: 
113: '''
114: BLAS Functions for Cython
115: =========================
116: 
117: Usable from Cython via::
118: 
119:     cimport scipy.linalg.cython_blas
120: 
121: These wrappers do not check for alignment of arrays.
122: Alignment should be checked before these wrappers are used.
123: 
124: Raw function pointers (Fortran-style pointer arguments):
125: 
126: - {}
127: 
128: 
129: '''
130: 
131: # Within scipy, these wrappers can be used via relative or absolute cimport.
132: # Examples:
133: # from ..linalg cimport cython_blas
134: # from scipy.linalg cimport cython_blas
135: # cimport scipy.linalg.cython_blas as cython_blas
136: # cimport ..linalg.cython_blas as cython_blas
137: 
138: # Within scipy, if BLAS functions are needed in C/C++/Fortran,
139: # these wrappers should not be used.
140: # The original libraries should be linked directly.
141: 
142: from __future__ import absolute_import
143: 
144: cdef extern from "fortran_defs.h":
145:     pass
146: 
147: from numpy cimport npy_complex64, npy_complex128
148: 
149: '''
150: 
151: 
152: def make_blas_pyx_preamble(all_sigs):
153:     names = [sig[0] for sig in all_sigs]
154:     return blas_pyx_preamble.format("\n- ".join(names))
155: 
156: lapack_pyx_preamble = ''''''
157: LAPACK functions for Cython
158: ===========================
159: 
160: Usable from Cython via::
161: 
162:     cimport scipy.linalg.cython_lapack
163: 
164: This module provides Cython-level wrappers for all primary routines included
165: in LAPACK 3.1.0 except for ``zcgesv`` since its interface is not consistent
166: from LAPACK 3.1.0 to 3.6.0. It also provides some of the
167: fixed-api auxiliary routines.
168: 
169: These wrappers do not check for alignment of arrays.
170: Alignment should be checked before these wrappers are used.
171: 
172: Raw function pointers (Fortran-style pointer arguments):
173: 
174: - {}
175: 
176: 
177: '''
178: 
179: # Within scipy, these wrappers can be used via relative or absolute cimport.
180: # Examples:
181: # from ..linalg cimport cython_lapack
182: # from scipy.linalg cimport cython_lapack
183: # cimport scipy.linalg.cython_lapack as cython_lapack
184: # cimport ..linalg.cython_lapack as cython_lapack
185: 
186: # Within scipy, if LAPACK functions are needed in C/C++/Fortran,
187: # these wrappers should not be used.
188: # The original libraries should be linked directly.
189: 
190: from __future__ import absolute_import
191: 
192: cdef extern from "fortran_defs.h":
193:     pass
194: 
195: from numpy cimport npy_complex64, npy_complex128
196: 
197: cdef extern from "_lapack_subroutines.h":
198:     # Function pointer type declarations for
199:     # gees and gges families of functions.
200:     ctypedef bint _cselect1(npy_complex64*)
201:     ctypedef bint _cselect2(npy_complex64*, npy_complex64*)
202:     ctypedef bint _dselect2(d*, d*)
203:     ctypedef bint _dselect3(d*, d*, d*)
204:     ctypedef bint _sselect2(s*, s*)
205:     ctypedef bint _sselect3(s*, s*, s*)
206:     ctypedef bint _zselect1(npy_complex128*)
207:     ctypedef bint _zselect2(npy_complex128*, npy_complex128*)
208: 
209: '''
210: 
211: 
212: def make_lapack_pyx_preamble(all_sigs):
213:     names = [sig[0] for sig in all_sigs]
214:     return lapack_pyx_preamble.format("\n- ".join(names))
215: 
216: blas_py_wrappers = '''
217: 
218: # Python-accessible wrappers for testing:
219: 
220: cdef inline bint _is_contiguous(double[:,:] a, int axis) nogil:
221:     return (a.strides[axis] == sizeof(a[0,0]) or a.shape[axis] == 1)
222: 
223: cpdef float complex _test_cdotc(float complex[:] cx, float complex[:] cy) nogil:
224:     cdef:
225:         int n = cx.shape[0]
226:         int incx = cx.strides[0] // sizeof(cx[0])
227:         int incy = cy.strides[0] // sizeof(cy[0])
228:     return cdotc(&n, &cx[0], &incx, &cy[0], &incy)
229: 
230: cpdef float complex _test_cdotu(float complex[:] cx, float complex[:] cy) nogil:
231:     cdef:
232:         int n = cx.shape[0]
233:         int incx = cx.strides[0] // sizeof(cx[0])
234:         int incy = cy.strides[0] // sizeof(cy[0])
235:     return cdotu(&n, &cx[0], &incx, &cy[0], &incy)
236: 
237: cpdef double _test_dasum(double[:] dx) nogil:
238:     cdef:
239:         int n = dx.shape[0]
240:         int incx = dx.strides[0] // sizeof(dx[0])
241:     return dasum(&n, &dx[0], &incx)
242: 
243: cpdef double _test_ddot(double[:] dx, double[:] dy) nogil:
244:     cdef:
245:         int n = dx.shape[0]
246:         int incx = dx.strides[0] // sizeof(dx[0])
247:         int incy = dy.strides[0] // sizeof(dy[0])
248:     return ddot(&n, &dx[0], &incx, &dy[0], &incy)
249: 
250: cpdef int _test_dgemm(double alpha, double[:,:] a, double[:,:] b, double beta,
251:                 double[:,:] c) nogil except -1:
252:     cdef:
253:         char *transa
254:         char *transb
255:         int m, n, k, lda, ldb, ldc
256:         double *a0=&a[0,0]
257:         double *b0=&b[0,0]
258:         double *c0=&c[0,0]
259:     # In the case that c is C contiguous, swap a and b and
260:     # swap whether or not each of them is transposed.
261:     # This can be done because a.dot(b) = b.T.dot(a.T).T.
262:     if _is_contiguous(c, 1):
263:         if _is_contiguous(a, 1):
264:             transb = 'n'
265:             ldb = (&a[1,0]) - a0 if a.shape[0] > 1 else 1
266:         elif _is_contiguous(a, 0):
267:             transb = 't'
268:             ldb = (&a[0,1]) - a0 if a.shape[1] > 1 else 1
269:         else:
270:             with gil:
271:                 raise ValueError("Input 'a' is neither C nor Fortran contiguous.")
272:         if _is_contiguous(b, 1):
273:             transa = 'n'
274:             lda = (&b[1,0]) - b0 if b.shape[0] > 1 else 1
275:         elif _is_contiguous(b, 0):
276:             transa = 't'
277:             lda = (&b[0,1]) - b0 if b.shape[1] > 1 else 1
278:         else:
279:             with gil:
280:                 raise ValueError("Input 'b' is neither C nor Fortran contiguous.")
281:         k = b.shape[0]
282:         if k != a.shape[1]:
283:             with gil:
284:                 raise ValueError("Shape mismatch in input arrays.")
285:         m = b.shape[1]
286:         n = a.shape[0]
287:         if n != c.shape[0] or m != c.shape[1]:
288:             with gil:
289:                 raise ValueError("Output array does not have the correct shape.")
290:         ldc = (&c[1,0]) - c0 if c.shape[0] > 1 else 1
291:         dgemm(transa, transb, &m, &n, &k, &alpha, b0, &lda, a0,
292:                    &ldb, &beta, c0, &ldc)
293:     elif _is_contiguous(c, 0):
294:         if _is_contiguous(a, 1):
295:             transa = 't'
296:             lda = (&a[1,0]) - a0 if a.shape[0] > 1 else 1
297:         elif _is_contiguous(a, 0):
298:             transa = 'n'
299:             lda = (&a[0,1]) - a0 if a.shape[1] > 1 else 1
300:         else:
301:             with gil:
302:                 raise ValueError("Input 'a' is neither C nor Fortran contiguous.")
303:         if _is_contiguous(b, 1):
304:             transb = 't'
305:             ldb = (&b[1,0]) - b0 if b.shape[0] > 1 else 1
306:         elif _is_contiguous(b, 0):
307:             transb = 'n'
308:             ldb = (&b[0,1]) - b0 if b.shape[1] > 1 else 1
309:         else:
310:             with gil:
311:                 raise ValueError("Input 'b' is neither C nor Fortran contiguous.")
312:         m = a.shape[0]
313:         k = a.shape[1]
314:         if k != b.shape[0]:
315:             with gil:
316:                 raise ValueError("Shape mismatch in input arrays.")
317:         n = b.shape[1]
318:         if m != c.shape[0] or n != c.shape[1]:
319:             with gil:
320:                 raise ValueError("Output array does not have the correct shape.")
321:         ldc = (&c[0,1]) - c0 if c.shape[1] > 1 else 1
322:         dgemm(transa, transb, &m, &n, &k, &alpha, a0, &lda, b0,
323:                    &ldb, &beta, c0, &ldc)
324:     else:
325:         with gil:
326:             raise ValueError("Input 'c' is neither C nor Fortran contiguous.")
327:     return 0
328: 
329: cpdef double _test_dnrm2(double[:] x) nogil:
330:     cdef:
331:         int n = x.shape[0]
332:         int incx = x.strides[0] // sizeof(x[0])
333:     return dnrm2(&n, &x[0], &incx)
334: 
335: cpdef double _test_dzasum(double complex[:] zx) nogil:
336:     cdef:
337:         int n = zx.shape[0]
338:         int incx = zx.strides[0] // sizeof(zx[0])
339:     return dzasum(&n, &zx[0], &incx)
340: 
341: cpdef double _test_dznrm2(double complex[:] x) nogil:
342:     cdef:
343:         int n = x.shape[0]
344:         int incx = x.strides[0] // sizeof(x[0])
345:     return dznrm2(&n, &x[0], &incx)
346: 
347: cpdef int _test_icamax(float complex[:] cx) nogil:
348:     cdef:
349:         int n = cx.shape[0]
350:         int incx = cx.strides[0] // sizeof(cx[0])
351:     return icamax(&n, &cx[0], &incx)
352: 
353: cpdef int _test_idamax(double[:] dx) nogil:
354:     cdef:
355:         int n = dx.shape[0]
356:         int incx = dx.strides[0] // sizeof(dx[0])
357:     return idamax(&n, &dx[0], &incx)
358: 
359: cpdef int _test_isamax(float[:] sx) nogil:
360:     cdef:
361:         int n = sx.shape[0]
362:         int incx = sx.strides[0] // sizeof(sx[0])
363:     return isamax(&n, &sx[0], &incx)
364: 
365: cpdef int _test_izamax(double complex[:] zx) nogil:
366:     cdef:
367:         int n = zx.shape[0]
368:         int incx = zx.strides[0] // sizeof(zx[0])
369:     return izamax(&n, &zx[0], &incx)
370: 
371: cpdef float _test_sasum(float[:] sx) nogil:
372:     cdef:
373:         int n = sx.shape[0]
374:         int incx = sx.shape[0] // sizeof(sx[0])
375:     return sasum(&n, &sx[0], &incx)
376: 
377: cpdef float _test_scasum(float complex[:] cx) nogil:
378:     cdef:
379:         int n = cx.shape[0]
380:         int incx = cx.strides[0] // sizeof(cx[0])
381:     return scasum(&n, &cx[0], &incx)
382: 
383: cpdef float _test_scnrm2(float complex[:] x) nogil:
384:     cdef:
385:         int n = x.shape[0]
386:         int incx = x.strides[0] // sizeof(x[0])
387:     return scnrm2(&n, &x[0], &incx)
388: 
389: cpdef float _test_sdot(float[:] sx, float[:] sy) nogil:
390:     cdef:
391:         int n = sx.shape[0]
392:         int incx = sx.strides[0] // sizeof(sx[0])
393:         int incy = sy.strides[0] // sizeof(sy[0])
394:     return sdot(&n, &sx[0], &incx, &sy[0], &incy)
395: 
396: cpdef float _test_snrm2(float[:] x) nogil:
397:     cdef:
398:         int n = x.shape[0]
399:         int incx = x.shape[0] // sizeof(x[0])
400:     return snrm2(&n, &x[0], &incx)
401: 
402: cpdef double complex _test_zdotc(double complex[:] zx, double complex[:] zy) nogil:
403:     cdef:
404:         int n = zx.shape[0]
405:         int incx = zx.strides[0] // sizeof(zx[0])
406:         int incy = zy.strides[0] // sizeof(zy[0])
407:     return zdotc(&n, &zx[0], &incx, &zy[0], &incy)
408: 
409: cpdef double complex _test_zdotu(double complex[:] zx, double complex[:] zy) nogil:
410:     cdef:
411:         int n = zx.shape[0]
412:         int incx = zx.strides[0] // sizeof(zx[0])
413:         int incy = zy.strides[0] // sizeof(zy[0])
414:     return zdotu(&n, &zx[0], &incx, &zy[0], &incy)
415: '''
416: 
417: 
418: def generate_blas_pyx(func_sigs, sub_sigs, all_sigs, header_name):
419:     funcs = "\n".join(pyx_decl_func(*(s+(header_name,))) for s in func_sigs)
420:     subs = "\n" + "\n".join(pyx_decl_sub(*(s[::2]+(header_name,)))
421:                             for s in sub_sigs)
422:     return make_blas_pyx_preamble(all_sigs) + funcs + subs + blas_py_wrappers
423: 
424: lapack_py_wrappers = '''
425: 
426: # Python accessible wrappers for testing:
427: 
428: def _test_dlamch(cmach):
429:     # This conversion is necessary to handle Python 3 strings.
430:     cmach_bytes = bytes(cmach)
431:     # Now that it is a bytes representation, a non-temporary variable
432:     # must be passed as a part of the function call.
433:     cdef char* cmach_char = cmach_bytes
434:     return dlamch(cmach_char)
435: 
436: def _test_slamch(cmach):
437:     # This conversion is necessary to handle Python 3 strings.
438:     cmach_bytes = bytes(cmach)
439:     # Now that it is a bytes representation, a non-temporary variable
440:     # must be passed as a part of the function call.
441:     cdef char* cmach_char = cmach_bytes
442:     return slamch(cmach_char)
443: '''
444: 
445: 
446: def generate_lapack_pyx(func_sigs, sub_sigs, all_sigs, header_name):
447:     funcs = "\n".join(pyx_decl_func(*(s+(header_name,))) for s in func_sigs)
448:     subs = "\n" + "\n".join(pyx_decl_sub(*(s[::2]+(header_name,)))
449:                             for s in sub_sigs)
450:     preamble = make_lapack_pyx_preamble(all_sigs)
451:     return preamble + funcs + subs + lapack_py_wrappers
452: 
453: pxd_template = '''ctypedef {ret_type} {name}_t({args}) nogil
454: cdef {name}_t *{name}_f
455: '''
456: pxd_template = '''cdef {ret_type} {name}({args}) nogil
457: '''
458: 
459: 
460: def pxd_decl(name, ret_type, args):
461:     args = args.replace('lambda', 'lambda_').replace('*in,', '*in_,')
462:     return pxd_template.format(name=name, ret_type=ret_type, args=args)
463: 
464: blas_pxd_preamble = '''# Within scipy, these wrappers can be used via relative or absolute cimport.
465: # Examples:
466: # from ..linalg cimport cython_blas
467: # from scipy.linalg cimport cython_blas
468: # cimport scipy.linalg.cython_blas as cython_blas
469: # cimport ..linalg.cython_blas as cython_blas
470: 
471: # Within scipy, if BLAS functions are needed in C/C++/Fortran,
472: # these wrappers should not be used.
473: # The original libraries should be linked directly.
474: 
475: ctypedef float s
476: ctypedef double d
477: ctypedef float complex c
478: ctypedef double complex z
479: 
480: '''
481: 
482: 
483: def generate_blas_pxd(all_sigs):
484:     body = '\n'.join(pxd_decl(*sig) for sig in all_sigs)
485:     return blas_pxd_preamble + body
486: 
487: lapack_pxd_preamble = '''# Within scipy, these wrappers can be used via relative or absolute cimport.
488: # Examples:
489: # from ..linalg cimport cython_lapack
490: # from scipy.linalg cimport cython_lapack
491: # cimport scipy.linalg.cython_lapack as cython_lapack
492: # cimport ..linalg.cython_lapack as cython_lapack
493: 
494: # Within scipy, if LAPACK functions are needed in C/C++/Fortran,
495: # these wrappers should not be used.
496: # The original libraries should be linked directly.
497: 
498: ctypedef float s
499: ctypedef double d
500: ctypedef float complex c
501: ctypedef double complex z
502: 
503: # Function pointer type declarations for
504: # gees and gges families of functions.
505: ctypedef bint cselect1(c*)
506: ctypedef bint cselect2(c*, c*)
507: ctypedef bint dselect2(d*, d*)
508: ctypedef bint dselect3(d*, d*, d*)
509: ctypedef bint sselect2(s*, s*)
510: ctypedef bint sselect3(s*, s*, s*)
511: ctypedef bint zselect1(z*)
512: ctypedef bint zselect2(z*, z*)
513: 
514: '''
515: 
516: 
517: def generate_lapack_pxd(all_sigs):
518:     return lapack_pxd_preamble + '\n'.join(pxd_decl(*sig) for sig in all_sigs)
519: 
520: fortran_template = '''      subroutine {name}wrp(ret, {argnames})
521:         external {wrapper}
522:         {ret_type} {wrapper}
523:         {ret_type} ret
524:         {argdecls}
525:         ret = {wrapper}({argnames})
526:       end
527: '''
528: 
529: dims = {'work': '(*)', 'ab': '(ldab,*)', 'a': '(lda,*)', 'dl': '(*)',
530:         'd': '(*)', 'du': '(*)', 'ap': '(*)', 'e': '(*)', 'lld': '(*)'}
531: 
532: 
533: def process_fortran_name(name, funcname):
534:     if 'inc' in name:
535:         return name
536:     xy_exclusions = ['ladiv', 'lapy2', 'lapy3']
537:     if ('x' in name or 'y' in name) and funcname[1:] not in xy_exclusions:
538:         return name + '(n)'
539:     if name in dims:
540:         return name + dims[name]
541:     return name
542: 
543: 
544: def fort_subroutine_wrapper(name, ret_type, args):
545:     if name[0] in ['c', 's'] or name in ['zladiv', 'zdotu', 'zdotc']:
546:         wrapper = 'w' + name
547:     else:
548:         wrapper = name
549:     types, names = arg_names_and_types(args)
550:     argnames = ', '.join(names)
551: 
552:     names = [process_fortran_name(n, name) for n in names]
553:     argdecls = '\n        '.join('{0} {1}'.format(fortran_types[t], n)
554:                                  for n, t in zip(names, types))
555:     return fortran_template.format(name=name, wrapper=wrapper,
556:                                    argnames=argnames, argdecls=argdecls,
557:                                    ret_type=fortran_types[ret_type])
558: 
559: 
560: def generate_fortran(func_sigs):
561:     return "\n".join(fort_subroutine_wrapper(*sig) for sig in func_sigs)
562: 
563: 
564: def make_c_args(args):
565:     types, names = arg_names_and_types(args)
566:     types = [c_types[arg] for arg in types]
567:     return ', '.join('{0} *{1}'.format(t, n) for t, n in zip(types, names))
568: 
569: c_func_template = "void F_FUNC({name}wrp, {upname}WRP)({return_type} *ret, {args});\n"
570: 
571: 
572: def c_func_decl(name, return_type, args):
573:     args = make_c_args(args)
574:     return_type = c_types[return_type]
575:     return c_func_template.format(name=name, upname=name.upper(),
576:                                   return_type=return_type, args=args)
577: 
578: c_sub_template = "void F_FUNC({name},{upname})({args});\n"
579: 
580: 
581: def c_sub_decl(name, return_type, args):
582:     args = make_c_args(args)
583:     return c_sub_template.format(name=name, upname=name.upper(), args=args)
584: 
585: c_preamble = '''#ifndef SCIPY_LINALG_{lib}_FORTRAN_WRAPPERS_H
586: #define SCIPY_LINALG_{lib}_FORTRAN_WRAPPERS_H
587: #include "fortran_defs.h"
588: #include "numpy/arrayobject.h"
589: '''
590: 
591: lapack_decls = '''
592: typedef int (*_cselect1)(npy_complex64*);
593: typedef int (*_cselect2)(npy_complex64*, npy_complex64*);
594: typedef int (*_dselect2)(double*, double*);
595: typedef int (*_dselect3)(double*, double*, double*);
596: typedef int (*_sselect2)(float*, float*);
597: typedef int (*_sselect3)(float*, float*, float*);
598: typedef int (*_zselect1)(npy_complex128*);
599: typedef int (*_zselect2)(npy_complex128*, npy_complex128*);
600: '''
601: 
602: cpp_guard = '''
603: #ifdef __cplusplus
604: extern "C" {
605: #endif
606: 
607: '''
608: 
609: c_end = '''
610: #ifdef __cplusplus
611: }
612: #endif
613: #endif
614: '''
615: 
616: 
617: def generate_c_header(func_sigs, sub_sigs, all_sigs, lib_name):
618:     funcs = "".join(c_func_decl(*sig) for sig in func_sigs)
619:     subs = "\n" + "".join(c_sub_decl(*sig) for sig in sub_sigs)
620:     if lib_name == 'LAPACK':
621:         preamble = (c_preamble.format(lib=lib_name) + lapack_decls)
622:     else:
623:         preamble = c_preamble.format(lib=lib_name)
624:     return "".join([preamble, cpp_guard, funcs, subs, c_end])
625: 
626: 
627: def split_signature(sig):
628:     name_and_type, args = sig[:-1].split('(')
629:     ret_type, name = name_and_type.split(' ')
630:     return name, ret_type, args
631: 
632: 
633: def filter_lines(ls):
634:     ls = [l.strip() for l in ls if l != '\n' and l[0] != '#']
635:     func_sigs = [split_signature(l) for l in ls if l.split(' ')[0] != 'void']
636:     sub_sigs = [split_signature(l) for l in ls if l.split(' ')[0] == 'void']
637:     all_sigs = list(sorted(func_sigs + sub_sigs, key=itemgetter(0)))
638:     return func_sigs, sub_sigs, all_sigs
639: 
640: 
641: def all_newer(src_files, dst_files):
642:     from distutils.dep_util import newer
643:     return all(os.path.exists(dst) and newer(dst, src)
644:                for dst in dst_files for src in src_files)
645: 
646: 
647: def make_all(blas_signature_file="cython_blas_signatures.txt",
648:              lapack_signature_file="cython_lapack_signatures.txt",
649:              blas_name="cython_blas",
650:              lapack_name="cython_lapack",
651:              blas_fortran_name="_blas_subroutine_wrappers.f",
652:              lapack_fortran_name="_lapack_subroutine_wrappers.f",
653:              blas_header_name="_blas_subroutines.h",
654:              lapack_header_name="_lapack_subroutines.h"):
655: 
656:     src_files = (os.path.abspath(__file__),
657:                  blas_signature_file,
658:                  lapack_signature_file)
659:     dst_files = (blas_name + '.pyx',
660:                  blas_name + '.pxd',
661:                  blas_fortran_name,
662:                  blas_header_name,
663:                  lapack_name + '.pyx',
664:                  lapack_name + '.pxd',
665:                  lapack_fortran_name,
666:                  lapack_header_name)
667: 
668:     os.chdir(BASE_DIR)
669: 
670:     if all_newer(src_files, dst_files):
671:         print("scipy/linalg/_generate_pyx.py: all files up-to-date")
672:         return
673: 
674:     comments = ["This file was generated by _generate_pyx.py.\n",
675:                 "Do not edit this file directly.\n"]
676:     ccomment = ''.join(['/* ' + line.rstrip() + ' */\n' for line in comments]) + '\n'
677:     pyxcomment = ''.join(['# ' + line for line in comments]) + '\n'
678:     fcomment = ''.join(['c     ' + line for line in comments]) + '\n'
679:     with open(blas_signature_file, 'r') as f:
680:         blas_sigs = f.readlines()
681:     blas_sigs = filter_lines(blas_sigs)
682:     blas_pyx = generate_blas_pyx(*(blas_sigs + (blas_header_name,)))
683:     with open(blas_name + '.pyx', 'w') as f:
684:         f.write(pyxcomment)
685:         f.write(blas_pyx)
686:     blas_pxd = generate_blas_pxd(blas_sigs[2])
687:     with open(blas_name + '.pxd', 'w') as f:
688:         f.write(pyxcomment)
689:         f.write(blas_pxd)
690:     blas_fortran = generate_fortran(blas_sigs[0])
691:     with open(blas_fortran_name, 'w') as f:
692:         f.write(fcomment)
693:         f.write(blas_fortran)
694:     blas_c_header = generate_c_header(*(blas_sigs + ('BLAS',)))
695:     with open(blas_header_name, 'w') as f:
696:         f.write(ccomment)
697:         f.write(blas_c_header)
698:     with open(lapack_signature_file, 'r') as f:
699:         lapack_sigs = f.readlines()
700:     lapack_sigs = filter_lines(lapack_sigs)
701:     lapack_pyx = generate_lapack_pyx(*(lapack_sigs + (lapack_header_name,)))
702:     with open(lapack_name + '.pyx', 'w') as f:
703:         f.write(pyxcomment)
704:         f.write(lapack_pyx)
705:     lapack_pxd = generate_lapack_pxd(lapack_sigs[2])
706:     with open(lapack_name + '.pxd', 'w') as f:
707:         f.write(pyxcomment)
708:         f.write(lapack_pxd)
709:     lapack_fortran = generate_fortran(lapack_sigs[0])
710:     with open(lapack_fortran_name, 'w') as f:
711:         f.write(fcomment)
712:         f.write(lapack_fortran)
713:     lapack_c_header = generate_c_header(*(lapack_sigs + ('LAPACK',)))
714:     with open(lapack_header_name, 'w') as f:
715:         f.write(ccomment)
716:         f.write(lapack_c_header)
717: 
718: if __name__ == '__main__':
719:     make_all()
720: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_28283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'str', '\nCode generator script to make the Cython BLAS and LAPACK wrappers\nfrom the files "cython_blas_signatures.txt" and\n"cython_lapack_signatures.txt" which contain the signatures for\nall the BLAS/LAPACK routines that should be included in the wrappers.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import os' statement (line 8)
import os

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from operator import itemgetter' statement (line 9)
try:
    from operator import itemgetter

except:
    itemgetter = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'operator', None, module_type_store, ['itemgetter'], [itemgetter])


# Assigning a Call to a Name (line 11):

# Assigning a Call to a Name (line 11):

# Call to abspath(...): (line 11)
# Processing the call arguments (line 11)

# Call to dirname(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of '__file__' (line 11)
file___28290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 43), '__file__', False)
# Processing the call keyword arguments (line 11)
kwargs_28291 = {}
# Getting the type of 'os' (line 11)
os_28287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 27), 'os', False)
# Obtaining the member 'path' of a type (line 11)
path_28288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 27), os_28287, 'path')
# Obtaining the member 'dirname' of a type (line 11)
dirname_28289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 27), path_28288, 'dirname')
# Calling dirname(args, kwargs) (line 11)
dirname_call_result_28292 = invoke(stypy.reporting.localization.Localization(__file__, 11, 27), dirname_28289, *[file___28290], **kwargs_28291)

# Processing the call keyword arguments (line 11)
kwargs_28293 = {}
# Getting the type of 'os' (line 11)
os_28284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'os', False)
# Obtaining the member 'path' of a type (line 11)
path_28285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 11), os_28284, 'path')
# Obtaining the member 'abspath' of a type (line 11)
abspath_28286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 11), path_28285, 'abspath')
# Calling abspath(args, kwargs) (line 11)
abspath_call_result_28294 = invoke(stypy.reporting.localization.Localization(__file__, 11, 11), abspath_28286, *[dirname_call_result_28292], **kwargs_28293)

# Assigning a type to the variable 'BASE_DIR' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'BASE_DIR', abspath_call_result_28294)

# Assigning a Dict to a Name (line 13):

# Assigning a Dict to a Name (line 13):

# Obtaining an instance of the builtin type 'dict' (line 13)
dict_28295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 13)
# Adding element type (key, value) (line 13)
str_28296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 17), 'str', 'int')
str_28297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 24), 'str', 'integer')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), dict_28295, (str_28296, str_28297))
# Adding element type (key, value) (line 13)
str_28298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 17), 'str', 'c')
str_28299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 22), 'str', 'complex')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), dict_28295, (str_28298, str_28299))
# Adding element type (key, value) (line 13)
str_28300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 17), 'str', 'd')
str_28301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 22), 'str', 'double precision')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), dict_28295, (str_28300, str_28301))
# Adding element type (key, value) (line 13)
str_28302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 17), 'str', 's')
str_28303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'str', 'real')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), dict_28295, (str_28302, str_28303))
# Adding element type (key, value) (line 13)
str_28304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 17), 'str', 'z')
str_28305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 22), 'str', 'complex*16')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), dict_28295, (str_28304, str_28305))
# Adding element type (key, value) (line 13)
str_28306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 17), 'str', 'char')
str_28307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'str', 'character')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), dict_28295, (str_28306, str_28307))
# Adding element type (key, value) (line 13)
str_28308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'str', 'bint')
str_28309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'str', 'logical')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), dict_28295, (str_28308, str_28309))

# Assigning a type to the variable 'fortran_types' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'fortran_types', dict_28295)

# Assigning a Dict to a Name (line 21):

# Assigning a Dict to a Name (line 21):

# Obtaining an instance of the builtin type 'dict' (line 21)
dict_28310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 21)
# Adding element type (key, value) (line 21)
str_28311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 11), 'str', 'int')
str_28312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 18), 'str', 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), dict_28310, (str_28311, str_28312))
# Adding element type (key, value) (line 21)
str_28313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'str', 'c')
str_28314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'str', 'npy_complex64')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), dict_28310, (str_28313, str_28314))
# Adding element type (key, value) (line 21)
str_28315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 11), 'str', 'd')
str_28316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 16), 'str', 'double')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), dict_28310, (str_28315, str_28316))
# Adding element type (key, value) (line 21)
str_28317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 11), 'str', 's')
str_28318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'str', 'float')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), dict_28310, (str_28317, str_28318))
# Adding element type (key, value) (line 21)
str_28319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 11), 'str', 'z')
str_28320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'str', 'npy_complex128')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), dict_28310, (str_28319, str_28320))
# Adding element type (key, value) (line 21)
str_28321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 11), 'str', 'char')
str_28322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'str', 'char')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), dict_28310, (str_28321, str_28322))
# Adding element type (key, value) (line 21)
str_28323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 11), 'str', 'bint')
str_28324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 19), 'str', 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), dict_28310, (str_28323, str_28324))
# Adding element type (key, value) (line 21)
str_28325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 11), 'str', 'cselect1')
str_28326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 23), 'str', '_cselect1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), dict_28310, (str_28325, str_28326))
# Adding element type (key, value) (line 21)
str_28327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 11), 'str', 'cselect2')
str_28328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 23), 'str', '_cselect2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), dict_28310, (str_28327, str_28328))
# Adding element type (key, value) (line 21)
str_28329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 11), 'str', 'dselect2')
str_28330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 23), 'str', '_dselect2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), dict_28310, (str_28329, str_28330))
# Adding element type (key, value) (line 21)
str_28331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 11), 'str', 'dselect3')
str_28332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 23), 'str', '_dselect3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), dict_28310, (str_28331, str_28332))
# Adding element type (key, value) (line 21)
str_28333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 11), 'str', 'sselect2')
str_28334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'str', '_sselect2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), dict_28310, (str_28333, str_28334))
# Adding element type (key, value) (line 21)
str_28335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 11), 'str', 'sselect3')
str_28336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 23), 'str', '_sselect3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), dict_28310, (str_28335, str_28336))
# Adding element type (key, value) (line 21)
str_28337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 11), 'str', 'zselect1')
str_28338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 23), 'str', '_zselect1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), dict_28310, (str_28337, str_28338))
# Adding element type (key, value) (line 21)
str_28339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 11), 'str', 'zselect2')
str_28340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'str', '_zselect2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), dict_28310, (str_28339, str_28340))

# Assigning a type to the variable 'c_types' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'c_types', dict_28310)

@norecursion
def arg_names_and_types(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'arg_names_and_types'
    module_type_store = module_type_store.open_function_context('arg_names_and_types', 38, 0, False)
    
    # Passed parameters checking function
    arg_names_and_types.stypy_localization = localization
    arg_names_and_types.stypy_type_of_self = None
    arg_names_and_types.stypy_type_store = module_type_store
    arg_names_and_types.stypy_function_name = 'arg_names_and_types'
    arg_names_and_types.stypy_param_names_list = ['args']
    arg_names_and_types.stypy_varargs_param_name = None
    arg_names_and_types.stypy_kwargs_param_name = None
    arg_names_and_types.stypy_call_defaults = defaults
    arg_names_and_types.stypy_call_varargs = varargs
    arg_names_and_types.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'arg_names_and_types', ['args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'arg_names_and_types', localization, ['args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'arg_names_and_types(...)' code ##################

    
    # Call to zip(...): (line 39)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to split(...): (line 39)
    # Processing the call arguments (line 39)
    str_28349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 55), 'str', ', ')
    # Processing the call keyword arguments (line 39)
    kwargs_28350 = {}
    # Getting the type of 'args' (line 39)
    args_28347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 44), 'args', False)
    # Obtaining the member 'split' of a type (line 39)
    split_28348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 44), args_28347, 'split')
    # Calling split(args, kwargs) (line 39)
    split_call_result_28351 = invoke(stypy.reporting.localization.Localization(__file__, 39, 44), split_28348, *[str_28349], **kwargs_28350)
    
    comprehension_28352 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 17), split_call_result_28351)
    # Assigning a type to the variable 'arg' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 'arg', comprehension_28352)
    
    # Call to split(...): (line 39)
    # Processing the call arguments (line 39)
    str_28344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 27), 'str', ' *')
    # Processing the call keyword arguments (line 39)
    kwargs_28345 = {}
    # Getting the type of 'arg' (line 39)
    arg_28342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 'arg', False)
    # Obtaining the member 'split' of a type (line 39)
    split_28343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 17), arg_28342, 'split')
    # Calling split(args, kwargs) (line 39)
    split_call_result_28346 = invoke(stypy.reporting.localization.Localization(__file__, 39, 17), split_28343, *[str_28344], **kwargs_28345)
    
    list_28353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 17), list_28353, split_call_result_28346)
    # Processing the call keyword arguments (line 39)
    kwargs_28354 = {}
    # Getting the type of 'zip' (line 39)
    zip_28341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'zip', False)
    # Calling zip(args, kwargs) (line 39)
    zip_call_result_28355 = invoke(stypy.reporting.localization.Localization(__file__, 39, 11), zip_28341, *[list_28353], **kwargs_28354)
    
    # Assigning a type to the variable 'stypy_return_type' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type', zip_call_result_28355)
    
    # ################# End of 'arg_names_and_types(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'arg_names_and_types' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_28356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28356)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'arg_names_and_types'
    return stypy_return_type_28356

# Assigning a type to the variable 'arg_names_and_types' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'arg_names_and_types', arg_names_and_types)

# Assigning a Str to a Name (line 41):

# Assigning a Str to a Name (line 41):
str_28357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, (-1)), 'str', '\ncdef extern from "{header_name}":\n    void _fortran_{name} "F_FUNC({name}wrp, {upname}WRP)"({ret_type} *out, {fort_args}) nogil\ncdef {ret_type} {name}({args}) nogil:\n    cdef {ret_type} out\n    _fortran_{name}(&out, {argnames})\n    return out\n')
# Assigning a type to the variable 'pyx_func_template' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'pyx_func_template', str_28357)

# Assigning a Dict to a Name (line 50):

# Assigning a Dict to a Name (line 50):

# Obtaining an instance of the builtin type 'dict' (line 50)
dict_28358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 50)
# Adding element type (key, value) (line 50)
str_28359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 13), 'str', 'c')
str_28360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 18), 'str', 'npy_complex64')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), dict_28358, (str_28359, str_28360))
# Adding element type (key, value) (line 50)
str_28361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 35), 'str', 'z')
str_28362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 40), 'str', 'npy_complex128')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), dict_28358, (str_28361, str_28362))
# Adding element type (key, value) (line 50)
str_28363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 13), 'str', 'cselect1')
str_28364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 25), 'str', '_cselect1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), dict_28358, (str_28363, str_28364))
# Adding element type (key, value) (line 50)
str_28365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 38), 'str', 'cselect2')
str_28366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 50), 'str', '_cselect2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), dict_28358, (str_28365, str_28366))
# Adding element type (key, value) (line 50)
str_28367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 13), 'str', 'dselect2')
str_28368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 25), 'str', '_dselect2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), dict_28358, (str_28367, str_28368))
# Adding element type (key, value) (line 50)
str_28369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 38), 'str', 'dselect3')
str_28370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 50), 'str', '_dselect3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), dict_28358, (str_28369, str_28370))
# Adding element type (key, value) (line 50)
str_28371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 13), 'str', 'sselect2')
str_28372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 25), 'str', '_sselect2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), dict_28358, (str_28371, str_28372))
# Adding element type (key, value) (line 50)
str_28373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 38), 'str', 'sselect3')
str_28374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 50), 'str', '_sselect3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), dict_28358, (str_28373, str_28374))
# Adding element type (key, value) (line 50)
str_28375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 13), 'str', 'zselect1')
str_28376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 25), 'str', '_zselect1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), dict_28358, (str_28375, str_28376))
# Adding element type (key, value) (line 50)
str_28377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 38), 'str', 'zselect2')
str_28378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 50), 'str', '_zselect2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), dict_28358, (str_28377, str_28378))

# Assigning a type to the variable 'npy_types' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'npy_types', dict_28358)

@norecursion
def arg_casts(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'arg_casts'
    module_type_store = module_type_store.open_function_context('arg_casts', 57, 0, False)
    
    # Passed parameters checking function
    arg_casts.stypy_localization = localization
    arg_casts.stypy_type_of_self = None
    arg_casts.stypy_type_store = module_type_store
    arg_casts.stypy_function_name = 'arg_casts'
    arg_casts.stypy_param_names_list = ['arg']
    arg_casts.stypy_varargs_param_name = None
    arg_casts.stypy_kwargs_param_name = None
    arg_casts.stypy_call_defaults = defaults
    arg_casts.stypy_call_varargs = varargs
    arg_casts.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'arg_casts', ['arg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'arg_casts', localization, ['arg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'arg_casts(...)' code ##################

    
    
    # Getting the type of 'arg' (line 58)
    arg_28379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 7), 'arg')
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_28380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    # Adding element type (line 58)
    str_28381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 15), 'str', 'npy_complex64')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 14), list_28380, str_28381)
    # Adding element type (line 58)
    str_28382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 32), 'str', 'npy_complex128')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 14), list_28380, str_28382)
    # Adding element type (line 58)
    str_28383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 50), 'str', '_cselect1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 14), list_28380, str_28383)
    # Adding element type (line 58)
    str_28384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 63), 'str', '_cselect2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 14), list_28380, str_28384)
    # Adding element type (line 58)
    str_28385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 15), 'str', '_dselect2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 14), list_28380, str_28385)
    # Adding element type (line 58)
    str_28386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 28), 'str', '_dselect3')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 14), list_28380, str_28386)
    # Adding element type (line 58)
    str_28387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 41), 'str', '_sselect2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 14), list_28380, str_28387)
    # Adding element type (line 58)
    str_28388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 54), 'str', '_sselect3')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 14), list_28380, str_28388)
    # Adding element type (line 58)
    str_28389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 15), 'str', '_zselect1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 14), list_28380, str_28389)
    # Adding element type (line 58)
    str_28390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 28), 'str', '_zselect2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 14), list_28380, str_28390)
    
    # Applying the binary operator 'in' (line 58)
    result_contains_28391 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 7), 'in', arg_28379, list_28380)
    
    # Testing the type of an if condition (line 58)
    if_condition_28392 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 4), result_contains_28391)
    # Assigning a type to the variable 'if_condition_28392' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'if_condition_28392', if_condition_28392)
    # SSA begins for if statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to format(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'arg' (line 61)
    arg_28395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 31), 'arg', False)
    # Processing the call keyword arguments (line 61)
    kwargs_28396 = {}
    str_28393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 15), 'str', '<{0}*>')
    # Obtaining the member 'format' of a type (line 61)
    format_28394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 15), str_28393, 'format')
    # Calling format(args, kwargs) (line 61)
    format_call_result_28397 = invoke(stypy.reporting.localization.Localization(__file__, 61, 15), format_28394, *[arg_28395], **kwargs_28396)
    
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'stypy_return_type', format_call_result_28397)
    # SSA join for if statement (line 58)
    module_type_store = module_type_store.join_ssa_context()
    
    str_28398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 11), 'str', '')
    # Assigning a type to the variable 'stypy_return_type' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type', str_28398)
    
    # ################# End of 'arg_casts(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'arg_casts' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_28399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28399)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'arg_casts'
    return stypy_return_type_28399

# Assigning a type to the variable 'arg_casts' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'arg_casts', arg_casts)

@norecursion
def pyx_decl_func(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pyx_decl_func'
    module_type_store = module_type_store.open_function_context('pyx_decl_func', 65, 0, False)
    
    # Passed parameters checking function
    pyx_decl_func.stypy_localization = localization
    pyx_decl_func.stypy_type_of_self = None
    pyx_decl_func.stypy_type_store = module_type_store
    pyx_decl_func.stypy_function_name = 'pyx_decl_func'
    pyx_decl_func.stypy_param_names_list = ['name', 'ret_type', 'args', 'header_name']
    pyx_decl_func.stypy_varargs_param_name = None
    pyx_decl_func.stypy_kwargs_param_name = None
    pyx_decl_func.stypy_call_defaults = defaults
    pyx_decl_func.stypy_call_varargs = varargs
    pyx_decl_func.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pyx_decl_func', ['name', 'ret_type', 'args', 'header_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pyx_decl_func', localization, ['name', 'ret_type', 'args', 'header_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pyx_decl_func(...)' code ##################

    
    # Assigning a Call to a Tuple (line 66):
    
    # Assigning a Subscript to a Name (line 66):
    
    # Obtaining the type of the subscript
    int_28400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 4), 'int')
    
    # Call to arg_names_and_types(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'args' (line 66)
    args_28402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 45), 'args', False)
    # Processing the call keyword arguments (line 66)
    kwargs_28403 = {}
    # Getting the type of 'arg_names_and_types' (line 66)
    arg_names_and_types_28401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'arg_names_and_types', False)
    # Calling arg_names_and_types(args, kwargs) (line 66)
    arg_names_and_types_call_result_28404 = invoke(stypy.reporting.localization.Localization(__file__, 66, 25), arg_names_and_types_28401, *[args_28402], **kwargs_28403)
    
    # Obtaining the member '__getitem__' of a type (line 66)
    getitem___28405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 4), arg_names_and_types_call_result_28404, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
    subscript_call_result_28406 = invoke(stypy.reporting.localization.Localization(__file__, 66, 4), getitem___28405, int_28400)
    
    # Assigning a type to the variable 'tuple_var_assignment_28271' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'tuple_var_assignment_28271', subscript_call_result_28406)
    
    # Assigning a Subscript to a Name (line 66):
    
    # Obtaining the type of the subscript
    int_28407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 4), 'int')
    
    # Call to arg_names_and_types(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'args' (line 66)
    args_28409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 45), 'args', False)
    # Processing the call keyword arguments (line 66)
    kwargs_28410 = {}
    # Getting the type of 'arg_names_and_types' (line 66)
    arg_names_and_types_28408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'arg_names_and_types', False)
    # Calling arg_names_and_types(args, kwargs) (line 66)
    arg_names_and_types_call_result_28411 = invoke(stypy.reporting.localization.Localization(__file__, 66, 25), arg_names_and_types_28408, *[args_28409], **kwargs_28410)
    
    # Obtaining the member '__getitem__' of a type (line 66)
    getitem___28412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 4), arg_names_and_types_call_result_28411, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
    subscript_call_result_28413 = invoke(stypy.reporting.localization.Localization(__file__, 66, 4), getitem___28412, int_28407)
    
    # Assigning a type to the variable 'tuple_var_assignment_28272' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'tuple_var_assignment_28272', subscript_call_result_28413)
    
    # Assigning a Name to a Name (line 66):
    # Getting the type of 'tuple_var_assignment_28271' (line 66)
    tuple_var_assignment_28271_28414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'tuple_var_assignment_28271')
    # Assigning a type to the variable 'argtypes' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'argtypes', tuple_var_assignment_28271_28414)
    
    # Assigning a Name to a Name (line 66):
    # Getting the type of 'tuple_var_assignment_28272' (line 66)
    tuple_var_assignment_28272_28415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'tuple_var_assignment_28272')
    # Assigning a type to the variable 'argnames' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 14), 'argnames', tuple_var_assignment_28272_28415)
    
    
    # Getting the type of 'ret_type' (line 71)
    ret_type_28416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 7), 'ret_type')
    # Getting the type of 'argnames' (line 71)
    argnames_28417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'argnames')
    # Applying the binary operator 'in' (line 71)
    result_contains_28418 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 7), 'in', ret_type_28416, argnames_28417)
    
    # Testing the type of an if condition (line 71)
    if_condition_28419 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 4), result_contains_28418)
    # Assigning a type to the variable 'if_condition_28419' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'if_condition_28419', if_condition_28419)
    # SSA begins for if statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a ListComp to a Name (line 72):
    
    # Assigning a ListComp to a Name (line 72):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'argnames' (line 72)
    argnames_28428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 68), 'argnames')
    comprehension_28429 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 20), argnames_28428)
    # Assigning a type to the variable 'n' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'n', comprehension_28429)
    
    
    # Getting the type of 'n' (line 72)
    n_28420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'n')
    # Getting the type of 'ret_type' (line 72)
    ret_type_28421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 30), 'ret_type')
    # Applying the binary operator '!=' (line 72)
    result_ne_28422 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 25), '!=', n_28420, ret_type_28421)
    
    # Testing the type of an if expression (line 72)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 20), result_ne_28422)
    # SSA begins for if expression (line 72)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'n' (line 72)
    n_28423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'n')
    # SSA branch for the else part of an if expression (line 72)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'ret_type' (line 72)
    ret_type_28424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 44), 'ret_type')
    str_28425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 55), 'str', '_')
    # Applying the binary operator '+' (line 72)
    result_add_28426 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 44), '+', ret_type_28424, str_28425)
    
    # SSA join for if expression (line 72)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_28427 = union_type.UnionType.add(n_28423, result_add_28426)
    
    list_28430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 20), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 20), list_28430, if_exp_28427)
    # Assigning a type to the variable 'argnames' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'argnames', list_28430)
    
    # Assigning a ListComp to a Name (line 73):
    
    # Assigning a ListComp to a Name (line 73):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'argnames' (line 74)
    argnames_28441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 29), 'argnames')
    comprehension_28442 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 20), argnames_28441)
    # Assigning a type to the variable 'n' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'n', comprehension_28442)
    
    
    # Getting the type of 'n' (line 73)
    n_28431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 25), 'n')
    
    # Obtaining an instance of the builtin type 'list' (line 73)
    list_28432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 73)
    # Adding element type (line 73)
    str_28433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 35), 'str', 'lambda')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 34), list_28432, str_28433)
    # Adding element type (line 73)
    str_28434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 45), 'str', 'in')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 34), list_28432, str_28434)
    
    # Applying the binary operator 'notin' (line 73)
    result_contains_28435 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 25), 'notin', n_28431, list_28432)
    
    # Testing the type of an if expression (line 73)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 20), result_contains_28435)
    # SSA begins for if expression (line 73)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'n' (line 73)
    n_28436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'n')
    # SSA branch for the else part of an if expression (line 73)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'n' (line 73)
    n_28437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 56), 'n')
    str_28438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 60), 'str', '_')
    # Applying the binary operator '+' (line 73)
    result_add_28439 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 56), '+', n_28437, str_28438)
    
    # SSA join for if expression (line 73)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_28440 = union_type.UnionType.add(n_28436, result_add_28439)
    
    list_28443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 20), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 20), list_28443, if_exp_28440)
    # Assigning a type to the variable 'argnames' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'argnames', list_28443)
    
    # Assigning a Call to a Name (line 75):
    
    # Assigning a Call to a Name (line 75):
    
    # Call to join(...): (line 75)
    # Processing the call arguments (line 75)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'argtypes' (line 76)
    argtypes_28454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 42), 'argtypes', False)
    # Getting the type of 'argnames' (line 76)
    argnames_28455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 52), 'argnames', False)
    # Processing the call keyword arguments (line 76)
    kwargs_28456 = {}
    # Getting the type of 'zip' (line 76)
    zip_28453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 38), 'zip', False)
    # Calling zip(args, kwargs) (line 76)
    zip_call_result_28457 = invoke(stypy.reporting.localization.Localization(__file__, 76, 38), zip_28453, *[argtypes_28454, argnames_28455], **kwargs_28456)
    
    comprehension_28458 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 26), zip_call_result_28457)
    # Assigning a type to the variable 'n' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 26), comprehension_28458))
    # Assigning a type to the variable 't' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 26), comprehension_28458))
    
    # Call to join(...): (line 75)
    # Processing the call arguments (line 75)
    
    # Obtaining an instance of the builtin type 'list' (line 75)
    list_28448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 75)
    # Adding element type (line 75)
    # Getting the type of 'n' (line 75)
    n_28449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 37), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 36), list_28448, n_28449)
    # Adding element type (line 75)
    # Getting the type of 't' (line 75)
    t_28450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 40), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 36), list_28448, t_28450)
    
    # Processing the call keyword arguments (line 75)
    kwargs_28451 = {}
    str_28446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 26), 'str', ' *')
    # Obtaining the member 'join' of a type (line 75)
    join_28447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 26), str_28446, 'join')
    # Calling join(args, kwargs) (line 75)
    join_call_result_28452 = invoke(stypy.reporting.localization.Localization(__file__, 75, 26), join_28447, *[list_28448], **kwargs_28451)
    
    list_28459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 26), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 26), list_28459, join_call_result_28452)
    # Processing the call keyword arguments (line 75)
    kwargs_28460 = {}
    str_28444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 15), 'str', ', ')
    # Obtaining the member 'join' of a type (line 75)
    join_28445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 15), str_28444, 'join')
    # Calling join(args, kwargs) (line 75)
    join_call_result_28461 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), join_28445, *[list_28459], **kwargs_28460)
    
    # Assigning a type to the variable 'args' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'args', join_call_result_28461)
    # SSA join for if statement (line 71)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Name (line 77):
    
    # Assigning a ListComp to a Name (line 77):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'argtypes' (line 77)
    argtypes_28468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 45), 'argtypes')
    comprehension_28469 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 16), argtypes_28468)
    # Assigning a type to the variable 't' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 't', comprehension_28469)
    
    # Call to get(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 't' (line 77)
    t_28464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 30), 't', False)
    # Getting the type of 't' (line 77)
    t_28465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 33), 't', False)
    # Processing the call keyword arguments (line 77)
    kwargs_28466 = {}
    # Getting the type of 'npy_types' (line 77)
    npy_types_28462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'npy_types', False)
    # Obtaining the member 'get' of a type (line 77)
    get_28463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 16), npy_types_28462, 'get')
    # Calling get(args, kwargs) (line 77)
    get_call_result_28467 = invoke(stypy.reporting.localization.Localization(__file__, 77, 16), get_28463, *[t_28464, t_28465], **kwargs_28466)
    
    list_28470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 16), list_28470, get_call_result_28467)
    # Assigning a type to the variable 'argtypes' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'argtypes', list_28470)
    
    # Assigning a Call to a Name (line 78):
    
    # Assigning a Call to a Name (line 78):
    
    # Call to join(...): (line 78)
    # Processing the call arguments (line 78)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'argtypes' (line 79)
    argtypes_28481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 43), 'argtypes', False)
    # Getting the type of 'argnames' (line 79)
    argnames_28482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 53), 'argnames', False)
    # Processing the call keyword arguments (line 79)
    kwargs_28483 = {}
    # Getting the type of 'zip' (line 79)
    zip_28480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 39), 'zip', False)
    # Calling zip(args, kwargs) (line 79)
    zip_call_result_28484 = invoke(stypy.reporting.localization.Localization(__file__, 79, 39), zip_28480, *[argtypes_28481, argnames_28482], **kwargs_28483)
    
    comprehension_28485 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 27), zip_call_result_28484)
    # Assigning a type to the variable 'n' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 27), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 27), comprehension_28485))
    # Assigning a type to the variable 't' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 27), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 27), comprehension_28485))
    
    # Call to join(...): (line 78)
    # Processing the call arguments (line 78)
    
    # Obtaining an instance of the builtin type 'list' (line 78)
    list_28475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 78)
    # Adding element type (line 78)
    # Getting the type of 'n' (line 78)
    n_28476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 38), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 37), list_28475, n_28476)
    # Adding element type (line 78)
    # Getting the type of 't' (line 78)
    t_28477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 41), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 37), list_28475, t_28477)
    
    # Processing the call keyword arguments (line 78)
    kwargs_28478 = {}
    str_28473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 27), 'str', ' *')
    # Obtaining the member 'join' of a type (line 78)
    join_28474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 27), str_28473, 'join')
    # Calling join(args, kwargs) (line 78)
    join_call_result_28479 = invoke(stypy.reporting.localization.Localization(__file__, 78, 27), join_28474, *[list_28475], **kwargs_28478)
    
    list_28486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 27), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 27), list_28486, join_call_result_28479)
    # Processing the call keyword arguments (line 78)
    kwargs_28487 = {}
    str_28471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 16), 'str', ', ')
    # Obtaining the member 'join' of a type (line 78)
    join_28472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 16), str_28471, 'join')
    # Calling join(args, kwargs) (line 78)
    join_call_result_28488 = invoke(stypy.reporting.localization.Localization(__file__, 78, 16), join_28472, *[list_28486], **kwargs_28487)
    
    # Assigning a type to the variable 'fort_args' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'fort_args', join_call_result_28488)
    
    # Assigning a ListComp to a Name (line 80):
    
    # Assigning a ListComp to a Name (line 80):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'argnames' (line 80)
    argnames_28496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 49), 'argnames', False)
    # Getting the type of 'argtypes' (line 80)
    argtypes_28497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 59), 'argtypes', False)
    # Processing the call keyword arguments (line 80)
    kwargs_28498 = {}
    # Getting the type of 'zip' (line 80)
    zip_28495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 45), 'zip', False)
    # Calling zip(args, kwargs) (line 80)
    zip_call_result_28499 = invoke(stypy.reporting.localization.Localization(__file__, 80, 45), zip_28495, *[argnames_28496, argtypes_28497], **kwargs_28498)
    
    comprehension_28500 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 16), zip_call_result_28499)
    # Assigning a type to the variable 'n' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 16), comprehension_28500))
    # Assigning a type to the variable 't' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 16), comprehension_28500))
    
    # Call to arg_casts(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 't' (line 80)
    t_28490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 26), 't', False)
    # Processing the call keyword arguments (line 80)
    kwargs_28491 = {}
    # Getting the type of 'arg_casts' (line 80)
    arg_casts_28489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'arg_casts', False)
    # Calling arg_casts(args, kwargs) (line 80)
    arg_casts_call_result_28492 = invoke(stypy.reporting.localization.Localization(__file__, 80, 16), arg_casts_28489, *[t_28490], **kwargs_28491)
    
    # Getting the type of 'n' (line 80)
    n_28493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 31), 'n')
    # Applying the binary operator '+' (line 80)
    result_add_28494 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 16), '+', arg_casts_call_result_28492, n_28493)
    
    list_28501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 16), list_28501, result_add_28494)
    # Assigning a type to the variable 'argnames' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'argnames', list_28501)
    
    # Assigning a Call to a Name (line 81):
    
    # Assigning a Call to a Name (line 81):
    
    # Call to join(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'argnames' (line 81)
    argnames_28504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 25), 'argnames', False)
    # Processing the call keyword arguments (line 81)
    kwargs_28505 = {}
    str_28502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 15), 'str', ', ')
    # Obtaining the member 'join' of a type (line 81)
    join_28503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 15), str_28502, 'join')
    # Calling join(args, kwargs) (line 81)
    join_call_result_28506 = invoke(stypy.reporting.localization.Localization(__file__, 81, 15), join_28503, *[argnames_28504], **kwargs_28505)
    
    # Assigning a type to the variable 'argnames' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'argnames', join_call_result_28506)
    
    # Assigning a Subscript to a Name (line 82):
    
    # Assigning a Subscript to a Name (line 82):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ret_type' (line 82)
    ret_type_28507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 25), 'ret_type')
    # Getting the type of 'c_types' (line 82)
    c_types_28508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), 'c_types')
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___28509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 17), c_types_28508, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 82)
    subscript_call_result_28510 = invoke(stypy.reporting.localization.Localization(__file__, 82, 17), getitem___28509, ret_type_28507)
    
    # Assigning a type to the variable 'c_ret_type' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'c_ret_type', subscript_call_result_28510)
    
    # Assigning a Call to a Name (line 83):
    
    # Assigning a Call to a Name (line 83):
    
    # Call to replace(...): (line 83)
    # Processing the call arguments (line 83)
    str_28513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 24), 'str', 'lambda')
    str_28514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 34), 'str', 'lambda_')
    # Processing the call keyword arguments (line 83)
    kwargs_28515 = {}
    # Getting the type of 'args' (line 83)
    args_28511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'args', False)
    # Obtaining the member 'replace' of a type (line 83)
    replace_28512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 11), args_28511, 'replace')
    # Calling replace(args, kwargs) (line 83)
    replace_call_result_28516 = invoke(stypy.reporting.localization.Localization(__file__, 83, 11), replace_28512, *[str_28513, str_28514], **kwargs_28515)
    
    # Assigning a type to the variable 'args' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'args', replace_call_result_28516)
    
    # Call to format(...): (line 84)
    # Processing the call keyword arguments (line 84)
    # Getting the type of 'name' (line 84)
    name_28519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 41), 'name', False)
    keyword_28520 = name_28519
    
    # Call to upper(...): (line 84)
    # Processing the call keyword arguments (line 84)
    kwargs_28523 = {}
    # Getting the type of 'name' (line 84)
    name_28521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 54), 'name', False)
    # Obtaining the member 'upper' of a type (line 84)
    upper_28522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 54), name_28521, 'upper')
    # Calling upper(args, kwargs) (line 84)
    upper_call_result_28524 = invoke(stypy.reporting.localization.Localization(__file__, 84, 54), upper_28522, *[], **kwargs_28523)
    
    keyword_28525 = upper_call_result_28524
    # Getting the type of 'args' (line 84)
    args_28526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 73), 'args', False)
    keyword_28527 = args_28526
    # Getting the type of 'fort_args' (line 85)
    fort_args_28528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 46), 'fort_args', False)
    keyword_28529 = fort_args_28528
    # Getting the type of 'ret_type' (line 85)
    ret_type_28530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 66), 'ret_type', False)
    keyword_28531 = ret_type_28530
    # Getting the type of 'c_ret_type' (line 86)
    c_ret_type_28532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 47), 'c_ret_type', False)
    keyword_28533 = c_ret_type_28532
    # Getting the type of 'argnames' (line 86)
    argnames_28534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 68), 'argnames', False)
    keyword_28535 = argnames_28534
    # Getting the type of 'header_name' (line 87)
    header_name_28536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 48), 'header_name', False)
    keyword_28537 = header_name_28536
    kwargs_28538 = {'fort_args': keyword_28529, 'header_name': keyword_28537, 'name': keyword_28520, 'ret_type': keyword_28531, 'args': keyword_28527, 'argnames': keyword_28535, 'c_ret_type': keyword_28533, 'upname': keyword_28525}
    # Getting the type of 'pyx_func_template' (line 84)
    pyx_func_template_28517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), 'pyx_func_template', False)
    # Obtaining the member 'format' of a type (line 84)
    format_28518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 11), pyx_func_template_28517, 'format')
    # Calling format(args, kwargs) (line 84)
    format_call_result_28539 = invoke(stypy.reporting.localization.Localization(__file__, 84, 11), format_28518, *[], **kwargs_28538)
    
    # Assigning a type to the variable 'stypy_return_type' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type', format_call_result_28539)
    
    # ################# End of 'pyx_decl_func(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pyx_decl_func' in the type store
    # Getting the type of 'stypy_return_type' (line 65)
    stypy_return_type_28540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28540)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pyx_decl_func'
    return stypy_return_type_28540

# Assigning a type to the variable 'pyx_decl_func' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'pyx_decl_func', pyx_decl_func)

# Assigning a Str to a Name (line 89):

# Assigning a Str to a Name (line 89):
str_28541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, (-1)), 'str', 'cdef extern from "{header_name}":\n    void _fortran_{name} "F_FUNC({name},{upname})"({fort_args}) nogil\ncdef void {name}({args}) nogil:\n    _fortran_{name}({argnames})\n')
# Assigning a type to the variable 'pyx_sub_template' (line 89)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'pyx_sub_template', str_28541)

@norecursion
def pyx_decl_sub(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pyx_decl_sub'
    module_type_store = module_type_store.open_function_context('pyx_decl_sub', 96, 0, False)
    
    # Passed parameters checking function
    pyx_decl_sub.stypy_localization = localization
    pyx_decl_sub.stypy_type_of_self = None
    pyx_decl_sub.stypy_type_store = module_type_store
    pyx_decl_sub.stypy_function_name = 'pyx_decl_sub'
    pyx_decl_sub.stypy_param_names_list = ['name', 'args', 'header_name']
    pyx_decl_sub.stypy_varargs_param_name = None
    pyx_decl_sub.stypy_kwargs_param_name = None
    pyx_decl_sub.stypy_call_defaults = defaults
    pyx_decl_sub.stypy_call_varargs = varargs
    pyx_decl_sub.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pyx_decl_sub', ['name', 'args', 'header_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pyx_decl_sub', localization, ['name', 'args', 'header_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pyx_decl_sub(...)' code ##################

    
    # Assigning a Call to a Tuple (line 97):
    
    # Assigning a Subscript to a Name (line 97):
    
    # Obtaining the type of the subscript
    int_28542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 4), 'int')
    
    # Call to arg_names_and_types(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'args' (line 97)
    args_28544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 45), 'args', False)
    # Processing the call keyword arguments (line 97)
    kwargs_28545 = {}
    # Getting the type of 'arg_names_and_types' (line 97)
    arg_names_and_types_28543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 25), 'arg_names_and_types', False)
    # Calling arg_names_and_types(args, kwargs) (line 97)
    arg_names_and_types_call_result_28546 = invoke(stypy.reporting.localization.Localization(__file__, 97, 25), arg_names_and_types_28543, *[args_28544], **kwargs_28545)
    
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___28547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 4), arg_names_and_types_call_result_28546, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_28548 = invoke(stypy.reporting.localization.Localization(__file__, 97, 4), getitem___28547, int_28542)
    
    # Assigning a type to the variable 'tuple_var_assignment_28273' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'tuple_var_assignment_28273', subscript_call_result_28548)
    
    # Assigning a Subscript to a Name (line 97):
    
    # Obtaining the type of the subscript
    int_28549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 4), 'int')
    
    # Call to arg_names_and_types(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'args' (line 97)
    args_28551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 45), 'args', False)
    # Processing the call keyword arguments (line 97)
    kwargs_28552 = {}
    # Getting the type of 'arg_names_and_types' (line 97)
    arg_names_and_types_28550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 25), 'arg_names_and_types', False)
    # Calling arg_names_and_types(args, kwargs) (line 97)
    arg_names_and_types_call_result_28553 = invoke(stypy.reporting.localization.Localization(__file__, 97, 25), arg_names_and_types_28550, *[args_28551], **kwargs_28552)
    
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___28554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 4), arg_names_and_types_call_result_28553, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_28555 = invoke(stypy.reporting.localization.Localization(__file__, 97, 4), getitem___28554, int_28549)
    
    # Assigning a type to the variable 'tuple_var_assignment_28274' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'tuple_var_assignment_28274', subscript_call_result_28555)
    
    # Assigning a Name to a Name (line 97):
    # Getting the type of 'tuple_var_assignment_28273' (line 97)
    tuple_var_assignment_28273_28556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'tuple_var_assignment_28273')
    # Assigning a type to the variable 'argtypes' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'argtypes', tuple_var_assignment_28273_28556)
    
    # Assigning a Name to a Name (line 97):
    # Getting the type of 'tuple_var_assignment_28274' (line 97)
    tuple_var_assignment_28274_28557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'tuple_var_assignment_28274')
    # Assigning a type to the variable 'argnames' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 14), 'argnames', tuple_var_assignment_28274_28557)
    
    # Assigning a ListComp to a Name (line 98):
    
    # Assigning a ListComp to a Name (line 98):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'argtypes' (line 98)
    argtypes_28564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 45), 'argtypes')
    comprehension_28565 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 16), argtypes_28564)
    # Assigning a type to the variable 't' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 't', comprehension_28565)
    
    # Call to get(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 't' (line 98)
    t_28560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 30), 't', False)
    # Getting the type of 't' (line 98)
    t_28561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 't', False)
    # Processing the call keyword arguments (line 98)
    kwargs_28562 = {}
    # Getting the type of 'npy_types' (line 98)
    npy_types_28558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'npy_types', False)
    # Obtaining the member 'get' of a type (line 98)
    get_28559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 16), npy_types_28558, 'get')
    # Calling get(args, kwargs) (line 98)
    get_call_result_28563 = invoke(stypy.reporting.localization.Localization(__file__, 98, 16), get_28559, *[t_28560, t_28561], **kwargs_28562)
    
    list_28566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 16), list_28566, get_call_result_28563)
    # Assigning a type to the variable 'argtypes' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'argtypes', list_28566)
    
    # Assigning a ListComp to a Name (line 99):
    
    # Assigning a ListComp to a Name (line 99):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'argnames' (line 99)
    argnames_28577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 69), 'argnames')
    comprehension_28578 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 16), argnames_28577)
    # Assigning a type to the variable 'n' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'n', comprehension_28578)
    
    
    # Getting the type of 'n' (line 99)
    n_28567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 21), 'n')
    
    # Obtaining an instance of the builtin type 'list' (line 99)
    list_28568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 99)
    # Adding element type (line 99)
    str_28569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 31), 'str', 'lambda')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 30), list_28568, str_28569)
    # Adding element type (line 99)
    str_28570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 41), 'str', 'in')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 30), list_28568, str_28570)
    
    # Applying the binary operator 'notin' (line 99)
    result_contains_28571 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 21), 'notin', n_28567, list_28568)
    
    # Testing the type of an if expression (line 99)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 16), result_contains_28571)
    # SSA begins for if expression (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    # Getting the type of 'n' (line 99)
    n_28572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'n')
    # SSA branch for the else part of an if expression (line 99)
    module_type_store.open_ssa_branch('if expression else')
    # Getting the type of 'n' (line 99)
    n_28573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 52), 'n')
    str_28574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 56), 'str', '_')
    # Applying the binary operator '+' (line 99)
    result_add_28575 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 52), '+', n_28573, str_28574)
    
    # SSA join for if expression (line 99)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_28576 = union_type.UnionType.add(n_28572, result_add_28575)
    
    list_28579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 16), list_28579, if_exp_28576)
    # Assigning a type to the variable 'argnames' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'argnames', list_28579)
    
    # Assigning a Call to a Name (line 100):
    
    # Assigning a Call to a Name (line 100):
    
    # Call to join(...): (line 100)
    # Processing the call arguments (line 100)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'argtypes' (line 101)
    argtypes_28590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 43), 'argtypes', False)
    # Getting the type of 'argnames' (line 101)
    argnames_28591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 53), 'argnames', False)
    # Processing the call keyword arguments (line 101)
    kwargs_28592 = {}
    # Getting the type of 'zip' (line 101)
    zip_28589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 39), 'zip', False)
    # Calling zip(args, kwargs) (line 101)
    zip_call_result_28593 = invoke(stypy.reporting.localization.Localization(__file__, 101, 39), zip_28589, *[argtypes_28590, argnames_28591], **kwargs_28592)
    
    comprehension_28594 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 27), zip_call_result_28593)
    # Assigning a type to the variable 'n' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 27), comprehension_28594))
    # Assigning a type to the variable 't' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 27), comprehension_28594))
    
    # Call to join(...): (line 100)
    # Processing the call arguments (line 100)
    
    # Obtaining an instance of the builtin type 'list' (line 100)
    list_28584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 100)
    # Adding element type (line 100)
    # Getting the type of 'n' (line 100)
    n_28585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 38), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 37), list_28584, n_28585)
    # Adding element type (line 100)
    # Getting the type of 't' (line 100)
    t_28586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 41), 't', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 37), list_28584, t_28586)
    
    # Processing the call keyword arguments (line 100)
    kwargs_28587 = {}
    str_28582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 27), 'str', ' *')
    # Obtaining the member 'join' of a type (line 100)
    join_28583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 27), str_28582, 'join')
    # Calling join(args, kwargs) (line 100)
    join_call_result_28588 = invoke(stypy.reporting.localization.Localization(__file__, 100, 27), join_28583, *[list_28584], **kwargs_28587)
    
    list_28595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 27), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 27), list_28595, join_call_result_28588)
    # Processing the call keyword arguments (line 100)
    kwargs_28596 = {}
    str_28580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 16), 'str', ', ')
    # Obtaining the member 'join' of a type (line 100)
    join_28581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 16), str_28580, 'join')
    # Calling join(args, kwargs) (line 100)
    join_call_result_28597 = invoke(stypy.reporting.localization.Localization(__file__, 100, 16), join_28581, *[list_28595], **kwargs_28596)
    
    # Assigning a type to the variable 'fort_args' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'fort_args', join_call_result_28597)
    
    # Assigning a ListComp to a Name (line 102):
    
    # Assigning a ListComp to a Name (line 102):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to zip(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'argnames' (line 102)
    argnames_28605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 49), 'argnames', False)
    # Getting the type of 'argtypes' (line 102)
    argtypes_28606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 59), 'argtypes', False)
    # Processing the call keyword arguments (line 102)
    kwargs_28607 = {}
    # Getting the type of 'zip' (line 102)
    zip_28604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 45), 'zip', False)
    # Calling zip(args, kwargs) (line 102)
    zip_call_result_28608 = invoke(stypy.reporting.localization.Localization(__file__, 102, 45), zip_28604, *[argnames_28605, argtypes_28606], **kwargs_28607)
    
    comprehension_28609 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 16), zip_call_result_28608)
    # Assigning a type to the variable 'n' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 16), comprehension_28609))
    # Assigning a type to the variable 't' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 16), comprehension_28609))
    
    # Call to arg_casts(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 't' (line 102)
    t_28599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 26), 't', False)
    # Processing the call keyword arguments (line 102)
    kwargs_28600 = {}
    # Getting the type of 'arg_casts' (line 102)
    arg_casts_28598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'arg_casts', False)
    # Calling arg_casts(args, kwargs) (line 102)
    arg_casts_call_result_28601 = invoke(stypy.reporting.localization.Localization(__file__, 102, 16), arg_casts_28598, *[t_28599], **kwargs_28600)
    
    # Getting the type of 'n' (line 102)
    n_28602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 31), 'n')
    # Applying the binary operator '+' (line 102)
    result_add_28603 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 16), '+', arg_casts_call_result_28601, n_28602)
    
    list_28610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 16), list_28610, result_add_28603)
    # Assigning a type to the variable 'argnames' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'argnames', list_28610)
    
    # Assigning a Call to a Name (line 103):
    
    # Assigning a Call to a Name (line 103):
    
    # Call to join(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'argnames' (line 103)
    argnames_28613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 25), 'argnames', False)
    # Processing the call keyword arguments (line 103)
    kwargs_28614 = {}
    str_28611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 15), 'str', ', ')
    # Obtaining the member 'join' of a type (line 103)
    join_28612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), str_28611, 'join')
    # Calling join(args, kwargs) (line 103)
    join_call_result_28615 = invoke(stypy.reporting.localization.Localization(__file__, 103, 15), join_28612, *[argnames_28613], **kwargs_28614)
    
    # Assigning a type to the variable 'argnames' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'argnames', join_call_result_28615)
    
    # Assigning a Call to a Name (line 104):
    
    # Assigning a Call to a Name (line 104):
    
    # Call to replace(...): (line 104)
    # Processing the call arguments (line 104)
    str_28623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 57), 'str', '*in,')
    str_28624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 65), 'str', '*in_,')
    # Processing the call keyword arguments (line 104)
    kwargs_28625 = {}
    
    # Call to replace(...): (line 104)
    # Processing the call arguments (line 104)
    str_28618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'str', '*lambda,')
    str_28619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 36), 'str', '*lambda_,')
    # Processing the call keyword arguments (line 104)
    kwargs_28620 = {}
    # Getting the type of 'args' (line 104)
    args_28616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'args', False)
    # Obtaining the member 'replace' of a type (line 104)
    replace_28617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 11), args_28616, 'replace')
    # Calling replace(args, kwargs) (line 104)
    replace_call_result_28621 = invoke(stypy.reporting.localization.Localization(__file__, 104, 11), replace_28617, *[str_28618, str_28619], **kwargs_28620)
    
    # Obtaining the member 'replace' of a type (line 104)
    replace_28622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 11), replace_call_result_28621, 'replace')
    # Calling replace(args, kwargs) (line 104)
    replace_call_result_28626 = invoke(stypy.reporting.localization.Localization(__file__, 104, 11), replace_28622, *[str_28623, str_28624], **kwargs_28625)
    
    # Assigning a type to the variable 'args' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'args', replace_call_result_28626)
    
    # Call to format(...): (line 105)
    # Processing the call keyword arguments (line 105)
    # Getting the type of 'name' (line 105)
    name_28629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 40), 'name', False)
    keyword_28630 = name_28629
    
    # Call to upper(...): (line 105)
    # Processing the call keyword arguments (line 105)
    kwargs_28633 = {}
    # Getting the type of 'name' (line 105)
    name_28631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 53), 'name', False)
    # Obtaining the member 'upper' of a type (line 105)
    upper_28632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 53), name_28631, 'upper')
    # Calling upper(args, kwargs) (line 105)
    upper_call_result_28634 = invoke(stypy.reporting.localization.Localization(__file__, 105, 53), upper_28632, *[], **kwargs_28633)
    
    keyword_28635 = upper_call_result_28634
    # Getting the type of 'args' (line 106)
    args_28636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 40), 'args', False)
    keyword_28637 = args_28636
    # Getting the type of 'fort_args' (line 106)
    fort_args_28638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 56), 'fort_args', False)
    keyword_28639 = fort_args_28638
    # Getting the type of 'argnames' (line 107)
    argnames_28640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 44), 'argnames', False)
    keyword_28641 = argnames_28640
    # Getting the type of 'header_name' (line 107)
    header_name_28642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 66), 'header_name', False)
    keyword_28643 = header_name_28642
    kwargs_28644 = {'fort_args': keyword_28639, 'header_name': keyword_28643, 'name': keyword_28630, 'args': keyword_28637, 'argnames': keyword_28641, 'upname': keyword_28635}
    # Getting the type of 'pyx_sub_template' (line 105)
    pyx_sub_template_28627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'pyx_sub_template', False)
    # Obtaining the member 'format' of a type (line 105)
    format_28628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 11), pyx_sub_template_28627, 'format')
    # Calling format(args, kwargs) (line 105)
    format_call_result_28645 = invoke(stypy.reporting.localization.Localization(__file__, 105, 11), format_28628, *[], **kwargs_28644)
    
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type', format_call_result_28645)
    
    # ################# End of 'pyx_decl_sub(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pyx_decl_sub' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_28646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28646)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pyx_decl_sub'
    return stypy_return_type_28646

# Assigning a type to the variable 'pyx_decl_sub' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'pyx_decl_sub', pyx_decl_sub)

# Assigning a Str to a Name (line 109):

# Assigning a Str to a Name (line 109):
str_28647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, (-1)), 'str', '# cython: boundscheck = False\n# cython: wraparound = False\n# cython: cdivision = True\n\n"""\nBLAS Functions for Cython\n=========================\n\nUsable from Cython via::\n\n    cimport scipy.linalg.cython_blas\n\nThese wrappers do not check for alignment of arrays.\nAlignment should be checked before these wrappers are used.\n\nRaw function pointers (Fortran-style pointer arguments):\n\n- {}\n\n\n"""\n\n# Within scipy, these wrappers can be used via relative or absolute cimport.\n# Examples:\n# from ..linalg cimport cython_blas\n# from scipy.linalg cimport cython_blas\n# cimport scipy.linalg.cython_blas as cython_blas\n# cimport ..linalg.cython_blas as cython_blas\n\n# Within scipy, if BLAS functions are needed in C/C++/Fortran,\n# these wrappers should not be used.\n# The original libraries should be linked directly.\n\nfrom __future__ import absolute_import\n\ncdef extern from "fortran_defs.h":\n    pass\n\nfrom numpy cimport npy_complex64, npy_complex128\n\n')
# Assigning a type to the variable 'blas_pyx_preamble' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'blas_pyx_preamble', str_28647)

@norecursion
def make_blas_pyx_preamble(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'make_blas_pyx_preamble'
    module_type_store = module_type_store.open_function_context('make_blas_pyx_preamble', 152, 0, False)
    
    # Passed parameters checking function
    make_blas_pyx_preamble.stypy_localization = localization
    make_blas_pyx_preamble.stypy_type_of_self = None
    make_blas_pyx_preamble.stypy_type_store = module_type_store
    make_blas_pyx_preamble.stypy_function_name = 'make_blas_pyx_preamble'
    make_blas_pyx_preamble.stypy_param_names_list = ['all_sigs']
    make_blas_pyx_preamble.stypy_varargs_param_name = None
    make_blas_pyx_preamble.stypy_kwargs_param_name = None
    make_blas_pyx_preamble.stypy_call_defaults = defaults
    make_blas_pyx_preamble.stypy_call_varargs = varargs
    make_blas_pyx_preamble.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_blas_pyx_preamble', ['all_sigs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_blas_pyx_preamble', localization, ['all_sigs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_blas_pyx_preamble(...)' code ##################

    
    # Assigning a ListComp to a Name (line 153):
    
    # Assigning a ListComp to a Name (line 153):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'all_sigs' (line 153)
    all_sigs_28652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 31), 'all_sigs')
    comprehension_28653 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 13), all_sigs_28652)
    # Assigning a type to the variable 'sig' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 13), 'sig', comprehension_28653)
    
    # Obtaining the type of the subscript
    int_28648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 17), 'int')
    # Getting the type of 'sig' (line 153)
    sig_28649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 13), 'sig')
    # Obtaining the member '__getitem__' of a type (line 153)
    getitem___28650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 13), sig_28649, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 153)
    subscript_call_result_28651 = invoke(stypy.reporting.localization.Localization(__file__, 153, 13), getitem___28650, int_28648)
    
    list_28654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 13), list_28654, subscript_call_result_28651)
    # Assigning a type to the variable 'names' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'names', list_28654)
    
    # Call to format(...): (line 154)
    # Processing the call arguments (line 154)
    
    # Call to join(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'names' (line 154)
    names_28659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 48), 'names', False)
    # Processing the call keyword arguments (line 154)
    kwargs_28660 = {}
    str_28657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 36), 'str', '\n- ')
    # Obtaining the member 'join' of a type (line 154)
    join_28658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 36), str_28657, 'join')
    # Calling join(args, kwargs) (line 154)
    join_call_result_28661 = invoke(stypy.reporting.localization.Localization(__file__, 154, 36), join_28658, *[names_28659], **kwargs_28660)
    
    # Processing the call keyword arguments (line 154)
    kwargs_28662 = {}
    # Getting the type of 'blas_pyx_preamble' (line 154)
    blas_pyx_preamble_28655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 11), 'blas_pyx_preamble', False)
    # Obtaining the member 'format' of a type (line 154)
    format_28656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 11), blas_pyx_preamble_28655, 'format')
    # Calling format(args, kwargs) (line 154)
    format_call_result_28663 = invoke(stypy.reporting.localization.Localization(__file__, 154, 11), format_28656, *[join_call_result_28661], **kwargs_28662)
    
    # Assigning a type to the variable 'stypy_return_type' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type', format_call_result_28663)
    
    # ################# End of 'make_blas_pyx_preamble(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_blas_pyx_preamble' in the type store
    # Getting the type of 'stypy_return_type' (line 152)
    stypy_return_type_28664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28664)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_blas_pyx_preamble'
    return stypy_return_type_28664

# Assigning a type to the variable 'make_blas_pyx_preamble' (line 152)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'make_blas_pyx_preamble', make_blas_pyx_preamble)

# Assigning a Str to a Name (line 156):

# Assigning a Str to a Name (line 156):
str_28665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, (-1)), 'str', '"""\nLAPACK functions for Cython\n===========================\n\nUsable from Cython via::\n\n    cimport scipy.linalg.cython_lapack\n\nThis module provides Cython-level wrappers for all primary routines included\nin LAPACK 3.1.0 except for ``zcgesv`` since its interface is not consistent\nfrom LAPACK 3.1.0 to 3.6.0. It also provides some of the\nfixed-api auxiliary routines.\n\nThese wrappers do not check for alignment of arrays.\nAlignment should be checked before these wrappers are used.\n\nRaw function pointers (Fortran-style pointer arguments):\n\n- {}\n\n\n"""\n\n# Within scipy, these wrappers can be used via relative or absolute cimport.\n# Examples:\n# from ..linalg cimport cython_lapack\n# from scipy.linalg cimport cython_lapack\n# cimport scipy.linalg.cython_lapack as cython_lapack\n# cimport ..linalg.cython_lapack as cython_lapack\n\n# Within scipy, if LAPACK functions are needed in C/C++/Fortran,\n# these wrappers should not be used.\n# The original libraries should be linked directly.\n\nfrom __future__ import absolute_import\n\ncdef extern from "fortran_defs.h":\n    pass\n\nfrom numpy cimport npy_complex64, npy_complex128\n\ncdef extern from "_lapack_subroutines.h":\n    # Function pointer type declarations for\n    # gees and gges families of functions.\n    ctypedef bint _cselect1(npy_complex64*)\n    ctypedef bint _cselect2(npy_complex64*, npy_complex64*)\n    ctypedef bint _dselect2(d*, d*)\n    ctypedef bint _dselect3(d*, d*, d*)\n    ctypedef bint _sselect2(s*, s*)\n    ctypedef bint _sselect3(s*, s*, s*)\n    ctypedef bint _zselect1(npy_complex128*)\n    ctypedef bint _zselect2(npy_complex128*, npy_complex128*)\n\n')
# Assigning a type to the variable 'lapack_pyx_preamble' (line 156)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 0), 'lapack_pyx_preamble', str_28665)

@norecursion
def make_lapack_pyx_preamble(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'make_lapack_pyx_preamble'
    module_type_store = module_type_store.open_function_context('make_lapack_pyx_preamble', 212, 0, False)
    
    # Passed parameters checking function
    make_lapack_pyx_preamble.stypy_localization = localization
    make_lapack_pyx_preamble.stypy_type_of_self = None
    make_lapack_pyx_preamble.stypy_type_store = module_type_store
    make_lapack_pyx_preamble.stypy_function_name = 'make_lapack_pyx_preamble'
    make_lapack_pyx_preamble.stypy_param_names_list = ['all_sigs']
    make_lapack_pyx_preamble.stypy_varargs_param_name = None
    make_lapack_pyx_preamble.stypy_kwargs_param_name = None
    make_lapack_pyx_preamble.stypy_call_defaults = defaults
    make_lapack_pyx_preamble.stypy_call_varargs = varargs
    make_lapack_pyx_preamble.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_lapack_pyx_preamble', ['all_sigs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_lapack_pyx_preamble', localization, ['all_sigs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_lapack_pyx_preamble(...)' code ##################

    
    # Assigning a ListComp to a Name (line 213):
    
    # Assigning a ListComp to a Name (line 213):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'all_sigs' (line 213)
    all_sigs_28670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 31), 'all_sigs')
    comprehension_28671 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 13), all_sigs_28670)
    # Assigning a type to the variable 'sig' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 13), 'sig', comprehension_28671)
    
    # Obtaining the type of the subscript
    int_28666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 17), 'int')
    # Getting the type of 'sig' (line 213)
    sig_28667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 13), 'sig')
    # Obtaining the member '__getitem__' of a type (line 213)
    getitem___28668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 13), sig_28667, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 213)
    subscript_call_result_28669 = invoke(stypy.reporting.localization.Localization(__file__, 213, 13), getitem___28668, int_28666)
    
    list_28672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 13), list_28672, subscript_call_result_28669)
    # Assigning a type to the variable 'names' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'names', list_28672)
    
    # Call to format(...): (line 214)
    # Processing the call arguments (line 214)
    
    # Call to join(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'names' (line 214)
    names_28677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 50), 'names', False)
    # Processing the call keyword arguments (line 214)
    kwargs_28678 = {}
    str_28675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 38), 'str', '\n- ')
    # Obtaining the member 'join' of a type (line 214)
    join_28676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 38), str_28675, 'join')
    # Calling join(args, kwargs) (line 214)
    join_call_result_28679 = invoke(stypy.reporting.localization.Localization(__file__, 214, 38), join_28676, *[names_28677], **kwargs_28678)
    
    # Processing the call keyword arguments (line 214)
    kwargs_28680 = {}
    # Getting the type of 'lapack_pyx_preamble' (line 214)
    lapack_pyx_preamble_28673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 11), 'lapack_pyx_preamble', False)
    # Obtaining the member 'format' of a type (line 214)
    format_28674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 11), lapack_pyx_preamble_28673, 'format')
    # Calling format(args, kwargs) (line 214)
    format_call_result_28681 = invoke(stypy.reporting.localization.Localization(__file__, 214, 11), format_28674, *[join_call_result_28679], **kwargs_28680)
    
    # Assigning a type to the variable 'stypy_return_type' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type', format_call_result_28681)
    
    # ################# End of 'make_lapack_pyx_preamble(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_lapack_pyx_preamble' in the type store
    # Getting the type of 'stypy_return_type' (line 212)
    stypy_return_type_28682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28682)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_lapack_pyx_preamble'
    return stypy_return_type_28682

# Assigning a type to the variable 'make_lapack_pyx_preamble' (line 212)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 0), 'make_lapack_pyx_preamble', make_lapack_pyx_preamble)

# Assigning a Str to a Name (line 216):

# Assigning a Str to a Name (line 216):
str_28683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, (-1)), 'str', '\n\n# Python-accessible wrappers for testing:\n\ncdef inline bint _is_contiguous(double[:,:] a, int axis) nogil:\n    return (a.strides[axis] == sizeof(a[0,0]) or a.shape[axis] == 1)\n\ncpdef float complex _test_cdotc(float complex[:] cx, float complex[:] cy) nogil:\n    cdef:\n        int n = cx.shape[0]\n        int incx = cx.strides[0] // sizeof(cx[0])\n        int incy = cy.strides[0] // sizeof(cy[0])\n    return cdotc(&n, &cx[0], &incx, &cy[0], &incy)\n\ncpdef float complex _test_cdotu(float complex[:] cx, float complex[:] cy) nogil:\n    cdef:\n        int n = cx.shape[0]\n        int incx = cx.strides[0] // sizeof(cx[0])\n        int incy = cy.strides[0] // sizeof(cy[0])\n    return cdotu(&n, &cx[0], &incx, &cy[0], &incy)\n\ncpdef double _test_dasum(double[:] dx) nogil:\n    cdef:\n        int n = dx.shape[0]\n        int incx = dx.strides[0] // sizeof(dx[0])\n    return dasum(&n, &dx[0], &incx)\n\ncpdef double _test_ddot(double[:] dx, double[:] dy) nogil:\n    cdef:\n        int n = dx.shape[0]\n        int incx = dx.strides[0] // sizeof(dx[0])\n        int incy = dy.strides[0] // sizeof(dy[0])\n    return ddot(&n, &dx[0], &incx, &dy[0], &incy)\n\ncpdef int _test_dgemm(double alpha, double[:,:] a, double[:,:] b, double beta,\n                double[:,:] c) nogil except -1:\n    cdef:\n        char *transa\n        char *transb\n        int m, n, k, lda, ldb, ldc\n        double *a0=&a[0,0]\n        double *b0=&b[0,0]\n        double *c0=&c[0,0]\n    # In the case that c is C contiguous, swap a and b and\n    # swap whether or not each of them is transposed.\n    # This can be done because a.dot(b) = b.T.dot(a.T).T.\n    if _is_contiguous(c, 1):\n        if _is_contiguous(a, 1):\n            transb = \'n\'\n            ldb = (&a[1,0]) - a0 if a.shape[0] > 1 else 1\n        elif _is_contiguous(a, 0):\n            transb = \'t\'\n            ldb = (&a[0,1]) - a0 if a.shape[1] > 1 else 1\n        else:\n            with gil:\n                raise ValueError("Input \'a\' is neither C nor Fortran contiguous.")\n        if _is_contiguous(b, 1):\n            transa = \'n\'\n            lda = (&b[1,0]) - b0 if b.shape[0] > 1 else 1\n        elif _is_contiguous(b, 0):\n            transa = \'t\'\n            lda = (&b[0,1]) - b0 if b.shape[1] > 1 else 1\n        else:\n            with gil:\n                raise ValueError("Input \'b\' is neither C nor Fortran contiguous.")\n        k = b.shape[0]\n        if k != a.shape[1]:\n            with gil:\n                raise ValueError("Shape mismatch in input arrays.")\n        m = b.shape[1]\n        n = a.shape[0]\n        if n != c.shape[0] or m != c.shape[1]:\n            with gil:\n                raise ValueError("Output array does not have the correct shape.")\n        ldc = (&c[1,0]) - c0 if c.shape[0] > 1 else 1\n        dgemm(transa, transb, &m, &n, &k, &alpha, b0, &lda, a0,\n                   &ldb, &beta, c0, &ldc)\n    elif _is_contiguous(c, 0):\n        if _is_contiguous(a, 1):\n            transa = \'t\'\n            lda = (&a[1,0]) - a0 if a.shape[0] > 1 else 1\n        elif _is_contiguous(a, 0):\n            transa = \'n\'\n            lda = (&a[0,1]) - a0 if a.shape[1] > 1 else 1\n        else:\n            with gil:\n                raise ValueError("Input \'a\' is neither C nor Fortran contiguous.")\n        if _is_contiguous(b, 1):\n            transb = \'t\'\n            ldb = (&b[1,0]) - b0 if b.shape[0] > 1 else 1\n        elif _is_contiguous(b, 0):\n            transb = \'n\'\n            ldb = (&b[0,1]) - b0 if b.shape[1] > 1 else 1\n        else:\n            with gil:\n                raise ValueError("Input \'b\' is neither C nor Fortran contiguous.")\n        m = a.shape[0]\n        k = a.shape[1]\n        if k != b.shape[0]:\n            with gil:\n                raise ValueError("Shape mismatch in input arrays.")\n        n = b.shape[1]\n        if m != c.shape[0] or n != c.shape[1]:\n            with gil:\n                raise ValueError("Output array does not have the correct shape.")\n        ldc = (&c[0,1]) - c0 if c.shape[1] > 1 else 1\n        dgemm(transa, transb, &m, &n, &k, &alpha, a0, &lda, b0,\n                   &ldb, &beta, c0, &ldc)\n    else:\n        with gil:\n            raise ValueError("Input \'c\' is neither C nor Fortran contiguous.")\n    return 0\n\ncpdef double _test_dnrm2(double[:] x) nogil:\n    cdef:\n        int n = x.shape[0]\n        int incx = x.strides[0] // sizeof(x[0])\n    return dnrm2(&n, &x[0], &incx)\n\ncpdef double _test_dzasum(double complex[:] zx) nogil:\n    cdef:\n        int n = zx.shape[0]\n        int incx = zx.strides[0] // sizeof(zx[0])\n    return dzasum(&n, &zx[0], &incx)\n\ncpdef double _test_dznrm2(double complex[:] x) nogil:\n    cdef:\n        int n = x.shape[0]\n        int incx = x.strides[0] // sizeof(x[0])\n    return dznrm2(&n, &x[0], &incx)\n\ncpdef int _test_icamax(float complex[:] cx) nogil:\n    cdef:\n        int n = cx.shape[0]\n        int incx = cx.strides[0] // sizeof(cx[0])\n    return icamax(&n, &cx[0], &incx)\n\ncpdef int _test_idamax(double[:] dx) nogil:\n    cdef:\n        int n = dx.shape[0]\n        int incx = dx.strides[0] // sizeof(dx[0])\n    return idamax(&n, &dx[0], &incx)\n\ncpdef int _test_isamax(float[:] sx) nogil:\n    cdef:\n        int n = sx.shape[0]\n        int incx = sx.strides[0] // sizeof(sx[0])\n    return isamax(&n, &sx[0], &incx)\n\ncpdef int _test_izamax(double complex[:] zx) nogil:\n    cdef:\n        int n = zx.shape[0]\n        int incx = zx.strides[0] // sizeof(zx[0])\n    return izamax(&n, &zx[0], &incx)\n\ncpdef float _test_sasum(float[:] sx) nogil:\n    cdef:\n        int n = sx.shape[0]\n        int incx = sx.shape[0] // sizeof(sx[0])\n    return sasum(&n, &sx[0], &incx)\n\ncpdef float _test_scasum(float complex[:] cx) nogil:\n    cdef:\n        int n = cx.shape[0]\n        int incx = cx.strides[0] // sizeof(cx[0])\n    return scasum(&n, &cx[0], &incx)\n\ncpdef float _test_scnrm2(float complex[:] x) nogil:\n    cdef:\n        int n = x.shape[0]\n        int incx = x.strides[0] // sizeof(x[0])\n    return scnrm2(&n, &x[0], &incx)\n\ncpdef float _test_sdot(float[:] sx, float[:] sy) nogil:\n    cdef:\n        int n = sx.shape[0]\n        int incx = sx.strides[0] // sizeof(sx[0])\n        int incy = sy.strides[0] // sizeof(sy[0])\n    return sdot(&n, &sx[0], &incx, &sy[0], &incy)\n\ncpdef float _test_snrm2(float[:] x) nogil:\n    cdef:\n        int n = x.shape[0]\n        int incx = x.shape[0] // sizeof(x[0])\n    return snrm2(&n, &x[0], &incx)\n\ncpdef double complex _test_zdotc(double complex[:] zx, double complex[:] zy) nogil:\n    cdef:\n        int n = zx.shape[0]\n        int incx = zx.strides[0] // sizeof(zx[0])\n        int incy = zy.strides[0] // sizeof(zy[0])\n    return zdotc(&n, &zx[0], &incx, &zy[0], &incy)\n\ncpdef double complex _test_zdotu(double complex[:] zx, double complex[:] zy) nogil:\n    cdef:\n        int n = zx.shape[0]\n        int incx = zx.strides[0] // sizeof(zx[0])\n        int incy = zy.strides[0] // sizeof(zy[0])\n    return zdotu(&n, &zx[0], &incx, &zy[0], &incy)\n')
# Assigning a type to the variable 'blas_py_wrappers' (line 216)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 0), 'blas_py_wrappers', str_28683)

@norecursion
def generate_blas_pyx(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'generate_blas_pyx'
    module_type_store = module_type_store.open_function_context('generate_blas_pyx', 418, 0, False)
    
    # Passed parameters checking function
    generate_blas_pyx.stypy_localization = localization
    generate_blas_pyx.stypy_type_of_self = None
    generate_blas_pyx.stypy_type_store = module_type_store
    generate_blas_pyx.stypy_function_name = 'generate_blas_pyx'
    generate_blas_pyx.stypy_param_names_list = ['func_sigs', 'sub_sigs', 'all_sigs', 'header_name']
    generate_blas_pyx.stypy_varargs_param_name = None
    generate_blas_pyx.stypy_kwargs_param_name = None
    generate_blas_pyx.stypy_call_defaults = defaults
    generate_blas_pyx.stypy_call_varargs = varargs
    generate_blas_pyx.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generate_blas_pyx', ['func_sigs', 'sub_sigs', 'all_sigs', 'header_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generate_blas_pyx', localization, ['func_sigs', 'sub_sigs', 'all_sigs', 'header_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generate_blas_pyx(...)' code ##################

    
    # Assigning a Call to a Name (line 419):
    
    # Assigning a Call to a Name (line 419):
    
    # Call to join(...): (line 419)
    # Processing the call arguments (line 419)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 419, 22, True)
    # Calculating comprehension expression
    # Getting the type of 'func_sigs' (line 419)
    func_sigs_28693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 66), 'func_sigs', False)
    comprehension_28694 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 22), func_sigs_28693)
    # Assigning a type to the variable 's' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 22), 's', comprehension_28694)
    
    # Call to pyx_decl_func(...): (line 419)
    # Getting the type of 's' (line 419)
    s_28687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 38), 's', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 419)
    tuple_28688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 419)
    # Adding element type (line 419)
    # Getting the type of 'header_name' (line 419)
    header_name_28689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 41), 'header_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 41), tuple_28688, header_name_28689)
    
    # Applying the binary operator '+' (line 419)
    result_add_28690 = python_operator(stypy.reporting.localization.Localization(__file__, 419, 38), '+', s_28687, tuple_28688)
    
    # Processing the call keyword arguments (line 419)
    kwargs_28691 = {}
    # Getting the type of 'pyx_decl_func' (line 419)
    pyx_decl_func_28686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 22), 'pyx_decl_func', False)
    # Calling pyx_decl_func(args, kwargs) (line 419)
    pyx_decl_func_call_result_28692 = invoke(stypy.reporting.localization.Localization(__file__, 419, 22), pyx_decl_func_28686, *[result_add_28690], **kwargs_28691)
    
    list_28695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 22), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 22), list_28695, pyx_decl_func_call_result_28692)
    # Processing the call keyword arguments (line 419)
    kwargs_28696 = {}
    str_28684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 12), 'str', '\n')
    # Obtaining the member 'join' of a type (line 419)
    join_28685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 12), str_28684, 'join')
    # Calling join(args, kwargs) (line 419)
    join_call_result_28697 = invoke(stypy.reporting.localization.Localization(__file__, 419, 12), join_28685, *[list_28695], **kwargs_28696)
    
    # Assigning a type to the variable 'funcs' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'funcs', join_call_result_28697)
    
    # Assigning a BinOp to a Name (line 420):
    
    # Assigning a BinOp to a Name (line 420):
    str_28698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 11), 'str', '\n')
    
    # Call to join(...): (line 420)
    # Processing the call arguments (line 420)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 420, 28, True)
    # Calculating comprehension expression
    # Getting the type of 'sub_sigs' (line 421)
    sub_sigs_28712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 37), 'sub_sigs', False)
    comprehension_28713 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 28), sub_sigs_28712)
    # Assigning a type to the variable 's' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 28), 's', comprehension_28713)
    
    # Call to pyx_decl_sub(...): (line 420)
    
    # Obtaining the type of the subscript
    int_28702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 47), 'int')
    slice_28703 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 420, 43), None, None, int_28702)
    # Getting the type of 's' (line 420)
    s_28704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 43), 's', False)
    # Obtaining the member '__getitem__' of a type (line 420)
    getitem___28705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 43), s_28704, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 420)
    subscript_call_result_28706 = invoke(stypy.reporting.localization.Localization(__file__, 420, 43), getitem___28705, slice_28703)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 420)
    tuple_28707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 51), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 420)
    # Adding element type (line 420)
    # Getting the type of 'header_name' (line 420)
    header_name_28708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 51), 'header_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 51), tuple_28707, header_name_28708)
    
    # Applying the binary operator '+' (line 420)
    result_add_28709 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 43), '+', subscript_call_result_28706, tuple_28707)
    
    # Processing the call keyword arguments (line 420)
    kwargs_28710 = {}
    # Getting the type of 'pyx_decl_sub' (line 420)
    pyx_decl_sub_28701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 28), 'pyx_decl_sub', False)
    # Calling pyx_decl_sub(args, kwargs) (line 420)
    pyx_decl_sub_call_result_28711 = invoke(stypy.reporting.localization.Localization(__file__, 420, 28), pyx_decl_sub_28701, *[result_add_28709], **kwargs_28710)
    
    list_28714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 28), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 420, 28), list_28714, pyx_decl_sub_call_result_28711)
    # Processing the call keyword arguments (line 420)
    kwargs_28715 = {}
    str_28699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 18), 'str', '\n')
    # Obtaining the member 'join' of a type (line 420)
    join_28700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 18), str_28699, 'join')
    # Calling join(args, kwargs) (line 420)
    join_call_result_28716 = invoke(stypy.reporting.localization.Localization(__file__, 420, 18), join_28700, *[list_28714], **kwargs_28715)
    
    # Applying the binary operator '+' (line 420)
    result_add_28717 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 11), '+', str_28698, join_call_result_28716)
    
    # Assigning a type to the variable 'subs' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'subs', result_add_28717)
    
    # Call to make_blas_pyx_preamble(...): (line 422)
    # Processing the call arguments (line 422)
    # Getting the type of 'all_sigs' (line 422)
    all_sigs_28719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 34), 'all_sigs', False)
    # Processing the call keyword arguments (line 422)
    kwargs_28720 = {}
    # Getting the type of 'make_blas_pyx_preamble' (line 422)
    make_blas_pyx_preamble_28718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 11), 'make_blas_pyx_preamble', False)
    # Calling make_blas_pyx_preamble(args, kwargs) (line 422)
    make_blas_pyx_preamble_call_result_28721 = invoke(stypy.reporting.localization.Localization(__file__, 422, 11), make_blas_pyx_preamble_28718, *[all_sigs_28719], **kwargs_28720)
    
    # Getting the type of 'funcs' (line 422)
    funcs_28722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 46), 'funcs')
    # Applying the binary operator '+' (line 422)
    result_add_28723 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 11), '+', make_blas_pyx_preamble_call_result_28721, funcs_28722)
    
    # Getting the type of 'subs' (line 422)
    subs_28724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 54), 'subs')
    # Applying the binary operator '+' (line 422)
    result_add_28725 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 52), '+', result_add_28723, subs_28724)
    
    # Getting the type of 'blas_py_wrappers' (line 422)
    blas_py_wrappers_28726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 61), 'blas_py_wrappers')
    # Applying the binary operator '+' (line 422)
    result_add_28727 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 59), '+', result_add_28725, blas_py_wrappers_28726)
    
    # Assigning a type to the variable 'stypy_return_type' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'stypy_return_type', result_add_28727)
    
    # ################# End of 'generate_blas_pyx(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generate_blas_pyx' in the type store
    # Getting the type of 'stypy_return_type' (line 418)
    stypy_return_type_28728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28728)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generate_blas_pyx'
    return stypy_return_type_28728

# Assigning a type to the variable 'generate_blas_pyx' (line 418)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 0), 'generate_blas_pyx', generate_blas_pyx)

# Assigning a Str to a Name (line 424):

# Assigning a Str to a Name (line 424):
str_28729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, (-1)), 'str', '\n\n# Python accessible wrappers for testing:\n\ndef _test_dlamch(cmach):\n    # This conversion is necessary to handle Python 3 strings.\n    cmach_bytes = bytes(cmach)\n    # Now that it is a bytes representation, a non-temporary variable\n    # must be passed as a part of the function call.\n    cdef char* cmach_char = cmach_bytes\n    return dlamch(cmach_char)\n\ndef _test_slamch(cmach):\n    # This conversion is necessary to handle Python 3 strings.\n    cmach_bytes = bytes(cmach)\n    # Now that it is a bytes representation, a non-temporary variable\n    # must be passed as a part of the function call.\n    cdef char* cmach_char = cmach_bytes\n    return slamch(cmach_char)\n')
# Assigning a type to the variable 'lapack_py_wrappers' (line 424)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 0), 'lapack_py_wrappers', str_28729)

@norecursion
def generate_lapack_pyx(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'generate_lapack_pyx'
    module_type_store = module_type_store.open_function_context('generate_lapack_pyx', 446, 0, False)
    
    # Passed parameters checking function
    generate_lapack_pyx.stypy_localization = localization
    generate_lapack_pyx.stypy_type_of_self = None
    generate_lapack_pyx.stypy_type_store = module_type_store
    generate_lapack_pyx.stypy_function_name = 'generate_lapack_pyx'
    generate_lapack_pyx.stypy_param_names_list = ['func_sigs', 'sub_sigs', 'all_sigs', 'header_name']
    generate_lapack_pyx.stypy_varargs_param_name = None
    generate_lapack_pyx.stypy_kwargs_param_name = None
    generate_lapack_pyx.stypy_call_defaults = defaults
    generate_lapack_pyx.stypy_call_varargs = varargs
    generate_lapack_pyx.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generate_lapack_pyx', ['func_sigs', 'sub_sigs', 'all_sigs', 'header_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generate_lapack_pyx', localization, ['func_sigs', 'sub_sigs', 'all_sigs', 'header_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generate_lapack_pyx(...)' code ##################

    
    # Assigning a Call to a Name (line 447):
    
    # Assigning a Call to a Name (line 447):
    
    # Call to join(...): (line 447)
    # Processing the call arguments (line 447)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 447, 22, True)
    # Calculating comprehension expression
    # Getting the type of 'func_sigs' (line 447)
    func_sigs_28739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 66), 'func_sigs', False)
    comprehension_28740 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 22), func_sigs_28739)
    # Assigning a type to the variable 's' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 22), 's', comprehension_28740)
    
    # Call to pyx_decl_func(...): (line 447)
    # Getting the type of 's' (line 447)
    s_28733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 38), 's', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 447)
    tuple_28734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 447)
    # Adding element type (line 447)
    # Getting the type of 'header_name' (line 447)
    header_name_28735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 41), 'header_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 41), tuple_28734, header_name_28735)
    
    # Applying the binary operator '+' (line 447)
    result_add_28736 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 38), '+', s_28733, tuple_28734)
    
    # Processing the call keyword arguments (line 447)
    kwargs_28737 = {}
    # Getting the type of 'pyx_decl_func' (line 447)
    pyx_decl_func_28732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 22), 'pyx_decl_func', False)
    # Calling pyx_decl_func(args, kwargs) (line 447)
    pyx_decl_func_call_result_28738 = invoke(stypy.reporting.localization.Localization(__file__, 447, 22), pyx_decl_func_28732, *[result_add_28736], **kwargs_28737)
    
    list_28741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 22), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 447, 22), list_28741, pyx_decl_func_call_result_28738)
    # Processing the call keyword arguments (line 447)
    kwargs_28742 = {}
    str_28730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 12), 'str', '\n')
    # Obtaining the member 'join' of a type (line 447)
    join_28731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 12), str_28730, 'join')
    # Calling join(args, kwargs) (line 447)
    join_call_result_28743 = invoke(stypy.reporting.localization.Localization(__file__, 447, 12), join_28731, *[list_28741], **kwargs_28742)
    
    # Assigning a type to the variable 'funcs' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'funcs', join_call_result_28743)
    
    # Assigning a BinOp to a Name (line 448):
    
    # Assigning a BinOp to a Name (line 448):
    str_28744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 11), 'str', '\n')
    
    # Call to join(...): (line 448)
    # Processing the call arguments (line 448)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 448, 28, True)
    # Calculating comprehension expression
    # Getting the type of 'sub_sigs' (line 449)
    sub_sigs_28758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 37), 'sub_sigs', False)
    comprehension_28759 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 28), sub_sigs_28758)
    # Assigning a type to the variable 's' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 28), 's', comprehension_28759)
    
    # Call to pyx_decl_sub(...): (line 448)
    
    # Obtaining the type of the subscript
    int_28748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 47), 'int')
    slice_28749 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 448, 43), None, None, int_28748)
    # Getting the type of 's' (line 448)
    s_28750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 43), 's', False)
    # Obtaining the member '__getitem__' of a type (line 448)
    getitem___28751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 43), s_28750, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 448)
    subscript_call_result_28752 = invoke(stypy.reporting.localization.Localization(__file__, 448, 43), getitem___28751, slice_28749)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 448)
    tuple_28753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 51), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 448)
    # Adding element type (line 448)
    # Getting the type of 'header_name' (line 448)
    header_name_28754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 51), 'header_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 51), tuple_28753, header_name_28754)
    
    # Applying the binary operator '+' (line 448)
    result_add_28755 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 43), '+', subscript_call_result_28752, tuple_28753)
    
    # Processing the call keyword arguments (line 448)
    kwargs_28756 = {}
    # Getting the type of 'pyx_decl_sub' (line 448)
    pyx_decl_sub_28747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 28), 'pyx_decl_sub', False)
    # Calling pyx_decl_sub(args, kwargs) (line 448)
    pyx_decl_sub_call_result_28757 = invoke(stypy.reporting.localization.Localization(__file__, 448, 28), pyx_decl_sub_28747, *[result_add_28755], **kwargs_28756)
    
    list_28760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 28), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 448, 28), list_28760, pyx_decl_sub_call_result_28757)
    # Processing the call keyword arguments (line 448)
    kwargs_28761 = {}
    str_28745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, 18), 'str', '\n')
    # Obtaining the member 'join' of a type (line 448)
    join_28746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 18), str_28745, 'join')
    # Calling join(args, kwargs) (line 448)
    join_call_result_28762 = invoke(stypy.reporting.localization.Localization(__file__, 448, 18), join_28746, *[list_28760], **kwargs_28761)
    
    # Applying the binary operator '+' (line 448)
    result_add_28763 = python_operator(stypy.reporting.localization.Localization(__file__, 448, 11), '+', str_28744, join_call_result_28762)
    
    # Assigning a type to the variable 'subs' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 4), 'subs', result_add_28763)
    
    # Assigning a Call to a Name (line 450):
    
    # Assigning a Call to a Name (line 450):
    
    # Call to make_lapack_pyx_preamble(...): (line 450)
    # Processing the call arguments (line 450)
    # Getting the type of 'all_sigs' (line 450)
    all_sigs_28765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 40), 'all_sigs', False)
    # Processing the call keyword arguments (line 450)
    kwargs_28766 = {}
    # Getting the type of 'make_lapack_pyx_preamble' (line 450)
    make_lapack_pyx_preamble_28764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 15), 'make_lapack_pyx_preamble', False)
    # Calling make_lapack_pyx_preamble(args, kwargs) (line 450)
    make_lapack_pyx_preamble_call_result_28767 = invoke(stypy.reporting.localization.Localization(__file__, 450, 15), make_lapack_pyx_preamble_28764, *[all_sigs_28765], **kwargs_28766)
    
    # Assigning a type to the variable 'preamble' (line 450)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'preamble', make_lapack_pyx_preamble_call_result_28767)
    # Getting the type of 'preamble' (line 451)
    preamble_28768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 11), 'preamble')
    # Getting the type of 'funcs' (line 451)
    funcs_28769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 22), 'funcs')
    # Applying the binary operator '+' (line 451)
    result_add_28770 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 11), '+', preamble_28768, funcs_28769)
    
    # Getting the type of 'subs' (line 451)
    subs_28771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 30), 'subs')
    # Applying the binary operator '+' (line 451)
    result_add_28772 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 28), '+', result_add_28770, subs_28771)
    
    # Getting the type of 'lapack_py_wrappers' (line 451)
    lapack_py_wrappers_28773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 37), 'lapack_py_wrappers')
    # Applying the binary operator '+' (line 451)
    result_add_28774 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 35), '+', result_add_28772, lapack_py_wrappers_28773)
    
    # Assigning a type to the variable 'stypy_return_type' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'stypy_return_type', result_add_28774)
    
    # ################# End of 'generate_lapack_pyx(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generate_lapack_pyx' in the type store
    # Getting the type of 'stypy_return_type' (line 446)
    stypy_return_type_28775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28775)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generate_lapack_pyx'
    return stypy_return_type_28775

# Assigning a type to the variable 'generate_lapack_pyx' (line 446)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 0), 'generate_lapack_pyx', generate_lapack_pyx)

# Assigning a Str to a Name (line 453):

# Assigning a Str to a Name (line 453):
str_28776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, (-1)), 'str', 'ctypedef {ret_type} {name}_t({args}) nogil\ncdef {name}_t *{name}_f\n')
# Assigning a type to the variable 'pxd_template' (line 453)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 0), 'pxd_template', str_28776)

# Assigning a Str to a Name (line 456):

# Assigning a Str to a Name (line 456):
str_28777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, (-1)), 'str', 'cdef {ret_type} {name}({args}) nogil\n')
# Assigning a type to the variable 'pxd_template' (line 456)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 0), 'pxd_template', str_28777)

@norecursion
def pxd_decl(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pxd_decl'
    module_type_store = module_type_store.open_function_context('pxd_decl', 460, 0, False)
    
    # Passed parameters checking function
    pxd_decl.stypy_localization = localization
    pxd_decl.stypy_type_of_self = None
    pxd_decl.stypy_type_store = module_type_store
    pxd_decl.stypy_function_name = 'pxd_decl'
    pxd_decl.stypy_param_names_list = ['name', 'ret_type', 'args']
    pxd_decl.stypy_varargs_param_name = None
    pxd_decl.stypy_kwargs_param_name = None
    pxd_decl.stypy_call_defaults = defaults
    pxd_decl.stypy_call_varargs = varargs
    pxd_decl.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pxd_decl', ['name', 'ret_type', 'args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pxd_decl', localization, ['name', 'ret_type', 'args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pxd_decl(...)' code ##################

    
    # Assigning a Call to a Name (line 461):
    
    # Assigning a Call to a Name (line 461):
    
    # Call to replace(...): (line 461)
    # Processing the call arguments (line 461)
    str_28785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 53), 'str', '*in,')
    str_28786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 61), 'str', '*in_,')
    # Processing the call keyword arguments (line 461)
    kwargs_28787 = {}
    
    # Call to replace(...): (line 461)
    # Processing the call arguments (line 461)
    str_28780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 24), 'str', 'lambda')
    str_28781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 34), 'str', 'lambda_')
    # Processing the call keyword arguments (line 461)
    kwargs_28782 = {}
    # Getting the type of 'args' (line 461)
    args_28778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 11), 'args', False)
    # Obtaining the member 'replace' of a type (line 461)
    replace_28779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 11), args_28778, 'replace')
    # Calling replace(args, kwargs) (line 461)
    replace_call_result_28783 = invoke(stypy.reporting.localization.Localization(__file__, 461, 11), replace_28779, *[str_28780, str_28781], **kwargs_28782)
    
    # Obtaining the member 'replace' of a type (line 461)
    replace_28784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 11), replace_call_result_28783, 'replace')
    # Calling replace(args, kwargs) (line 461)
    replace_call_result_28788 = invoke(stypy.reporting.localization.Localization(__file__, 461, 11), replace_28784, *[str_28785, str_28786], **kwargs_28787)
    
    # Assigning a type to the variable 'args' (line 461)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'args', replace_call_result_28788)
    
    # Call to format(...): (line 462)
    # Processing the call keyword arguments (line 462)
    # Getting the type of 'name' (line 462)
    name_28791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 36), 'name', False)
    keyword_28792 = name_28791
    # Getting the type of 'ret_type' (line 462)
    ret_type_28793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 51), 'ret_type', False)
    keyword_28794 = ret_type_28793
    # Getting the type of 'args' (line 462)
    args_28795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 66), 'args', False)
    keyword_28796 = args_28795
    kwargs_28797 = {'ret_type': keyword_28794, 'args': keyword_28796, 'name': keyword_28792}
    # Getting the type of 'pxd_template' (line 462)
    pxd_template_28789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 11), 'pxd_template', False)
    # Obtaining the member 'format' of a type (line 462)
    format_28790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 11), pxd_template_28789, 'format')
    # Calling format(args, kwargs) (line 462)
    format_call_result_28798 = invoke(stypy.reporting.localization.Localization(__file__, 462, 11), format_28790, *[], **kwargs_28797)
    
    # Assigning a type to the variable 'stypy_return_type' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'stypy_return_type', format_call_result_28798)
    
    # ################# End of 'pxd_decl(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pxd_decl' in the type store
    # Getting the type of 'stypy_return_type' (line 460)
    stypy_return_type_28799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28799)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pxd_decl'
    return stypy_return_type_28799

# Assigning a type to the variable 'pxd_decl' (line 460)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 0), 'pxd_decl', pxd_decl)

# Assigning a Str to a Name (line 464):

# Assigning a Str to a Name (line 464):
str_28800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, (-1)), 'str', '# Within scipy, these wrappers can be used via relative or absolute cimport.\n# Examples:\n# from ..linalg cimport cython_blas\n# from scipy.linalg cimport cython_blas\n# cimport scipy.linalg.cython_blas as cython_blas\n# cimport ..linalg.cython_blas as cython_blas\n\n# Within scipy, if BLAS functions are needed in C/C++/Fortran,\n# these wrappers should not be used.\n# The original libraries should be linked directly.\n\nctypedef float s\nctypedef double d\nctypedef float complex c\nctypedef double complex z\n\n')
# Assigning a type to the variable 'blas_pxd_preamble' (line 464)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 0), 'blas_pxd_preamble', str_28800)

@norecursion
def generate_blas_pxd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'generate_blas_pxd'
    module_type_store = module_type_store.open_function_context('generate_blas_pxd', 483, 0, False)
    
    # Passed parameters checking function
    generate_blas_pxd.stypy_localization = localization
    generate_blas_pxd.stypy_type_of_self = None
    generate_blas_pxd.stypy_type_store = module_type_store
    generate_blas_pxd.stypy_function_name = 'generate_blas_pxd'
    generate_blas_pxd.stypy_param_names_list = ['all_sigs']
    generate_blas_pxd.stypy_varargs_param_name = None
    generate_blas_pxd.stypy_kwargs_param_name = None
    generate_blas_pxd.stypy_call_defaults = defaults
    generate_blas_pxd.stypy_call_varargs = varargs
    generate_blas_pxd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generate_blas_pxd', ['all_sigs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generate_blas_pxd', localization, ['all_sigs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generate_blas_pxd(...)' code ##################

    
    # Assigning a Call to a Name (line 484):
    
    # Assigning a Call to a Name (line 484):
    
    # Call to join(...): (line 484)
    # Processing the call arguments (line 484)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 484, 21, True)
    # Calculating comprehension expression
    # Getting the type of 'all_sigs' (line 484)
    all_sigs_28807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 47), 'all_sigs', False)
    comprehension_28808 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 21), all_sigs_28807)
    # Assigning a type to the variable 'sig' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 21), 'sig', comprehension_28808)
    
    # Call to pxd_decl(...): (line 484)
    # Getting the type of 'sig' (line 484)
    sig_28804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 31), 'sig', False)
    # Processing the call keyword arguments (line 484)
    kwargs_28805 = {}
    # Getting the type of 'pxd_decl' (line 484)
    pxd_decl_28803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 21), 'pxd_decl', False)
    # Calling pxd_decl(args, kwargs) (line 484)
    pxd_decl_call_result_28806 = invoke(stypy.reporting.localization.Localization(__file__, 484, 21), pxd_decl_28803, *[sig_28804], **kwargs_28805)
    
    list_28809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 21), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 484, 21), list_28809, pxd_decl_call_result_28806)
    # Processing the call keyword arguments (line 484)
    kwargs_28810 = {}
    str_28801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 11), 'str', '\n')
    # Obtaining the member 'join' of a type (line 484)
    join_28802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 11), str_28801, 'join')
    # Calling join(args, kwargs) (line 484)
    join_call_result_28811 = invoke(stypy.reporting.localization.Localization(__file__, 484, 11), join_28802, *[list_28809], **kwargs_28810)
    
    # Assigning a type to the variable 'body' (line 484)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 4), 'body', join_call_result_28811)
    # Getting the type of 'blas_pxd_preamble' (line 485)
    blas_pxd_preamble_28812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 11), 'blas_pxd_preamble')
    # Getting the type of 'body' (line 485)
    body_28813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 31), 'body')
    # Applying the binary operator '+' (line 485)
    result_add_28814 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 11), '+', blas_pxd_preamble_28812, body_28813)
    
    # Assigning a type to the variable 'stypy_return_type' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 4), 'stypy_return_type', result_add_28814)
    
    # ################# End of 'generate_blas_pxd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generate_blas_pxd' in the type store
    # Getting the type of 'stypy_return_type' (line 483)
    stypy_return_type_28815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28815)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generate_blas_pxd'
    return stypy_return_type_28815

# Assigning a type to the variable 'generate_blas_pxd' (line 483)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 0), 'generate_blas_pxd', generate_blas_pxd)

# Assigning a Str to a Name (line 487):

# Assigning a Str to a Name (line 487):
str_28816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, (-1)), 'str', '# Within scipy, these wrappers can be used via relative or absolute cimport.\n# Examples:\n# from ..linalg cimport cython_lapack\n# from scipy.linalg cimport cython_lapack\n# cimport scipy.linalg.cython_lapack as cython_lapack\n# cimport ..linalg.cython_lapack as cython_lapack\n\n# Within scipy, if LAPACK functions are needed in C/C++/Fortran,\n# these wrappers should not be used.\n# The original libraries should be linked directly.\n\nctypedef float s\nctypedef double d\nctypedef float complex c\nctypedef double complex z\n\n# Function pointer type declarations for\n# gees and gges families of functions.\nctypedef bint cselect1(c*)\nctypedef bint cselect2(c*, c*)\nctypedef bint dselect2(d*, d*)\nctypedef bint dselect3(d*, d*, d*)\nctypedef bint sselect2(s*, s*)\nctypedef bint sselect3(s*, s*, s*)\nctypedef bint zselect1(z*)\nctypedef bint zselect2(z*, z*)\n\n')
# Assigning a type to the variable 'lapack_pxd_preamble' (line 487)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 0), 'lapack_pxd_preamble', str_28816)

@norecursion
def generate_lapack_pxd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'generate_lapack_pxd'
    module_type_store = module_type_store.open_function_context('generate_lapack_pxd', 517, 0, False)
    
    # Passed parameters checking function
    generate_lapack_pxd.stypy_localization = localization
    generate_lapack_pxd.stypy_type_of_self = None
    generate_lapack_pxd.stypy_type_store = module_type_store
    generate_lapack_pxd.stypy_function_name = 'generate_lapack_pxd'
    generate_lapack_pxd.stypy_param_names_list = ['all_sigs']
    generate_lapack_pxd.stypy_varargs_param_name = None
    generate_lapack_pxd.stypy_kwargs_param_name = None
    generate_lapack_pxd.stypy_call_defaults = defaults
    generate_lapack_pxd.stypy_call_varargs = varargs
    generate_lapack_pxd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generate_lapack_pxd', ['all_sigs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generate_lapack_pxd', localization, ['all_sigs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generate_lapack_pxd(...)' code ##################

    # Getting the type of 'lapack_pxd_preamble' (line 518)
    lapack_pxd_preamble_28817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 11), 'lapack_pxd_preamble')
    
    # Call to join(...): (line 518)
    # Processing the call arguments (line 518)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 518, 43, True)
    # Calculating comprehension expression
    # Getting the type of 'all_sigs' (line 518)
    all_sigs_28824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 69), 'all_sigs', False)
    comprehension_28825 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 43), all_sigs_28824)
    # Assigning a type to the variable 'sig' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 43), 'sig', comprehension_28825)
    
    # Call to pxd_decl(...): (line 518)
    # Getting the type of 'sig' (line 518)
    sig_28821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 53), 'sig', False)
    # Processing the call keyword arguments (line 518)
    kwargs_28822 = {}
    # Getting the type of 'pxd_decl' (line 518)
    pxd_decl_28820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 43), 'pxd_decl', False)
    # Calling pxd_decl(args, kwargs) (line 518)
    pxd_decl_call_result_28823 = invoke(stypy.reporting.localization.Localization(__file__, 518, 43), pxd_decl_28820, *[sig_28821], **kwargs_28822)
    
    list_28826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 43), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 43), list_28826, pxd_decl_call_result_28823)
    # Processing the call keyword arguments (line 518)
    kwargs_28827 = {}
    str_28818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 33), 'str', '\n')
    # Obtaining the member 'join' of a type (line 518)
    join_28819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 33), str_28818, 'join')
    # Calling join(args, kwargs) (line 518)
    join_call_result_28828 = invoke(stypy.reporting.localization.Localization(__file__, 518, 33), join_28819, *[list_28826], **kwargs_28827)
    
    # Applying the binary operator '+' (line 518)
    result_add_28829 = python_operator(stypy.reporting.localization.Localization(__file__, 518, 11), '+', lapack_pxd_preamble_28817, join_call_result_28828)
    
    # Assigning a type to the variable 'stypy_return_type' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 'stypy_return_type', result_add_28829)
    
    # ################# End of 'generate_lapack_pxd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generate_lapack_pxd' in the type store
    # Getting the type of 'stypy_return_type' (line 517)
    stypy_return_type_28830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28830)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generate_lapack_pxd'
    return stypy_return_type_28830

# Assigning a type to the variable 'generate_lapack_pxd' (line 517)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 0), 'generate_lapack_pxd', generate_lapack_pxd)

# Assigning a Str to a Name (line 520):

# Assigning a Str to a Name (line 520):
str_28831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, (-1)), 'str', '      subroutine {name}wrp(ret, {argnames})\n        external {wrapper}\n        {ret_type} {wrapper}\n        {ret_type} ret\n        {argdecls}\n        ret = {wrapper}({argnames})\n      end\n')
# Assigning a type to the variable 'fortran_template' (line 520)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 0), 'fortran_template', str_28831)

# Assigning a Dict to a Name (line 529):

# Assigning a Dict to a Name (line 529):

# Obtaining an instance of the builtin type 'dict' (line 529)
dict_28832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 7), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 529)
# Adding element type (key, value) (line 529)
str_28833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 8), 'str', 'work')
str_28834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 16), 'str', '(*)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 7), dict_28832, (str_28833, str_28834))
# Adding element type (key, value) (line 529)
str_28835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 23), 'str', 'ab')
str_28836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 29), 'str', '(ldab,*)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 7), dict_28832, (str_28835, str_28836))
# Adding element type (key, value) (line 529)
str_28837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 41), 'str', 'a')
str_28838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 46), 'str', '(lda,*)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 7), dict_28832, (str_28837, str_28838))
# Adding element type (key, value) (line 529)
str_28839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 57), 'str', 'dl')
str_28840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 63), 'str', '(*)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 7), dict_28832, (str_28839, str_28840))
# Adding element type (key, value) (line 529)
str_28841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 8), 'str', 'd')
str_28842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 13), 'str', '(*)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 7), dict_28832, (str_28841, str_28842))
# Adding element type (key, value) (line 529)
str_28843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 20), 'str', 'du')
str_28844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 26), 'str', '(*)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 7), dict_28832, (str_28843, str_28844))
# Adding element type (key, value) (line 529)
str_28845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 33), 'str', 'ap')
str_28846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 39), 'str', '(*)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 7), dict_28832, (str_28845, str_28846))
# Adding element type (key, value) (line 529)
str_28847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 46), 'str', 'e')
str_28848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 51), 'str', '(*)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 7), dict_28832, (str_28847, str_28848))
# Adding element type (key, value) (line 529)
str_28849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 58), 'str', 'lld')
str_28850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 65), 'str', '(*)')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 7), dict_28832, (str_28849, str_28850))

# Assigning a type to the variable 'dims' (line 529)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 0), 'dims', dict_28832)

@norecursion
def process_fortran_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'process_fortran_name'
    module_type_store = module_type_store.open_function_context('process_fortran_name', 533, 0, False)
    
    # Passed parameters checking function
    process_fortran_name.stypy_localization = localization
    process_fortran_name.stypy_type_of_self = None
    process_fortran_name.stypy_type_store = module_type_store
    process_fortran_name.stypy_function_name = 'process_fortran_name'
    process_fortran_name.stypy_param_names_list = ['name', 'funcname']
    process_fortran_name.stypy_varargs_param_name = None
    process_fortran_name.stypy_kwargs_param_name = None
    process_fortran_name.stypy_call_defaults = defaults
    process_fortran_name.stypy_call_varargs = varargs
    process_fortran_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'process_fortran_name', ['name', 'funcname'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'process_fortran_name', localization, ['name', 'funcname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'process_fortran_name(...)' code ##################

    
    
    str_28851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 7), 'str', 'inc')
    # Getting the type of 'name' (line 534)
    name_28852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 16), 'name')
    # Applying the binary operator 'in' (line 534)
    result_contains_28853 = python_operator(stypy.reporting.localization.Localization(__file__, 534, 7), 'in', str_28851, name_28852)
    
    # Testing the type of an if condition (line 534)
    if_condition_28854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 534, 4), result_contains_28853)
    # Assigning a type to the variable 'if_condition_28854' (line 534)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 4), 'if_condition_28854', if_condition_28854)
    # SSA begins for if statement (line 534)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'name' (line 535)
    name_28855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 15), 'name')
    # Assigning a type to the variable 'stypy_return_type' (line 535)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'stypy_return_type', name_28855)
    # SSA join for if statement (line 534)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 536):
    
    # Assigning a List to a Name (line 536):
    
    # Obtaining an instance of the builtin type 'list' (line 536)
    list_28856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 536)
    # Adding element type (line 536)
    str_28857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 21), 'str', 'ladiv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 20), list_28856, str_28857)
    # Adding element type (line 536)
    str_28858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 30), 'str', 'lapy2')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 20), list_28856, str_28858)
    # Adding element type (line 536)
    str_28859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 39), 'str', 'lapy3')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 536, 20), list_28856, str_28859)
    
    # Assigning a type to the variable 'xy_exclusions' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'xy_exclusions', list_28856)
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    str_28860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 8), 'str', 'x')
    # Getting the type of 'name' (line 537)
    name_28861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 15), 'name')
    # Applying the binary operator 'in' (line 537)
    result_contains_28862 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 8), 'in', str_28860, name_28861)
    
    
    str_28863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 23), 'str', 'y')
    # Getting the type of 'name' (line 537)
    name_28864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 30), 'name')
    # Applying the binary operator 'in' (line 537)
    result_contains_28865 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 23), 'in', str_28863, name_28864)
    
    # Applying the binary operator 'or' (line 537)
    result_or_keyword_28866 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 8), 'or', result_contains_28862, result_contains_28865)
    
    
    
    # Obtaining the type of the subscript
    int_28867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 49), 'int')
    slice_28868 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 537, 40), int_28867, None, None)
    # Getting the type of 'funcname' (line 537)
    funcname_28869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 40), 'funcname')
    # Obtaining the member '__getitem__' of a type (line 537)
    getitem___28870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 40), funcname_28869, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 537)
    subscript_call_result_28871 = invoke(stypy.reporting.localization.Localization(__file__, 537, 40), getitem___28870, slice_28868)
    
    # Getting the type of 'xy_exclusions' (line 537)
    xy_exclusions_28872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 60), 'xy_exclusions')
    # Applying the binary operator 'notin' (line 537)
    result_contains_28873 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 40), 'notin', subscript_call_result_28871, xy_exclusions_28872)
    
    # Applying the binary operator 'and' (line 537)
    result_and_keyword_28874 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 7), 'and', result_or_keyword_28866, result_contains_28873)
    
    # Testing the type of an if condition (line 537)
    if_condition_28875 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 537, 4), result_and_keyword_28874)
    # Assigning a type to the variable 'if_condition_28875' (line 537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'if_condition_28875', if_condition_28875)
    # SSA begins for if statement (line 537)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'name' (line 538)
    name_28876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 15), 'name')
    str_28877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 22), 'str', '(n)')
    # Applying the binary operator '+' (line 538)
    result_add_28878 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 15), '+', name_28876, str_28877)
    
    # Assigning a type to the variable 'stypy_return_type' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'stypy_return_type', result_add_28878)
    # SSA join for if statement (line 537)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'name' (line 539)
    name_28879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 7), 'name')
    # Getting the type of 'dims' (line 539)
    dims_28880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 15), 'dims')
    # Applying the binary operator 'in' (line 539)
    result_contains_28881 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 7), 'in', name_28879, dims_28880)
    
    # Testing the type of an if condition (line 539)
    if_condition_28882 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 539, 4), result_contains_28881)
    # Assigning a type to the variable 'if_condition_28882' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'if_condition_28882', if_condition_28882)
    # SSA begins for if statement (line 539)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'name' (line 540)
    name_28883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 15), 'name')
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 540)
    name_28884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 27), 'name')
    # Getting the type of 'dims' (line 540)
    dims_28885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 22), 'dims')
    # Obtaining the member '__getitem__' of a type (line 540)
    getitem___28886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 22), dims_28885, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 540)
    subscript_call_result_28887 = invoke(stypy.reporting.localization.Localization(__file__, 540, 22), getitem___28886, name_28884)
    
    # Applying the binary operator '+' (line 540)
    result_add_28888 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 15), '+', name_28883, subscript_call_result_28887)
    
    # Assigning a type to the variable 'stypy_return_type' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'stypy_return_type', result_add_28888)
    # SSA join for if statement (line 539)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'name' (line 541)
    name_28889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 11), 'name')
    # Assigning a type to the variable 'stypy_return_type' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'stypy_return_type', name_28889)
    
    # ################# End of 'process_fortran_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'process_fortran_name' in the type store
    # Getting the type of 'stypy_return_type' (line 533)
    stypy_return_type_28890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28890)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'process_fortran_name'
    return stypy_return_type_28890

# Assigning a type to the variable 'process_fortran_name' (line 533)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 0), 'process_fortran_name', process_fortran_name)

@norecursion
def fort_subroutine_wrapper(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fort_subroutine_wrapper'
    module_type_store = module_type_store.open_function_context('fort_subroutine_wrapper', 544, 0, False)
    
    # Passed parameters checking function
    fort_subroutine_wrapper.stypy_localization = localization
    fort_subroutine_wrapper.stypy_type_of_self = None
    fort_subroutine_wrapper.stypy_type_store = module_type_store
    fort_subroutine_wrapper.stypy_function_name = 'fort_subroutine_wrapper'
    fort_subroutine_wrapper.stypy_param_names_list = ['name', 'ret_type', 'args']
    fort_subroutine_wrapper.stypy_varargs_param_name = None
    fort_subroutine_wrapper.stypy_kwargs_param_name = None
    fort_subroutine_wrapper.stypy_call_defaults = defaults
    fort_subroutine_wrapper.stypy_call_varargs = varargs
    fort_subroutine_wrapper.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fort_subroutine_wrapper', ['name', 'ret_type', 'args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fort_subroutine_wrapper', localization, ['name', 'ret_type', 'args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fort_subroutine_wrapper(...)' code ##################

    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_28891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 12), 'int')
    # Getting the type of 'name' (line 545)
    name_28892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 7), 'name')
    # Obtaining the member '__getitem__' of a type (line 545)
    getitem___28893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 7), name_28892, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 545)
    subscript_call_result_28894 = invoke(stypy.reporting.localization.Localization(__file__, 545, 7), getitem___28893, int_28891)
    
    
    # Obtaining an instance of the builtin type 'list' (line 545)
    list_28895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 545)
    # Adding element type (line 545)
    str_28896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 19), 'str', 'c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 18), list_28895, str_28896)
    # Adding element type (line 545)
    str_28897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 24), 'str', 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 18), list_28895, str_28897)
    
    # Applying the binary operator 'in' (line 545)
    result_contains_28898 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 7), 'in', subscript_call_result_28894, list_28895)
    
    
    # Getting the type of 'name' (line 545)
    name_28899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 32), 'name')
    
    # Obtaining an instance of the builtin type 'list' (line 545)
    list_28900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 545)
    # Adding element type (line 545)
    str_28901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 41), 'str', 'zladiv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 40), list_28900, str_28901)
    # Adding element type (line 545)
    str_28902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 51), 'str', 'zdotu')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 40), list_28900, str_28902)
    # Adding element type (line 545)
    str_28903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 60), 'str', 'zdotc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 545, 40), list_28900, str_28903)
    
    # Applying the binary operator 'in' (line 545)
    result_contains_28904 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 32), 'in', name_28899, list_28900)
    
    # Applying the binary operator 'or' (line 545)
    result_or_keyword_28905 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 7), 'or', result_contains_28898, result_contains_28904)
    
    # Testing the type of an if condition (line 545)
    if_condition_28906 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 545, 4), result_or_keyword_28905)
    # Assigning a type to the variable 'if_condition_28906' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'if_condition_28906', if_condition_28906)
    # SSA begins for if statement (line 545)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 546):
    
    # Assigning a BinOp to a Name (line 546):
    str_28907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 18), 'str', 'w')
    # Getting the type of 'name' (line 546)
    name_28908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 24), 'name')
    # Applying the binary operator '+' (line 546)
    result_add_28909 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 18), '+', str_28907, name_28908)
    
    # Assigning a type to the variable 'wrapper' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'wrapper', result_add_28909)
    # SSA branch for the else part of an if statement (line 545)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 548):
    
    # Assigning a Name to a Name (line 548):
    # Getting the type of 'name' (line 548)
    name_28910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 18), 'name')
    # Assigning a type to the variable 'wrapper' (line 548)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'wrapper', name_28910)
    # SSA join for if statement (line 545)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 549):
    
    # Assigning a Subscript to a Name (line 549):
    
    # Obtaining the type of the subscript
    int_28911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 4), 'int')
    
    # Call to arg_names_and_types(...): (line 549)
    # Processing the call arguments (line 549)
    # Getting the type of 'args' (line 549)
    args_28913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 39), 'args', False)
    # Processing the call keyword arguments (line 549)
    kwargs_28914 = {}
    # Getting the type of 'arg_names_and_types' (line 549)
    arg_names_and_types_28912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 19), 'arg_names_and_types', False)
    # Calling arg_names_and_types(args, kwargs) (line 549)
    arg_names_and_types_call_result_28915 = invoke(stypy.reporting.localization.Localization(__file__, 549, 19), arg_names_and_types_28912, *[args_28913], **kwargs_28914)
    
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___28916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 4), arg_names_and_types_call_result_28915, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_28917 = invoke(stypy.reporting.localization.Localization(__file__, 549, 4), getitem___28916, int_28911)
    
    # Assigning a type to the variable 'tuple_var_assignment_28275' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_var_assignment_28275', subscript_call_result_28917)
    
    # Assigning a Subscript to a Name (line 549):
    
    # Obtaining the type of the subscript
    int_28918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 4), 'int')
    
    # Call to arg_names_and_types(...): (line 549)
    # Processing the call arguments (line 549)
    # Getting the type of 'args' (line 549)
    args_28920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 39), 'args', False)
    # Processing the call keyword arguments (line 549)
    kwargs_28921 = {}
    # Getting the type of 'arg_names_and_types' (line 549)
    arg_names_and_types_28919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 19), 'arg_names_and_types', False)
    # Calling arg_names_and_types(args, kwargs) (line 549)
    arg_names_and_types_call_result_28922 = invoke(stypy.reporting.localization.Localization(__file__, 549, 19), arg_names_and_types_28919, *[args_28920], **kwargs_28921)
    
    # Obtaining the member '__getitem__' of a type (line 549)
    getitem___28923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 549, 4), arg_names_and_types_call_result_28922, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 549)
    subscript_call_result_28924 = invoke(stypy.reporting.localization.Localization(__file__, 549, 4), getitem___28923, int_28918)
    
    # Assigning a type to the variable 'tuple_var_assignment_28276' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_var_assignment_28276', subscript_call_result_28924)
    
    # Assigning a Name to a Name (line 549):
    # Getting the type of 'tuple_var_assignment_28275' (line 549)
    tuple_var_assignment_28275_28925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_var_assignment_28275')
    # Assigning a type to the variable 'types' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'types', tuple_var_assignment_28275_28925)
    
    # Assigning a Name to a Name (line 549):
    # Getting the type of 'tuple_var_assignment_28276' (line 549)
    tuple_var_assignment_28276_28926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 4), 'tuple_var_assignment_28276')
    # Assigning a type to the variable 'names' (line 549)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 11), 'names', tuple_var_assignment_28276_28926)
    
    # Assigning a Call to a Name (line 550):
    
    # Assigning a Call to a Name (line 550):
    
    # Call to join(...): (line 550)
    # Processing the call arguments (line 550)
    # Getting the type of 'names' (line 550)
    names_28929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 25), 'names', False)
    # Processing the call keyword arguments (line 550)
    kwargs_28930 = {}
    str_28927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 15), 'str', ', ')
    # Obtaining the member 'join' of a type (line 550)
    join_28928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 15), str_28927, 'join')
    # Calling join(args, kwargs) (line 550)
    join_call_result_28931 = invoke(stypy.reporting.localization.Localization(__file__, 550, 15), join_28928, *[names_28929], **kwargs_28930)
    
    # Assigning a type to the variable 'argnames' (line 550)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 4), 'argnames', join_call_result_28931)
    
    # Assigning a ListComp to a Name (line 552):
    
    # Assigning a ListComp to a Name (line 552):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'names' (line 552)
    names_28937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 52), 'names')
    comprehension_28938 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 13), names_28937)
    # Assigning a type to the variable 'n' (line 552)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 13), 'n', comprehension_28938)
    
    # Call to process_fortran_name(...): (line 552)
    # Processing the call arguments (line 552)
    # Getting the type of 'n' (line 552)
    n_28933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 34), 'n', False)
    # Getting the type of 'name' (line 552)
    name_28934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 37), 'name', False)
    # Processing the call keyword arguments (line 552)
    kwargs_28935 = {}
    # Getting the type of 'process_fortran_name' (line 552)
    process_fortran_name_28932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 13), 'process_fortran_name', False)
    # Calling process_fortran_name(args, kwargs) (line 552)
    process_fortran_name_call_result_28936 = invoke(stypy.reporting.localization.Localization(__file__, 552, 13), process_fortran_name_28932, *[n_28933, name_28934], **kwargs_28935)
    
    list_28939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 552, 13), list_28939, process_fortran_name_call_result_28936)
    # Assigning a type to the variable 'names' (line 552)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 4), 'names', list_28939)
    
    # Assigning a Call to a Name (line 553):
    
    # Assigning a Call to a Name (line 553):
    
    # Call to join(...): (line 553)
    # Processing the call arguments (line 553)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 553, 33, True)
    # Calculating comprehension expression
    
    # Call to zip(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'names' (line 554)
    names_28952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 49), 'names', False)
    # Getting the type of 'types' (line 554)
    types_28953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 56), 'types', False)
    # Processing the call keyword arguments (line 554)
    kwargs_28954 = {}
    # Getting the type of 'zip' (line 554)
    zip_28951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 45), 'zip', False)
    # Calling zip(args, kwargs) (line 554)
    zip_call_result_28955 = invoke(stypy.reporting.localization.Localization(__file__, 554, 45), zip_28951, *[names_28952, types_28953], **kwargs_28954)
    
    comprehension_28956 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 33), zip_call_result_28955)
    # Assigning a type to the variable 'n' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 33), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 33), comprehension_28956))
    # Assigning a type to the variable 't' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 33), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 33), comprehension_28956))
    
    # Call to format(...): (line 553)
    # Processing the call arguments (line 553)
    
    # Obtaining the type of the subscript
    # Getting the type of 't' (line 553)
    t_28944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 64), 't', False)
    # Getting the type of 'fortran_types' (line 553)
    fortran_types_28945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 50), 'fortran_types', False)
    # Obtaining the member '__getitem__' of a type (line 553)
    getitem___28946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 50), fortran_types_28945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 553)
    subscript_call_result_28947 = invoke(stypy.reporting.localization.Localization(__file__, 553, 50), getitem___28946, t_28944)
    
    # Getting the type of 'n' (line 553)
    n_28948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 68), 'n', False)
    # Processing the call keyword arguments (line 553)
    kwargs_28949 = {}
    str_28942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 33), 'str', '{0} {1}')
    # Obtaining the member 'format' of a type (line 553)
    format_28943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 33), str_28942, 'format')
    # Calling format(args, kwargs) (line 553)
    format_call_result_28950 = invoke(stypy.reporting.localization.Localization(__file__, 553, 33), format_28943, *[subscript_call_result_28947, n_28948], **kwargs_28949)
    
    list_28957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 33), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 553, 33), list_28957, format_call_result_28950)
    # Processing the call keyword arguments (line 553)
    kwargs_28958 = {}
    str_28940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 553, 15), 'str', '\n        ')
    # Obtaining the member 'join' of a type (line 553)
    join_28941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 15), str_28940, 'join')
    # Calling join(args, kwargs) (line 553)
    join_call_result_28959 = invoke(stypy.reporting.localization.Localization(__file__, 553, 15), join_28941, *[list_28957], **kwargs_28958)
    
    # Assigning a type to the variable 'argdecls' (line 553)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 4), 'argdecls', join_call_result_28959)
    
    # Call to format(...): (line 555)
    # Processing the call keyword arguments (line 555)
    # Getting the type of 'name' (line 555)
    name_28962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 40), 'name', False)
    keyword_28963 = name_28962
    # Getting the type of 'wrapper' (line 555)
    wrapper_28964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 54), 'wrapper', False)
    keyword_28965 = wrapper_28964
    # Getting the type of 'argnames' (line 556)
    argnames_28966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 44), 'argnames', False)
    keyword_28967 = argnames_28966
    # Getting the type of 'argdecls' (line 556)
    argdecls_28968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 63), 'argdecls', False)
    keyword_28969 = argdecls_28968
    
    # Obtaining the type of the subscript
    # Getting the type of 'ret_type' (line 557)
    ret_type_28970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 58), 'ret_type', False)
    # Getting the type of 'fortran_types' (line 557)
    fortran_types_28971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 44), 'fortran_types', False)
    # Obtaining the member '__getitem__' of a type (line 557)
    getitem___28972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 44), fortran_types_28971, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 557)
    subscript_call_result_28973 = invoke(stypy.reporting.localization.Localization(__file__, 557, 44), getitem___28972, ret_type_28970)
    
    keyword_28974 = subscript_call_result_28973
    kwargs_28975 = {'ret_type': keyword_28974, 'argdecls': keyword_28969, 'argnames': keyword_28967, 'name': keyword_28963, 'wrapper': keyword_28965}
    # Getting the type of 'fortran_template' (line 555)
    fortran_template_28960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 11), 'fortran_template', False)
    # Obtaining the member 'format' of a type (line 555)
    format_28961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 11), fortran_template_28960, 'format')
    # Calling format(args, kwargs) (line 555)
    format_call_result_28976 = invoke(stypy.reporting.localization.Localization(__file__, 555, 11), format_28961, *[], **kwargs_28975)
    
    # Assigning a type to the variable 'stypy_return_type' (line 555)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'stypy_return_type', format_call_result_28976)
    
    # ################# End of 'fort_subroutine_wrapper(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fort_subroutine_wrapper' in the type store
    # Getting the type of 'stypy_return_type' (line 544)
    stypy_return_type_28977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28977)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fort_subroutine_wrapper'
    return stypy_return_type_28977

# Assigning a type to the variable 'fort_subroutine_wrapper' (line 544)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 0), 'fort_subroutine_wrapper', fort_subroutine_wrapper)

@norecursion
def generate_fortran(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'generate_fortran'
    module_type_store = module_type_store.open_function_context('generate_fortran', 560, 0, False)
    
    # Passed parameters checking function
    generate_fortran.stypy_localization = localization
    generate_fortran.stypy_type_of_self = None
    generate_fortran.stypy_type_store = module_type_store
    generate_fortran.stypy_function_name = 'generate_fortran'
    generate_fortran.stypy_param_names_list = ['func_sigs']
    generate_fortran.stypy_varargs_param_name = None
    generate_fortran.stypy_kwargs_param_name = None
    generate_fortran.stypy_call_defaults = defaults
    generate_fortran.stypy_call_varargs = varargs
    generate_fortran.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generate_fortran', ['func_sigs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generate_fortran', localization, ['func_sigs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generate_fortran(...)' code ##################

    
    # Call to join(...): (line 561)
    # Processing the call arguments (line 561)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 561, 21, True)
    # Calculating comprehension expression
    # Getting the type of 'func_sigs' (line 561)
    func_sigs_28984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 62), 'func_sigs', False)
    comprehension_28985 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 21), func_sigs_28984)
    # Assigning a type to the variable 'sig' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 21), 'sig', comprehension_28985)
    
    # Call to fort_subroutine_wrapper(...): (line 561)
    # Getting the type of 'sig' (line 561)
    sig_28981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 46), 'sig', False)
    # Processing the call keyword arguments (line 561)
    kwargs_28982 = {}
    # Getting the type of 'fort_subroutine_wrapper' (line 561)
    fort_subroutine_wrapper_28980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 21), 'fort_subroutine_wrapper', False)
    # Calling fort_subroutine_wrapper(args, kwargs) (line 561)
    fort_subroutine_wrapper_call_result_28983 = invoke(stypy.reporting.localization.Localization(__file__, 561, 21), fort_subroutine_wrapper_28980, *[sig_28981], **kwargs_28982)
    
    list_28986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 21), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 21), list_28986, fort_subroutine_wrapper_call_result_28983)
    # Processing the call keyword arguments (line 561)
    kwargs_28987 = {}
    str_28978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 11), 'str', '\n')
    # Obtaining the member 'join' of a type (line 561)
    join_28979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 11), str_28978, 'join')
    # Calling join(args, kwargs) (line 561)
    join_call_result_28988 = invoke(stypy.reporting.localization.Localization(__file__, 561, 11), join_28979, *[list_28986], **kwargs_28987)
    
    # Assigning a type to the variable 'stypy_return_type' (line 561)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'stypy_return_type', join_call_result_28988)
    
    # ################# End of 'generate_fortran(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generate_fortran' in the type store
    # Getting the type of 'stypy_return_type' (line 560)
    stypy_return_type_28989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28989)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generate_fortran'
    return stypy_return_type_28989

# Assigning a type to the variable 'generate_fortran' (line 560)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 0), 'generate_fortran', generate_fortran)

@norecursion
def make_c_args(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'make_c_args'
    module_type_store = module_type_store.open_function_context('make_c_args', 564, 0, False)
    
    # Passed parameters checking function
    make_c_args.stypy_localization = localization
    make_c_args.stypy_type_of_self = None
    make_c_args.stypy_type_store = module_type_store
    make_c_args.stypy_function_name = 'make_c_args'
    make_c_args.stypy_param_names_list = ['args']
    make_c_args.stypy_varargs_param_name = None
    make_c_args.stypy_kwargs_param_name = None
    make_c_args.stypy_call_defaults = defaults
    make_c_args.stypy_call_varargs = varargs
    make_c_args.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_c_args', ['args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_c_args', localization, ['args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_c_args(...)' code ##################

    
    # Assigning a Call to a Tuple (line 565):
    
    # Assigning a Subscript to a Name (line 565):
    
    # Obtaining the type of the subscript
    int_28990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 4), 'int')
    
    # Call to arg_names_and_types(...): (line 565)
    # Processing the call arguments (line 565)
    # Getting the type of 'args' (line 565)
    args_28992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 39), 'args', False)
    # Processing the call keyword arguments (line 565)
    kwargs_28993 = {}
    # Getting the type of 'arg_names_and_types' (line 565)
    arg_names_and_types_28991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 19), 'arg_names_and_types', False)
    # Calling arg_names_and_types(args, kwargs) (line 565)
    arg_names_and_types_call_result_28994 = invoke(stypy.reporting.localization.Localization(__file__, 565, 19), arg_names_and_types_28991, *[args_28992], **kwargs_28993)
    
    # Obtaining the member '__getitem__' of a type (line 565)
    getitem___28995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 4), arg_names_and_types_call_result_28994, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 565)
    subscript_call_result_28996 = invoke(stypy.reporting.localization.Localization(__file__, 565, 4), getitem___28995, int_28990)
    
    # Assigning a type to the variable 'tuple_var_assignment_28277' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'tuple_var_assignment_28277', subscript_call_result_28996)
    
    # Assigning a Subscript to a Name (line 565):
    
    # Obtaining the type of the subscript
    int_28997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 4), 'int')
    
    # Call to arg_names_and_types(...): (line 565)
    # Processing the call arguments (line 565)
    # Getting the type of 'args' (line 565)
    args_28999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 39), 'args', False)
    # Processing the call keyword arguments (line 565)
    kwargs_29000 = {}
    # Getting the type of 'arg_names_and_types' (line 565)
    arg_names_and_types_28998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 19), 'arg_names_and_types', False)
    # Calling arg_names_and_types(args, kwargs) (line 565)
    arg_names_and_types_call_result_29001 = invoke(stypy.reporting.localization.Localization(__file__, 565, 19), arg_names_and_types_28998, *[args_28999], **kwargs_29000)
    
    # Obtaining the member '__getitem__' of a type (line 565)
    getitem___29002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 4), arg_names_and_types_call_result_29001, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 565)
    subscript_call_result_29003 = invoke(stypy.reporting.localization.Localization(__file__, 565, 4), getitem___29002, int_28997)
    
    # Assigning a type to the variable 'tuple_var_assignment_28278' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'tuple_var_assignment_28278', subscript_call_result_29003)
    
    # Assigning a Name to a Name (line 565):
    # Getting the type of 'tuple_var_assignment_28277' (line 565)
    tuple_var_assignment_28277_29004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'tuple_var_assignment_28277')
    # Assigning a type to the variable 'types' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'types', tuple_var_assignment_28277_29004)
    
    # Assigning a Name to a Name (line 565):
    # Getting the type of 'tuple_var_assignment_28278' (line 565)
    tuple_var_assignment_28278_29005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'tuple_var_assignment_28278')
    # Assigning a type to the variable 'names' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 11), 'names', tuple_var_assignment_28278_29005)
    
    # Assigning a ListComp to a Name (line 566):
    
    # Assigning a ListComp to a Name (line 566):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'types' (line 566)
    types_29010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 37), 'types')
    comprehension_29011 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 13), types_29010)
    # Assigning a type to the variable 'arg' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 13), 'arg', comprehension_29011)
    
    # Obtaining the type of the subscript
    # Getting the type of 'arg' (line 566)
    arg_29006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 21), 'arg')
    # Getting the type of 'c_types' (line 566)
    c_types_29007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 13), 'c_types')
    # Obtaining the member '__getitem__' of a type (line 566)
    getitem___29008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 13), c_types_29007, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 566)
    subscript_call_result_29009 = invoke(stypy.reporting.localization.Localization(__file__, 566, 13), getitem___29008, arg_29006)
    
    list_29012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 13), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 566, 13), list_29012, subscript_call_result_29009)
    # Assigning a type to the variable 'types' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'types', list_29012)
    
    # Call to join(...): (line 567)
    # Processing the call arguments (line 567)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 567, 21, True)
    # Calculating comprehension expression
    
    # Call to zip(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'types' (line 567)
    types_29022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 61), 'types', False)
    # Getting the type of 'names' (line 567)
    names_29023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 68), 'names', False)
    # Processing the call keyword arguments (line 567)
    kwargs_29024 = {}
    # Getting the type of 'zip' (line 567)
    zip_29021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 57), 'zip', False)
    # Calling zip(args, kwargs) (line 567)
    zip_call_result_29025 = invoke(stypy.reporting.localization.Localization(__file__, 567, 57), zip_29021, *[types_29022, names_29023], **kwargs_29024)
    
    comprehension_29026 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 21), zip_call_result_29025)
    # Assigning a type to the variable 't' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 21), 't', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 21), comprehension_29026))
    # Assigning a type to the variable 'n' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 21), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 21), comprehension_29026))
    
    # Call to format(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 't' (line 567)
    t_29017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 39), 't', False)
    # Getting the type of 'n' (line 567)
    n_29018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 42), 'n', False)
    # Processing the call keyword arguments (line 567)
    kwargs_29019 = {}
    str_29015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 21), 'str', '{0} *{1}')
    # Obtaining the member 'format' of a type (line 567)
    format_29016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 21), str_29015, 'format')
    # Calling format(args, kwargs) (line 567)
    format_call_result_29020 = invoke(stypy.reporting.localization.Localization(__file__, 567, 21), format_29016, *[t_29017, n_29018], **kwargs_29019)
    
    list_29027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 21), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 567, 21), list_29027, format_call_result_29020)
    # Processing the call keyword arguments (line 567)
    kwargs_29028 = {}
    str_29013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 11), 'str', ', ')
    # Obtaining the member 'join' of a type (line 567)
    join_29014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 11), str_29013, 'join')
    # Calling join(args, kwargs) (line 567)
    join_call_result_29029 = invoke(stypy.reporting.localization.Localization(__file__, 567, 11), join_29014, *[list_29027], **kwargs_29028)
    
    # Assigning a type to the variable 'stypy_return_type' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'stypy_return_type', join_call_result_29029)
    
    # ################# End of 'make_c_args(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_c_args' in the type store
    # Getting the type of 'stypy_return_type' (line 564)
    stypy_return_type_29030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29030)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_c_args'
    return stypy_return_type_29030

# Assigning a type to the variable 'make_c_args' (line 564)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 0), 'make_c_args', make_c_args)

# Assigning a Str to a Name (line 569):

# Assigning a Str to a Name (line 569):
str_29031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 18), 'str', 'void F_FUNC({name}wrp, {upname}WRP)({return_type} *ret, {args});\n')
# Assigning a type to the variable 'c_func_template' (line 569)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 0), 'c_func_template', str_29031)

@norecursion
def c_func_decl(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'c_func_decl'
    module_type_store = module_type_store.open_function_context('c_func_decl', 572, 0, False)
    
    # Passed parameters checking function
    c_func_decl.stypy_localization = localization
    c_func_decl.stypy_type_of_self = None
    c_func_decl.stypy_type_store = module_type_store
    c_func_decl.stypy_function_name = 'c_func_decl'
    c_func_decl.stypy_param_names_list = ['name', 'return_type', 'args']
    c_func_decl.stypy_varargs_param_name = None
    c_func_decl.stypy_kwargs_param_name = None
    c_func_decl.stypy_call_defaults = defaults
    c_func_decl.stypy_call_varargs = varargs
    c_func_decl.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'c_func_decl', ['name', 'return_type', 'args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'c_func_decl', localization, ['name', 'return_type', 'args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'c_func_decl(...)' code ##################

    
    # Assigning a Call to a Name (line 573):
    
    # Assigning a Call to a Name (line 573):
    
    # Call to make_c_args(...): (line 573)
    # Processing the call arguments (line 573)
    # Getting the type of 'args' (line 573)
    args_29033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 23), 'args', False)
    # Processing the call keyword arguments (line 573)
    kwargs_29034 = {}
    # Getting the type of 'make_c_args' (line 573)
    make_c_args_29032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 11), 'make_c_args', False)
    # Calling make_c_args(args, kwargs) (line 573)
    make_c_args_call_result_29035 = invoke(stypy.reporting.localization.Localization(__file__, 573, 11), make_c_args_29032, *[args_29033], **kwargs_29034)
    
    # Assigning a type to the variable 'args' (line 573)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 4), 'args', make_c_args_call_result_29035)
    
    # Assigning a Subscript to a Name (line 574):
    
    # Assigning a Subscript to a Name (line 574):
    
    # Obtaining the type of the subscript
    # Getting the type of 'return_type' (line 574)
    return_type_29036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 26), 'return_type')
    # Getting the type of 'c_types' (line 574)
    c_types_29037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 18), 'c_types')
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___29038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 18), c_types_29037, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 574)
    subscript_call_result_29039 = invoke(stypy.reporting.localization.Localization(__file__, 574, 18), getitem___29038, return_type_29036)
    
    # Assigning a type to the variable 'return_type' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'return_type', subscript_call_result_29039)
    
    # Call to format(...): (line 575)
    # Processing the call keyword arguments (line 575)
    # Getting the type of 'name' (line 575)
    name_29042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 39), 'name', False)
    keyword_29043 = name_29042
    
    # Call to upper(...): (line 575)
    # Processing the call keyword arguments (line 575)
    kwargs_29046 = {}
    # Getting the type of 'name' (line 575)
    name_29044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 52), 'name', False)
    # Obtaining the member 'upper' of a type (line 575)
    upper_29045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 52), name_29044, 'upper')
    # Calling upper(args, kwargs) (line 575)
    upper_call_result_29047 = invoke(stypy.reporting.localization.Localization(__file__, 575, 52), upper_29045, *[], **kwargs_29046)
    
    keyword_29048 = upper_call_result_29047
    # Getting the type of 'return_type' (line 576)
    return_type_29049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 46), 'return_type', False)
    keyword_29050 = return_type_29049
    # Getting the type of 'args' (line 576)
    args_29051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 64), 'args', False)
    keyword_29052 = args_29051
    kwargs_29053 = {'return_type': keyword_29050, 'args': keyword_29052, 'upname': keyword_29048, 'name': keyword_29043}
    # Getting the type of 'c_func_template' (line 575)
    c_func_template_29040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 11), 'c_func_template', False)
    # Obtaining the member 'format' of a type (line 575)
    format_29041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 11), c_func_template_29040, 'format')
    # Calling format(args, kwargs) (line 575)
    format_call_result_29054 = invoke(stypy.reporting.localization.Localization(__file__, 575, 11), format_29041, *[], **kwargs_29053)
    
    # Assigning a type to the variable 'stypy_return_type' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'stypy_return_type', format_call_result_29054)
    
    # ################# End of 'c_func_decl(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'c_func_decl' in the type store
    # Getting the type of 'stypy_return_type' (line 572)
    stypy_return_type_29055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29055)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'c_func_decl'
    return stypy_return_type_29055

# Assigning a type to the variable 'c_func_decl' (line 572)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 0), 'c_func_decl', c_func_decl)

# Assigning a Str to a Name (line 578):

# Assigning a Str to a Name (line 578):
str_29056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 17), 'str', 'void F_FUNC({name},{upname})({args});\n')
# Assigning a type to the variable 'c_sub_template' (line 578)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 0), 'c_sub_template', str_29056)

@norecursion
def c_sub_decl(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'c_sub_decl'
    module_type_store = module_type_store.open_function_context('c_sub_decl', 581, 0, False)
    
    # Passed parameters checking function
    c_sub_decl.stypy_localization = localization
    c_sub_decl.stypy_type_of_self = None
    c_sub_decl.stypy_type_store = module_type_store
    c_sub_decl.stypy_function_name = 'c_sub_decl'
    c_sub_decl.stypy_param_names_list = ['name', 'return_type', 'args']
    c_sub_decl.stypy_varargs_param_name = None
    c_sub_decl.stypy_kwargs_param_name = None
    c_sub_decl.stypy_call_defaults = defaults
    c_sub_decl.stypy_call_varargs = varargs
    c_sub_decl.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'c_sub_decl', ['name', 'return_type', 'args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'c_sub_decl', localization, ['name', 'return_type', 'args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'c_sub_decl(...)' code ##################

    
    # Assigning a Call to a Name (line 582):
    
    # Assigning a Call to a Name (line 582):
    
    # Call to make_c_args(...): (line 582)
    # Processing the call arguments (line 582)
    # Getting the type of 'args' (line 582)
    args_29058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 23), 'args', False)
    # Processing the call keyword arguments (line 582)
    kwargs_29059 = {}
    # Getting the type of 'make_c_args' (line 582)
    make_c_args_29057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 11), 'make_c_args', False)
    # Calling make_c_args(args, kwargs) (line 582)
    make_c_args_call_result_29060 = invoke(stypy.reporting.localization.Localization(__file__, 582, 11), make_c_args_29057, *[args_29058], **kwargs_29059)
    
    # Assigning a type to the variable 'args' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'args', make_c_args_call_result_29060)
    
    # Call to format(...): (line 583)
    # Processing the call keyword arguments (line 583)
    # Getting the type of 'name' (line 583)
    name_29063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 38), 'name', False)
    keyword_29064 = name_29063
    
    # Call to upper(...): (line 583)
    # Processing the call keyword arguments (line 583)
    kwargs_29067 = {}
    # Getting the type of 'name' (line 583)
    name_29065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 51), 'name', False)
    # Obtaining the member 'upper' of a type (line 583)
    upper_29066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 51), name_29065, 'upper')
    # Calling upper(args, kwargs) (line 583)
    upper_call_result_29068 = invoke(stypy.reporting.localization.Localization(__file__, 583, 51), upper_29066, *[], **kwargs_29067)
    
    keyword_29069 = upper_call_result_29068
    # Getting the type of 'args' (line 583)
    args_29070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 70), 'args', False)
    keyword_29071 = args_29070
    kwargs_29072 = {'args': keyword_29071, 'upname': keyword_29069, 'name': keyword_29064}
    # Getting the type of 'c_sub_template' (line 583)
    c_sub_template_29061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 11), 'c_sub_template', False)
    # Obtaining the member 'format' of a type (line 583)
    format_29062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 11), c_sub_template_29061, 'format')
    # Calling format(args, kwargs) (line 583)
    format_call_result_29073 = invoke(stypy.reporting.localization.Localization(__file__, 583, 11), format_29062, *[], **kwargs_29072)
    
    # Assigning a type to the variable 'stypy_return_type' (line 583)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 4), 'stypy_return_type', format_call_result_29073)
    
    # ################# End of 'c_sub_decl(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'c_sub_decl' in the type store
    # Getting the type of 'stypy_return_type' (line 581)
    stypy_return_type_29074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29074)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'c_sub_decl'
    return stypy_return_type_29074

# Assigning a type to the variable 'c_sub_decl' (line 581)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 0), 'c_sub_decl', c_sub_decl)

# Assigning a Str to a Name (line 585):

# Assigning a Str to a Name (line 585):
str_29075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, (-1)), 'str', '#ifndef SCIPY_LINALG_{lib}_FORTRAN_WRAPPERS_H\n#define SCIPY_LINALG_{lib}_FORTRAN_WRAPPERS_H\n#include "fortran_defs.h"\n#include "numpy/arrayobject.h"\n')
# Assigning a type to the variable 'c_preamble' (line 585)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 0), 'c_preamble', str_29075)

# Assigning a Str to a Name (line 591):

# Assigning a Str to a Name (line 591):
str_29076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, (-1)), 'str', '\ntypedef int (*_cselect1)(npy_complex64*);\ntypedef int (*_cselect2)(npy_complex64*, npy_complex64*);\ntypedef int (*_dselect2)(double*, double*);\ntypedef int (*_dselect3)(double*, double*, double*);\ntypedef int (*_sselect2)(float*, float*);\ntypedef int (*_sselect3)(float*, float*, float*);\ntypedef int (*_zselect1)(npy_complex128*);\ntypedef int (*_zselect2)(npy_complex128*, npy_complex128*);\n')
# Assigning a type to the variable 'lapack_decls' (line 591)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 0), 'lapack_decls', str_29076)

# Assigning a Str to a Name (line 602):

# Assigning a Str to a Name (line 602):
str_29077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 607, (-1)), 'str', '\n#ifdef __cplusplus\nextern "C" {\n#endif\n\n')
# Assigning a type to the variable 'cpp_guard' (line 602)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 0), 'cpp_guard', str_29077)

# Assigning a Str to a Name (line 609):

# Assigning a Str to a Name (line 609):
str_29078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, (-1)), 'str', '\n#ifdef __cplusplus\n}\n#endif\n#endif\n')
# Assigning a type to the variable 'c_end' (line 609)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 0), 'c_end', str_29078)

@norecursion
def generate_c_header(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'generate_c_header'
    module_type_store = module_type_store.open_function_context('generate_c_header', 617, 0, False)
    
    # Passed parameters checking function
    generate_c_header.stypy_localization = localization
    generate_c_header.stypy_type_of_self = None
    generate_c_header.stypy_type_store = module_type_store
    generate_c_header.stypy_function_name = 'generate_c_header'
    generate_c_header.stypy_param_names_list = ['func_sigs', 'sub_sigs', 'all_sigs', 'lib_name']
    generate_c_header.stypy_varargs_param_name = None
    generate_c_header.stypy_kwargs_param_name = None
    generate_c_header.stypy_call_defaults = defaults
    generate_c_header.stypy_call_varargs = varargs
    generate_c_header.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generate_c_header', ['func_sigs', 'sub_sigs', 'all_sigs', 'lib_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generate_c_header', localization, ['func_sigs', 'sub_sigs', 'all_sigs', 'lib_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generate_c_header(...)' code ##################

    
    # Assigning a Call to a Name (line 618):
    
    # Assigning a Call to a Name (line 618):
    
    # Call to join(...): (line 618)
    # Processing the call arguments (line 618)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 618, 20, True)
    # Calculating comprehension expression
    # Getting the type of 'func_sigs' (line 618)
    func_sigs_29085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 49), 'func_sigs', False)
    comprehension_29086 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 20), func_sigs_29085)
    # Assigning a type to the variable 'sig' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 20), 'sig', comprehension_29086)
    
    # Call to c_func_decl(...): (line 618)
    # Getting the type of 'sig' (line 618)
    sig_29082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 33), 'sig', False)
    # Processing the call keyword arguments (line 618)
    kwargs_29083 = {}
    # Getting the type of 'c_func_decl' (line 618)
    c_func_decl_29081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 20), 'c_func_decl', False)
    # Calling c_func_decl(args, kwargs) (line 618)
    c_func_decl_call_result_29084 = invoke(stypy.reporting.localization.Localization(__file__, 618, 20), c_func_decl_29081, *[sig_29082], **kwargs_29083)
    
    list_29087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 20), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 618, 20), list_29087, c_func_decl_call_result_29084)
    # Processing the call keyword arguments (line 618)
    kwargs_29088 = {}
    str_29079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 12), 'str', '')
    # Obtaining the member 'join' of a type (line 618)
    join_29080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 12), str_29079, 'join')
    # Calling join(args, kwargs) (line 618)
    join_call_result_29089 = invoke(stypy.reporting.localization.Localization(__file__, 618, 12), join_29080, *[list_29087], **kwargs_29088)
    
    # Assigning a type to the variable 'funcs' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'funcs', join_call_result_29089)
    
    # Assigning a BinOp to a Name (line 619):
    
    # Assigning a BinOp to a Name (line 619):
    str_29090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 11), 'str', '\n')
    
    # Call to join(...): (line 619)
    # Processing the call arguments (line 619)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 619, 26, True)
    # Calculating comprehension expression
    # Getting the type of 'sub_sigs' (line 619)
    sub_sigs_29097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 54), 'sub_sigs', False)
    comprehension_29098 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 26), sub_sigs_29097)
    # Assigning a type to the variable 'sig' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 26), 'sig', comprehension_29098)
    
    # Call to c_sub_decl(...): (line 619)
    # Getting the type of 'sig' (line 619)
    sig_29094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 38), 'sig', False)
    # Processing the call keyword arguments (line 619)
    kwargs_29095 = {}
    # Getting the type of 'c_sub_decl' (line 619)
    c_sub_decl_29093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 26), 'c_sub_decl', False)
    # Calling c_sub_decl(args, kwargs) (line 619)
    c_sub_decl_call_result_29096 = invoke(stypy.reporting.localization.Localization(__file__, 619, 26), c_sub_decl_29093, *[sig_29094], **kwargs_29095)
    
    list_29099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 26), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 619, 26), list_29099, c_sub_decl_call_result_29096)
    # Processing the call keyword arguments (line 619)
    kwargs_29100 = {}
    str_29091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 18), 'str', '')
    # Obtaining the member 'join' of a type (line 619)
    join_29092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 18), str_29091, 'join')
    # Calling join(args, kwargs) (line 619)
    join_call_result_29101 = invoke(stypy.reporting.localization.Localization(__file__, 619, 18), join_29092, *[list_29099], **kwargs_29100)
    
    # Applying the binary operator '+' (line 619)
    result_add_29102 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 11), '+', str_29090, join_call_result_29101)
    
    # Assigning a type to the variable 'subs' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'subs', result_add_29102)
    
    
    # Getting the type of 'lib_name' (line 620)
    lib_name_29103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 7), 'lib_name')
    str_29104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 19), 'str', 'LAPACK')
    # Applying the binary operator '==' (line 620)
    result_eq_29105 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 7), '==', lib_name_29103, str_29104)
    
    # Testing the type of an if condition (line 620)
    if_condition_29106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 620, 4), result_eq_29105)
    # Assigning a type to the variable 'if_condition_29106' (line 620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 4), 'if_condition_29106', if_condition_29106)
    # SSA begins for if statement (line 620)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 621):
    
    # Assigning a BinOp to a Name (line 621):
    
    # Call to format(...): (line 621)
    # Processing the call keyword arguments (line 621)
    # Getting the type of 'lib_name' (line 621)
    lib_name_29109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 42), 'lib_name', False)
    keyword_29110 = lib_name_29109
    kwargs_29111 = {'lib': keyword_29110}
    # Getting the type of 'c_preamble' (line 621)
    c_preamble_29107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 20), 'c_preamble', False)
    # Obtaining the member 'format' of a type (line 621)
    format_29108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 20), c_preamble_29107, 'format')
    # Calling format(args, kwargs) (line 621)
    format_call_result_29112 = invoke(stypy.reporting.localization.Localization(__file__, 621, 20), format_29108, *[], **kwargs_29111)
    
    # Getting the type of 'lapack_decls' (line 621)
    lapack_decls_29113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 54), 'lapack_decls')
    # Applying the binary operator '+' (line 621)
    result_add_29114 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 20), '+', format_call_result_29112, lapack_decls_29113)
    
    # Assigning a type to the variable 'preamble' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'preamble', result_add_29114)
    # SSA branch for the else part of an if statement (line 620)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 623):
    
    # Assigning a Call to a Name (line 623):
    
    # Call to format(...): (line 623)
    # Processing the call keyword arguments (line 623)
    # Getting the type of 'lib_name' (line 623)
    lib_name_29117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 41), 'lib_name', False)
    keyword_29118 = lib_name_29117
    kwargs_29119 = {'lib': keyword_29118}
    # Getting the type of 'c_preamble' (line 623)
    c_preamble_29115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 19), 'c_preamble', False)
    # Obtaining the member 'format' of a type (line 623)
    format_29116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 19), c_preamble_29115, 'format')
    # Calling format(args, kwargs) (line 623)
    format_call_result_29120 = invoke(stypy.reporting.localization.Localization(__file__, 623, 19), format_29116, *[], **kwargs_29119)
    
    # Assigning a type to the variable 'preamble' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 8), 'preamble', format_call_result_29120)
    # SSA join for if statement (line 620)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to join(...): (line 624)
    # Processing the call arguments (line 624)
    
    # Obtaining an instance of the builtin type 'list' (line 624)
    list_29123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 624)
    # Adding element type (line 624)
    # Getting the type of 'preamble' (line 624)
    preamble_29124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 20), 'preamble', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 19), list_29123, preamble_29124)
    # Adding element type (line 624)
    # Getting the type of 'cpp_guard' (line 624)
    cpp_guard_29125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 30), 'cpp_guard', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 19), list_29123, cpp_guard_29125)
    # Adding element type (line 624)
    # Getting the type of 'funcs' (line 624)
    funcs_29126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 41), 'funcs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 19), list_29123, funcs_29126)
    # Adding element type (line 624)
    # Getting the type of 'subs' (line 624)
    subs_29127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 48), 'subs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 19), list_29123, subs_29127)
    # Adding element type (line 624)
    # Getting the type of 'c_end' (line 624)
    c_end_29128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 54), 'c_end', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 19), list_29123, c_end_29128)
    
    # Processing the call keyword arguments (line 624)
    kwargs_29129 = {}
    str_29121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 11), 'str', '')
    # Obtaining the member 'join' of a type (line 624)
    join_29122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 11), str_29121, 'join')
    # Calling join(args, kwargs) (line 624)
    join_call_result_29130 = invoke(stypy.reporting.localization.Localization(__file__, 624, 11), join_29122, *[list_29123], **kwargs_29129)
    
    # Assigning a type to the variable 'stypy_return_type' (line 624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 4), 'stypy_return_type', join_call_result_29130)
    
    # ################# End of 'generate_c_header(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generate_c_header' in the type store
    # Getting the type of 'stypy_return_type' (line 617)
    stypy_return_type_29131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29131)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generate_c_header'
    return stypy_return_type_29131

# Assigning a type to the variable 'generate_c_header' (line 617)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 0), 'generate_c_header', generate_c_header)

@norecursion
def split_signature(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'split_signature'
    module_type_store = module_type_store.open_function_context('split_signature', 627, 0, False)
    
    # Passed parameters checking function
    split_signature.stypy_localization = localization
    split_signature.stypy_type_of_self = None
    split_signature.stypy_type_store = module_type_store
    split_signature.stypy_function_name = 'split_signature'
    split_signature.stypy_param_names_list = ['sig']
    split_signature.stypy_varargs_param_name = None
    split_signature.stypy_kwargs_param_name = None
    split_signature.stypy_call_defaults = defaults
    split_signature.stypy_call_varargs = varargs
    split_signature.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'split_signature', ['sig'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'split_signature', localization, ['sig'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'split_signature(...)' code ##################

    
    # Assigning a Call to a Tuple (line 628):
    
    # Assigning a Subscript to a Name (line 628):
    
    # Obtaining the type of the subscript
    int_29132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 4), 'int')
    
    # Call to split(...): (line 628)
    # Processing the call arguments (line 628)
    str_29139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 41), 'str', '(')
    # Processing the call keyword arguments (line 628)
    kwargs_29140 = {}
    
    # Obtaining the type of the subscript
    int_29133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 31), 'int')
    slice_29134 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 628, 26), None, int_29133, None)
    # Getting the type of 'sig' (line 628)
    sig_29135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 26), 'sig', False)
    # Obtaining the member '__getitem__' of a type (line 628)
    getitem___29136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 26), sig_29135, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 628)
    subscript_call_result_29137 = invoke(stypy.reporting.localization.Localization(__file__, 628, 26), getitem___29136, slice_29134)
    
    # Obtaining the member 'split' of a type (line 628)
    split_29138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 26), subscript_call_result_29137, 'split')
    # Calling split(args, kwargs) (line 628)
    split_call_result_29141 = invoke(stypy.reporting.localization.Localization(__file__, 628, 26), split_29138, *[str_29139], **kwargs_29140)
    
    # Obtaining the member '__getitem__' of a type (line 628)
    getitem___29142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 4), split_call_result_29141, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 628)
    subscript_call_result_29143 = invoke(stypy.reporting.localization.Localization(__file__, 628, 4), getitem___29142, int_29132)
    
    # Assigning a type to the variable 'tuple_var_assignment_28279' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'tuple_var_assignment_28279', subscript_call_result_29143)
    
    # Assigning a Subscript to a Name (line 628):
    
    # Obtaining the type of the subscript
    int_29144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 4), 'int')
    
    # Call to split(...): (line 628)
    # Processing the call arguments (line 628)
    str_29151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 41), 'str', '(')
    # Processing the call keyword arguments (line 628)
    kwargs_29152 = {}
    
    # Obtaining the type of the subscript
    int_29145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 31), 'int')
    slice_29146 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 628, 26), None, int_29145, None)
    # Getting the type of 'sig' (line 628)
    sig_29147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 26), 'sig', False)
    # Obtaining the member '__getitem__' of a type (line 628)
    getitem___29148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 26), sig_29147, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 628)
    subscript_call_result_29149 = invoke(stypy.reporting.localization.Localization(__file__, 628, 26), getitem___29148, slice_29146)
    
    # Obtaining the member 'split' of a type (line 628)
    split_29150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 26), subscript_call_result_29149, 'split')
    # Calling split(args, kwargs) (line 628)
    split_call_result_29153 = invoke(stypy.reporting.localization.Localization(__file__, 628, 26), split_29150, *[str_29151], **kwargs_29152)
    
    # Obtaining the member '__getitem__' of a type (line 628)
    getitem___29154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 4), split_call_result_29153, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 628)
    subscript_call_result_29155 = invoke(stypy.reporting.localization.Localization(__file__, 628, 4), getitem___29154, int_29144)
    
    # Assigning a type to the variable 'tuple_var_assignment_28280' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'tuple_var_assignment_28280', subscript_call_result_29155)
    
    # Assigning a Name to a Name (line 628):
    # Getting the type of 'tuple_var_assignment_28279' (line 628)
    tuple_var_assignment_28279_29156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'tuple_var_assignment_28279')
    # Assigning a type to the variable 'name_and_type' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'name_and_type', tuple_var_assignment_28279_29156)
    
    # Assigning a Name to a Name (line 628):
    # Getting the type of 'tuple_var_assignment_28280' (line 628)
    tuple_var_assignment_28280_29157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 4), 'tuple_var_assignment_28280')
    # Assigning a type to the variable 'args' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 19), 'args', tuple_var_assignment_28280_29157)
    
    # Assigning a Call to a Tuple (line 629):
    
    # Assigning a Subscript to a Name (line 629):
    
    # Obtaining the type of the subscript
    int_29158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 4), 'int')
    
    # Call to split(...): (line 629)
    # Processing the call arguments (line 629)
    str_29161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 41), 'str', ' ')
    # Processing the call keyword arguments (line 629)
    kwargs_29162 = {}
    # Getting the type of 'name_and_type' (line 629)
    name_and_type_29159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 21), 'name_and_type', False)
    # Obtaining the member 'split' of a type (line 629)
    split_29160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 21), name_and_type_29159, 'split')
    # Calling split(args, kwargs) (line 629)
    split_call_result_29163 = invoke(stypy.reporting.localization.Localization(__file__, 629, 21), split_29160, *[str_29161], **kwargs_29162)
    
    # Obtaining the member '__getitem__' of a type (line 629)
    getitem___29164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 4), split_call_result_29163, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 629)
    subscript_call_result_29165 = invoke(stypy.reporting.localization.Localization(__file__, 629, 4), getitem___29164, int_29158)
    
    # Assigning a type to the variable 'tuple_var_assignment_28281' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 4), 'tuple_var_assignment_28281', subscript_call_result_29165)
    
    # Assigning a Subscript to a Name (line 629):
    
    # Obtaining the type of the subscript
    int_29166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 4), 'int')
    
    # Call to split(...): (line 629)
    # Processing the call arguments (line 629)
    str_29169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 41), 'str', ' ')
    # Processing the call keyword arguments (line 629)
    kwargs_29170 = {}
    # Getting the type of 'name_and_type' (line 629)
    name_and_type_29167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 21), 'name_and_type', False)
    # Obtaining the member 'split' of a type (line 629)
    split_29168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 21), name_and_type_29167, 'split')
    # Calling split(args, kwargs) (line 629)
    split_call_result_29171 = invoke(stypy.reporting.localization.Localization(__file__, 629, 21), split_29168, *[str_29169], **kwargs_29170)
    
    # Obtaining the member '__getitem__' of a type (line 629)
    getitem___29172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 4), split_call_result_29171, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 629)
    subscript_call_result_29173 = invoke(stypy.reporting.localization.Localization(__file__, 629, 4), getitem___29172, int_29166)
    
    # Assigning a type to the variable 'tuple_var_assignment_28282' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 4), 'tuple_var_assignment_28282', subscript_call_result_29173)
    
    # Assigning a Name to a Name (line 629):
    # Getting the type of 'tuple_var_assignment_28281' (line 629)
    tuple_var_assignment_28281_29174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 4), 'tuple_var_assignment_28281')
    # Assigning a type to the variable 'ret_type' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 4), 'ret_type', tuple_var_assignment_28281_29174)
    
    # Assigning a Name to a Name (line 629):
    # Getting the type of 'tuple_var_assignment_28282' (line 629)
    tuple_var_assignment_28282_29175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 4), 'tuple_var_assignment_28282')
    # Assigning a type to the variable 'name' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 14), 'name', tuple_var_assignment_28282_29175)
    
    # Obtaining an instance of the builtin type 'tuple' (line 630)
    tuple_29176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 630)
    # Adding element type (line 630)
    # Getting the type of 'name' (line 630)
    name_29177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 11), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 11), tuple_29176, name_29177)
    # Adding element type (line 630)
    # Getting the type of 'ret_type' (line 630)
    ret_type_29178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 17), 'ret_type')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 11), tuple_29176, ret_type_29178)
    # Adding element type (line 630)
    # Getting the type of 'args' (line 630)
    args_29179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 27), 'args')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 11), tuple_29176, args_29179)
    
    # Assigning a type to the variable 'stypy_return_type' (line 630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'stypy_return_type', tuple_29176)
    
    # ################# End of 'split_signature(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'split_signature' in the type store
    # Getting the type of 'stypy_return_type' (line 627)
    stypy_return_type_29180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29180)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'split_signature'
    return stypy_return_type_29180

# Assigning a type to the variable 'split_signature' (line 627)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 0), 'split_signature', split_signature)

@norecursion
def filter_lines(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'filter_lines'
    module_type_store = module_type_store.open_function_context('filter_lines', 633, 0, False)
    
    # Passed parameters checking function
    filter_lines.stypy_localization = localization
    filter_lines.stypy_type_of_self = None
    filter_lines.stypy_type_store = module_type_store
    filter_lines.stypy_function_name = 'filter_lines'
    filter_lines.stypy_param_names_list = ['ls']
    filter_lines.stypy_varargs_param_name = None
    filter_lines.stypy_kwargs_param_name = None
    filter_lines.stypy_call_defaults = defaults
    filter_lines.stypy_call_varargs = varargs
    filter_lines.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'filter_lines', ['ls'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'filter_lines', localization, ['ls'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'filter_lines(...)' code ##################

    
    # Assigning a ListComp to a Name (line 634):
    
    # Assigning a ListComp to a Name (line 634):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'ls' (line 634)
    ls_29195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 29), 'ls')
    comprehension_29196 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 634, 10), ls_29195)
    # Assigning a type to the variable 'l' (line 634)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 10), 'l', comprehension_29196)
    
    # Evaluating a boolean operation
    
    # Getting the type of 'l' (line 634)
    l_29185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 35), 'l')
    str_29186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 40), 'str', '\n')
    # Applying the binary operator '!=' (line 634)
    result_ne_29187 = python_operator(stypy.reporting.localization.Localization(__file__, 634, 35), '!=', l_29185, str_29186)
    
    
    
    # Obtaining the type of the subscript
    int_29188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 51), 'int')
    # Getting the type of 'l' (line 634)
    l_29189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 49), 'l')
    # Obtaining the member '__getitem__' of a type (line 634)
    getitem___29190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 49), l_29189, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 634)
    subscript_call_result_29191 = invoke(stypy.reporting.localization.Localization(__file__, 634, 49), getitem___29190, int_29188)
    
    str_29192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 57), 'str', '#')
    # Applying the binary operator '!=' (line 634)
    result_ne_29193 = python_operator(stypy.reporting.localization.Localization(__file__, 634, 49), '!=', subscript_call_result_29191, str_29192)
    
    # Applying the binary operator 'and' (line 634)
    result_and_keyword_29194 = python_operator(stypy.reporting.localization.Localization(__file__, 634, 35), 'and', result_ne_29187, result_ne_29193)
    
    
    # Call to strip(...): (line 634)
    # Processing the call keyword arguments (line 634)
    kwargs_29183 = {}
    # Getting the type of 'l' (line 634)
    l_29181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 10), 'l', False)
    # Obtaining the member 'strip' of a type (line 634)
    strip_29182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 10), l_29181, 'strip')
    # Calling strip(args, kwargs) (line 634)
    strip_call_result_29184 = invoke(stypy.reporting.localization.Localization(__file__, 634, 10), strip_29182, *[], **kwargs_29183)
    
    list_29197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 634, 10), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 634, 10), list_29197, strip_call_result_29184)
    # Assigning a type to the variable 'ls' (line 634)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 634, 4), 'ls', list_29197)
    
    # Assigning a ListComp to a Name (line 635):
    
    # Assigning a ListComp to a Name (line 635):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'ls' (line 635)
    ls_29212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 45), 'ls')
    comprehension_29213 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 635, 17), ls_29212)
    # Assigning a type to the variable 'l' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 17), 'l', comprehension_29213)
    
    
    # Obtaining the type of the subscript
    int_29202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 64), 'int')
    
    # Call to split(...): (line 635)
    # Processing the call arguments (line 635)
    str_29205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 59), 'str', ' ')
    # Processing the call keyword arguments (line 635)
    kwargs_29206 = {}
    # Getting the type of 'l' (line 635)
    l_29203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 51), 'l', False)
    # Obtaining the member 'split' of a type (line 635)
    split_29204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 51), l_29203, 'split')
    # Calling split(args, kwargs) (line 635)
    split_call_result_29207 = invoke(stypy.reporting.localization.Localization(__file__, 635, 51), split_29204, *[str_29205], **kwargs_29206)
    
    # Obtaining the member '__getitem__' of a type (line 635)
    getitem___29208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 51), split_call_result_29207, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 635)
    subscript_call_result_29209 = invoke(stypy.reporting.localization.Localization(__file__, 635, 51), getitem___29208, int_29202)
    
    str_29210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 70), 'str', 'void')
    # Applying the binary operator '!=' (line 635)
    result_ne_29211 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 51), '!=', subscript_call_result_29209, str_29210)
    
    
    # Call to split_signature(...): (line 635)
    # Processing the call arguments (line 635)
    # Getting the type of 'l' (line 635)
    l_29199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 33), 'l', False)
    # Processing the call keyword arguments (line 635)
    kwargs_29200 = {}
    # Getting the type of 'split_signature' (line 635)
    split_signature_29198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 17), 'split_signature', False)
    # Calling split_signature(args, kwargs) (line 635)
    split_signature_call_result_29201 = invoke(stypy.reporting.localization.Localization(__file__, 635, 17), split_signature_29198, *[l_29199], **kwargs_29200)
    
    list_29214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 635, 17), list_29214, split_signature_call_result_29201)
    # Assigning a type to the variable 'func_sigs' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'func_sigs', list_29214)
    
    # Assigning a ListComp to a Name (line 636):
    
    # Assigning a ListComp to a Name (line 636):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'ls' (line 636)
    ls_29229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 44), 'ls')
    comprehension_29230 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 16), ls_29229)
    # Assigning a type to the variable 'l' (line 636)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 16), 'l', comprehension_29230)
    
    
    # Obtaining the type of the subscript
    int_29219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 63), 'int')
    
    # Call to split(...): (line 636)
    # Processing the call arguments (line 636)
    str_29222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 58), 'str', ' ')
    # Processing the call keyword arguments (line 636)
    kwargs_29223 = {}
    # Getting the type of 'l' (line 636)
    l_29220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 50), 'l', False)
    # Obtaining the member 'split' of a type (line 636)
    split_29221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 50), l_29220, 'split')
    # Calling split(args, kwargs) (line 636)
    split_call_result_29224 = invoke(stypy.reporting.localization.Localization(__file__, 636, 50), split_29221, *[str_29222], **kwargs_29223)
    
    # Obtaining the member '__getitem__' of a type (line 636)
    getitem___29225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 50), split_call_result_29224, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 636)
    subscript_call_result_29226 = invoke(stypy.reporting.localization.Localization(__file__, 636, 50), getitem___29225, int_29219)
    
    str_29227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 69), 'str', 'void')
    # Applying the binary operator '==' (line 636)
    result_eq_29228 = python_operator(stypy.reporting.localization.Localization(__file__, 636, 50), '==', subscript_call_result_29226, str_29227)
    
    
    # Call to split_signature(...): (line 636)
    # Processing the call arguments (line 636)
    # Getting the type of 'l' (line 636)
    l_29216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 32), 'l', False)
    # Processing the call keyword arguments (line 636)
    kwargs_29217 = {}
    # Getting the type of 'split_signature' (line 636)
    split_signature_29215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 16), 'split_signature', False)
    # Calling split_signature(args, kwargs) (line 636)
    split_signature_call_result_29218 = invoke(stypy.reporting.localization.Localization(__file__, 636, 16), split_signature_29215, *[l_29216], **kwargs_29217)
    
    list_29231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 636, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 636, 16), list_29231, split_signature_call_result_29218)
    # Assigning a type to the variable 'sub_sigs' (line 636)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 4), 'sub_sigs', list_29231)
    
    # Assigning a Call to a Name (line 637):
    
    # Assigning a Call to a Name (line 637):
    
    # Call to list(...): (line 637)
    # Processing the call arguments (line 637)
    
    # Call to sorted(...): (line 637)
    # Processing the call arguments (line 637)
    # Getting the type of 'func_sigs' (line 637)
    func_sigs_29234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 27), 'func_sigs', False)
    # Getting the type of 'sub_sigs' (line 637)
    sub_sigs_29235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 39), 'sub_sigs', False)
    # Applying the binary operator '+' (line 637)
    result_add_29236 = python_operator(stypy.reporting.localization.Localization(__file__, 637, 27), '+', func_sigs_29234, sub_sigs_29235)
    
    # Processing the call keyword arguments (line 637)
    
    # Call to itemgetter(...): (line 637)
    # Processing the call arguments (line 637)
    int_29238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 64), 'int')
    # Processing the call keyword arguments (line 637)
    kwargs_29239 = {}
    # Getting the type of 'itemgetter' (line 637)
    itemgetter_29237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 53), 'itemgetter', False)
    # Calling itemgetter(args, kwargs) (line 637)
    itemgetter_call_result_29240 = invoke(stypy.reporting.localization.Localization(__file__, 637, 53), itemgetter_29237, *[int_29238], **kwargs_29239)
    
    keyword_29241 = itemgetter_call_result_29240
    kwargs_29242 = {'key': keyword_29241}
    # Getting the type of 'sorted' (line 637)
    sorted_29233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 20), 'sorted', False)
    # Calling sorted(args, kwargs) (line 637)
    sorted_call_result_29243 = invoke(stypy.reporting.localization.Localization(__file__, 637, 20), sorted_29233, *[result_add_29236], **kwargs_29242)
    
    # Processing the call keyword arguments (line 637)
    kwargs_29244 = {}
    # Getting the type of 'list' (line 637)
    list_29232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 15), 'list', False)
    # Calling list(args, kwargs) (line 637)
    list_call_result_29245 = invoke(stypy.reporting.localization.Localization(__file__, 637, 15), list_29232, *[sorted_call_result_29243], **kwargs_29244)
    
    # Assigning a type to the variable 'all_sigs' (line 637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 4), 'all_sigs', list_call_result_29245)
    
    # Obtaining an instance of the builtin type 'tuple' (line 638)
    tuple_29246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 638)
    # Adding element type (line 638)
    # Getting the type of 'func_sigs' (line 638)
    func_sigs_29247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 11), 'func_sigs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 11), tuple_29246, func_sigs_29247)
    # Adding element type (line 638)
    # Getting the type of 'sub_sigs' (line 638)
    sub_sigs_29248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 22), 'sub_sigs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 11), tuple_29246, sub_sigs_29248)
    # Adding element type (line 638)
    # Getting the type of 'all_sigs' (line 638)
    all_sigs_29249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 32), 'all_sigs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 11), tuple_29246, all_sigs_29249)
    
    # Assigning a type to the variable 'stypy_return_type' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'stypy_return_type', tuple_29246)
    
    # ################# End of 'filter_lines(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'filter_lines' in the type store
    # Getting the type of 'stypy_return_type' (line 633)
    stypy_return_type_29250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29250)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'filter_lines'
    return stypy_return_type_29250

# Assigning a type to the variable 'filter_lines' (line 633)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 0), 'filter_lines', filter_lines)

@norecursion
def all_newer(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'all_newer'
    module_type_store = module_type_store.open_function_context('all_newer', 641, 0, False)
    
    # Passed parameters checking function
    all_newer.stypy_localization = localization
    all_newer.stypy_type_of_self = None
    all_newer.stypy_type_store = module_type_store
    all_newer.stypy_function_name = 'all_newer'
    all_newer.stypy_param_names_list = ['src_files', 'dst_files']
    all_newer.stypy_varargs_param_name = None
    all_newer.stypy_kwargs_param_name = None
    all_newer.stypy_call_defaults = defaults
    all_newer.stypy_call_varargs = varargs
    all_newer.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'all_newer', ['src_files', 'dst_files'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'all_newer', localization, ['src_files', 'dst_files'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'all_newer(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 642, 4))
    
    # 'from distutils.dep_util import newer' statement (line 642)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_29251 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 642, 4), 'distutils.dep_util')

    if (type(import_29251) is not StypyTypeError):

        if (import_29251 != 'pyd_module'):
            __import__(import_29251)
            sys_modules_29252 = sys.modules[import_29251]
            import_from_module(stypy.reporting.localization.Localization(__file__, 642, 4), 'distutils.dep_util', sys_modules_29252.module_type_store, module_type_store, ['newer'])
            nest_module(stypy.reporting.localization.Localization(__file__, 642, 4), __file__, sys_modules_29252, sys_modules_29252.module_type_store, module_type_store)
        else:
            from distutils.dep_util import newer

            import_from_module(stypy.reporting.localization.Localization(__file__, 642, 4), 'distutils.dep_util', None, module_type_store, ['newer'], [newer])

    else:
        # Assigning a type to the variable 'distutils.dep_util' (line 642)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 4), 'distutils.dep_util', import_29251)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    
    # Call to all(...): (line 643)
    # Processing the call arguments (line 643)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 643, 15, True)
    # Calculating comprehension expression
    # Getting the type of 'dst_files' (line 644)
    dst_files_29266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 26), 'dst_files', False)
    comprehension_29267 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 643, 15), dst_files_29266)
    # Assigning a type to the variable 'dst' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 15), 'dst', comprehension_29267)
    # Calculating comprehension expression
    # Getting the type of 'src_files' (line 644)
    src_files_29268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 47), 'src_files', False)
    comprehension_29269 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 643, 15), src_files_29268)
    # Assigning a type to the variable 'src' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 15), 'src', comprehension_29269)
    
    # Evaluating a boolean operation
    
    # Call to exists(...): (line 643)
    # Processing the call arguments (line 643)
    # Getting the type of 'dst' (line 643)
    dst_29257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 30), 'dst', False)
    # Processing the call keyword arguments (line 643)
    kwargs_29258 = {}
    # Getting the type of 'os' (line 643)
    os_29254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 643)
    path_29255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 15), os_29254, 'path')
    # Obtaining the member 'exists' of a type (line 643)
    exists_29256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 15), path_29255, 'exists')
    # Calling exists(args, kwargs) (line 643)
    exists_call_result_29259 = invoke(stypy.reporting.localization.Localization(__file__, 643, 15), exists_29256, *[dst_29257], **kwargs_29258)
    
    
    # Call to newer(...): (line 643)
    # Processing the call arguments (line 643)
    # Getting the type of 'dst' (line 643)
    dst_29261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 45), 'dst', False)
    # Getting the type of 'src' (line 643)
    src_29262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 50), 'src', False)
    # Processing the call keyword arguments (line 643)
    kwargs_29263 = {}
    # Getting the type of 'newer' (line 643)
    newer_29260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 39), 'newer', False)
    # Calling newer(args, kwargs) (line 643)
    newer_call_result_29264 = invoke(stypy.reporting.localization.Localization(__file__, 643, 39), newer_29260, *[dst_29261, src_29262], **kwargs_29263)
    
    # Applying the binary operator 'and' (line 643)
    result_and_keyword_29265 = python_operator(stypy.reporting.localization.Localization(__file__, 643, 15), 'and', exists_call_result_29259, newer_call_result_29264)
    
    list_29270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 15), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 643, 15), list_29270, result_and_keyword_29265)
    # Processing the call keyword arguments (line 643)
    kwargs_29271 = {}
    # Getting the type of 'all' (line 643)
    all_29253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 11), 'all', False)
    # Calling all(args, kwargs) (line 643)
    all_call_result_29272 = invoke(stypy.reporting.localization.Localization(__file__, 643, 11), all_29253, *[list_29270], **kwargs_29271)
    
    # Assigning a type to the variable 'stypy_return_type' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 4), 'stypy_return_type', all_call_result_29272)
    
    # ################# End of 'all_newer(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'all_newer' in the type store
    # Getting the type of 'stypy_return_type' (line 641)
    stypy_return_type_29273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29273)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'all_newer'
    return stypy_return_type_29273

# Assigning a type to the variable 'all_newer' (line 641)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 0), 'all_newer', all_newer)

@norecursion
def make_all(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_29274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 33), 'str', 'cython_blas_signatures.txt')
    str_29275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 648, 35), 'str', 'cython_lapack_signatures.txt')
    str_29276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 23), 'str', 'cython_blas')
    str_29277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 650, 25), 'str', 'cython_lapack')
    str_29278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 31), 'str', '_blas_subroutine_wrappers.f')
    str_29279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 33), 'str', '_lapack_subroutine_wrappers.f')
    str_29280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 30), 'str', '_blas_subroutines.h')
    str_29281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 654, 32), 'str', '_lapack_subroutines.h')
    defaults = [str_29274, str_29275, str_29276, str_29277, str_29278, str_29279, str_29280, str_29281]
    # Create a new context for function 'make_all'
    module_type_store = module_type_store.open_function_context('make_all', 647, 0, False)
    
    # Passed parameters checking function
    make_all.stypy_localization = localization
    make_all.stypy_type_of_self = None
    make_all.stypy_type_store = module_type_store
    make_all.stypy_function_name = 'make_all'
    make_all.stypy_param_names_list = ['blas_signature_file', 'lapack_signature_file', 'blas_name', 'lapack_name', 'blas_fortran_name', 'lapack_fortran_name', 'blas_header_name', 'lapack_header_name']
    make_all.stypy_varargs_param_name = None
    make_all.stypy_kwargs_param_name = None
    make_all.stypy_call_defaults = defaults
    make_all.stypy_call_varargs = varargs
    make_all.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_all', ['blas_signature_file', 'lapack_signature_file', 'blas_name', 'lapack_name', 'blas_fortran_name', 'lapack_fortran_name', 'blas_header_name', 'lapack_header_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_all', localization, ['blas_signature_file', 'lapack_signature_file', 'blas_name', 'lapack_name', 'blas_fortran_name', 'lapack_fortran_name', 'blas_header_name', 'lapack_header_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_all(...)' code ##################

    
    # Assigning a Tuple to a Name (line 656):
    
    # Assigning a Tuple to a Name (line 656):
    
    # Obtaining an instance of the builtin type 'tuple' (line 656)
    tuple_29282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 656, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 656)
    # Adding element type (line 656)
    
    # Call to abspath(...): (line 656)
    # Processing the call arguments (line 656)
    # Getting the type of '__file__' (line 656)
    file___29286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 33), '__file__', False)
    # Processing the call keyword arguments (line 656)
    kwargs_29287 = {}
    # Getting the type of 'os' (line 656)
    os_29283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 17), 'os', False)
    # Obtaining the member 'path' of a type (line 656)
    path_29284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 17), os_29283, 'path')
    # Obtaining the member 'abspath' of a type (line 656)
    abspath_29285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 656, 17), path_29284, 'abspath')
    # Calling abspath(args, kwargs) (line 656)
    abspath_call_result_29288 = invoke(stypy.reporting.localization.Localization(__file__, 656, 17), abspath_29285, *[file___29286], **kwargs_29287)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 656, 17), tuple_29282, abspath_call_result_29288)
    # Adding element type (line 656)
    # Getting the type of 'blas_signature_file' (line 657)
    blas_signature_file_29289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 17), 'blas_signature_file')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 656, 17), tuple_29282, blas_signature_file_29289)
    # Adding element type (line 656)
    # Getting the type of 'lapack_signature_file' (line 658)
    lapack_signature_file_29290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 17), 'lapack_signature_file')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 656, 17), tuple_29282, lapack_signature_file_29290)
    
    # Assigning a type to the variable 'src_files' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 4), 'src_files', tuple_29282)
    
    # Assigning a Tuple to a Name (line 659):
    
    # Assigning a Tuple to a Name (line 659):
    
    # Obtaining an instance of the builtin type 'tuple' (line 659)
    tuple_29291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 659)
    # Adding element type (line 659)
    # Getting the type of 'blas_name' (line 659)
    blas_name_29292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 17), 'blas_name')
    str_29293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 659, 29), 'str', '.pyx')
    # Applying the binary operator '+' (line 659)
    result_add_29294 = python_operator(stypy.reporting.localization.Localization(__file__, 659, 17), '+', blas_name_29292, str_29293)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 659, 17), tuple_29291, result_add_29294)
    # Adding element type (line 659)
    # Getting the type of 'blas_name' (line 660)
    blas_name_29295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 17), 'blas_name')
    str_29296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 29), 'str', '.pxd')
    # Applying the binary operator '+' (line 660)
    result_add_29297 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 17), '+', blas_name_29295, str_29296)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 659, 17), tuple_29291, result_add_29297)
    # Adding element type (line 659)
    # Getting the type of 'blas_fortran_name' (line 661)
    blas_fortran_name_29298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 17), 'blas_fortran_name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 659, 17), tuple_29291, blas_fortran_name_29298)
    # Adding element type (line 659)
    # Getting the type of 'blas_header_name' (line 662)
    blas_header_name_29299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 17), 'blas_header_name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 659, 17), tuple_29291, blas_header_name_29299)
    # Adding element type (line 659)
    # Getting the type of 'lapack_name' (line 663)
    lapack_name_29300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 17), 'lapack_name')
    str_29301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 31), 'str', '.pyx')
    # Applying the binary operator '+' (line 663)
    result_add_29302 = python_operator(stypy.reporting.localization.Localization(__file__, 663, 17), '+', lapack_name_29300, str_29301)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 659, 17), tuple_29291, result_add_29302)
    # Adding element type (line 659)
    # Getting the type of 'lapack_name' (line 664)
    lapack_name_29303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 17), 'lapack_name')
    str_29304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 31), 'str', '.pxd')
    # Applying the binary operator '+' (line 664)
    result_add_29305 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 17), '+', lapack_name_29303, str_29304)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 659, 17), tuple_29291, result_add_29305)
    # Adding element type (line 659)
    # Getting the type of 'lapack_fortran_name' (line 665)
    lapack_fortran_name_29306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 17), 'lapack_fortran_name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 659, 17), tuple_29291, lapack_fortran_name_29306)
    # Adding element type (line 659)
    # Getting the type of 'lapack_header_name' (line 666)
    lapack_header_name_29307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 17), 'lapack_header_name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 659, 17), tuple_29291, lapack_header_name_29307)
    
    # Assigning a type to the variable 'dst_files' (line 659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 4), 'dst_files', tuple_29291)
    
    # Call to chdir(...): (line 668)
    # Processing the call arguments (line 668)
    # Getting the type of 'BASE_DIR' (line 668)
    BASE_DIR_29310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 13), 'BASE_DIR', False)
    # Processing the call keyword arguments (line 668)
    kwargs_29311 = {}
    # Getting the type of 'os' (line 668)
    os_29308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 4), 'os', False)
    # Obtaining the member 'chdir' of a type (line 668)
    chdir_29309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 4), os_29308, 'chdir')
    # Calling chdir(args, kwargs) (line 668)
    chdir_call_result_29312 = invoke(stypy.reporting.localization.Localization(__file__, 668, 4), chdir_29309, *[BASE_DIR_29310], **kwargs_29311)
    
    
    
    # Call to all_newer(...): (line 670)
    # Processing the call arguments (line 670)
    # Getting the type of 'src_files' (line 670)
    src_files_29314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 17), 'src_files', False)
    # Getting the type of 'dst_files' (line 670)
    dst_files_29315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 28), 'dst_files', False)
    # Processing the call keyword arguments (line 670)
    kwargs_29316 = {}
    # Getting the type of 'all_newer' (line 670)
    all_newer_29313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 7), 'all_newer', False)
    # Calling all_newer(args, kwargs) (line 670)
    all_newer_call_result_29317 = invoke(stypy.reporting.localization.Localization(__file__, 670, 7), all_newer_29313, *[src_files_29314, dst_files_29315], **kwargs_29316)
    
    # Testing the type of an if condition (line 670)
    if_condition_29318 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 670, 4), all_newer_call_result_29317)
    # Assigning a type to the variable 'if_condition_29318' (line 670)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 4), 'if_condition_29318', if_condition_29318)
    # SSA begins for if statement (line 670)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_29319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 671, 14), 'str', 'scipy/linalg/_generate_pyx.py: all files up-to-date')
    # Assigning a type to the variable 'stypy_return_type' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 670)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 674):
    
    # Assigning a List to a Name (line 674):
    
    # Obtaining an instance of the builtin type 'list' (line 674)
    list_29320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 674)
    # Adding element type (line 674)
    str_29321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 674, 16), 'str', 'This file was generated by _generate_pyx.py.\n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 674, 15), list_29320, str_29321)
    # Adding element type (line 674)
    str_29322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 16), 'str', 'Do not edit this file directly.\n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 674, 15), list_29320, str_29322)
    
    # Assigning a type to the variable 'comments' (line 674)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 674, 4), 'comments', list_29320)
    
    # Assigning a BinOp to a Name (line 676):
    
    # Assigning a BinOp to a Name (line 676):
    
    # Call to join(...): (line 676)
    # Processing the call arguments (line 676)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'comments' (line 676)
    comments_29333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 68), 'comments', False)
    comprehension_29334 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 24), comments_29333)
    # Assigning a type to the variable 'line' (line 676)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 24), 'line', comprehension_29334)
    str_29325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 24), 'str', '/* ')
    
    # Call to rstrip(...): (line 676)
    # Processing the call keyword arguments (line 676)
    kwargs_29328 = {}
    # Getting the type of 'line' (line 676)
    line_29326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 32), 'line', False)
    # Obtaining the member 'rstrip' of a type (line 676)
    rstrip_29327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 32), line_29326, 'rstrip')
    # Calling rstrip(args, kwargs) (line 676)
    rstrip_call_result_29329 = invoke(stypy.reporting.localization.Localization(__file__, 676, 32), rstrip_29327, *[], **kwargs_29328)
    
    # Applying the binary operator '+' (line 676)
    result_add_29330 = python_operator(stypy.reporting.localization.Localization(__file__, 676, 24), '+', str_29325, rstrip_call_result_29329)
    
    str_29331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 48), 'str', ' */\n')
    # Applying the binary operator '+' (line 676)
    result_add_29332 = python_operator(stypy.reporting.localization.Localization(__file__, 676, 46), '+', result_add_29330, str_29331)
    
    list_29335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 24), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 24), list_29335, result_add_29332)
    # Processing the call keyword arguments (line 676)
    kwargs_29336 = {}
    str_29323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 15), 'str', '')
    # Obtaining the member 'join' of a type (line 676)
    join_29324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 676, 15), str_29323, 'join')
    # Calling join(args, kwargs) (line 676)
    join_call_result_29337 = invoke(stypy.reporting.localization.Localization(__file__, 676, 15), join_29324, *[list_29335], **kwargs_29336)
    
    str_29338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 81), 'str', '\n')
    # Applying the binary operator '+' (line 676)
    result_add_29339 = python_operator(stypy.reporting.localization.Localization(__file__, 676, 15), '+', join_call_result_29337, str_29338)
    
    # Assigning a type to the variable 'ccomment' (line 676)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 4), 'ccomment', result_add_29339)
    
    # Assigning a BinOp to a Name (line 677):
    
    # Assigning a BinOp to a Name (line 677):
    
    # Call to join(...): (line 677)
    # Processing the call arguments (line 677)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'comments' (line 677)
    comments_29345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 50), 'comments', False)
    comprehension_29346 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 677, 26), comments_29345)
    # Assigning a type to the variable 'line' (line 677)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 26), 'line', comprehension_29346)
    str_29342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 26), 'str', '# ')
    # Getting the type of 'line' (line 677)
    line_29343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 33), 'line', False)
    # Applying the binary operator '+' (line 677)
    result_add_29344 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 26), '+', str_29342, line_29343)
    
    list_29347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 26), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 677, 26), list_29347, result_add_29344)
    # Processing the call keyword arguments (line 677)
    kwargs_29348 = {}
    str_29340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 17), 'str', '')
    # Obtaining the member 'join' of a type (line 677)
    join_29341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 677, 17), str_29340, 'join')
    # Calling join(args, kwargs) (line 677)
    join_call_result_29349 = invoke(stypy.reporting.localization.Localization(__file__, 677, 17), join_29341, *[list_29347], **kwargs_29348)
    
    str_29350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 63), 'str', '\n')
    # Applying the binary operator '+' (line 677)
    result_add_29351 = python_operator(stypy.reporting.localization.Localization(__file__, 677, 17), '+', join_call_result_29349, str_29350)
    
    # Assigning a type to the variable 'pyxcomment' (line 677)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 4), 'pyxcomment', result_add_29351)
    
    # Assigning a BinOp to a Name (line 678):
    
    # Assigning a BinOp to a Name (line 678):
    
    # Call to join(...): (line 678)
    # Processing the call arguments (line 678)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'comments' (line 678)
    comments_29357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 52), 'comments', False)
    comprehension_29358 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 678, 24), comments_29357)
    # Assigning a type to the variable 'line' (line 678)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 24), 'line', comprehension_29358)
    str_29354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 24), 'str', 'c     ')
    # Getting the type of 'line' (line 678)
    line_29355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 35), 'line', False)
    # Applying the binary operator '+' (line 678)
    result_add_29356 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 24), '+', str_29354, line_29355)
    
    list_29359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 24), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 678, 24), list_29359, result_add_29356)
    # Processing the call keyword arguments (line 678)
    kwargs_29360 = {}
    str_29352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 15), 'str', '')
    # Obtaining the member 'join' of a type (line 678)
    join_29353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 678, 15), str_29352, 'join')
    # Calling join(args, kwargs) (line 678)
    join_call_result_29361 = invoke(stypy.reporting.localization.Localization(__file__, 678, 15), join_29353, *[list_29359], **kwargs_29360)
    
    str_29362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 678, 65), 'str', '\n')
    # Applying the binary operator '+' (line 678)
    result_add_29363 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 15), '+', join_call_result_29361, str_29362)
    
    # Assigning a type to the variable 'fcomment' (line 678)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 4), 'fcomment', result_add_29363)
    
    # Call to open(...): (line 679)
    # Processing the call arguments (line 679)
    # Getting the type of 'blas_signature_file' (line 679)
    blas_signature_file_29365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 14), 'blas_signature_file', False)
    str_29366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 679, 35), 'str', 'r')
    # Processing the call keyword arguments (line 679)
    kwargs_29367 = {}
    # Getting the type of 'open' (line 679)
    open_29364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 9), 'open', False)
    # Calling open(args, kwargs) (line 679)
    open_call_result_29368 = invoke(stypy.reporting.localization.Localization(__file__, 679, 9), open_29364, *[blas_signature_file_29365, str_29366], **kwargs_29367)
    
    with_29369 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 679, 9), open_call_result_29368, 'with parameter', '__enter__', '__exit__')

    if with_29369:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 679)
        enter___29370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 9), open_call_result_29368, '__enter__')
        with_enter_29371 = invoke(stypy.reporting.localization.Localization(__file__, 679, 9), enter___29370)
        # Assigning a type to the variable 'f' (line 679)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 9), 'f', with_enter_29371)
        
        # Assigning a Call to a Name (line 680):
        
        # Assigning a Call to a Name (line 680):
        
        # Call to readlines(...): (line 680)
        # Processing the call keyword arguments (line 680)
        kwargs_29374 = {}
        # Getting the type of 'f' (line 680)
        f_29372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 20), 'f', False)
        # Obtaining the member 'readlines' of a type (line 680)
        readlines_29373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 20), f_29372, 'readlines')
        # Calling readlines(args, kwargs) (line 680)
        readlines_call_result_29375 = invoke(stypy.reporting.localization.Localization(__file__, 680, 20), readlines_29373, *[], **kwargs_29374)
        
        # Assigning a type to the variable 'blas_sigs' (line 680)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 8), 'blas_sigs', readlines_call_result_29375)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 679)
        exit___29376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 679, 9), open_call_result_29368, '__exit__')
        with_exit_29377 = invoke(stypy.reporting.localization.Localization(__file__, 679, 9), exit___29376, None, None, None)

    
    # Assigning a Call to a Name (line 681):
    
    # Assigning a Call to a Name (line 681):
    
    # Call to filter_lines(...): (line 681)
    # Processing the call arguments (line 681)
    # Getting the type of 'blas_sigs' (line 681)
    blas_sigs_29379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 29), 'blas_sigs', False)
    # Processing the call keyword arguments (line 681)
    kwargs_29380 = {}
    # Getting the type of 'filter_lines' (line 681)
    filter_lines_29378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 16), 'filter_lines', False)
    # Calling filter_lines(args, kwargs) (line 681)
    filter_lines_call_result_29381 = invoke(stypy.reporting.localization.Localization(__file__, 681, 16), filter_lines_29378, *[blas_sigs_29379], **kwargs_29380)
    
    # Assigning a type to the variable 'blas_sigs' (line 681)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 4), 'blas_sigs', filter_lines_call_result_29381)
    
    # Assigning a Call to a Name (line 682):
    
    # Assigning a Call to a Name (line 682):
    
    # Call to generate_blas_pyx(...): (line 682)
    # Getting the type of 'blas_sigs' (line 682)
    blas_sigs_29383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 35), 'blas_sigs', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 682)
    tuple_29384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 682)
    # Adding element type (line 682)
    # Getting the type of 'blas_header_name' (line 682)
    blas_header_name_29385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 48), 'blas_header_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 682, 48), tuple_29384, blas_header_name_29385)
    
    # Applying the binary operator '+' (line 682)
    result_add_29386 = python_operator(stypy.reporting.localization.Localization(__file__, 682, 35), '+', blas_sigs_29383, tuple_29384)
    
    # Processing the call keyword arguments (line 682)
    kwargs_29387 = {}
    # Getting the type of 'generate_blas_pyx' (line 682)
    generate_blas_pyx_29382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 15), 'generate_blas_pyx', False)
    # Calling generate_blas_pyx(args, kwargs) (line 682)
    generate_blas_pyx_call_result_29388 = invoke(stypy.reporting.localization.Localization(__file__, 682, 15), generate_blas_pyx_29382, *[result_add_29386], **kwargs_29387)
    
    # Assigning a type to the variable 'blas_pyx' (line 682)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 4), 'blas_pyx', generate_blas_pyx_call_result_29388)
    
    # Call to open(...): (line 683)
    # Processing the call arguments (line 683)
    # Getting the type of 'blas_name' (line 683)
    blas_name_29390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 14), 'blas_name', False)
    str_29391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 26), 'str', '.pyx')
    # Applying the binary operator '+' (line 683)
    result_add_29392 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 14), '+', blas_name_29390, str_29391)
    
    str_29393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 34), 'str', 'w')
    # Processing the call keyword arguments (line 683)
    kwargs_29394 = {}
    # Getting the type of 'open' (line 683)
    open_29389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 9), 'open', False)
    # Calling open(args, kwargs) (line 683)
    open_call_result_29395 = invoke(stypy.reporting.localization.Localization(__file__, 683, 9), open_29389, *[result_add_29392, str_29393], **kwargs_29394)
    
    with_29396 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 683, 9), open_call_result_29395, 'with parameter', '__enter__', '__exit__')

    if with_29396:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 683)
        enter___29397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 9), open_call_result_29395, '__enter__')
        with_enter_29398 = invoke(stypy.reporting.localization.Localization(__file__, 683, 9), enter___29397)
        # Assigning a type to the variable 'f' (line 683)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 9), 'f', with_enter_29398)
        
        # Call to write(...): (line 684)
        # Processing the call arguments (line 684)
        # Getting the type of 'pyxcomment' (line 684)
        pyxcomment_29401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 16), 'pyxcomment', False)
        # Processing the call keyword arguments (line 684)
        kwargs_29402 = {}
        # Getting the type of 'f' (line 684)
        f_29399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 684)
        write_29400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 684, 8), f_29399, 'write')
        # Calling write(args, kwargs) (line 684)
        write_call_result_29403 = invoke(stypy.reporting.localization.Localization(__file__, 684, 8), write_29400, *[pyxcomment_29401], **kwargs_29402)
        
        
        # Call to write(...): (line 685)
        # Processing the call arguments (line 685)
        # Getting the type of 'blas_pyx' (line 685)
        blas_pyx_29406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 16), 'blas_pyx', False)
        # Processing the call keyword arguments (line 685)
        kwargs_29407 = {}
        # Getting the type of 'f' (line 685)
        f_29404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 685)
        write_29405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 685, 8), f_29404, 'write')
        # Calling write(args, kwargs) (line 685)
        write_call_result_29408 = invoke(stypy.reporting.localization.Localization(__file__, 685, 8), write_29405, *[blas_pyx_29406], **kwargs_29407)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 683)
        exit___29409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 683, 9), open_call_result_29395, '__exit__')
        with_exit_29410 = invoke(stypy.reporting.localization.Localization(__file__, 683, 9), exit___29409, None, None, None)

    
    # Assigning a Call to a Name (line 686):
    
    # Assigning a Call to a Name (line 686):
    
    # Call to generate_blas_pxd(...): (line 686)
    # Processing the call arguments (line 686)
    
    # Obtaining the type of the subscript
    int_29412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 43), 'int')
    # Getting the type of 'blas_sigs' (line 686)
    blas_sigs_29413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 33), 'blas_sigs', False)
    # Obtaining the member '__getitem__' of a type (line 686)
    getitem___29414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 33), blas_sigs_29413, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 686)
    subscript_call_result_29415 = invoke(stypy.reporting.localization.Localization(__file__, 686, 33), getitem___29414, int_29412)
    
    # Processing the call keyword arguments (line 686)
    kwargs_29416 = {}
    # Getting the type of 'generate_blas_pxd' (line 686)
    generate_blas_pxd_29411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 15), 'generate_blas_pxd', False)
    # Calling generate_blas_pxd(args, kwargs) (line 686)
    generate_blas_pxd_call_result_29417 = invoke(stypy.reporting.localization.Localization(__file__, 686, 15), generate_blas_pxd_29411, *[subscript_call_result_29415], **kwargs_29416)
    
    # Assigning a type to the variable 'blas_pxd' (line 686)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 4), 'blas_pxd', generate_blas_pxd_call_result_29417)
    
    # Call to open(...): (line 687)
    # Processing the call arguments (line 687)
    # Getting the type of 'blas_name' (line 687)
    blas_name_29419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 14), 'blas_name', False)
    str_29420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 26), 'str', '.pxd')
    # Applying the binary operator '+' (line 687)
    result_add_29421 = python_operator(stypy.reporting.localization.Localization(__file__, 687, 14), '+', blas_name_29419, str_29420)
    
    str_29422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 34), 'str', 'w')
    # Processing the call keyword arguments (line 687)
    kwargs_29423 = {}
    # Getting the type of 'open' (line 687)
    open_29418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 9), 'open', False)
    # Calling open(args, kwargs) (line 687)
    open_call_result_29424 = invoke(stypy.reporting.localization.Localization(__file__, 687, 9), open_29418, *[result_add_29421, str_29422], **kwargs_29423)
    
    with_29425 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 687, 9), open_call_result_29424, 'with parameter', '__enter__', '__exit__')

    if with_29425:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 687)
        enter___29426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 9), open_call_result_29424, '__enter__')
        with_enter_29427 = invoke(stypy.reporting.localization.Localization(__file__, 687, 9), enter___29426)
        # Assigning a type to the variable 'f' (line 687)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 9), 'f', with_enter_29427)
        
        # Call to write(...): (line 688)
        # Processing the call arguments (line 688)
        # Getting the type of 'pyxcomment' (line 688)
        pyxcomment_29430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 16), 'pyxcomment', False)
        # Processing the call keyword arguments (line 688)
        kwargs_29431 = {}
        # Getting the type of 'f' (line 688)
        f_29428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 688)
        write_29429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 688, 8), f_29428, 'write')
        # Calling write(args, kwargs) (line 688)
        write_call_result_29432 = invoke(stypy.reporting.localization.Localization(__file__, 688, 8), write_29429, *[pyxcomment_29430], **kwargs_29431)
        
        
        # Call to write(...): (line 689)
        # Processing the call arguments (line 689)
        # Getting the type of 'blas_pxd' (line 689)
        blas_pxd_29435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 16), 'blas_pxd', False)
        # Processing the call keyword arguments (line 689)
        kwargs_29436 = {}
        # Getting the type of 'f' (line 689)
        f_29433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 689)
        write_29434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 8), f_29433, 'write')
        # Calling write(args, kwargs) (line 689)
        write_call_result_29437 = invoke(stypy.reporting.localization.Localization(__file__, 689, 8), write_29434, *[blas_pxd_29435], **kwargs_29436)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 687)
        exit___29438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 9), open_call_result_29424, '__exit__')
        with_exit_29439 = invoke(stypy.reporting.localization.Localization(__file__, 687, 9), exit___29438, None, None, None)

    
    # Assigning a Call to a Name (line 690):
    
    # Assigning a Call to a Name (line 690):
    
    # Call to generate_fortran(...): (line 690)
    # Processing the call arguments (line 690)
    
    # Obtaining the type of the subscript
    int_29441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 46), 'int')
    # Getting the type of 'blas_sigs' (line 690)
    blas_sigs_29442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 36), 'blas_sigs', False)
    # Obtaining the member '__getitem__' of a type (line 690)
    getitem___29443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 36), blas_sigs_29442, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 690)
    subscript_call_result_29444 = invoke(stypy.reporting.localization.Localization(__file__, 690, 36), getitem___29443, int_29441)
    
    # Processing the call keyword arguments (line 690)
    kwargs_29445 = {}
    # Getting the type of 'generate_fortran' (line 690)
    generate_fortran_29440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 19), 'generate_fortran', False)
    # Calling generate_fortran(args, kwargs) (line 690)
    generate_fortran_call_result_29446 = invoke(stypy.reporting.localization.Localization(__file__, 690, 19), generate_fortran_29440, *[subscript_call_result_29444], **kwargs_29445)
    
    # Assigning a type to the variable 'blas_fortran' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'blas_fortran', generate_fortran_call_result_29446)
    
    # Call to open(...): (line 691)
    # Processing the call arguments (line 691)
    # Getting the type of 'blas_fortran_name' (line 691)
    blas_fortran_name_29448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 14), 'blas_fortran_name', False)
    str_29449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 33), 'str', 'w')
    # Processing the call keyword arguments (line 691)
    kwargs_29450 = {}
    # Getting the type of 'open' (line 691)
    open_29447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 9), 'open', False)
    # Calling open(args, kwargs) (line 691)
    open_call_result_29451 = invoke(stypy.reporting.localization.Localization(__file__, 691, 9), open_29447, *[blas_fortran_name_29448, str_29449], **kwargs_29450)
    
    with_29452 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 691, 9), open_call_result_29451, 'with parameter', '__enter__', '__exit__')

    if with_29452:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 691)
        enter___29453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 9), open_call_result_29451, '__enter__')
        with_enter_29454 = invoke(stypy.reporting.localization.Localization(__file__, 691, 9), enter___29453)
        # Assigning a type to the variable 'f' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 9), 'f', with_enter_29454)
        
        # Call to write(...): (line 692)
        # Processing the call arguments (line 692)
        # Getting the type of 'fcomment' (line 692)
        fcomment_29457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 16), 'fcomment', False)
        # Processing the call keyword arguments (line 692)
        kwargs_29458 = {}
        # Getting the type of 'f' (line 692)
        f_29455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 692)
        write_29456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 8), f_29455, 'write')
        # Calling write(args, kwargs) (line 692)
        write_call_result_29459 = invoke(stypy.reporting.localization.Localization(__file__, 692, 8), write_29456, *[fcomment_29457], **kwargs_29458)
        
        
        # Call to write(...): (line 693)
        # Processing the call arguments (line 693)
        # Getting the type of 'blas_fortran' (line 693)
        blas_fortran_29462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 16), 'blas_fortran', False)
        # Processing the call keyword arguments (line 693)
        kwargs_29463 = {}
        # Getting the type of 'f' (line 693)
        f_29460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 693)
        write_29461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 8), f_29460, 'write')
        # Calling write(args, kwargs) (line 693)
        write_call_result_29464 = invoke(stypy.reporting.localization.Localization(__file__, 693, 8), write_29461, *[blas_fortran_29462], **kwargs_29463)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 691)
        exit___29465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 9), open_call_result_29451, '__exit__')
        with_exit_29466 = invoke(stypy.reporting.localization.Localization(__file__, 691, 9), exit___29465, None, None, None)

    
    # Assigning a Call to a Name (line 694):
    
    # Assigning a Call to a Name (line 694):
    
    # Call to generate_c_header(...): (line 694)
    # Getting the type of 'blas_sigs' (line 694)
    blas_sigs_29468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 40), 'blas_sigs', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 694)
    tuple_29469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 694)
    # Adding element type (line 694)
    str_29470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 53), 'str', 'BLAS')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 694, 53), tuple_29469, str_29470)
    
    # Applying the binary operator '+' (line 694)
    result_add_29471 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 40), '+', blas_sigs_29468, tuple_29469)
    
    # Processing the call keyword arguments (line 694)
    kwargs_29472 = {}
    # Getting the type of 'generate_c_header' (line 694)
    generate_c_header_29467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 20), 'generate_c_header', False)
    # Calling generate_c_header(args, kwargs) (line 694)
    generate_c_header_call_result_29473 = invoke(stypy.reporting.localization.Localization(__file__, 694, 20), generate_c_header_29467, *[result_add_29471], **kwargs_29472)
    
    # Assigning a type to the variable 'blas_c_header' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'blas_c_header', generate_c_header_call_result_29473)
    
    # Call to open(...): (line 695)
    # Processing the call arguments (line 695)
    # Getting the type of 'blas_header_name' (line 695)
    blas_header_name_29475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 14), 'blas_header_name', False)
    str_29476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 32), 'str', 'w')
    # Processing the call keyword arguments (line 695)
    kwargs_29477 = {}
    # Getting the type of 'open' (line 695)
    open_29474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 9), 'open', False)
    # Calling open(args, kwargs) (line 695)
    open_call_result_29478 = invoke(stypy.reporting.localization.Localization(__file__, 695, 9), open_29474, *[blas_header_name_29475, str_29476], **kwargs_29477)
    
    with_29479 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 695, 9), open_call_result_29478, 'with parameter', '__enter__', '__exit__')

    if with_29479:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 695)
        enter___29480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 9), open_call_result_29478, '__enter__')
        with_enter_29481 = invoke(stypy.reporting.localization.Localization(__file__, 695, 9), enter___29480)
        # Assigning a type to the variable 'f' (line 695)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 9), 'f', with_enter_29481)
        
        # Call to write(...): (line 696)
        # Processing the call arguments (line 696)
        # Getting the type of 'ccomment' (line 696)
        ccomment_29484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 16), 'ccomment', False)
        # Processing the call keyword arguments (line 696)
        kwargs_29485 = {}
        # Getting the type of 'f' (line 696)
        f_29482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 696)
        write_29483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 696, 8), f_29482, 'write')
        # Calling write(args, kwargs) (line 696)
        write_call_result_29486 = invoke(stypy.reporting.localization.Localization(__file__, 696, 8), write_29483, *[ccomment_29484], **kwargs_29485)
        
        
        # Call to write(...): (line 697)
        # Processing the call arguments (line 697)
        # Getting the type of 'blas_c_header' (line 697)
        blas_c_header_29489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'blas_c_header', False)
        # Processing the call keyword arguments (line 697)
        kwargs_29490 = {}
        # Getting the type of 'f' (line 697)
        f_29487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 697)
        write_29488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 8), f_29487, 'write')
        # Calling write(args, kwargs) (line 697)
        write_call_result_29491 = invoke(stypy.reporting.localization.Localization(__file__, 697, 8), write_29488, *[blas_c_header_29489], **kwargs_29490)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 695)
        exit___29492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 9), open_call_result_29478, '__exit__')
        with_exit_29493 = invoke(stypy.reporting.localization.Localization(__file__, 695, 9), exit___29492, None, None, None)

    
    # Call to open(...): (line 698)
    # Processing the call arguments (line 698)
    # Getting the type of 'lapack_signature_file' (line 698)
    lapack_signature_file_29495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 14), 'lapack_signature_file', False)
    str_29496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 37), 'str', 'r')
    # Processing the call keyword arguments (line 698)
    kwargs_29497 = {}
    # Getting the type of 'open' (line 698)
    open_29494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 9), 'open', False)
    # Calling open(args, kwargs) (line 698)
    open_call_result_29498 = invoke(stypy.reporting.localization.Localization(__file__, 698, 9), open_29494, *[lapack_signature_file_29495, str_29496], **kwargs_29497)
    
    with_29499 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 698, 9), open_call_result_29498, 'with parameter', '__enter__', '__exit__')

    if with_29499:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 698)
        enter___29500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 9), open_call_result_29498, '__enter__')
        with_enter_29501 = invoke(stypy.reporting.localization.Localization(__file__, 698, 9), enter___29500)
        # Assigning a type to the variable 'f' (line 698)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 9), 'f', with_enter_29501)
        
        # Assigning a Call to a Name (line 699):
        
        # Assigning a Call to a Name (line 699):
        
        # Call to readlines(...): (line 699)
        # Processing the call keyword arguments (line 699)
        kwargs_29504 = {}
        # Getting the type of 'f' (line 699)
        f_29502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 22), 'f', False)
        # Obtaining the member 'readlines' of a type (line 699)
        readlines_29503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 699, 22), f_29502, 'readlines')
        # Calling readlines(args, kwargs) (line 699)
        readlines_call_result_29505 = invoke(stypy.reporting.localization.Localization(__file__, 699, 22), readlines_29503, *[], **kwargs_29504)
        
        # Assigning a type to the variable 'lapack_sigs' (line 699)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 8), 'lapack_sigs', readlines_call_result_29505)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 698)
        exit___29506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 9), open_call_result_29498, '__exit__')
        with_exit_29507 = invoke(stypy.reporting.localization.Localization(__file__, 698, 9), exit___29506, None, None, None)

    
    # Assigning a Call to a Name (line 700):
    
    # Assigning a Call to a Name (line 700):
    
    # Call to filter_lines(...): (line 700)
    # Processing the call arguments (line 700)
    # Getting the type of 'lapack_sigs' (line 700)
    lapack_sigs_29509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 31), 'lapack_sigs', False)
    # Processing the call keyword arguments (line 700)
    kwargs_29510 = {}
    # Getting the type of 'filter_lines' (line 700)
    filter_lines_29508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 18), 'filter_lines', False)
    # Calling filter_lines(args, kwargs) (line 700)
    filter_lines_call_result_29511 = invoke(stypy.reporting.localization.Localization(__file__, 700, 18), filter_lines_29508, *[lapack_sigs_29509], **kwargs_29510)
    
    # Assigning a type to the variable 'lapack_sigs' (line 700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 4), 'lapack_sigs', filter_lines_call_result_29511)
    
    # Assigning a Call to a Name (line 701):
    
    # Assigning a Call to a Name (line 701):
    
    # Call to generate_lapack_pyx(...): (line 701)
    # Getting the type of 'lapack_sigs' (line 701)
    lapack_sigs_29513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 39), 'lapack_sigs', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 701)
    tuple_29514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 54), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 701)
    # Adding element type (line 701)
    # Getting the type of 'lapack_header_name' (line 701)
    lapack_header_name_29515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 54), 'lapack_header_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 701, 54), tuple_29514, lapack_header_name_29515)
    
    # Applying the binary operator '+' (line 701)
    result_add_29516 = python_operator(stypy.reporting.localization.Localization(__file__, 701, 39), '+', lapack_sigs_29513, tuple_29514)
    
    # Processing the call keyword arguments (line 701)
    kwargs_29517 = {}
    # Getting the type of 'generate_lapack_pyx' (line 701)
    generate_lapack_pyx_29512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 17), 'generate_lapack_pyx', False)
    # Calling generate_lapack_pyx(args, kwargs) (line 701)
    generate_lapack_pyx_call_result_29518 = invoke(stypy.reporting.localization.Localization(__file__, 701, 17), generate_lapack_pyx_29512, *[result_add_29516], **kwargs_29517)
    
    # Assigning a type to the variable 'lapack_pyx' (line 701)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 4), 'lapack_pyx', generate_lapack_pyx_call_result_29518)
    
    # Call to open(...): (line 702)
    # Processing the call arguments (line 702)
    # Getting the type of 'lapack_name' (line 702)
    lapack_name_29520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 14), 'lapack_name', False)
    str_29521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 28), 'str', '.pyx')
    # Applying the binary operator '+' (line 702)
    result_add_29522 = python_operator(stypy.reporting.localization.Localization(__file__, 702, 14), '+', lapack_name_29520, str_29521)
    
    str_29523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 36), 'str', 'w')
    # Processing the call keyword arguments (line 702)
    kwargs_29524 = {}
    # Getting the type of 'open' (line 702)
    open_29519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 9), 'open', False)
    # Calling open(args, kwargs) (line 702)
    open_call_result_29525 = invoke(stypy.reporting.localization.Localization(__file__, 702, 9), open_29519, *[result_add_29522, str_29523], **kwargs_29524)
    
    with_29526 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 702, 9), open_call_result_29525, 'with parameter', '__enter__', '__exit__')

    if with_29526:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 702)
        enter___29527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 9), open_call_result_29525, '__enter__')
        with_enter_29528 = invoke(stypy.reporting.localization.Localization(__file__, 702, 9), enter___29527)
        # Assigning a type to the variable 'f' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 9), 'f', with_enter_29528)
        
        # Call to write(...): (line 703)
        # Processing the call arguments (line 703)
        # Getting the type of 'pyxcomment' (line 703)
        pyxcomment_29531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 16), 'pyxcomment', False)
        # Processing the call keyword arguments (line 703)
        kwargs_29532 = {}
        # Getting the type of 'f' (line 703)
        f_29529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 703)
        write_29530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 703, 8), f_29529, 'write')
        # Calling write(args, kwargs) (line 703)
        write_call_result_29533 = invoke(stypy.reporting.localization.Localization(__file__, 703, 8), write_29530, *[pyxcomment_29531], **kwargs_29532)
        
        
        # Call to write(...): (line 704)
        # Processing the call arguments (line 704)
        # Getting the type of 'lapack_pyx' (line 704)
        lapack_pyx_29536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 16), 'lapack_pyx', False)
        # Processing the call keyword arguments (line 704)
        kwargs_29537 = {}
        # Getting the type of 'f' (line 704)
        f_29534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 704)
        write_29535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 704, 8), f_29534, 'write')
        # Calling write(args, kwargs) (line 704)
        write_call_result_29538 = invoke(stypy.reporting.localization.Localization(__file__, 704, 8), write_29535, *[lapack_pyx_29536], **kwargs_29537)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 702)
        exit___29539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 9), open_call_result_29525, '__exit__')
        with_exit_29540 = invoke(stypy.reporting.localization.Localization(__file__, 702, 9), exit___29539, None, None, None)

    
    # Assigning a Call to a Name (line 705):
    
    # Assigning a Call to a Name (line 705):
    
    # Call to generate_lapack_pxd(...): (line 705)
    # Processing the call arguments (line 705)
    
    # Obtaining the type of the subscript
    int_29542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 49), 'int')
    # Getting the type of 'lapack_sigs' (line 705)
    lapack_sigs_29543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 37), 'lapack_sigs', False)
    # Obtaining the member '__getitem__' of a type (line 705)
    getitem___29544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 37), lapack_sigs_29543, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 705)
    subscript_call_result_29545 = invoke(stypy.reporting.localization.Localization(__file__, 705, 37), getitem___29544, int_29542)
    
    # Processing the call keyword arguments (line 705)
    kwargs_29546 = {}
    # Getting the type of 'generate_lapack_pxd' (line 705)
    generate_lapack_pxd_29541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 17), 'generate_lapack_pxd', False)
    # Calling generate_lapack_pxd(args, kwargs) (line 705)
    generate_lapack_pxd_call_result_29547 = invoke(stypy.reporting.localization.Localization(__file__, 705, 17), generate_lapack_pxd_29541, *[subscript_call_result_29545], **kwargs_29546)
    
    # Assigning a type to the variable 'lapack_pxd' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'lapack_pxd', generate_lapack_pxd_call_result_29547)
    
    # Call to open(...): (line 706)
    # Processing the call arguments (line 706)
    # Getting the type of 'lapack_name' (line 706)
    lapack_name_29549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 14), 'lapack_name', False)
    str_29550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 28), 'str', '.pxd')
    # Applying the binary operator '+' (line 706)
    result_add_29551 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 14), '+', lapack_name_29549, str_29550)
    
    str_29552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 36), 'str', 'w')
    # Processing the call keyword arguments (line 706)
    kwargs_29553 = {}
    # Getting the type of 'open' (line 706)
    open_29548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 9), 'open', False)
    # Calling open(args, kwargs) (line 706)
    open_call_result_29554 = invoke(stypy.reporting.localization.Localization(__file__, 706, 9), open_29548, *[result_add_29551, str_29552], **kwargs_29553)
    
    with_29555 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 706, 9), open_call_result_29554, 'with parameter', '__enter__', '__exit__')

    if with_29555:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 706)
        enter___29556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 9), open_call_result_29554, '__enter__')
        with_enter_29557 = invoke(stypy.reporting.localization.Localization(__file__, 706, 9), enter___29556)
        # Assigning a type to the variable 'f' (line 706)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 9), 'f', with_enter_29557)
        
        # Call to write(...): (line 707)
        # Processing the call arguments (line 707)
        # Getting the type of 'pyxcomment' (line 707)
        pyxcomment_29560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 16), 'pyxcomment', False)
        # Processing the call keyword arguments (line 707)
        kwargs_29561 = {}
        # Getting the type of 'f' (line 707)
        f_29558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 707)
        write_29559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 707, 8), f_29558, 'write')
        # Calling write(args, kwargs) (line 707)
        write_call_result_29562 = invoke(stypy.reporting.localization.Localization(__file__, 707, 8), write_29559, *[pyxcomment_29560], **kwargs_29561)
        
        
        # Call to write(...): (line 708)
        # Processing the call arguments (line 708)
        # Getting the type of 'lapack_pxd' (line 708)
        lapack_pxd_29565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 16), 'lapack_pxd', False)
        # Processing the call keyword arguments (line 708)
        kwargs_29566 = {}
        # Getting the type of 'f' (line 708)
        f_29563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 708)
        write_29564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 708, 8), f_29563, 'write')
        # Calling write(args, kwargs) (line 708)
        write_call_result_29567 = invoke(stypy.reporting.localization.Localization(__file__, 708, 8), write_29564, *[lapack_pxd_29565], **kwargs_29566)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 706)
        exit___29568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 9), open_call_result_29554, '__exit__')
        with_exit_29569 = invoke(stypy.reporting.localization.Localization(__file__, 706, 9), exit___29568, None, None, None)

    
    # Assigning a Call to a Name (line 709):
    
    # Assigning a Call to a Name (line 709):
    
    # Call to generate_fortran(...): (line 709)
    # Processing the call arguments (line 709)
    
    # Obtaining the type of the subscript
    int_29571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 50), 'int')
    # Getting the type of 'lapack_sigs' (line 709)
    lapack_sigs_29572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 38), 'lapack_sigs', False)
    # Obtaining the member '__getitem__' of a type (line 709)
    getitem___29573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 709, 38), lapack_sigs_29572, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 709)
    subscript_call_result_29574 = invoke(stypy.reporting.localization.Localization(__file__, 709, 38), getitem___29573, int_29571)
    
    # Processing the call keyword arguments (line 709)
    kwargs_29575 = {}
    # Getting the type of 'generate_fortran' (line 709)
    generate_fortran_29570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 21), 'generate_fortran', False)
    # Calling generate_fortran(args, kwargs) (line 709)
    generate_fortran_call_result_29576 = invoke(stypy.reporting.localization.Localization(__file__, 709, 21), generate_fortran_29570, *[subscript_call_result_29574], **kwargs_29575)
    
    # Assigning a type to the variable 'lapack_fortran' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'lapack_fortran', generate_fortran_call_result_29576)
    
    # Call to open(...): (line 710)
    # Processing the call arguments (line 710)
    # Getting the type of 'lapack_fortran_name' (line 710)
    lapack_fortran_name_29578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 14), 'lapack_fortran_name', False)
    str_29579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 35), 'str', 'w')
    # Processing the call keyword arguments (line 710)
    kwargs_29580 = {}
    # Getting the type of 'open' (line 710)
    open_29577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 9), 'open', False)
    # Calling open(args, kwargs) (line 710)
    open_call_result_29581 = invoke(stypy.reporting.localization.Localization(__file__, 710, 9), open_29577, *[lapack_fortran_name_29578, str_29579], **kwargs_29580)
    
    with_29582 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 710, 9), open_call_result_29581, 'with parameter', '__enter__', '__exit__')

    if with_29582:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 710)
        enter___29583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 9), open_call_result_29581, '__enter__')
        with_enter_29584 = invoke(stypy.reporting.localization.Localization(__file__, 710, 9), enter___29583)
        # Assigning a type to the variable 'f' (line 710)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 9), 'f', with_enter_29584)
        
        # Call to write(...): (line 711)
        # Processing the call arguments (line 711)
        # Getting the type of 'fcomment' (line 711)
        fcomment_29587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 16), 'fcomment', False)
        # Processing the call keyword arguments (line 711)
        kwargs_29588 = {}
        # Getting the type of 'f' (line 711)
        f_29585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 711)
        write_29586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 711, 8), f_29585, 'write')
        # Calling write(args, kwargs) (line 711)
        write_call_result_29589 = invoke(stypy.reporting.localization.Localization(__file__, 711, 8), write_29586, *[fcomment_29587], **kwargs_29588)
        
        
        # Call to write(...): (line 712)
        # Processing the call arguments (line 712)
        # Getting the type of 'lapack_fortran' (line 712)
        lapack_fortran_29592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 16), 'lapack_fortran', False)
        # Processing the call keyword arguments (line 712)
        kwargs_29593 = {}
        # Getting the type of 'f' (line 712)
        f_29590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 712)
        write_29591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 8), f_29590, 'write')
        # Calling write(args, kwargs) (line 712)
        write_call_result_29594 = invoke(stypy.reporting.localization.Localization(__file__, 712, 8), write_29591, *[lapack_fortran_29592], **kwargs_29593)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 710)
        exit___29595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 9), open_call_result_29581, '__exit__')
        with_exit_29596 = invoke(stypy.reporting.localization.Localization(__file__, 710, 9), exit___29595, None, None, None)

    
    # Assigning a Call to a Name (line 713):
    
    # Assigning a Call to a Name (line 713):
    
    # Call to generate_c_header(...): (line 713)
    # Getting the type of 'lapack_sigs' (line 713)
    lapack_sigs_29598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 42), 'lapack_sigs', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 713)
    tuple_29599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 713)
    # Adding element type (line 713)
    str_29600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 57), 'str', 'LAPACK')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 57), tuple_29599, str_29600)
    
    # Applying the binary operator '+' (line 713)
    result_add_29601 = python_operator(stypy.reporting.localization.Localization(__file__, 713, 42), '+', lapack_sigs_29598, tuple_29599)
    
    # Processing the call keyword arguments (line 713)
    kwargs_29602 = {}
    # Getting the type of 'generate_c_header' (line 713)
    generate_c_header_29597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 22), 'generate_c_header', False)
    # Calling generate_c_header(args, kwargs) (line 713)
    generate_c_header_call_result_29603 = invoke(stypy.reporting.localization.Localization(__file__, 713, 22), generate_c_header_29597, *[result_add_29601], **kwargs_29602)
    
    # Assigning a type to the variable 'lapack_c_header' (line 713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 4), 'lapack_c_header', generate_c_header_call_result_29603)
    
    # Call to open(...): (line 714)
    # Processing the call arguments (line 714)
    # Getting the type of 'lapack_header_name' (line 714)
    lapack_header_name_29605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 14), 'lapack_header_name', False)
    str_29606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 34), 'str', 'w')
    # Processing the call keyword arguments (line 714)
    kwargs_29607 = {}
    # Getting the type of 'open' (line 714)
    open_29604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 9), 'open', False)
    # Calling open(args, kwargs) (line 714)
    open_call_result_29608 = invoke(stypy.reporting.localization.Localization(__file__, 714, 9), open_29604, *[lapack_header_name_29605, str_29606], **kwargs_29607)
    
    with_29609 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 714, 9), open_call_result_29608, 'with parameter', '__enter__', '__exit__')

    if with_29609:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 714)
        enter___29610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 9), open_call_result_29608, '__enter__')
        with_enter_29611 = invoke(stypy.reporting.localization.Localization(__file__, 714, 9), enter___29610)
        # Assigning a type to the variable 'f' (line 714)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 9), 'f', with_enter_29611)
        
        # Call to write(...): (line 715)
        # Processing the call arguments (line 715)
        # Getting the type of 'ccomment' (line 715)
        ccomment_29614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 16), 'ccomment', False)
        # Processing the call keyword arguments (line 715)
        kwargs_29615 = {}
        # Getting the type of 'f' (line 715)
        f_29612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 715)
        write_29613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 8), f_29612, 'write')
        # Calling write(args, kwargs) (line 715)
        write_call_result_29616 = invoke(stypy.reporting.localization.Localization(__file__, 715, 8), write_29613, *[ccomment_29614], **kwargs_29615)
        
        
        # Call to write(...): (line 716)
        # Processing the call arguments (line 716)
        # Getting the type of 'lapack_c_header' (line 716)
        lapack_c_header_29619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 16), 'lapack_c_header', False)
        # Processing the call keyword arguments (line 716)
        kwargs_29620 = {}
        # Getting the type of 'f' (line 716)
        f_29617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 716, 8), 'f', False)
        # Obtaining the member 'write' of a type (line 716)
        write_29618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 716, 8), f_29617, 'write')
        # Calling write(args, kwargs) (line 716)
        write_call_result_29621 = invoke(stypy.reporting.localization.Localization(__file__, 716, 8), write_29618, *[lapack_c_header_29619], **kwargs_29620)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 714)
        exit___29622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 714, 9), open_call_result_29608, '__exit__')
        with_exit_29623 = invoke(stypy.reporting.localization.Localization(__file__, 714, 9), exit___29622, None, None, None)

    
    # ################# End of 'make_all(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_all' in the type store
    # Getting the type of 'stypy_return_type' (line 647)
    stypy_return_type_29624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29624)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_all'
    return stypy_return_type_29624

# Assigning a type to the variable 'make_all' (line 647)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 0), 'make_all', make_all)

if (__name__ == '__main__'):
    
    # Call to make_all(...): (line 719)
    # Processing the call keyword arguments (line 719)
    kwargs_29626 = {}
    # Getting the type of 'make_all' (line 719)
    make_all_29625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 719, 4), 'make_all', False)
    # Calling make_all(args, kwargs) (line 719)
    make_all_call_result_29627 = invoke(stypy.reporting.localization.Localization(__file__, 719, 4), make_all_29625, *[], **kwargs_29626)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
